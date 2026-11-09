"""Diagnose alpha_stand's face-up freeze attractor from a baseline rj_fu log.

What this does, in order:

1. Loads the baseline rj_fu log (output of rj_fu_eval.py).  The log stores per
   tick: normalized actor obs, actions, trunk_z, gravity_z, tilt, joint_pos.

2. Partitions trials into STUCK vs RECOVERED (already in the log).

3. From STUCK trials: picks the "attractor" window per trial — the tail 2 s
   (ticks t >= 11.0 s). From RECOVERED trials: picks an early-high-tilt window
   (t in [0.5, 2.0] s where tilt > 90°) as the matched "trying to rise"
   control. Both are at similar tilt but very different behavior.

4. Runs the actor forward on both populations, recording intermediate
   activations (post-ELU) at mlp.0, mlp.2, mlp.4, and the pre-activation action
   vector.  Computes:
     - per-unit mean activation in each population
     - per-unit (activation, action-L1-norm) correlation
     - per-unit action-saliency = |output projection of unit onto predicted
       action deviation from zero|, summed per unit
   ...and ranks units by how much they preferentially engage in the STUCK
   population AND drive the action toward ≈ 0 ("stay at HOME").

5. Sanity check: the first-layer fan-in column mass of the top STUCK-preferring
   units in mlp.2 and mlp.4 should land predominantly in the joint-obs cols of
   mlp.0.weight (cols 6..47), consistent with the published v8-era claim that
   the attractor lives in the teacher's response to the joint config.

6. Prints a report; writes a .pt with the ranking.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch


# Obs channel layout, from PLAN_teacher_fix.md section 1.5:
OBS_LAYOUT = {
    "base_ang_vel":     (0,  3),
    "projected_gravity":(3,  6),
    "joint_pos_rel":    (6, 20),
    "joint_vel":        (20, 34),
    "last_action":      (34, 48),
    "twist":            (48, 51),
    "head_cmd":         (51, 55),
    "body_cmd":         (55, 61),
}


def build_actor(actor_state: dict) -> torch.nn.Sequential:
    """Rebuild the 61→512→256→128→14 MLP from the state dict, with the obs
    normalizer in front (so we take raw 61-d obs and pass through)."""
    # Linear modules.
    def lin(in_f, out_f, w, b):
        m = torch.nn.Linear(in_f, out_f)
        with torch.no_grad():
            m.weight.copy_(w)
            m.bias.copy_(b)
        return m

    net = torch.nn.Sequential(
        lin(61, 512, actor_state["mlp.0.weight"], actor_state["mlp.0.bias"]),
        torch.nn.ELU(),
        lin(512, 256, actor_state["mlp.2.weight"], actor_state["mlp.2.bias"]),
        torch.nn.ELU(),
        lin(256, 128, actor_state["mlp.4.weight"], actor_state["mlp.4.bias"]),
        torch.nn.ELU(),
        lin(128, 14, actor_state["mlp.6.weight"], actor_state["mlp.6.bias"]),
    )
    return net


def forward_record(net: torch.nn.Sequential, obs_norm: torch.Tensor) -> dict:
    """Run the actor on already-normalized observations, returning the output
    and the post-ELU activations at each hidden layer."""
    h = obs_norm
    l0 = net[0](h)
    a0 = net[1](l0)
    l2 = net[2](a0)
    a2 = net[3](l2)
    l4 = net[4](a2)
    a4 = net[5](l4)
    out = net[6](a4)
    return {"h0": a0, "h1": a2, "h2": a4, "out": out, "pre0": l0, "pre1": l2, "pre2": l4}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("baseline_log", type=Path,
                    help="Output of rj_fu_eval.py — baseline run's .pt log")
    ap.add_argument("checkpoint", type=Path,
                    help="alpha_stand model_9750.pt")
    ap.add_argument("--out", type=Path, required=True,
                    help="Where to write the diagnosis dict")
    args = ap.parse_args()

    log = torch.load(args.baseline_log, map_location="cpu", weights_only=False)
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    actor = ckpt["actor_state_dict"]
    net = build_actor(actor).eval()

    obs = log["obs"]            # (T, N, 61) — already normalized (actor's view)
    action = log["action"]      # (T, N, 14)
    trunk_z = log["trunk_z"]    # (T, N)
    grav_z = log["grav_z"]
    tilt = log["tilt_rad"]
    stuck = log["stuck_mask"]   # (N,)
    recovered = log["recovered_mask"]
    T, N = obs.shape[:2]
    dt = log["dt"]

    print(f"Baseline: {int(stuck.sum())}/{N} stuck   recovered={int(recovered.sum())}")

    # Window selection.
    t = torch.arange(T) * dt
    stuck_window = (t >= 11.0).view(-1, 1) & stuck.view(1, -1)       # (T, N)
    rec_window = (t >= 0.5) & (t <= 2.0)
    high_tilt = tilt > torch.tensor(3.14159 / 2)                     # > 90°
    rec_window = rec_window.view(-1, 1) & high_tilt & recovered.view(1, -1)

    obs_stuck = obs[stuck_window]                                    # (M_s, 61)
    obs_rec   = obs[rec_window]                                      # (M_r, 61)
    act_stuck = action[stuck_window]                                 # (M_s, 14)
    act_rec   = action[rec_window]                                   # (M_r, 14)
    print(f"  stuck-window samples:     {obs_stuck.shape[0]}")
    print(f"  rising-matched samples:   {obs_rec.shape[0]}")

    if obs_stuck.shape[0] == 0:
        print("No stuck samples — abort.")
        return

    with torch.inference_mode():
        rec_s = forward_record(net, obs_stuck)
        rec_r = forward_record(net, obs_rec)

    # Per-unit mean activation in each population.
    mu_s = {k: v.mean(dim=0) for k, v in rec_s.items()}
    mu_r = {k: v.mean(dim=0) for k, v in rec_r.items()}

    # Attribution to "near-zero action" output: per unit k in layer L with
    # activation a_k, its *linear* contribution to the final 14-d pre-activation
    # action via downstream weights is W_down[:, k] (effective weight matrix
    # from layer L to the output).  A unit is a "freeze contributor" if its
    # fan-out direction, times its STUCK-minus-RECOVERED differential activation,
    # pulls the output DOWN in magnitude relative to its magnitude in rising.
    #
    # Effective output weights from each hidden layer:
    W6 = actor["mlp.6.weight"]       # (14, 128)
    W4 = actor["mlp.4.weight"]       # (128, 256)
    W2 = actor["mlp.2.weight"]       # (256, 512)
    Wh2_to_out = W6                                     # (14, 128)
    Wh1_to_out = W6 @ W4                                 # (14, 256)   (ignores ELU; linearized)
    Wh0_to_out = W6 @ W4 @ W2                            # (14, 512)

    # STUCK-minus-RECOVERED diff of mean activation per layer.
    d0 = mu_s["h0"] - mu_r["h0"]        # (512,)
    d1 = mu_s["h1"] - mu_r["h1"]        # (256,)
    d2 = mu_s["h2"] - mu_r["h2"]        # (128,)

    # Per-unit "freeze score": magnitude of the output change the unit's
    # activation differential would cause in the LINEARIZED network — units with
    # large |d * ||W_down[:, k]||_2| are the ones whose stuck-vs-rising
    # activation flip accounts for the most output shift.
    score0 = d0.abs() * Wh0_to_out.norm(dim=0)          # (512,)
    score1 = d1.abs() * Wh1_to_out.norm(dim=0)          # (256,)
    score2 = d2.abs() * Wh2_to_out.norm(dim=0)          # (128,)

    def topk(score, k=20):
        v, i = torch.topk(score, k)
        return [(int(ii), float(vv)) for ii, vv in zip(i, v)]

    print("\nTop freeze contributors (unit_idx, score):")
    print(f"  mlp.0 (post-ELU, 512 units): {topk(score0, 15)}")
    print(f"  mlp.2 (post-ELU, 256 units): {topk(score1, 15)}")
    print(f"  mlp.4 (post-ELU, 128 units): {topk(score2, 15)}")

    # Verify top units in later layers draw their fan-in predominantly from the
    # joint-obs columns (cols 6..47 of mlp.0.weight), which is the pathway the
    # runtime-mask fix lives in.  For a mlp.2 unit u, its "obs-side" fan-in
    # mass via the chain mlp.2[u] · ELU · mlp.0 is approximately
    #     | W2[u, :] @ W0 | per obs column j.
    W0 = actor["mlp.0.weight"]       # (512, 61)
    # Linearized obs → h1 effective matrix:
    W_obs_to_h1 = W2 @ W0            # (256, 61)

    print("\nObs-column fan-in share for top mlp.2 freeze units:")
    print(f"  (cols 6..47 = joint_pos+joint_vel+last_action — the runtime-mask block)")
    top_h1 = [i for i, _ in topk(score1, 10)]
    hdr = "  unit  jointcols-share  gravcols-share  twistcols-share"
    print(hdr)
    for u in top_h1:
        row = W_obs_to_h1[u].abs()
        tot = row.sum().clamp_min(1e-9)
        joint_share = row[6:48].sum() / tot
        grav_share  = row[3:6].sum()  / tot
        twist_share = row[48:51].sum() / tot
        print(f"   {u:4d}       {joint_share.item():.3f}            "
              f"{grav_share.item():.3f}           {twist_share.item():.3f}")

    # Breakdown by SUB-block of the joint-obs block:
    print("\n  … same top units, broken down within the joint-obs block:")
    print("  unit  hip_roll_pos(c=7,12)  head_yaw_pos(c=13)  joint_pos(c=6-19)  joint_vel(c=20-33)  last_action(c=34-47)")
    for u in top_h1:
        row = W_obs_to_h1[u].abs()
        tot = row.sum().clamp_min(1e-9)
        hip_roll_pos = (row[7] + row[12]) / tot
        head_yaw_pos = row[13] / tot
        jp = row[6:20].sum() / tot
        jv = row[20:34].sum() / tot
        la = row[34:48].sum() / tot
        print(f"   {u:4d}       {hip_roll_pos.item():.3f}              "
              f"{head_yaw_pos.item():.3f}            {jp.item():.3f}             "
              f"{jv.item():.3f}            {la.item():.3f}")

    # Sanity: "near-zero action" claim. Mean |action| in stuck vs rising.
    print(f"\n  mean |action|  stuck  = {act_stuck.abs().mean().item():.4f}")
    print(f"  mean |action|  rising = {act_rec.abs().mean().item():.4f}")

    out = {
        "n_stuck":       int(stuck.sum()),
        "n_recovered":   int(recovered.sum()),
        "stuck_samples": int(obs_stuck.shape[0]),
        "rising_samples": int(obs_rec.shape[0]),
        "score_h0":      score0,
        "score_h1":      score1,
        "score_h2":      score2,
        "d0":            d0,
        "d1":            d1,
        "d2":            d2,
        "mu_stuck":      mu_s,
        "mu_recovered":  mu_r,
        "W_obs_to_h1":   W_obs_to_h1,
        "top_h1":        top_h1,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, args.out)
    print(f"\nDiagnosis written → {args.out}")


if __name__ == "__main__":
    main()
