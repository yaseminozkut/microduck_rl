"""rj_fu eval harness: load an alpha_stand .pt, run 256 envs in
Mjlab-StandUp-Flat-Backlash-MicroDuck with face-up random-joint spawns, 13 s
episodes at seed 42, classify recovered vs stuck.

Spawn (`rj_fu`): the standup env's set_ground_state event is REPLACED at cfg
build time with maybe_set_random_prone_orientation configured for 100 % face-up
with 100 % random-joint resampling at 0.9 of range. z in [0.05, 0.09].

Classifier: at any tick with t_s >= 11.0, recovered iff
    trunk_z > 0.095 AND gravity_z < -0.9 .

Usage:
    uv run python claude_experiments/alpha_stand_fix/rj_fu_eval.py \
        <checkpoint.pt> --out <log.pt>
"""
from __future__ import annotations

import argparse
import math
from dataclasses import asdict
from pathlib import Path

import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers import EventTermCfg
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.torch import configure_torch_backends
from rsl_rl.runners import OnPolicyRunner

import mjlab_microduck.tasks  # noqa: F401
from mjlab_microduck.tasks import mdp as microduck_mdp


TASK_ID = "Mjlab-StandUp-Flat-Backlash-MicroDuck"


def make_rj_fu_env_cfg(seed: int, num_envs: int, duration_s: float):
    """Build the env cfg with the rj_fu spawn event replacing the stock
    standup `set_ground_state` event."""
    env_cfg = load_env_cfg(TASK_ID, play=True)
    env_cfg.seed = seed
    env_cfg.scene.num_envs = num_envs
    env_cfg.episode_length_s = duration_s + 2.0  # buffer so no reset mid-rollout

    # Pre-instantiation cfg writes work (the manager-deepcopy footgun applies
    # only to post-instantiation cfg writes).  Replace the standup event.
    env_cfg.events.pop("set_ground_state", None)
    env_cfg.events["rj_fu_init"] = EventTermCfg(
        func=microduck_mdp.maybe_set_random_prone_orientation,
        mode="reset",
        params={
            "prone_prob":        1.0,   # every env prone
            "face_down_prob":    0.0,   # among prone: face-up only
            "side_prob":         0.0,   # no on-side
            "crouch_prob":       0.0,
            "sit_prob":          0.0,
            "so3_prob":          0.0,
            "joint_random_prob": 1.0,   # every env random-joint
            "joint_range_frac":  0.9,
            "prone_z_min":       0.05,
            "prone_z_max":       0.09,
            "joint_random_extra_z": 0.06,
        },
    )

    # Disable the ground_state_mix curriculum — it mutates the deleted event.
    env_cfg.curriculum.pop("ground_state_mix", None)
    # In play mode fell_over is already removed (see make_microduck_standup_env_cfg
    # — in play, terminations have no fall term; nan_state is the only one).
    return env_cfg


def run_policy(checkpoint: Path, seed: int = 42, num_envs: int = 256,
               duration_s: float = 13.0, device: str = "cpu") -> dict:
    """Roll out ``checkpoint`` from rj_fu starts and record per-tick physics."""
    configure_torch_backends()
    env_cfg = make_rj_fu_env_cfg(seed=seed, num_envs=num_envs, duration_s=duration_s)
    agent_cfg = load_rl_cfg(TASK_ID)

    raw = ManagerBasedRlEnv(cfg=env_cfg, device=device)
    env = RslRlVecEnvWrapper(raw, clip_actions=agent_cfg.clip_actions)
    runner_cls = load_runner_cls(TASK_ID) or OnPolicyRunner
    runner = runner_cls(env, asdict(agent_cfg), device=device)
    runner.load(str(checkpoint), map_location=device)
    policy = runner.get_inference_policy(device=device)

    robot = raw.scene["robot"]
    dt = raw.step_dt
    n_steps = int(math.ceil(duration_s / dt))

    # Pre-allocate per-tick logs.
    trunk_z_hist   = torch.zeros(n_steps, num_envs)
    grav_z_hist    = torch.zeros(n_steps, num_envs)
    tilt_rad_hist  = torch.zeros(n_steps, num_envs)
    # For failure diagnosis later we'll log full state for a subset of envs.
    joint_pos_hist = torch.zeros(n_steps, num_envs, 14)
    action_hist    = torch.zeros(n_steps, num_envs, 14)
    obs_hist       = torch.zeros(n_steps, num_envs, 61)

    # Servo joint ids (identity on plain models; passive-aware on backlash).
    servo_ids = torch.as_tensor(
        microduck_mdp._servo_joint_ids(raw, robot),
        device=device, dtype=torch.long,
    )

    actor_obs_cur = env.get_observations()
    # actor_obs_cur is a TensorDict with the actor's obs groups; flatten for logs.
    actor_mod = runner.alg.actor
    obs_groups = list(actor_mod.obs_groups)

    def _flat(td):
        return torch.cat([td[g] for g in obs_groups], dim=-1)

    for step in range(n_steps):
        trunk_z_hist[step]   = robot.data.root_link_pos_w[:, 2].cpu()
        pg = robot.data.projected_gravity_b
        grav_z_hist[step]    = pg[:, 2].cpu()
        cos_tilt = (-pg[:, 2]).clamp(-1.0, 1.0)
        tilt_rad_hist[step]  = torch.acos(cos_tilt).cpu()
        joint_pos_hist[step] = robot.data.joint_pos[:, servo_ids].cpu()

        with torch.inference_mode():
            actions = policy(actor_obs_cur)
        action_hist[step] = actions.cpu()
        obs_hist[step]    = _flat(actor_obs_cur).cpu()

        actor_obs_cur, _, _, _ = env.step(actions)

    # Classify.
    t_axis = torch.arange(n_steps) * dt
    classify_mask = t_axis >= 11.0           # indices of ticks at/after 11 s
    tz = trunk_z_hist[classify_mask]         # (K, N)
    gz = grav_z_hist[classify_mask]
    recovered_any = ((tz > 0.095) & (gz < -0.9)).any(dim=0)   # (N,) per-env
    stuck = ~recovered_any
    n = int(num_envs)
    out = {
        "checkpoint":       str(checkpoint),
        "seed":             seed,
        "num_envs":         num_envs,
        "duration_s":       duration_s,
        "dt":               dt,
        "n_steps":          n_steps,
        "n_recovered":      int(recovered_any.sum()),
        "n_stuck":          int(stuck.sum()),
        "frac_stuck":       float(stuck.float().mean()),
        "trunk_z":          trunk_z_hist,
        "grav_z":           grav_z_hist,
        "tilt_rad":         tilt_rad_hist,
        "joint_pos":        joint_pos_hist,
        "action":           action_hist,
        "obs":              obs_hist,
        "stuck_mask":       stuck,
        "recovered_mask":   recovered_any,
    }
    raw.close()
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint", type=Path)
    ap.add_argument("--out",      type=Path, required=True)
    ap.add_argument("--seed",     type=int, default=42)
    ap.add_argument("--num-envs", type=int, default=256)
    ap.add_argument("--duration-s", type=float, default=13.0)
    ap.add_argument("--device",   default="cpu")
    args = ap.parse_args()

    result = run_policy(args.checkpoint, args.seed, args.num_envs,
                        args.duration_s, args.device)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(result, args.out)
    print(f"\n=== rj_fu eval — {args.checkpoint.name} ===")
    print(f"  seed={result['seed']}  num_envs={result['num_envs']}  "
          f"duration={result['duration_s']}s  dt={result['dt']:.4f}s")
    print(f"  recovered = {result['n_recovered']:4d}/{result['num_envs']}")
    print(f"  stuck     = {result['n_stuck']:4d}/{result['num_envs']}  "
          f"({100*result['frac_stuck']:.1f} %)")
    print(f"  log → {args.out}")


if __name__ == "__main__":
    main()
