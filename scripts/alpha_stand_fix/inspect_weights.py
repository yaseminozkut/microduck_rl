"""Load alpha_stand .pt and verify shapes / dump normalizer.

Confirms the plan's claims against the actual tensors in the checkpoint:
- obs_normalizer._mean (1, 61), ._std (1, 61)
- mlp.{0,2,4,6}.{weight,bias} with shapes (512,61) (512,) (256,512) (256,)
  (128,256) (128,) (14,128) (14,)
- Total actor params = 197,896.
"""
from __future__ import annotations

from pathlib import Path

import torch

CKPT = Path(
    "/Users/yaseminozkut/.cache/huggingface/hub/models--pollen-robotics--microduck-training/"
    "snapshots/64d28ce092253c509b51031ec7ccb53790a00d0d/"
    "bundles/microduck_stand/69u48n8l_9750/model_9750.pt"
)

EXPECTED = {
    "obs_normalizer._mean": (1, 61),
    "obs_normalizer._std": (1, 61),
    "mlp.0.weight": (512, 61),
    "mlp.0.bias": (512,),
    "mlp.2.weight": (256, 512),
    "mlp.2.bias": (256,),
    "mlp.4.weight": (128, 256),
    "mlp.4.bias": (128,),
    "mlp.6.weight": (14, 128),
    "mlp.6.bias": (14,),
}


def main() -> None:
    assert CKPT.exists(), CKPT
    ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
    print("top-level keys:", list(ckpt.keys()))
    if "iter" in ckpt:
        print(f"  iter = {ckpt['iter']}")
    actor = ckpt["actor_state_dict"]
    print(f"actor_state_dict keys ({len(actor)}):")
    total = 0
    for k, v in actor.items():
        tag = "OK" if actor[k].shape == EXPECTED.get(k, actor[k].shape) else "MISMATCH"
        exp = EXPECTED.get(k, "?")
        n = v.numel()
        total += n
        print(f"  {k:30s} {str(tuple(v.shape)):18s} {n:>8d}  expected={exp}  [{tag}]")
    print(f"TOTAL actor params: {total:,}  (plan says 197,896)")
    mean = actor["obs_normalizer._mean"].reshape(-1)
    std = actor["obs_normalizer._std"].reshape(-1)
    # Plan quotes "integers × 1e-3"; print rounded to verify index 5 (gravity_z) ≈ -0.949.
    print("\nobs_normalizer._mean (×1e3, rounded):")
    ints = (mean * 1000).round().long().tolist()
    for i in range(0, 61, 15):
        print(" ", " ".join(f"{v:>5d}" for v in ints[i : i + 15]))
    print(f"  mean[5] (gravity_z): {mean[5].item():+.4f}  (plan: ~-0.949)")
    print("\nobs_normalizer._std (×1e3, rounded):")
    ints = (std * 1000).round().long().tolist()
    for i in range(0, 61, 15):
        print(" ", " ".join(f"{v:>5d}" for v in ints[i : i + 15]))

    # Hold on: the plan specifies the twist-command normalizer std is tiny (0.005)
    # because the stand teacher saw ~zero twist in training. Print std[48:51].
    print(f"\nstd[48:51] twist (vx,vy,vyaw): {std[48:51].tolist()}  "
          "(plan/distill.py says ~0.005 for twist — teacher never saw it)")


if __name__ == "__main__":
    main()
