"""Audit the baseline rj_fu log's initial tick: is the spawn actually
face-up with randomized joints?

Expected on rj_fu:
  tilt  @ t=0:   about pi (face-up) — lots of envs > 2.5 rad
  grav_z @ t=0:  roughly +1 (back on floor)
  trunk_z @ t=0: in [0.05, 0.09] + small clearance
  joint_pos spread: high variance across envs (random) — std per joint
                    should be a non-trivial fraction of each joint's range
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("log", type=Path)
    args = ap.parse_args()
    d = torch.load(args.log, map_location="cpu", weights_only=False)

    t0_tilt = d["tilt_rad"][0]
    t0_gz   = d["grav_z"][0]
    t0_z    = d["trunk_z"][0]
    t0_jp   = d["joint_pos"][0]  # (N, 14)

    print(f"log: {args.log}")
    print(f"  N envs = {t0_tilt.shape[0]}")
    print()
    print(f"t=0 tilt (rad):   min={t0_tilt.min():.3f}  "
          f"median={t0_tilt.median():.3f}  max={t0_tilt.max():.3f}  "
          f"share > 2.5 rad = {(t0_tilt > 2.5).float().mean().item():.2%}")
    print(f"t=0 gravity_z:    min={t0_gz.min():+.3f}  "
          f"median={t0_gz.median():+.3f}  max={t0_gz.max():+.3f}")
    print(f"t=0 trunk_z (m):  min={t0_z.min():.3f}  "
          f"median={t0_z.median():.3f}  max={t0_z.max():.3f}")
    print()
    print(f"t=0 joint_pos per-joint (across {t0_jp.shape[0]} envs):")
    names = [
        "left_hip_yaw", "left_hip_roll", "left_hip_pitch", "left_knee", "left_ankle",
        "neck_pitch", "head_pitch", "head_yaw", "head_roll",
        "right_hip_yaw", "right_hip_roll", "right_hip_pitch", "right_knee", "right_ankle",
    ]
    for i, name in enumerate(names):
        col = t0_jp[:, i]
        print(f"  {i:2d} {name:18s}  min={col.min():+.3f}  "
              f"med={col.median():+.3f}  max={col.max():+.3f}  "
              f"std={col.std():.3f}")

    # Per-env attractor scoring.  The published fingerprint:
    #   tilt ~1.68 rad (96°),  head_yaw < -1.3, left_hip_roll < -0.3
    # Count trials whose INITIAL state falls in that basin.
    face_up = t0_tilt > 2.5
    head_yaw_neg = t0_jp[:, 7] < -1.3
    lhr_pinned = t0_jp[:, 1] < -0.3
    print()
    print(f"face-up (tilt>2.5 rad):         {face_up.sum().item()}/{t0_tilt.shape[0]}")
    print(f"  of which head_yaw < -1.3:     {(face_up & head_yaw_neg).sum().item()}")
    print(f"  of which left_hip_roll < -0.3:{(face_up & lhr_pinned).sum().item()}")
    print(f"  of which BOTH (attractor hit):{(face_up & head_yaw_neg & lhr_pinned).sum().item()}")

    # End-state sanity: how upright is each env at the LAST tick?
    T = d["trunk_z"].shape[0]
    end_z  = d["trunk_z"][T - 1]
    end_gz = d["grav_z"][T - 1]
    end_upright = (end_z > 0.095) & (end_gz < -0.9)
    print()
    print(f"END tick trunk_z:  min={end_z.min():.3f}  med={end_z.median():.3f}  max={end_z.max():.3f}")
    print(f"END tick grav_z:   min={end_gz.min():+.3f}  med={end_gz.median():+.3f}  max={end_gz.max():+.3f}")
    print(f"END tick upright (same classifier): {end_upright.sum().item()}/{end_z.shape[0]}")


if __name__ == "__main__":
    main()
