"""Apply a declared weight edit to an rsl_rl actor .pt and save a new checkpoint.

Edit grammar (one spec per --edit flag, pipe-separated fields):
    <tensor>|<op>|<slice>|<scalar>

Supported tensors:
    mlp.{0,2,4,6}.weight  mlp.{0,2,4,6}.bias

Supported ops:
    scale    multiply the slice by <scalar>
    zero     set the slice to zero   (<scalar> ignored, pass 0)
    set      set the slice to <scalar>

Slice syntax:
    :                      — every entry
    r=<list>               — rows only (comma-separated, no spaces)
    c=<list>               — columns only
    r=<list>,c=<list>      — tensor-product of rows × cols
    r=<list>,c=<a>:<b>     — tensor-product of rows × contiguous col range
    r=<a>:<b>              — contiguous row range
    c=<a>:<b>              — contiguous col range
    r=<a>:<b>,c=<a>:<b>    — contiguous row × col range

Index ranges are python-style (half-open) and apply to the tensor's first two dims.
For 1-d tensors (biases), use r=... only.

The edit summary (tensors touched, parameters changed) is stored in
    ckpt["infos"]["weight_edit"]
so the write is self-documenting.
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import torch


ALLOWED_TENSORS = {
    f"mlp.{i}.{k}" for i in (0, 2, 4, 6) for k in ("weight", "bias")
}


def _parse_idx_list(s: str) -> list[int]:
    if ":" in s:
        a, b = s.split(":")
        a = int(a) if a else None
        b = int(b) if b else None
        return list(range(a if a is not None else 0, b if b is not None else -1))
    return [int(x) for x in s.split(",") if x]


def _slice_mask(tensor: torch.Tensor, spec: str) -> tuple[torch.Tensor, int]:
    """Return a boolean mask of the same shape as ``tensor`` marking the
    entries to touch, and the integer count of marked entries."""
    mask = torch.zeros_like(tensor, dtype=torch.bool)
    if spec == ":":
        mask[:] = True
    else:
        rows = cols = None
        for part in spec.split(","):
            if part.startswith("r="):
                rows = _parse_idx_list(part[2:])
            elif part.startswith("c="):
                cols = _parse_idx_list(part[2:])
            else:
                raise ValueError(f"bad slice part {part!r}")
        if tensor.ndim == 2:
            rows = rows if rows is not None else list(range(tensor.shape[0]))
            cols = cols if cols is not None else list(range(tensor.shape[1]))
            idx_r = torch.tensor(rows, dtype=torch.long)
            idx_c = torch.tensor(cols, dtype=torch.long)
            mask[idx_r.unsqueeze(1), idx_c.unsqueeze(0)] = True
        else:
            rows = rows if rows is not None else list(range(tensor.shape[0]))
            idx_r = torch.tensor(rows, dtype=torch.long)
            mask[idx_r] = True
    return mask, int(mask.sum().item())


def apply_one(actor: dict, edit: str) -> dict:
    tensor_name, op, slice_spec, scalar_s = edit.split("|", 3)
    if tensor_name not in ALLOWED_TENSORS:
        raise ValueError(f"bad tensor {tensor_name!r}")
    if op not in ("scale", "zero", "set"):
        raise ValueError(f"bad op {op!r}")
    scalar = float(scalar_s)
    t = actor[tensor_name]
    mask, n = _slice_mask(t, slice_spec)
    if op == "scale":
        t[mask] = t[mask] * scalar
    elif op == "zero":
        t[mask] = 0.0
    elif op == "set":
        t[mask] = scalar
    return {"tensor": tensor_name, "op": op, "slice": slice_spec,
            "scalar": scalar, "params_touched": n}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint", type=Path)
    ap.add_argument("output",     type=Path)
    ap.add_argument("--edit", action="append", required=True,
                    help="<tensor>|<op>|<slice>|<scalar> ; may repeat")
    ap.add_argument("--note", default="", help="free text stored in infos.weight_edit.note")
    args = ap.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    actor = ckpt["actor_state_dict"]
    total_params = sum(v.numel() for k, v in actor.items() if k.startswith("mlp."))
    applied = []
    for e in args.edit:
        applied.append(apply_one(actor, e))

    ckpt.setdefault("infos", {})["weight_edit"] = {
        "source":          str(args.checkpoint),
        "edits":           applied,
        "total_mlp_params": total_params,
        "params_touched":  sum(e["params_touched"] for e in applied),
        "note":            args.note,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, args.output)
    digest = hashlib.sha256(args.output.read_bytes()).hexdigest()[:16]
    print(f"WROTE {args.output}  sha256={digest}")
    print(f"  total MLP params    = {total_params:,}")
    tot = sum(e['params_touched'] for e in applied)
    print(f"  params touched      = {tot:,}  ({100*tot/total_params:.3f} %)")
    for e in applied:
        print(f"  {e['tensor']:16s} {e['op']:5s} {e['slice']:30s} "
              f"scalar={e['scalar']:<8g} ({e['params_touched']} params)")


if __name__ == "__main__":
    main()
