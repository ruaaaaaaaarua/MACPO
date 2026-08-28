#!/usr/bin/env python3
"""Export actor-only weights from Orbax MAPPO checkpoints to portable .npz files."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np
from flax.core import freeze, unfreeze
from orbax.checkpoint import checkpointer
from orbax.checkpoint.pytree_checkpoint_handler import PyTreeCheckpointHandler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _is_actor_key(key_path: str) -> bool:
    key_lower = key_path.lower()
    if "critic" in key_lower:
        return False
    actor_markers = (
        "actor_",
        "actor_hypernet",
        "agent_embeddings",
        "log_std",
    )
    return any(m in key_lower for m in actor_markers)


def _flatten_params(params: Any, prefix: str = "") -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}
    if isinstance(params, dict):
        for k, v in params.items():
            path = f"{prefix}/{k}" if prefix else str(k)
            out.update(_flatten_params(v, path))
    else:
        out[prefix] = np.asarray(params)
    return out


def export_orbax_actor(checkpoint_dir: Path, out_npz: Path, meta: dict) -> None:
    ckpt = checkpointer.Checkpointer(
        PyTreeCheckpointHandler(aggregate_filename="checkpoints")
    )
    params = ckpt.restore(str(checkpoint_dir))
    flat = _flatten_params(unfreeze(params))
    actor_flat = {k: v for k, v in flat.items() if _is_actor_key(k)}
    if not actor_flat:
        raise RuntimeError(f"No actor keys found in {checkpoint_dir}")

    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_npz, **actor_flat)
    meta = {
        **meta,
        "format": "npz_actor_only",
        "num_actor_tensors": len(actor_flat),
        "actor_keys": sorted(actor_flat.keys()),
        "source_checkpoint": str(checkpoint_dir),
    }
    meta_path = out_npz.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Exported {len(actor_flat)} actor tensors -> {out_npz}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--out-npz", type=Path, required=True)
    parser.add_argument("--algorithm", type=str, required=True)
    parser.add_argument("--seed", type=int, default=30)
    args = parser.parse_args()

    step_match = re.search(r"(\d+)_steps", args.checkpoint_dir.name)
    global_step = int(step_match.group(1)) if step_match else None
    export_orbax_actor(
        args.checkpoint_dir,
        args.out_npz,
        {
            "algorithm": args.algorithm,
            "seed": args.seed,
            "global_step": global_step,
        },
    )


if __name__ == "__main__":
    main()
