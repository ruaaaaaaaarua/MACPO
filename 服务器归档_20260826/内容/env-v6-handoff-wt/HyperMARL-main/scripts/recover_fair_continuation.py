#!/usr/bin/env python3
"""Resume the fair pipeline after the CUDA RNG checkpoint compatibility fix."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.run_fair_pipeline_continuation import (
    _resume_matd3_spec,
    _run_parallel,
    _stas_quality,
    _stas_spec,
)
from scripts.run_fair_stas_pipeline import DEFAULT_OUTPUT, _write_json


def run(root: Path) -> None:
    stage1 = json.loads((root / "selection_stage1.json").read_text())
    selected_mappo = stage1["selected_mappo"]
    width = 128 if selected_mappo.endswith("128") else 256
    activation = "tanh" if width == 128 else "relu"

    stas_10k = {
        "stas_causal": _stas_spec(root, "stas_causal", False, 10000, width, activation),
        "stas_bidirectional": _stas_spec(
            root, "stas_bidirectional", True, 10000, width, activation
        ),
    }
    qualities = {name: _stas_quality(spec) for name, spec in stas_10k.items()}
    eligible = [name for name, quality in qualities.items() if quality["eligible"]]
    pool = eligible or list(qualities)
    selected_stas = max(pool, key=lambda name: qualities[name]["score"])
    _write_json(
        root / "selection_stage2.json",
        {
            "selected_stas": selected_stas,
            "qualities": qualities,
            "quality_gate_passed": bool(eligible),
            "recovery_note": "MATD3 CUDA RNG tensors normalized to CPU before restore",
        },
    )

    stas_30k = _stas_spec(
        root,
        selected_stas,
        selected_stas == "stas_bidirectional",
        30000,
        width,
        activation,
        resume=True,
    )
    _run_parallel(
        root,
        "recovery_matd3_and_stas_30k",
        {
            "matd3_30k": _resume_matd3_spec(root),
            selected_stas: stas_30k,
        },
    )
    _write_json(
        root / "training_complete.json",
        {
            "selected_mappo": selected_mappo,
            "selected_stas": selected_stas,
            "mappo_scores_10k": stage1["scores"],
            "stas_quality_10k": qualities,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    run(args.root.resolve())


if __name__ == "__main__":
    main()
