#!/usr/bin/env python3
"""Continue the fair pipeline after stage-one: select, resume, and run STAS."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

try:
    from run_fair_stas_pipeline import (
        DEFAULT_OUTPUT,
        ROOT,
        RunSpec,
        _environment,
        _matd3_command,
        _ppo_command,
        _validate,
        _write_json,
    )
except ModuleNotFoundError:
    from scripts.run_fair_stas_pipeline import (
        DEFAULT_OUTPUT,
        ROOT,
        RunSpec,
        _environment,
        _matd3_command,
        _ppo_command,
        _validate,
        _write_json,
    )


def _last_three_score(path: Path) -> float:
    records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    by_episode = {int(row["training_episode"]): row for row in records}
    required = (9000, 9500, 10000)
    missing = [episode for episode in required if episode not in by_episode]
    if missing:
        raise ValueError(f"missing eval episodes {missing} in {path}")
    return float(
        np.mean([by_episode[episode]["summary"]["return_mean"] for episode in required])
    )


def select_mappo_name(eval_128: Path, eval_256: Path) -> tuple[str, dict[str, float]]:
    scores = {
        "stable_mappo_128": _last_three_score(eval_128),
        "stable_mappo_256": _last_three_score(eval_256),
    }
    if abs(scores["stable_mappo_128"] - scores["stable_mappo_256"]) <= 2.0:
        return "stable_mappo_128", scores
    return max(scores, key=scores.get), scores


def _wait_for_stage_one(root: Path) -> list[dict[str, Any]]:
    gate_path = root / "stage1_10k" / "gate_results.json"
    while not gate_path.exists():
        time.sleep(30)
    results = json.loads(gate_path.read_text())
    failures = [row for row in results if row.get("status") != "success"]
    if failures:
        raise RuntimeError(f"stage-one failed: {failures}")
    return results


def _resume_mappo_spec(root: Path, selected: str, width: int, activation: str) -> RunSpec:
    output = root / "stage1_10k" / selected / "output"
    spec = _ppo_command(
        name=selected,
        algorithm="Stable-MAPPO-128" if width == 128 else "Stable-MAPPO-256-ReLU",
        output_dir=output,
        episodes=30000,
        stable=True,
        width=width,
        activation=activation,
    )
    checkpoint = output / "checkpoints" / "training_state.msgpack"
    return RunSpec(
        **{
            **spec.__dict__,
            "command": [*spec.command, f"+TRAINING_CHECKPOINT_LOAD_PATH={checkpoint}"],
        }
    )


def _resume_matd3_spec(root: Path) -> RunSpec:
    output = root / "stage1_10k" / "matd3_256" / "output"
    spec = _matd3_command(output, episodes=30000)
    checkpoint = output / "checkpoints" / spec.algorithm / "matd3_episode_10000.pt"
    return RunSpec(
        **{
            **spec.__dict__,
            "command": [*spec.command, "--resume-checkpoint", str(checkpoint)],
        }
    )


def _stas_spec(
    root: Path,
    name: str,
    bidirectional: bool,
    episodes: int,
    width: int,
    activation: str,
    resume: bool = False,
) -> RunSpec:
    output = root / "stage2_stas" / name / "output"
    spec = _ppo_command(
        name=name,
        algorithm=(
            "Conserved-Bidirectional-STAS"
            if bidirectional else "Conserved-Causal-STAS"
        ),
        output_dir=output,
        episodes=episodes,
        stable=True,
        width=width,
        activation=activation,
        stas=True,
        bidirectional=bidirectional,
    )
    if not resume:
        return spec
    jax_checkpoint, stas_checkpoint = spec.checkpoints
    return RunSpec(
        **{
            **spec.__dict__,
            "command": [
                *spec.command,
                f"+TRAINING_CHECKPOINT_LOAD_PATH={jax_checkpoint}",
                f"+STAS.CHECKPOINT_LOAD_PATH={stas_checkpoint}",
            ],
        }
    )


def _run_parallel(root: Path, label: str, specs: dict[str, RunSpec]) -> list[dict[str, Any]]:
    run_dir = root / "continuation_logs" / label
    run_dir.mkdir(parents=True, exist_ok=True)
    processes: dict[str, tuple[subprocess.Popen[Any], Any, RunSpec]] = {}
    for name, spec in specs.items():
        spec.output_dir.mkdir(parents=True, exist_ok=True)
        log = (run_dir / f"{name}.log").open("w", encoding="utf-8")
        process = subprocess.Popen(
            spec.command,
            cwd=ROOT,
            env=_environment(spec),
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        processes[name] = (process, log, spec)
        (run_dir / f"{name}.pid").write_text(str(process.pid))
        print(f"[{label} launch] {name} pid={process.pid}", flush=True)
    results = []
    for name, (process, log, spec) in processes.items():
        returncode = process.wait()
        log.close()
        result = _validate(spec, returncode)
        results.append(result)
        _write_json(run_dir / f"{name}.result.json", result)
        print(f"[{label} {result['status']}] {name}: {result['errors']}", flush=True)
    _write_json(run_dir / "results.json", results)
    failures = [row for row in results if row["status"] != "success"]
    if failures:
        raise RuntimeError(f"{label} failed: {failures}")
    return results


def _stas_quality(spec: RunSpec) -> dict[str, Any]:
    state = torch.load(spec.checkpoints[1], map_location="cpu", weights_only=False)
    explained_variance = float(state.get("last_explained_variance", float("nan")))
    conservation_error = float(state.get("last_conservation_error", float("inf")))
    mix_coef = float(state.get("last_mix_coef", 0.0))
    gate_disabled = bool(state.get("gate", {}).get("disabled", False))
    return {
        "score": _last_three_score(spec.eval_jsonl),
        "explained_variance": explained_variance,
        "conservation_error": conservation_error,
        "mix_coef": mix_coef,
        "gate_disabled": gate_disabled,
        "eligible": _stas_checkpoint_eligible(state),
    }


def _stas_checkpoint_eligible(state: dict[str, Any]) -> bool:
    explained_variance = float(state.get("last_explained_variance", float("nan")))
    conservation_error = float(state.get("last_conservation_error", float("inf")))
    mix_coef = float(state.get("last_mix_coef", 0.0))
    gate_disabled = bool(state.get("gate", {}).get("disabled", False))
    return bool(
        not gate_disabled
        and mix_coef > 0.0
        and np.isfinite(explained_variance)
        and explained_variance >= 0.2
        and np.isfinite(conservation_error)
        and conservation_error < 1e-4
    )


def run(root: Path) -> None:
    _wait_for_stage_one(root)
    stage = root / "stage1_10k"
    selected_mappo, mappo_scores = select_mappo_name(
        stage / "stable_mappo_128" / "output" / "validation_eval.jsonl",
        stage / "stable_mappo_256" / "output" / "validation_eval.jsonl",
    )
    width = 128 if selected_mappo.endswith("128") else 256
    activation = "tanh" if width == 128 else "relu"
    _write_json(
        root / "selection_stage1.json",
        {"selected_mappo": selected_mappo, "scores": mappo_scores},
    )

    stas_10k = {
        "stas_causal": _stas_spec(root, "stas_causal", False, 10000, width, activation),
        "stas_bidirectional": _stas_spec(
            root, "stas_bidirectional", True, 10000, width, activation
        ),
    }
    concurrent = {
        "mappo_30k": _resume_mappo_spec(root, selected_mappo, width, activation),
        "matd3_30k": _resume_matd3_spec(root),
        **stas_10k,
    }
    _run_parallel(root, "selected_baselines_30k_and_stas_10k", concurrent)

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
        },
    )

    best_is_bidirectional = selected_stas == "stas_bidirectional"
    stas_30k = _stas_spec(
        root,
        selected_stas,
        best_is_bidirectional,
        30000,
        width,
        activation,
        resume=True,
    )
    _run_parallel(root, "selected_stas_30k", {selected_stas: stas_30k})
    _write_json(
        root / "training_complete.json",
        {
            "selected_mappo": selected_mappo,
            "selected_stas": selected_stas,
            "mappo_scores_10k": mappo_scores,
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
