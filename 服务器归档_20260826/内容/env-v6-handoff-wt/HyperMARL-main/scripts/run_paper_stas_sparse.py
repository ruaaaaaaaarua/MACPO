#!/usr/bin/env python3
"""Run the paper-style STAS sparse-reward comparison without touching old runs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from run_fair_stas_pipeline import (
        ROOT,
        RunSpec,
        _environment,
        _git_metadata,
        _matd3_command,
        _ppo_command,
        _run,
        _validate,
        _write_json,
    )
    from run_traffic_experiment import traffic_group_abc_overrides
except ModuleNotFoundError:
    from scripts.run_fair_stas_pipeline import (
        ROOT,
        RunSpec,
        _environment,
        _git_metadata,
        _matd3_command,
        _ppo_command,
        _run,
        _validate,
        _write_json,
    )
    from scripts.run_traffic_experiment import traffic_group_abc_overrides


LHV_H2 = 33.33
DEFAULT_ROOT = ROOT.parents[1] / "traffic-stas-paper-sparse-20260716-10k-v1"


def sparse_traffic_overrides() -> dict[str, Any]:
    overrides = traffic_group_abc_overrides()
    overrides.update(
        {
            "reward_emission_mode": "terminal_total",
            "gamma": 1.0,
            "lambda_h2_buy": 45.0 / LHV_H2,
            "external_h2_dependency_penalty_enable": True,
            "external_h2_dependency_penalty_kg": 15.0,
            "terminal_h2_shortfall_value_enable": False,
            "terminal_h2_settlement_in_reward_enable": False,
            "h2_transport_loss": 0.0,
        }
    )
    return overrides


def _set(command: list[str], key: str, value: str) -> None:
    prefix = f"{key}="
    command[:] = [item for item in command if not item.startswith(prefix)]
    command.append(f"{key}={value}")


def _set_option(command: list[str], option: str, value: str) -> None:
    while option in command:
        index = command.index(option)
        del command[index : index + 2]
    command.extend([option, value])


def _paper_stas_args(smoke: bool) -> list[str]:
    return [
        "STAS.MODE=paper",
        "STAS.CUDA=true",
        "STAS.DEVICE=cuda",
        "STAS.LR=0.0005",
        "+STAS.WEIGHT_DECAY=1e-05",
        "STAS.EMB_DIM=128",
        "STAS.N_HEADS=4",
        "STAS.N_LAYERS=3",
        "STAS.SAMPLE_NUM=5",
        "STAS.BUFFER_SIZE=15000",
        f"STAS.BATCH_SIZE={16 if smoke else 256}",
        f"+STAS.POLICY_WARMUP_EPISODES={40 if smoke else 4000}",
        "+STAS.REWARD_MODEL_UPDATE_INTERVAL_EPISODES="
        f"{8 if smoke else 800}",
        "+STAS.REWARD_MODEL_UPDATES_PER_INTERVAL="
        f"{5 if smoke else 50}",
    ]


def build_specs(root: Path, episodes: int, smoke: bool) -> dict[str, RunSpec]:
    root = Path(root)
    overrides = sparse_traffic_overrides()

    mappo = _ppo_command(
        name="mappo",
        algorithm="Sparse-Terminal-MAPPO",
        output_dir=root / "mappo" / "output",
        episodes=episodes,
        stable=True,
        width=256,
        activation="relu",
        env_overrides=overrides,
    )
    _set(mappo.command, "GAMMA", "1.0")
    _set(mappo.command, "ENT_COEF", "0.01")
    mappo.command.append(
        f"+BEST_VALIDATION_CHECKPOINT_DIR={mappo.output_dir / 'checkpoints' / 'best_validation'}"
    )

    stas = _ppo_command(
        name="stas",
        algorithm="Sparse-Terminal-Paper-STAS",
        output_dir=root / "stas" / "output",
        episodes=episodes,
        stable=True,
        width=256,
        activation="relu",
        stas=True,
        env_overrides=overrides,
    )
    stas.command[:] = [
        item
        for item in stas.command
        if not item.startswith("STAS.")
        and not item.startswith("+STAS.")
        and not item.startswith("+BEST_VALIDATION_CHECKPOINT_DIR=")
    ]
    _set(stas.command, "GAMMA", "1.0")
    _set(stas.command, "ENT_COEF", "0.01")
    stas.command.extend(_paper_stas_args(smoke))
    stas.command.extend(
        [
            f"+STAS.CHECKPOINT_PATH={stas.output_dir / 'checkpoints' / 'stas_credit.pt'}",
            f"+BEST_VALIDATION_CHECKPOINT_DIR={stas.output_dir / 'checkpoints' / 'best_validation'}",
        ]
    )

    matd3 = _matd3_command(
        root / "matd3" / "output", episodes, env_overrides=overrides
    )
    _set_option(matd3.command, "--gamma", "1.0")
    _set_option(
        matd3.command,
        "--best-validation-checkpoint-dir",
        str(matd3.output_dir / "checkpoints" / "best_validation"),
    )
    matd3 = replace(
        matd3,
        name="matd3",
        algorithm="Sparse-Terminal-MATD3",
        returns_file=matd3.output_dir
        / "returns"
        / "returns_microgrid_Sparse-Terminal-MATD3.npy",
        checkpoints=(
            matd3.output_dir
            / "checkpoints"
            / "Sparse-Terminal-MATD3"
            / f"matd3_episode_{episodes}.pt",
        ),
    )
    _set_option(matd3.command, "--alg", matd3.algorithm)
    return {"stas": stas, "mappo": mappo, "matd3": matd3}


def write_manifest(
    root: Path,
    episodes: int,
    specs: dict[str, RunSpec],
    smoke: bool,
) -> dict[str, Any]:
    root = Path(root)
    branch, commit = _git_metadata()
    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "branch": branch,
        "commit": commit,
        "exploratory_seed_30": True,
        "episodes": int(episodes),
        "smoke": bool(smoke),
        "reward_emission_mode": "terminal_total",
        "gamma": 1.0,
        "nominal_external_h2_cost_yuan_per_kg": 45.0,
        "dependency_penalty_yuan_per_kg": 15.0,
        "effective_external_h2_cost_yuan_per_kg": 60.0,
        "selection_split": "validation",
        "test_accessed": False,
        "validation_days": [8, 17, 21, 23],
        "validation_noise_seed": 4200,
        "locked_test_days_after_selection": [1, 7, 14, 24],
        "locked_test_noise_seed": 5200,
        "environment_overrides": sparse_traffic_overrides(),
        "runs": {
            name: {
                "command": spec.command,
                "output_dir": str(spec.output_dir),
                "validation_jsonl": str(spec.eval_jsonl),
                "latest_checkpoints": [str(path) for path in spec.checkpoints],
            }
            for name, spec in specs.items()
        },
    }
    _write_json(root / "manifest.json", payload)
    return payload


def _write_hashes(root: Path) -> None:
    root = Path(root)
    hashes = {}
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        if path.name == "sha256.json":
            continue
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        hashes[str(path.relative_to(root))] = digest.hexdigest()
    _write_json(root / "sha256.json", hashes)


def _run_specs_parallel(root: Path, specs: dict[str, RunSpec]) -> list[dict]:
    processes = {}
    for name, spec in specs.items():
        spec.output_dir.mkdir(parents=True, exist_ok=True)
        log_path = Path(root) / "logs" / f"{name}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log = log_path.open("w", encoding="utf-8")
        process = subprocess.Popen(
            spec.command,
            cwd=ROOT,
            env=_environment(spec),
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        processes[name] = (process, log, spec)
        (Path(root) / f"{name}.pid").write_text(str(process.pid))
        print(f"[launch] {name} pid={process.pid}", flush=True)
    results = []
    for name, (process, log, spec) in processes.items():
        returncode = process.wait()
        log.close()
        result = _validate(spec, returncode)
        results.append(result)
        _write_json(Path(root) / f"{name}.result.json", result)
        print(f"[{result['status']}] {name}: {result['errors']}", flush=True)
    _write_json(Path(root) / "gate_results.json", results)
    return results


def _best_validation_episode(path: Path) -> int:
    records = [json.loads(line) for line in path.read_text().splitlines() if line]
    return int(
        max(records, key=lambda row: float(row["summary"]["return_mean"]))[
            "training_episode"
        ]
    )


def _resume_to_20k(specs: dict[str, RunSpec]) -> dict[str, RunSpec]:
    resumed = {}
    for name, spec in specs.items():
        command = list(spec.command)
        if name == "matd3":
            _set_option(command, "--episodes", "20000")
            _set_option(command, "--checkpoint-interval", "20000")
            _set_option(command, "--resume-checkpoint", str(spec.checkpoints[0]))
            checkpoint = spec.checkpoints[0].with_name("matd3_episode_20000.pt")
            resumed[name] = replace(spec, command=command, checkpoints=(checkpoint,))
        else:
            _set(command, "TOTAL_TIMESTEPS", str(20000 * 24))
            _set(command, "CHECKPOINT_INTERVAL", str(20000 * 24))
            command.append(f"+TRAINING_CHECKPOINT_LOAD_PATH={spec.checkpoints[0]}")
            if name == "stas":
                command.append(f"+STAS.CHECKPOINT_LOAD_PATH={spec.checkpoints[1]}")
            resumed[name] = replace(spec, command=command)
    return resumed


def run(root: Path, phase: str) -> None:
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    if phase in {"smoke", "all"}:
        smoke_root = root / "smoke_100"
        if smoke_root.exists() and any(smoke_root.iterdir()):
            raise FileExistsError(f"refusing to overwrite {smoke_root}")
        specs = build_specs(smoke_root, 100, smoke=True)
        write_manifest(smoke_root, 100, specs, smoke=True)
        results = []
        for name, spec in specs.items():
            result = _run(spec, smoke_root / "logs" / f"{name}.log")
            results.append(result)
            if result["status"] != "success":
                raise RuntimeError(f"smoke failed for {name}: {result['errors']}")
        _write_json(smoke_root / "gate_results.json", results)
        _write_hashes(smoke_root)

    if phase in {"train", "all"}:
        train_root = root / "train_10k_20k"
        if train_root.exists() and any(train_root.iterdir()):
            raise FileExistsError(f"refusing to overwrite {train_root}")
        specs = build_specs(train_root, 10000, smoke=False)
        write_manifest(train_root, 10000, specs, smoke=False)
        results = _run_specs_parallel(train_root, specs)
        if any(result["status"] != "success" for result in results):
            raise RuntimeError("at least one 10k run failed")
        best_episodes = {
            name: _best_validation_episode(spec.eval_jsonl)
            for name, spec in specs.items()
        }
        _write_json(train_root / "best_validation_episodes.json", best_episodes)
        if any(episode >= 9500 for episode in best_episodes.values()):
            resumed = _resume_to_20k(specs)
            results = _run_specs_parallel(train_root / "continuation", resumed)
            if any(result["status"] != "success" for result in results):
                raise RuntimeError("at least one 20k continuation failed")
        _write_hashes(train_root)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument(
        "--phase", choices=("smoke", "train", "all", "spec"), default="all"
    )
    args = parser.parse_args()
    if args.phase == "spec":
        root = args.root.resolve()
        root.mkdir(parents=True, exist_ok=True)
        specs = build_specs(root, 10000, smoke=False)
        write_manifest(root, 10000, specs, smoke=False)
        return
    run(args.root, args.phase)


if __name__ == "__main__":
    main()
