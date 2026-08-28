#!/usr/bin/env python3
"""Fair MAPPO/MATD3/STAS smoke gate and stage-one launcher."""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

try:
    from run_abc_multialg_parallel import group_abc_spec, safe_tag
    from run_stas_mechanism_ablation import LHV_H2, microgrid_override_arg
except ModuleNotFoundError:
    from scripts.run_abc_multialg_parallel import group_abc_spec, safe_tag
    from scripts.run_stas_mechanism_ablation import LHV_H2, microgrid_override_arg


ROOT = Path(__file__).resolve().parents[1]
TRAIN_SEED = 30
VALIDATION_NOISE_SEED = 4200
TEST_NOISE_SEED = 5200
SEED = TRAIN_SEED
TRAIN_DAYS = (0, 2, 3, 4, 5, 6, 9, 10, 11, 12, 13, 15, 16, 18, 19, 20, 22, 25, 26, 27)
VALIDATION_DAYS = (8, 17, 21, 23)
TEST_DAYS = (1, 7, 14, 24)
DEFAULT_OUTPUT = ROOT.parent / "fair-stas-h2-action-order-20260715"


def _fair_group_abc_overrides() -> dict[str, Any]:
    """Return the complete locked training environment contract."""

    overrides = dict(group_abc_spec().env_overrides)
    overrides.update(
        {
            "italian_split_enable": True,
            "italian_split_strategy": "manifest",
            "italian_split_name": "train",
            "h2_learnable_rolling_order_enable": True,
            "h2_learnable_rolling_order_agent_indices": [0, 1, 2, 3],
            "h2_buyer_reservation_demand_enable": False,
        }
    )
    return overrides


@dataclass(frozen=True)
class RunSpec:
    name: str
    kind: str
    algorithm: str
    command: list[str]
    output_dir: Path
    eval_jsonl: Path
    returns_file: Path
    checkpoints: tuple[Path, ...]


def _ppo_command(
    *,
    name: str,
    algorithm: str,
    output_dir: Path,
    episodes: int,
    stable: bool,
    width: int,
    activation: str,
    stas: bool = False,
    bidirectional: bool = False,
    env_overrides: dict[str, Any] | None = None,
    stas_max_mix_coef: float = 0.1,
    stas_ramp_episodes: int = 8000,
) -> RunSpec:
    total_timesteps = episodes * 24
    eval_interval = min(episodes, 500) * 24
    eval_jsonl = output_dir / "validation_eval.jsonl"
    jax_checkpoint = output_dir / "checkpoints" / "training_state.msgpack"
    stas_checkpoint = output_dir / "checkpoints" / "stas_credit.pt"
    entry = (
        "baselines/STAS-MAPPO/mappo_stas.py"
        if stas
        else "baselines/MAPPO/mappo_ff_shared_weights.py"
    )
    config = (
        "stas_mappo_microgrid"
        if stas
        else "mappo_ff_independent_actors_microgrid"
    )
    command = [
        sys.executable,
        entry,
        f"--config-name={config}",
        f"ALG={algorithm}",
        f"EXP_NAME={name}",
        f"RUN_NAME={name}__seed{SEED}",
        f"SEED={SEED}",
        f"TOTAL_TIMESTEPS={total_timesteps}",
        "NUM_ENVS=4",
        "NUM_STEPS=24",
        f"EVAL_INTERVAL={eval_interval}",
        "WANDB_MODE=disabled",
        "CAPTURE_VIDEO_INTERVAL=null",
        "CHECKPOINT=True",
        f"CHECKPOINT_INTERVAL={total_timesteps}",
        f"+FIXED_EVAL_OUTPUT={eval_jsonl}",
        "+FIXED_EVAL_SPLIT=validation",
        f"+FIXED_EVAL_NOISE_SEED={VALIDATION_NOISE_SEED}",
        f"+TRAINING_CHECKPOINT_PATH={jax_checkpoint}",
        f"ACTOR_LAYERS=[{width},{width}]",
        f"CRITIC_LAYERS=[{width},{width}]",
        f"ACTIVATION={activation}",
        microgrid_override_arg(
            _fair_group_abc_overrides() if env_overrides is None else env_overrides
        ),
    ]
    if stable:
        command.extend(
            [
                "+POLICY_MODE=squashed_gaussian",
                "+LOG_STD_MIN=-2.5",
                "+LOG_STD_MAX=-0.5",
                "LOG_STD_INIT=-1.0",
                "ENT_COEF=0.0",
            ]
        )
    if stas:
        command.extend(
            [
                f"+STAS.CHECKPOINT_PATH={stas_checkpoint}",
                "+STAS.CONSERVE_DISCOUNTED=true",
                "+STAS.QUALITY_GATE_ENABLE=true",
                f"+STAS.BIDIRECTIONAL={'true' if bidirectional else 'false'}",
                "+STAS.WARMUP_EPISODES=2000",
                f"+STAS.RAMP_EPISODES={int(stas_ramp_episodes)}",
                f"+STAS.MAX_MIX_COEF={float(stas_max_mix_coef):g}",
                f"+BEST_VALIDATION_CHECKPOINT_DIR={output_dir / 'checkpoints' / 'best_validation'}",
                "+STAS.EXPLAINED_VARIANCE_THRESHOLD=0.2",
                "+STAS.NEGATIVE_PATIENCE=3",
                "STAS.MIX_COEF=0.0",
            ]
        )
    checkpoints = (jax_checkpoint, stas_checkpoint) if stas else (jax_checkpoint,)
    return RunSpec(
        name=name,
        kind="stas" if stas else "jax",
        algorithm=algorithm,
        command=command,
        output_dir=output_dir,
        eval_jsonl=eval_jsonl,
        returns_file=output_dir / "returns" / f"returns_microgrid_{algorithm}.npy",
        checkpoints=checkpoints,
    )


def _matd3_command(
    output_dir: Path,
    episodes: int,
    env_overrides: dict[str, Any] | None = None,
) -> RunSpec:
    algorithm = "MATD3-Fair-256"
    eval_jsonl = output_dir / "validation_eval.jsonl"
    checkpoint = (
        output_dir
        / "checkpoints"
        / safe_tag(algorithm)
        / f"matd3_episode_{episodes}.pt"
    )
    command = [
        sys.executable,
        "baselines/MATD3/train_matd3_microgrid.py",
        "--seed",
        str(SEED),
        "--episodes",
        str(episodes),
        "--episode-length",
        "24",
        "--alg",
        algorithm,
        "--hidden-dim",
        "256",
        "--fixed-eval-output",
        str(eval_jsonl),
        "--fixed-eval-split",
        "validation",
        "--fixed-eval-noise-seed",
        str(VALIDATION_NOISE_SEED),
        "--eval-interval-episodes",
        str(min(episodes, 500)),
        "--checkpoint-interval",
        str(episodes),
        "--microgrid-overrides-json",
        json.dumps(
            _fair_group_abc_overrides() if env_overrides is None else env_overrides,
            separators=(",", ":"),
        ),
    ]
    return RunSpec(
        name="matd3_256",
        kind="torch",
        algorithm=algorithm,
        command=command,
        output_dir=output_dir,
        eval_jsonl=eval_jsonl,
        returns_file=output_dir / "returns" / f"returns_microgrid_{safe_tag(algorithm)}.npy",
        checkpoints=(checkpoint,),
    )


def build_specs(root: Path, episodes: int) -> dict[str, RunSpec]:
    def out(name: str) -> Path:
        return root / name / "output"

    specs = {
        "legacy_mappo_128": _ppo_command(
            name="legacy_mappo_128",
            algorithm="Legacy-MAPPO-128",
            output_dir=out("legacy_mappo_128"),
            episodes=episodes,
            stable=False,
            width=128,
            activation="tanh",
        ),
        "stable_mappo_128": _ppo_command(
            name="stable_mappo_128",
            algorithm="Stable-MAPPO-128",
            output_dir=out("stable_mappo_128"),
            episodes=episodes,
            stable=True,
            width=128,
            activation="tanh",
        ),
        "stable_mappo_256": _ppo_command(
            name="stable_mappo_256",
            algorithm="Stable-MAPPO-256-ReLU",
            output_dir=out("stable_mappo_256"),
            episodes=episodes,
            stable=True,
            width=256,
            activation="relu",
        ),
        "matd3_256": _matd3_command(out("matd3_256"), episodes),
        "stas_causal": _ppo_command(
            name="stas_causal",
            algorithm="Conserved-Causal-STAS",
            output_dir=out("stas_causal"),
            episodes=episodes,
            stable=True,
            width=128,
            activation="tanh",
            stas=True,
        ),
        "stas_bidirectional": _ppo_command(
            name="stas_bidirectional",
            algorithm="Conserved-Bidirectional-STAS",
            output_dir=out("stas_bidirectional"),
            episodes=episodes,
            stable=True,
            width=128,
            activation="tanh",
            stas=True,
            bidirectional=True,
        ),
    }
    return specs


def _environment(spec: RunSpec) -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{ROOT}:{env.get('PYTHONPATH', '')}"
    env["WANDB_MODE"] = "disabled"
    env["HYPERMARL_OUTPUT_DIR"] = str(spec.output_dir)
    env["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    env["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.20"
    return env


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _ensure_fresh_output_root(root: Path) -> None:
    root = Path(root)
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(
            f"refusing to mix fair-pipeline artifacts with non-empty {root}"
        )


def _git_metadata() -> tuple[str, str]:
    branch = subprocess.check_output(
        ["git", "branch", "--show-current"],
        cwd=ROOT,
        text=True,
        stderr=subprocess.DEVNULL,
    ).strip()
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        text=True,
        stderr=subprocess.DEVNULL,
    ).strip()
    return branch, commit


def _h2_price_metadata(overrides: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "lambda_h2",
        "lambda_h2_buy",
        "lambda_h2_sell",
        "h2_price_min",
        "h2_price_max",
        "h2_price_init",
    )
    internal = {key: float(overrides[key]) for key in keys}
    return {
        "internal_unit": "yuan/kWh-H2",
        "display_unit": "yuan/kg",
        "lhv_kwh_per_kg": float(LHV_H2),
        "yuan_per_kwh_h2": internal,
        "yuan_per_kg": {
            key: value * float(LHV_H2) for key, value in internal.items()
        },
    }


def _manifest(root: Path, episodes: int, specs: dict[str, RunSpec]) -> None:
    branch, commit = _git_metadata()
    overrides = _fair_group_abc_overrides()
    _write_json(
        root / "manifest.json",
        {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "branch": branch,
            "commit": commit,
            "output_root": str(root),
            "seed": SEED,
            "seeds": {
                "training": TRAIN_SEED,
                "validation": VALIDATION_NOISE_SEED,
                "test": TEST_NOISE_SEED,
            },
            "episodes": episodes,
            "qmax_basis": "peak_heat_hour",
            "train_days": list(TRAIN_DAYS),
            "validation_days": list(VALIDATION_DAYS),
            "test_days": list(TEST_DAYS),
            "validation_noise_seeds": [VALIDATION_NOISE_SEED],
            "test_noise_seeds": [TEST_NOISE_SEED],
            "evaluation_grids": {
                "validation": [
                    {"day": day, "seed": VALIDATION_NOISE_SEED}
                    for day in VALIDATION_DAYS
                ],
                "test": [
                    {"day": day, "seed": TEST_NOISE_SEED}
                    for day in TEST_DAYS
                ],
            },
            "abc_overrides": overrides,
            "h2_prices": _h2_price_metadata(overrides),
            "runs": {
                name: {
                    "command": spec.command,
                    "output_dir": str(spec.output_dir),
                    "eval_jsonl": str(spec.eval_jsonl),
                    "checkpoints": [str(path) for path in spec.checkpoints],
                }
                for name, spec in specs.items()
            },
        },
    )


def _validate(spec: RunSpec, returncode: int) -> dict[str, Any]:
    errors: list[str] = []
    if returncode != 0:
        errors.append(f"returncode={returncode}")
    if not spec.returns_file.exists():
        errors.append("missing returns")
    else:
        values = np.load(spec.returns_file)
        if values.size == 0 or not np.isfinite(values).all():
            errors.append("invalid returns")
    if not spec.eval_jsonl.exists():
        errors.append("missing eval jsonl")
    else:
        records = [json.loads(line) for line in spec.eval_jsonl.read_text().splitlines()]
        if not records:
            errors.append("empty eval jsonl")
        for record in records:
            summary = record.get("summary", {})
            if not math.isfinite(float(summary.get("return_mean", float("nan")))):
                errors.append("non-finite eval return")
    for checkpoint in spec.checkpoints:
        if not checkpoint.exists():
            errors.append(f"missing checkpoint: {checkpoint}")
    return {
        "name": spec.name,
        "status": "success" if not errors else "failed",
        "returncode": returncode,
        "errors": errors,
    }


def _run(spec: RunSpec, log_path: Path) -> dict[str, Any]:
    spec.output_dir.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        completed = subprocess.run(
            spec.command,
            cwd=ROOT,
            env=_environment(spec),
            stdout=log,
            stderr=subprocess.STDOUT,
        )
    result = _validate(spec, completed.returncode)
    _write_json(log_path.parent / f"{spec.name}.result.json", result)
    return result


def run_smoke_and_stage_one(root: Path) -> None:
    smoke_root = root / "smoke_100"
    smoke_specs = build_specs(smoke_root, episodes=100)
    _manifest(smoke_root, 100, smoke_specs)
    smoke_results = []
    for name, spec in smoke_specs.items():
        print(f"[smoke launch] {name}", flush=True)
        result = _run(spec, smoke_root / "logs" / f"{name}.log")
        smoke_results.append(result)
        print(f"[smoke {result['status']}] {name}: {result['errors']}", flush=True)
        if result["status"] != "success":
            raise RuntimeError(f"smoke gate failed: {name}: {result['errors']}")
    _write_json(smoke_root / "gate_results.json", smoke_results)

    stage_root = root / "stage1_10k"
    all_specs = build_specs(stage_root, episodes=10000)
    specs = {name: all_specs[name] for name in (
        "legacy_mappo_128", "stable_mappo_128", "stable_mappo_256", "matd3_256"
    )}
    _manifest(stage_root, 10000, specs)
    processes: dict[str, tuple[subprocess.Popen[Any], Any, RunSpec]] = {}
    for name, spec in specs.items():
        spec.output_dir.mkdir(parents=True, exist_ok=True)
        log_path = stage_root / "logs" / f"{name}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log = log_path.open("w", encoding="utf-8")
        process = subprocess.Popen(
            spec.command, cwd=ROOT, env=_environment(spec), stdout=log,
            stderr=subprocess.STDOUT,
        )
        processes[name] = (process, log, spec)
        (stage_root / f"{name}.pid").write_text(str(process.pid))
        print(f"[stage1 launch] {name} pid={process.pid}", flush=True)
    results = []
    for name, (process, log, spec) in processes.items():
        returncode = process.wait()
        log.close()
        result = _validate(spec, returncode)
        results.append(result)
        _write_json(stage_root / f"{name}.result.json", result)
        print(f"[stage1 {result['status']}] {name}: {result['errors']}", flush=True)
    _write_json(stage_root / "gate_results.json", results)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--spec-only", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    _ensure_fresh_output_root(root)
    if args.spec_only:
        specs = build_specs(root, 100)
        _manifest(root, 100, specs)
        return
    run_smoke_and_stage_one(root)


if __name__ == "__main__":
    main()
