#!/usr/bin/env python3
"""Run and audit the two-phase 100-episode corrected STAS smoke gate."""

from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping, Sequence

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from baselines.utils.fixed_scenario_eval import (  # noqa: E402
    DEFAULT_NOISE_SEEDS,
    TEST_DAYS,
    VALIDATION_DAYS,
)
from scripts.run_stas_mechanism_ablation import (  # noqa: E402
    microgrid_override_arg,
    planned_experiments,
)


SCHEMA_VERSION = 1
LOCKED_SEED = 30
LOCKED_EPISODES = 100
LOCKED_RESUME_SPLIT = 60
NUM_ENVS = 4
NUM_STEPS = 24
EXPECTED_UPDATES = LOCKED_EPISODES // NUM_ENVS
EXPECTED_PHASE1_UPDATES = LOCKED_RESUME_SPLIT // NUM_ENVS


class SmokeRootNotFreshError(FileExistsError):
    """Raised before execution when a smoke root already contains evidence."""


@dataclass(frozen=True)
class PhaseSpec:
    name: str
    command: tuple[str, ...]
    output_dir: Path
    log: Path
    diagnostics: Path
    jax_checkpoint: Path
    stas_checkpoint: Path
    validation: Path


def _group_ab_overrides() -> dict[str, Any]:
    for candidate in planned_experiments():
        if candidate.group == "group_ab":
            overrides = copy.deepcopy(candidate.env_overrides)
            break
    else:
        raise RuntimeError("frozen A+B environment specification is unavailable")
    if overrides.get("italian_split_name") != "train":
        raise ValueError("smoke training must use the training split")
    if "italian_day_indices" in overrides:
        raise ValueError("smoke training must not select fixed validation/test days")
    return overrides


def _phase_command(
    *,
    name: str,
    total_episodes: int,
    endpoint_global_step: int,
    output_dir: Path,
    diagnostics: Path,
    jax_checkpoint: Path,
    stas_checkpoint: Path,
    validation: Path | None,
    load_jax: Path | None,
    load_stas: Path | None,
) -> tuple[str, ...]:
    command = [
        sys.executable,
        "baselines/STAS-MAPPO/mappo_stas.py",
        "--config-name=stas_mappo_microgrid",
        "ALG=Corrected-Causal-STAS-Smoke",
        f"EXP_NAME=corrected_stas_smoke_{name}",
        f"RUN_NAME=corrected_stas_smoke_{name}__seed{LOCKED_SEED}",
        f"SEED={LOCKED_SEED}",
        f"TOTAL_TIMESTEPS={total_episodes * NUM_STEPS}",
        f"NUM_ENVS={NUM_ENVS}",
        f"NUM_STEPS={NUM_STEPS}",
        "ACTOR_LAYERS=[256,256]",
        "CRITIC_LAYERS=[256,256]",
        "ACTIVATION=relu",
        "+POLICY_MODE=squashed_gaussian",
        "+LOG_STD_MIN=-2.5",
        "+LOG_STD_MAX=-0.5",
        "LOG_STD_INIT=-1.0",
        "ENT_COEF=0",
        "WANDB_MODE=disabled",
        "CAPTURE_VIDEO_INTERVAL=null",
        "EVAL_PARALLEL=False",
        f"EVAL_INTERVAL={LOCKED_EPISODES * NUM_STEPS}",
        "LOG_INTERVAL=1",
        "CHECKPOINT=True",
        f"CHECKPOINT_INTERVAL={endpoint_global_step}",
        f"+TRAINING_CHECKPOINT_PATH={jax_checkpoint}",
        f"+STAS.CHECKPOINT_PATH={stas_checkpoint}",
        f"STAS.DIAGNOSTICS_PATH={diagnostics}",
        "+STAS.CONSERVE_DISCOUNTED=true",
        "+STAS.QUALITY_GATE_ENABLE=true",
        "+STAS.BIDIRECTIONAL=false",
        "+STAS.WARMUP_EPISODES=200",
        "+STAS.RAMP_EPISODES=800",
        "+STAS.MAX_MIX_COEF=0.1",
        "+STAS.EXPLAINED_VARIANCE_THRESHOLD=0.2",
        "STAS.MIX_COEF=0.0",
        microgrid_override_arg(_group_ab_overrides()),
    ]
    if validation is not None:
        command.append(f"+FIXED_EVAL_OUTPUT={validation}")
    if load_jax is not None:
        command.append(f"+TRAINING_CHECKPOINT_LOAD_PATH={load_jax}")
    if load_stas is not None:
        command.append(f"+STAS.CHECKPOINT_LOAD_PATH={load_stas}")
    return tuple(command)


def build_phase_specs(
    root: Path,
    *,
    seed: int = LOCKED_SEED,
    episodes: int = LOCKED_EPISODES,
    resume_split: int = LOCKED_RESUME_SPLIT,
) -> dict[str, PhaseSpec]:
    if (seed, episodes, resume_split) != (
        LOCKED_SEED,
        LOCKED_EPISODES,
        LOCKED_RESUME_SPLIT,
    ):
        raise ValueError("correctness smoke seed/episode/resume split are locked")
    root = root.expanduser().resolve()
    phase1_root = root / "phase1_000_060"
    phase2_root = root / "phase2_060_100"

    def paths(name: str, phase_root: Path) -> dict[str, Path]:
        return {
            "output_dir": phase_root / "output",
            "log": phase_root / "train.log",
            "diagnostics": phase_root / "stas_diagnostics.jsonl",
            "jax_checkpoint": phase_root / "checkpoints" / "training_state.msgpack",
            "stas_checkpoint": phase_root / "checkpoints" / "stas_credit.pt",
            "validation": phase_root / "validation_eval.jsonl",
        }

    phase1_paths = paths("phase1", phase1_root)
    phase2_paths = paths("phase2", phase2_root)
    phase1 = PhaseSpec(
        name="phase1",
        command=_phase_command(
            name="phase1",
            total_episodes=resume_split,
            endpoint_global_step=resume_split * NUM_STEPS,
            validation=None,
            load_jax=None,
            load_stas=None,
            output_dir=phase1_paths["output_dir"],
            diagnostics=phase1_paths["diagnostics"],
            jax_checkpoint=phase1_paths["jax_checkpoint"],
            stas_checkpoint=phase1_paths["stas_checkpoint"],
        ),
        **phase1_paths,
    )
    phase2 = PhaseSpec(
        name="phase2",
        command=_phase_command(
            name="phase2",
            total_episodes=episodes,
            endpoint_global_step=episodes * NUM_STEPS,
            validation=phase2_paths["validation"],
            load_jax=phase1.jax_checkpoint,
            load_stas=phase1.stas_checkpoint,
            output_dir=phase2_paths["output_dir"],
            diagnostics=phase2_paths["diagnostics"],
            jax_checkpoint=phase2_paths["jax_checkpoint"],
            stas_checkpoint=phase2_paths["stas_checkpoint"],
        ),
        **phase2_paths,
    )
    return {"phase1": phase1, "phase2": phase2}


def _environment(spec: PhaseSpec) -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{ROOT}:{env.get('PYTHONPATH', '')}"
    env["WANDB_MODE"] = "disabled"
    env["WANDB_DIR"] = str(spec.output_dir / "wandb")
    env["HYPERMARL_OUTPUT_DIR"] = str(spec.output_dir)
    env["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    env["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.30"
    return env


def _run_phase(spec: PhaseSpec) -> int:
    spec.output_dir.mkdir(parents=True, exist_ok=True)
    spec.log.parent.mkdir(parents=True, exist_ok=True)
    (spec.output_dir / "wandb").mkdir(parents=True, exist_ok=True)
    with spec.log.open("w", encoding="utf-8") as stream:
        completed = subprocess.run(
            spec.command,
            cwd=ROOT,
            env=_environment(spec),
            stdout=stream,
            stderr=subprocess.STDOUT,
        )
    return int(completed.returncode)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    records: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"non-object JSONL record in {path}")
            records.append(payload)
    return records


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()


def _target_consistency_error(state: Mapping[str, Any]) -> float:
    try:
        gamma = float(state["config"]["gamma"])
        maximum = 0.0
        for key in ("buffer_storage", "holdout_buffer_storage"):
            for item in state.get(key, []):
                rewards = np.asarray(item[2], dtype=np.float64)
                weights = np.power(gamma, np.arange(rewards.shape[1], dtype=np.float64))
                actual = float(np.sum(rewards * weights[None, :]))
                maximum = max(maximum, abs(actual - float(item[4])))
        return float(maximum)
    except (KeyError, TypeError, ValueError, IndexError):
        return float("inf")


def _finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _expected_artifacts(specs: Mapping[str, PhaseSpec]) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for phase_name, spec in specs.items():
        paths.update(
            {
                f"{phase_name}.log": spec.log,
                f"{phase_name}.diagnostics": spec.diagnostics,
                f"{phase_name}.jax_checkpoint": spec.jax_checkpoint,
                f"{phase_name}.stas_checkpoint": spec.stas_checkpoint,
            }
        )
    paths["phase2.validation"] = specs["phase2"].validation
    return paths


def audit_smoke_artifacts(
    specs: Mapping[str, PhaseSpec],
    *,
    exit_codes: Mapping[str, int],
    source_commit: str,
    invocation: Sequence[str],
) -> dict[str, Any]:
    phase1_records: list[dict[str, Any]] = []
    phase2_records: list[dict[str, Any]] = []
    load_error: str | None = None
    try:
        phase1_records = _load_jsonl(specs["phase1"].diagnostics)
        phase2_records = _load_jsonl(specs["phase2"].diagnostics)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        load_error = f"diagnostics: {type(error).__name__}: {error}"
    combined = phase1_records + phase2_records

    phase2_log = (
        specs["phase2"].log.read_text(encoding="utf-8", errors="replace")
        if specs["phase2"].log.is_file()
        else ""
    )
    final_state: Mapping[str, Any] = {}
    checkpoint_error: str | None = None
    try:
        loaded = torch.load(
            specs["phase2"].stas_checkpoint,
            map_location="cpu",
            weights_only=False,
        )
        if not isinstance(loaded, dict):
            raise ValueError("final STAS checkpoint is not a dictionary")
        final_state = loaded
    except (OSError, RuntimeError, ValueError) as error:
        checkpoint_error = f"checkpoint: {type(error).__name__}: {error}"

    target_error = _target_consistency_error(final_state)
    checkpoint_sample_count = sum(
        len(final_state.get(key, []))
        for key in ("buffer_storage", "holdout_buffer_storage")
    )
    conservation_error = final_state.get("last_conservation_error")
    loss_values = [record.get("reward_model_loss") for record in combined]
    first_loss = next(
        (index for index, value in enumerate(loss_values) if value is not None),
        None,
    )
    loss_is_valid = bool(
        first_loss is not None
        and all(value is not None and _finite(value) for value in loss_values[first_loss:])
    )

    validation_records: list[dict[str, Any]] = []
    validation_error: str | None = None
    try:
        validation_records = _load_jsonl(specs["phase2"].validation)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        validation_error = f"validation: {type(error).__name__}: {error}"
    validation_episodes = (
        validation_records[-1].get("episodes", []) if validation_records else []
    )
    actual_scenarios = {
        (int(item.get("day", -1)), int(item.get("seed", -1)))
        for item in validation_episodes
        if isinstance(item, dict)
    }
    expected_scenarios = {
        (int(day), int(seed))
        for day in VALIDATION_DAYS
        for seed in DEFAULT_NOISE_SEEDS
    }
    expected_validation_count = len(expected_scenarios)

    expected_updates = list(range(1, EXPECTED_UPDATES + 1))
    expected_episodes = [update * NUM_ENVS for update in expected_updates]
    expected_steps = [update * NUM_ENVS * NUM_STEPS for update in expected_updates]
    record_numeric_fields_finite = all(
        all(
            _finite(record.get(field))
            for field in (
                "update",
                "episode",
                "global_step",
                "explained_variance",
                "mix_coef",
                "target_error",
                "conservation_error",
            )
        )
        for record in combined
    )
    assertions = {
        "subprocesses_exit_zero": dict(exit_codes)
        == {"phase1": 0, "phase2": 0},
        "phase2_restored_jax_training_state": bool(
            "Loaded full training checkpoint" in phase2_log
            and "episode=60, update=15" in phase2_log
        ),
        "phase2_restored_complete_stas_state": bool(
            "Loaded full STAS state" in phase2_log
            and "global_step=1440" in phase2_log
            and "rollouts_seen=15" in phase2_log
        ),
        "diagnostic_record_counts": bool(
            len(phase1_records) == EXPECTED_PHASE1_UPDATES
            and len(phase2_records) == EXPECTED_UPDATES - EXPECTED_PHASE1_UPDATES
        ),
        "diagnostic_schema_and_numeric_fields": bool(
            len(combined) == EXPECTED_UPDATES
            and all(record.get("schema_version") == 1 for record in combined)
            and record_numeric_fields_finite
        ),
        "diagnostic_update_sequence": [record.get("update") for record in combined]
        == expected_updates,
        "diagnostic_episode_sequence": [record.get("episode") for record in combined]
        == expected_episodes,
        "diagnostic_global_step_sequence": [
            record.get("global_step") for record in combined
        ]
        == expected_steps,
        "final_checkpoint_metadata": all(
            int(final_state.get(key, -1)) == expected
            for key, expected in (
                ("update", 25),
                ("episode", 100),
                ("global_step", 2400),
            )
        ),
        "final_assigner_counters": bool(
            int(final_state.get("rollouts_seen", -1)) == 25
            and int(final_state.get("episodes_seen", -1)) == 100
            and int(final_state.get("normalizer", {}).get("count", -1)) == 100
        ),
        "checkpoint_buffers_nonempty": checkpoint_sample_count > 0,
        "target_consistency": bool(
            checkpoint_sample_count > 0
            and _finite(target_error)
            and target_error <= 1e-6
        ),
        "conservation": bool(
            _finite(conservation_error) and abs(float(conservation_error)) <= 1e-5
        ),
        "loss_finite_once_available": loss_is_valid,
        "warmup_mix_zero_and_gate_inactive": bool(
            len(combined) == EXPECTED_UPDATES
            and all(
                float(record.get("mix_coef", float("nan"))) == 0.0
                and not bool(record.get("gate", {}).get("active", True))
                for record in combined
            )
        ),
        "validation_scenarios_exact": bool(
            len(validation_episodes) == expected_validation_count
            and actual_scenarios == expected_scenarios
        ),
        "validation_steps_exactly_24": bool(
            len(validation_episodes) == expected_validation_count
            and all(int(item.get("steps", -1)) == 24 for item in validation_episodes)
        ),
        "validation_metrics_finite": bool(
            len(validation_episodes) == expected_validation_count
            and all(
                _finite(item.get("return")) and _finite(item.get("base_cost"))
                for item in validation_episodes
            )
        ),
        "final_validation_contains_no_test_scenarios": bool(
            len(validation_episodes) == expected_validation_count
            and all(int(item.get("day", -1)) not in TEST_DAYS for item in validation_episodes)
        ),
    }

    expected_artifacts = _expected_artifacts(specs)
    artifacts = {
        name: {"path": str(path), "sha256": _sha256(path)}
        for name, path in expected_artifacts.items()
        if path.is_file()
    }
    assertions["all_artifacts_present_and_hashed"] = (
        set(artifacts) == set(expected_artifacts)
        and all(len(item["sha256"]) == 64 for item in artifacts.values())
    )
    errors = [
        error for error in (load_error, checkpoint_error, validation_error) if error
    ]
    errors.extend(name for name, passed in assertions.items() if not passed)
    safe_target_error = target_error if _finite(target_error) else None
    safe_conservation = (
        float(conservation_error) if _finite(conservation_error) else None
    )
    resolved_config = {
        "seed": LOCKED_SEED,
        "episodes": LOCKED_EPISODES,
        "resume_split": LOCKED_RESUME_SPLIT,
        "num_envs": NUM_ENVS,
        "num_steps": NUM_STEPS,
        "actor_layers": [256, 256],
        "critic_layers": [256, 256],
        "activation": "relu",
        "policy_mode": "squashed_gaussian",
        "log_std_min": -2.5,
        "log_std_max": -0.5,
        "log_std_init": -1.0,
        "ent_coef": 0.0,
        "stas": {
            "conserve_discounted": True,
            "bidirectional": False,
            "warmup_episodes": 200,
            "ramp_episodes": 800,
            "max_mix_coef": 0.1,
            "explained_variance_threshold": 0.2,
        },
        "environment": _group_ab_overrides(),
        "validation_days": list(VALIDATION_DAYS),
        "validation_noise_seeds": list(DEFAULT_NOISE_SEEDS),
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "source_commit": source_commit,
        "command": list(invocation),
        "commands": {name: list(spec.command) for name, spec in specs.items()},
        "resolved_config": resolved_config,
        "exit_codes": dict(exit_codes),
        "diagnostic_counts": {
            "phase1": len(phase1_records),
            "phase2": len(phase2_records),
        },
        "target_consistency_max_error": safe_target_error,
        "conservation_error": safe_conservation,
        "artifacts": artifacts,
        "assertions": assertions,
        "errors": errors,
        "passed": not errors and all(assertions.values()),
    }


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _assert_fresh_root(root: Path) -> None:
    for relative in ("phase1_000_060", "phase2_060_100", "gate_results.json"):
        if (root / relative).exists():
            raise SmokeRootNotFreshError(
                f"refusing to mix smoke artifacts with existing {root / relative}"
            )


def run_smoke(args: argparse.Namespace) -> dict[str, Any]:
    root = args.root.expanduser().resolve()
    _assert_fresh_root(root)
    specs = build_phase_specs(
        args.root,
        seed=args.seed,
        episodes=args.episodes,
        resume_split=args.resume_split,
    )
    root.mkdir(parents=True, exist_ok=True)
    exit_codes: dict[str, int] = {}
    exit_codes["phase1"] = _run_phase(specs["phase1"])
    if exit_codes["phase1"] == 0:
        exit_codes["phase2"] = _run_phase(specs["phase2"])
    else:
        specs["phase2"].log.parent.mkdir(parents=True, exist_ok=True)
        specs["phase2"].log.write_text(
            "phase2 skipped because phase1 failed\n", encoding="utf-8"
        )
        exit_codes["phase2"] = -1
    payload = audit_smoke_artifacts(
        specs,
        exit_codes=exit_codes,
        source_commit=_source_commit(),
        invocation=[sys.executable, *sys.argv],
    )
    _write_json(root / "gate_results.json", payload)
    return payload


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=LOCKED_SEED)
    parser.add_argument("--episodes", type=int, default=LOCKED_EPISODES)
    parser.add_argument("--resume-split", type=int, default=LOCKED_RESUME_SPLIT)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        payload = run_smoke(args)
    except SmokeRootNotFreshError as error:
        print(f"{type(error).__name__}: {error}", file=sys.stderr)
        return 1
    except BaseException as error:
        root = args.root.expanduser().resolve()
        payload = {
            "schema_version": SCHEMA_VERSION,
            "source_commit": _source_commit(),
            "command": [sys.executable, *sys.argv],
            "commands": {},
            "resolved_config": {
                "seed": args.seed,
                "episodes": args.episodes,
                "resume_split": args.resume_split,
            },
            "exit_codes": {},
            "artifacts": {},
            "assertions": {"execution_completed": False},
            "errors": [f"{type(error).__name__}: {error}"],
            "passed": False,
        }
        _write_json(root / "gate_results.json", payload)
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
