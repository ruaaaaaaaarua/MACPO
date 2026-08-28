"""Run the gated Env-v6 Swiss MV no-communication experiment matrix."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
RUNNER = ROOT / "scripts" / "run_env_v3_safe_matrix.py"
BENCHMARK = ROOT / "scripts" / "benchmark_env_v6_rollout.py"
ANALYZER = ROOT / "scripts" / "analyze_env_v6_swiss.py"

MAPPO = "v6_nocomm_gru_mappo"
PENALTY_MAPPO = "v6_nocomm_gru_mappo_penalty"
MACPO = "v6_nocomm_gru_macpo"
VARIANTS = (MAPPO, PENALTY_MAPPO, MACPO)
CPU_SETS = {MAPPO: "0-4", PENALTY_MAPPO: "5-9", MACPO: "10-14"}


def runtime_env(cache_dir: Path) -> dict[str, str]:
    env = dict(os.environ)
    env.update(
        {
            "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
            "XLA_PYTHON_CLIENT_MEM_FRACTION": "0.30",
            "JAX_COMPILATION_CACHE_DIR": str(cache_dir),
            "OMP_NUM_THREADS": "5",
            "OPENBLAS_NUM_THREADS": "5",
            "MKL_NUM_THREADS": "5",
        }
    )
    return env


def training_command(
    variant: str,
    *,
    updates: int,
    run_dir: Path,
    calibration_path: Path,
    cpu_set: str | None = None,
    resume: Path | None = None,
) -> list[str]:
    command: list[str] = []
    if cpu_set is not None:
        command.extend(("taskset", "-c", cpu_set))
    command.extend(
        (
            sys.executable,
            str(RUNNER),
            variant,
            "--updates",
            str(int(updates)),
            "--run-dir",
            str(run_dir),
            "--checkpoint-interval",
            "25",
            "--validation-interval",
            "100",
            "--env-v6-calibration",
            str(calibration_path),
        )
    )
    if resume is not None:
        command.extend(("--resume", str(resume)))
    return command


def _finite(value: Any) -> bool:
    if value is None or isinstance(value, (bool, str)):
        return True
    if isinstance(value, (int, float)):
        return math.isfinite(float(value))
    if isinstance(value, list):
        return all(_finite(item) for item in value)
    if isinstance(value, dict):
        return all(_finite(item) for item in value.values())
    return False


def _latest_checkpoint(run_dir: Path, variant: str) -> Path:
    checkpoints = sorted((run_dir / "checkpoints" / variant).glob("update_*.msgpack"))
    if not checkpoints:
        raise RuntimeError(f"{variant}: checkpoint is missing")
    return checkpoints[-1]


def validate_smoke(
    run_dir: Path,
    variant: str,
    *,
    expected_updates: int,
    calibration: dict[str, Any],
) -> dict[str, Any]:
    from baselines.MAPPO.safe_gru_trainer import SafeGRUMAPPOTrainer
    from scripts.run_env_v3_safe_matrix import (
        EXPERIMENTS,
        apply_env_v6_calibration,
        build_gru_config,
    )

    metrics_path = run_dir / f"{variant}.metrics.jsonl"
    result_path = run_dir / f"{variant}.json"
    rows = (
        [json.loads(line) for line in metrics_path.read_text().splitlines() if line]
        if metrics_path.exists()
        else []
    )
    if len(rows) != int(expected_updates) or not all(_finite(row) for row in rows):
        raise RuntimeError(f"{variant}: smoke metrics missing or non-finite")
    cost_scale = float(calibration["training_cost_scale"])
    reward_scale = float(calibration["economic_reward_scale_yuan"])
    for row in rows:
        raw = float(row["daily_voltage_cost_raw"])
        normalized = float(row["daily_voltage_cost_normalized"])
        if not math.isclose(raw / cost_scale, normalized, rel_tol=5e-7, abs_tol=1e-7):
            raise RuntimeError(f"{variant}: raw/normalized cost mismatch")
        if not math.isclose(float(row["cost_budget_raw"]), 0.02, abs_tol=1e-12):
            raise RuntimeError(f"{variant}: raw budget mismatch")
        if not math.isclose(float(row["cost_budget_normalized"]), 1.0, abs_tol=1e-12):
            raise RuntimeError(f"{variant}: normalized budget mismatch")
        raw_reward = float(row["daily_economic_return_raw_yuan"])
        normalized_reward = float(row["daily_economic_return_normalized"])
        if not math.isclose(
            raw_reward / reward_scale,
            normalized_reward,
            rel_tol=5e-7,
            abs_tol=1e-7,
        ):
            raise RuntimeError(f"{variant}: raw/normalized reward mismatch")
    if variant == MACPO:
        kls = [float(row["kl_after"]) for row in rows]
        if not kls or max(kls) > 0.010001:
            raise RuntimeError(f"{variant}: KL exceeded 0.01")
    else:
        kls = []
    if not result_path.exists():
        raise RuntimeError(f"{variant}: result is missing")
    result = json.loads(result_path.read_text())
    summary = result["deterministic_rollout"]["summary"]
    dimensions = result["dimensions"]
    if int(summary["steps"]) != 24 or float(summary["pf_failure_rate"]) != 0.0:
        raise RuntimeError(f"{variant}: evaluation power flow did not converge for 24h")
    if int(dimensions["num_agents"]) != 4 or int(dimensions["action_dim"]) <= 0:
        raise RuntimeError(f"{variant}: invalid action dimensions")

    config = build_gru_config(variant, updates=expected_updates)
    apply_env_v6_calibration(config, calibration)
    if (
        config["include_transaction_message"]
        or config["two_stage_intent"]
        or config["h2_supply_intent_message_enable"]
        or config["env_parallel_backend"] != "process"
        or config["num_envs"] != 2
    ):
        raise RuntimeError(f"{variant}: communication or parallel config is wrong")
    checkpoint = _latest_checkpoint(run_dir, variant)
    trainer = SafeGRUMAPPOTrainer(config)
    try:
        restored = trainer.load_checkpoint(
            checkpoint, algorithm=str(EXPERIMENTS[variant]["algorithm"])
        )
    finally:
        trainer.close()
    if restored != int(expected_updates):
        raise RuntimeError(f"{variant}: restored update {restored}, expected {expected_updates}")
    return {
        "metrics_rows": len(rows),
        "checkpoint": checkpoint.name,
        "restored_update": restored,
        "max_kl": max(kls) if kls else None,
        "dimensions": dimensions,
        "pf_failure_rate": summary["pf_failure_rate"],
    }


def _write_phase(run_dir: Path, report: dict[str, Any]) -> dict[str, Any]:
    (run_dir / "phase_launch.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    return report


def launch(
    *,
    run_dir: Path,
    calibration_path: Path,
    smoke_updates: int = 100,
    long_updates: int = 1000,
    benchmark_rollouts: int = 5,
) -> dict[str, Any]:
    run_dir.mkdir(parents=True, exist_ok=True)
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    if (
        calibration.get("environment") != "env-v6-swiss"
        or not calibration.get("feasible")
        or calibration.get("selection") is None
    ):
        return _write_phase(
            run_dir,
            {
                "environment": "env-v6-swiss",
                "training_started": False,
                "failure_layer": "physical_feasibility_gate",
                "calibration": str(calibration_path),
                "static_summary": calibration.get("static_summary"),
                "dynamic_candidates": calibration.get("dynamic_candidates", []),
            },
        )

    process_env = runtime_env(run_dir / "jax_cache")
    performance_path = run_dir / "performance_gate.json"
    performance_log = run_dir / "performance_gate.log"
    performance = (
        json.loads(performance_path.read_text()) if performance_path.exists() else None
    )
    benchmark_returncode = 0
    if not performance or not performance.get("passed"):
        with performance_log.open("w", encoding="utf-8") as handle:
            completed = subprocess.run(
                [
                    sys.executable,
                    str(BENCHMARK),
                    "--calibration",
                    str(calibration_path),
                    "--output",
                    str(performance_path),
                    "--rollouts",
                    str(int(benchmark_rollouts)),
                ],
                cwd=ROOT,
                env=process_env,
                stdout=handle,
                stderr=subprocess.STDOUT,
            )
        benchmark_returncode = completed.returncode
        performance = (
            json.loads(performance_path.read_text()) if performance_path.exists() else None
        )
    if benchmark_returncode != 0 or not performance or not performance.get("passed"):
        return _write_phase(
            run_dir,
            {
                "environment": "env-v6-swiss",
                "training_started": False,
                "failure_layer": "rollout_performance_gate",
                "performance": performance,
                "log": str(performance_log),
            },
        )

    smoke_dir = run_dir / "smoke"
    smoke_dir.mkdir(exist_ok=True)
    smokes: dict[str, Any] = {}
    for variant in VARIANTS:
        metrics_path = smoke_dir / f"{variant}.metrics.jsonl"
        result_path = smoke_dir / f"{variant}.json"
        existing_rows = (
            len([line for line in metrics_path.read_text().splitlines() if line])
            if metrics_path.exists()
            else 0
        )
        if result_path.exists() and existing_rows == int(smoke_updates):
            smokes[variant] = validate_smoke(
                smoke_dir,
                variant,
                expected_updates=smoke_updates,
                calibration=calibration,
            )
            continue
        resume = None
        if existing_rows:
            checkpoints = sorted(
                (smoke_dir / "checkpoints" / variant).glob("update_*.msgpack")
            )
            if not checkpoints:
                return _write_phase(
                    run_dir,
                    {
                        "environment": "env-v6-swiss",
                        "training_started": False,
                        "failure_layer": "smoke_resume",
                        "failed_variant": variant,
                        "reason": "partial metrics exist without a checkpoint",
                    },
                )
            resume = checkpoints[-1]
        log_path = smoke_dir / f"{variant}.launch.log"
        with log_path.open("w", encoding="utf-8") as handle:
            completed = subprocess.run(
                training_command(
                    variant,
                    updates=smoke_updates,
                    run_dir=smoke_dir,
                    calibration_path=calibration_path,
                    resume=resume,
                ),
                cwd=ROOT,
                env=process_env,
                stdout=handle,
                stderr=subprocess.STDOUT,
            )
        if completed.returncode != 0:
            return _write_phase(
                run_dir,
                {
                    "environment": "env-v6-swiss",
                    "training_started": False,
                    "failure_layer": "smoke_training",
                    "failed_variant": variant,
                    "log": str(log_path),
                    "smokes": smokes,
                },
            )
        try:
            smokes[variant] = validate_smoke(
                smoke_dir,
                variant,
                expected_updates=smoke_updates,
                calibration=calibration,
            )
        except Exception as error:
            return _write_phase(
                run_dir,
                {
                    "environment": "env-v6-swiss",
                    "training_started": False,
                    "failure_layer": "smoke_validation",
                    "failed_variant": variant,
                    "reason": str(error),
                    "smokes": smokes,
                },
            )

    long_dir = run_dir / "long"
    long_dir.mkdir(exist_ok=True)
    launched: dict[str, tuple[subprocess.Popen[Any], Any]] = {}
    for variant in VARIANTS:
        handle = (long_dir / f"{variant}.launch.log").open("w", encoding="utf-8")
        process = subprocess.Popen(
            training_command(
                variant,
                updates=long_updates,
                run_dir=long_dir,
                calibration_path=calibration_path,
                cpu_set=CPU_SETS[variant],
            ),
            cwd=ROOT,
            env=process_env,
            stdout=handle,
            stderr=subprocess.STDOUT,
        )
        launched[variant] = (process, handle)
    exit_codes = {variant: process.wait() for variant, (process, _) in launched.items()}
    for _, handle in launched.values():
        handle.close()
    report: dict[str, Any] = {
        "environment": "env-v6-swiss",
        "training_started": True,
        "calibration": calibration["selection"],
        "performance": performance,
        "smokes": smokes,
        "long_updates": int(long_updates),
        "exit_codes": exit_codes,
    }
    if all(code == 0 for code in exit_codes.values()):
        analysis = subprocess.run(
            [
                sys.executable,
                str(ANALYZER),
                "--run-dir",
                str(long_dir),
                "--updates",
                str(int(long_updates)),
                "--calibration",
                str(calibration_path),
            ],
            cwd=ROOT,
            env=process_env,
        )
        report["analysis_exit_code"] = analysis.returncode
        report["behavior_report"] = str(long_dir / "env_v6_behavior_report.json")
    else:
        report["failure_layer"] = "long_training_process"
    return _write_phase(run_dir, report)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--smoke-updates", type=int, default=100)
    parser.add_argument("--long-updates", type=int, default=1000)
    parser.add_argument("--benchmark-rollouts", type=int, default=5)
    args = parser.parse_args()
    print(
        json.dumps(
            launch(
                run_dir=args.run_dir,
                calibration_path=args.calibration,
                smoke_updates=args.smoke_updates,
                long_updates=args.long_updates,
                benchmark_rollouts=args.benchmark_rollouts,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
