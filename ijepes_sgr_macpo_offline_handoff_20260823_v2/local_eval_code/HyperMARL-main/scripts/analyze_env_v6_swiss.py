"""Evaluate Env-v6 Swiss MV safety, timing, economics, and local history use."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from baselines.MAPPO.safe_gru_trainer import SafeGRUMAPPOTrainer
from scripts.run_env_v3_safe_matrix import (
    EXPERIMENTS,
    apply_env_v6_calibration,
    build_gru_config,
)


MAPPO = "v6_nocomm_gru_mappo"
PENALTY_MAPPO = "v6_nocomm_gru_mappo_penalty"
MACPO = "v6_nocomm_gru_macpo"
VARIANTS = (MAPPO, PENALTY_MAPPO, MACPO)
EVALUATION_SEEDS = (30, 31, 32)
RAW_BUDGET = 0.02


def _latest_checkpoint(run_dir: Path, variant: str) -> Path:
    checkpoints = sorted((run_dir / "checkpoints" / variant).glob("update_*.msgpack"))
    if not checkpoints:
        raise FileNotFoundError(f"no checkpoint found for {variant}")
    return checkpoints[-1]


def _tail_training_summary(run_dir: Path, variant: str) -> dict[str, Any]:
    path = run_dir / f"{variant}.metrics.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines() if line]
    tail = rows[-200:]
    recovery = [str(row.get("mode")) == "cost_recovery" for row in tail]
    return {
        "metrics_rows": len(rows),
        "tail_rows": len(tail),
        "tail_cost_recovery_share": float(np.mean(recovery)) if recovery else None,
        "tail_raw_voltage_cost_mean": (
            float(np.mean([row["daily_voltage_cost_raw"] for row in tail]))
            if tail
            else None
        ),
        "tail_raw_economic_cost_mean": (
            float(np.mean([row["daily_economic_cost_raw_yuan"] for row in tail]))
            if tail and "daily_economic_cost_raw_yuan" in tail[0]
            else None
        ),
    }


def evaluate_variant(
    run_dir: Path,
    variant: str,
    *,
    updates: int,
    calibration: dict[str, Any],
) -> dict[str, Any]:
    config = build_gru_config(variant, updates=updates)
    apply_env_v6_calibration(config, calibration)
    trainer = SafeGRUMAPPOTrainer(config)
    try:
        checkpoint = _latest_checkpoint(run_dir, variant)
        restored = trainer.load_checkpoint(
            checkpoint, algorithm=str(EXPERIMENTS[variant]["algorithm"])
        )
        rollouts = {
            str(seed): trainer.deterministic_rollout(seed=seed)
            for seed in EVALUATION_SEEDS
        }
        counterfactuals: dict[str, Any] = {}
        if variant == MACPO:
            counterfactuals = {
                "gru_hidden_off": trainer.deterministic_rollout(
                    seed=30, gru_hidden_off=True
                ),
                "previous_action_off": trainer.deterministic_rollout(
                    seed=30, previous_action_off=True
                ),
                "eta_plus_2h": trainer.deterministic_rollout(
                    seed=30, eta_delay_hours=2
                ),
            }
    finally:
        trainer.close()
    summaries = {seed: report["summary"] for seed, report in rollouts.items()}
    safe_days = sum(
        float(summary["daily_voltage_cost"]) <= RAW_BUDGET
        and float(summary["pf_failure_rate"]) == 0.0
        and int(summary["steps"]) == 24
        for summary in summaries.values()
    )
    return {
        "variant": variant,
        "algorithm": EXPERIMENTS[variant]["algorithm"],
        "checkpoint": checkpoint.name,
        "restored_update": restored,
        "safe_days": int(safe_days),
        "three_day_voltage_cost": float(
            sum(float(summary["daily_voltage_cost"]) for summary in summaries.values())
        ),
        "all_power_flows_converged": all(
            float(summary["pf_failure_rate"]) == 0.0
            and int(summary["steps"]) == 24
            for summary in summaries.values()
        ),
        "training": _tail_training_summary(run_dir, variant),
        "summaries": summaries,
        "rollouts": rollouts,
        "counterfactuals": counterfactuals,
    }


def analyze(
    *, run_dir: Path, updates: int, calibration_path: Path
) -> dict[str, Any]:
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    if not calibration.get("feasible") or calibration.get("selection") is None:
        raise ValueError("Env-v6 analysis requires a passing calibration")
    variants = {
        variant: evaluate_variant(
            run_dir, variant, updates=updates, calibration=calibration
        )
        for variant in VARIANTS
    }
    macpo = variants[MACPO]
    recovery = macpo["training"]["tail_cost_recovery_share"]
    criteria = {
        "macpo_seed30_within_raw_budget": (
            float(macpo["summaries"]["30"]["daily_voltage_cost"]) <= RAW_BUDGET
        ),
        "macpo_at_least_two_safe_days": int(macpo["safe_days"]) >= 2,
        "macpo_lower_three_day_violation_than_controls": all(
            float(macpo["three_day_voltage_cost"])
            < float(variants[control]["three_day_voltage_cost"])
            for control in (MAPPO, PENALTY_MAPPO)
        ),
        "all_evaluation_power_flows_converged": all(
            bool(result["all_power_flows_converged"])
            for result in variants.values()
        ),
        "macpo_tail_cost_recovery_below_90_percent": (
            recovery is not None and float(recovery) < 0.9
        ),
    }
    passed = all(criteria.values())
    return {
        "environment": "env-v6-swiss",
        "raw_voltage_budget": RAW_BUDGET,
        "calibration": calibration["selection"],
        "variants": variants,
        "success_criteria": criteria,
        "passed": bool(passed),
        "failure_layer": None if passed else "trained_policy_behavior",
        "next_action": (
            "plan_multi_seed_training"
            if passed
            else "stop_without_more_updates_and_inspect_recorded_trajectories"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--updates", type=int, default=1000)
    parser.add_argument("--calibration", type=Path, required=True)
    args = parser.parse_args()
    report = analyze(
        run_dir=args.run_dir,
        updates=int(args.updates),
        calibration_path=args.calibration,
    )
    output = args.run_dir / "env_v6_behavior_report.json"
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "report": str(output),
                "passed": report["passed"],
                "success_criteria": report["success_criteria"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
