#!/usr/bin/env python3
"""Run isolated STAS microgrid mechanism experiments A/B/C."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


HM_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = HM_ROOT.parent
DEFAULT_ROOT = REPO_ROOT / "result" / "stas_mechanism_ablation_kg30_20260709"
SEED = 30
LONGRUN_TIMESTEPS = 240_000
SMOKE_TIMESTEPS = 96
LHV_H2 = 33.33


BEST_STAS_OVERRIDES = {
    "LR": 0.0002,
    "ANNEAL_LR": False,
    "UPDATE_EPOCHS": 8,
    "NUM_MINIBATCHES": 4,
    "CLIP_EPS": 0.2,
    "ENT_COEF": 0.01,
    "GAE_LAMBDA": 0.98,
    "LOG_STD_INIT": -0.5,
    "MAX_GRAD_NORM": 10.0,
    "STAS.MIX_COEF": 0.1,
    "STAS.LR": 0.001,
    "STAS.BATCH_SIZE": 32,
    "STAS.UPDATE_FREQ": 4,
    "STAS.UPDATES_PER_STEP": 1,
    "STAS.WARMUP_ROLLOUTS": 8,
    "STAS.DROPOUT": 0.2,
}


BASE_ENV = {
    "episode_length": 24,
    "multi_day_episode_enable": False,
    "episode_days": 1,
    "day_boundary_interval": 24,
    "day_boundary_info_enable": True,
    "daily_truncation_enable": False,
    "italian_split_enable": True,
    "italian_split_name": "train",
    "terminal_h2_shortfall_value_enable": False,
    "pv_cap": [7500.0, 1500.0, 500.0, 2000.0],
    "wt_cap": [1500.0, 6000.0, 3000.0, 500.0],
    "load_h_peak": [750.0, 600.0, 2925.0, 3656.25],
    "elec_internal_cda_enable": True,
    "h2_internal_cda_enable": True,
    "gas_network_enable": False,
    "gas_price_dynamic_enable": False,
    "gas_price_bidirectional_enable": False,
    "gas_price_obs_enable": False,
    "gas_pressure_obs_enable": False,
    "h2_transport_loss": 0.0,
    "h2_market_schedule_enable": False,
    "h2_market_lag_enable": True,
    "h2_delivery_lag": 4,
    "h2_pending_obs_enable": True,
    "h2_pending_obs_horizon": 4,
    "h2_pending_summary_obs_enable": True,
    "h2_cap_aware_buy_enable": True,
    "h2_delivery_reservation_enable": True,
    "h2_delivery_reservation_horizon": 4,
    "h2_delivery_reservation_ratio": 1.0,
    "h2_buyer_reservation_demand_enable": True,
    "h2_buyer_reservation_agent_indices": [2, 3],
    "h2_buyer_reservation_target_ratios": [0.0, 0.0, 0.35, 0.45],
    "h2_buyer_reservation_demand_gain": 1.0,
    "h2_buyer_reservation_max_order_fraction": 0.25,
}


KG30_PRICE = {
    "lambda_h2": 16.5 / LHV_H2,
    "lambda_h2_buy": 30.0 / LHV_H2,
    "lambda_h2_sell": 3.0 / LHV_H2,
    "h2_price_min": 3.0 / LHV_H2,
    "h2_price_max": 30.0 / LHV_H2,
    "h2_price_init": 16.5 / LHV_H2,
}


@dataclass(frozen=True)
class ExperimentSpec:
    group: str
    description: str
    env_overrides: dict[str, Any]
    stas_overrides: dict[str, Any]
    rollout_profile: str
    learnable_rolling: bool = False


def planned_experiments() -> list[ExperimentSpec]:
    group_b_feature_env = {
        "h2_buyer_reservation_demand_enable": False,
        "h2_learnable_rolling_order_enable": True,
        "h2_action_order_max_peak_hours": 1.0,
        "h2_learnable_rolling_order_agent_indices": [0, 1, 2, 3],
    }
    group_a_env = {
        **BASE_ENV,
        **KG30_PRICE,
        "lambda_h2_buy": 45.0 / LHV_H2,
        "external_h2_dependency_penalty_enable": True,
        "external_h2_dependency_penalty_kg": 15.0,
    }
    group_b_env = {
        **BASE_ENV,
        **KG30_PRICE,
        **group_b_feature_env,
    }
    group_ab_env = {**group_a_env, **group_b_feature_env}
    group_abc_env = {
        **group_ab_env,
        "italian_split_strategy": "manifest",
        "penalty_enable": False,
        "low_inventory_penalty_enable": False,
        "terminal_h2_floor_penalty_enable": False,
        "terminal_h2_shortfall_value_enable": False,
        "terminal_soc_floor_penalty_enable": False,
        "terminal_battery_salvage_enable": False,
        "stepwise_h2_floor_penalty_enable": False,
        "action_reg_enable": False,
        "h2_internal_trade_bonus_enable": False,
    }
    group_c_stas = {
        **BEST_STAS_OVERRIDES,
        "STAS.MIX_COEF": 0.02,
        "+STAS.MIX_COEF_SCHEDULE": [
            {"episode": 0, "coef": 0.02},
            {"episode": 2000, "coef": 0.02},
            {"episode": 6000, "coef": 0.10},
            {"episode": 10000, "coef": 0.20},
        ],
    }
    return [
        ExperimentSpec(
            "group_a",
            "kg30 internal price + emergency external H2 + external dependency penalty",
            group_a_env,
            BEST_STAS_OVERRIDES,
            "group_a",
        ),
        ExperimentSpec(
            "group_b",
            "kg30 price + complete action-controlled H2 buy order, no heuristic reservation",
            group_b_env,
            BEST_STAS_OVERRIDES,
            "group_b",
            learnable_rolling=True,
        ),
        ExperimentSpec(
            "group_c",
            "kg30 price + STAS mix schedule only",
            {**BASE_ENV, **KG30_PRICE},
            group_c_stas,
            "group_c",
        ),
        ExperimentSpec(
            "group_ab",
            "Group A external H2 penalty + Group B action-controlled H2 orders",
            group_ab_env,
            BEST_STAS_OVERRIDES,
            "group_ab",
            learnable_rolling=True,
        ),
        ExperimentSpec(
            "group_bc",
            "Group B action-controlled H2 orders + Group C STAS mix schedule",
            group_b_env,
            group_c_stas,
            "group_bc",
            learnable_rolling=True,
        ),
        ExperimentSpec(
            "group_abc",
            "Group A external H2 penalty + Group B action-controlled H2 orders + Group C STAS mix schedule",
            group_abc_env,
            group_c_stas,
            "group_abc",
            learnable_rolling=True,
        ),
    ]


def hydra_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, list):
        return "[" + ",".join(hydra_value(item) for item in value) + "]"
    if isinstance(value, dict):
        return "{" + ",".join(f"{key}:{hydra_value(val)}" for key, val in value.items()) + "}"
    return str(value)


def microgrid_override_arg(overrides: dict[str, Any]) -> str:
    body = ",".join(f"{key}:{hydra_value(value)}" for key, value in overrides.items())
    return f"+MICROGRID_CONFIG_OVERRIDES={{{body}}}"


def stas_override_args(overrides: dict[str, Any]) -> list[str]:
    return [f"{key}={hydra_value(value)}" for key, value in overrides.items()]


def timestamp() -> str:
    return datetime.utcnow().isoformat() + "Z"


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")


def experiment_dir(root: Path, spec: ExperimentSpec, smoke: bool) -> Path:
    return root / ("smoke" if smoke else "longrun") / spec.group


def train_command(spec: ExperimentSpec, out_dir: Path, total_timesteps: int, smoke: bool) -> list[str]:
    run_tag = "smoke" if smoke else "10k"
    alg = f"STAS-MAPPO-{spec.group}-{run_tag}"
    cmd = [
        sys.executable,
        "baselines/STAS-MAPPO/mappo_stas.py",
        "--config-name=stas_mappo_microgrid",
        f"ALG={alg}",
        f"EXP_NAME=stas_mechanism_{spec.group}",
        f"RUN_NAME=microgrid__{alg}__seed{SEED}",
        f"SEED={SEED}",
        f"TOTAL_TIMESTEPS={total_timesteps}",
        "WANDB_MODE=disabled",
        "EVAL_INTERVAL=100000000",
        "CAPTURE_VIDEO_INTERVAL=null",
        "CHECKPOINT=True",
        f"CHECKPOINT_INTERVAL={96 if smoke else 96000}",
        microgrid_override_arg(spec.env_overrides),
        *stas_override_args(spec.stas_overrides),
    ]
    return cmd


def run_command(command: list[str], cwd: Path, env: dict[str, str], log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.run(command, cwd=cwd, env=env, stdout=log, stderr=subprocess.STDOUT)
    return int(proc.returncode)


def score_returns(path: Path) -> dict[str, float]:
    arr = np.load(path).reshape(-1).astype(np.float64)
    window = min(arr.size, 125)
    rolling = np.convolve(arr, np.ones(window) / window, mode="valid")
    return {
        "num_points": int(arr.size),
        "final_return": float(arr[-1]),
        "mean_return": float(np.mean(arr)),
        "final_500_mean": float(np.mean(arr[-window:])),
        "best_rolling_500": float(np.max(rolling)),
    }


def expected_returns_file(out_dir: Path, spec: ExperimentSpec, smoke: bool) -> Path:
    run_tag = "smoke" if smoke else "10k"
    alg = f"STAS-MAPPO-{spec.group}-{run_tag}"
    return out_dir / "returns" / f"returns_microgrid_{alg}.npy"


def latest_checkpoint(spec: ExperimentSpec) -> Path:
    model_dir = Path(f"/tmp/models/microgrid__STAS-MAPPO-{spec.group}-10k__seed{SEED}")
    return model_dir / f"stas_mechanism_{spec.group}_240000_steps_2499_updates.agent_{SEED}_seed"


def run_training(root: Path, spec: ExperimentSpec, smoke: bool, force: bool) -> dict[str, Any]:
    tdir = experiment_dir(root, spec, smoke)
    result_path = tdir / "train_result.json"
    if result_path.exists() and not force:
        existing = json.loads(result_path.read_text(encoding="utf-8"))
        if existing.get("status") == "success":
            print(f"[skip] {spec.group} {'smoke' if smoke else 'longrun'} already succeeded")
            return existing
    out_dir = (tdir / "output").resolve()
    progress_log = (tdir / "progress.jsonl").resolve()
    total_timesteps = SMOKE_TIMESTEPS if smoke else LONGRUN_TIMESTEPS
    command = train_command(spec, out_dir, total_timesteps, smoke)
    write_json(
        tdir / "trial_config.json",
        {
            "created_at": timestamp(),
            "spec": asdict(spec),
            "command": command,
            "total_timesteps": total_timesteps,
            "output_dir": str(out_dir),
            "progress_log": str(progress_log),
        },
    )
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{HM_ROOT}:{env.get('PYTHONPATH', '')}"
    env["WANDB_MODE"] = "disabled"
    env["HYPERMARL_OUTPUT_DIR"] = str(out_dir)
    env["HYPERMARL_PROGRESS_LOG"] = str(progress_log)
    env["XLA_PYTHON_CLIENT_MEM_FRACTION"] = env.get("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.70")
    start = time.time()
    rc = run_command(command, HM_ROOT, env, tdir / "train.log")
    elapsed = time.time() - start
    returns_file = expected_returns_file(out_dir, spec, smoke)
    status = "success" if rc == 0 and returns_file.exists() else "failed"
    result: dict[str, Any] = {
        "group": spec.group,
        "phase": "smoke" if smoke else "longrun",
        "status": status,
        "returncode": rc,
        "elapsed_seconds": elapsed,
        "returns_file": str(returns_file),
        "train_log": str(tdir / "train.log"),
        "finished_at": timestamp(),
    }
    if returns_file.exists():
        result.update(score_returns(returns_file))
    write_json(result_path, result)
    append_jsonl(root / "manifest.jsonl", {"event": "train_finished", **result})
    print(f"[{status}] {spec.group} {'smoke' if smoke else 'longrun'}")
    return result


def run_rollout(root: Path, spec: ExperimentSpec, force: bool) -> dict[str, Any]:
    rdir = (root / "rollout" / spec.group).resolve()
    result_path = rdir / "rollout_result.json"
    if result_path.exists() and not force:
        existing = json.loads(result_path.read_text(encoding="utf-8"))
        if existing.get("status") == "success":
            print(f"[skip] {spec.group} rollout already succeeded")
            return existing
    checkpoint = latest_checkpoint(spec)
    command = [
        sys.executable,
        "scripts/rollout_best_stas_delay_analysis.py",
        "--episodes",
        "12",
        "--h2-price-mode",
        "kg30",
        "--experiment-profile",
        spec.rollout_profile,
        "--checkpoint",
        str(checkpoint),
        "--out-dir",
        str(rdir),
    ]
    if spec.learnable_rolling:
        command.append("--learnable-rolling")
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{HM_ROOT}:{env.get('PYTHONPATH', '')}"
    start = time.time()
    rc = run_command(command, HM_ROOT, env, rdir / "rollout.log")
    elapsed = time.time() - start
    summary_path = rdir / "summary.json"
    status = "success" if rc == 0 and summary_path.exists() else "failed"
    result = {
        "group": spec.group,
        "phase": "rollout",
        "status": status,
        "returncode": rc,
        "elapsed_seconds": elapsed,
        "checkpoint": str(checkpoint),
        "summary": str(summary_path),
        "rollout_log": str(rdir / "rollout.log"),
        "finished_at": timestamp(),
    }
    write_json(result_path, result)
    append_jsonl(root / "manifest.jsonl", {"event": "rollout_finished", **result})
    print(f"[{status}] {spec.group} rollout")
    return result


def summarize(root: Path) -> None:
    rows = []
    for result_file in sorted((root / "longrun").glob("*/train_result.json")):
        row = json.loads(result_file.read_text(encoding="utf-8"))
        rows.append(row)
    write_json(root / "summary.json", {"created_at": timestamp(), "longruns": rows})
    lines = ["# STAS Mechanism Ablation Summary", ""]
    lines.append("| group | status | final_500 | best_rolling_500 | final_return | returns |")
    lines.append("|---|---|---:|---:|---:|---|")
    for row in rows:
        lines.append(
            "| {group} | {status} | {final_500:.3f} | {best:.3f} | {final:.3f} | `{returns}` |".format(
                group=row.get("group"),
                status=row.get("status"),
                final_500=row.get("final_500_mean", float("nan")),
                best=row.get("best_rolling_500", float("nan")),
                final=row.get("final_return", float("nan")),
                returns=row.get("returns_file", ""),
            )
        )
    (root / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["dry-run", "smoke", "longrun", "rollout", "all", "summarize"])
    parser.add_argument("--root", default=str(DEFAULT_ROOT))
    parser.add_argument("--groups", default="group_a,group_b,group_c")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    specs = [spec for spec in planned_experiments() if spec.group in set(args.groups.split(","))]
    write_json(root / "experiment_plan.json", {"created_at": timestamp(), "experiments": [asdict(spec) for spec in specs]})
    if args.mode == "dry-run":
        for spec in specs:
            print(" ".join(train_command(spec, experiment_dir(root, spec, False) / "output", LONGRUN_TIMESTEPS, False)))
        return
    if args.mode in {"smoke", "all"}:
        for spec in specs:
            result = run_training(root, spec, smoke=True, force=args.force)
            if result["status"] != "success":
                raise SystemExit(f"smoke failed for {spec.group}")
    if args.mode in {"longrun", "all"}:
        for spec in specs:
            result = run_training(root, spec, smoke=False, force=args.force)
            if result["status"] != "success":
                raise SystemExit(f"longrun failed for {spec.group}")
    if args.mode in {"rollout", "all"}:
        for spec in specs:
            result = run_rollout(root, spec, force=args.force)
            if result["status"] != "success":
                raise SystemExit(f"rollout failed for {spec.group}")
    if args.mode in {"summarize", "all"}:
        summarize(root)


if __name__ == "__main__":
    main()
