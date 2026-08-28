"""Single-seed runner definitions for the exploratory env-v3-safe comparison."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from baselines.MAPPO.safe_gru_trainer import SafeGRUMAPPOTrainer
from scripts.env_v3_safe_overrides import env_v3_safe_overrides, hydra_override_arg


# Derived from the final twenty stochastic daily costs of the env-v3 balanced
# run (median / deterministic cost = 0.495, clipped to rho=1.0) and its
# empirical balanced reference budget.  Keeping both values explicit makes
# the v4 course reproducible without reading historical run files at launch.
V4_CURRICULUM_RHO = 1.0
V4_EVAL_BALANCED_BUDGET = 0.4662967833789416
V4_CURRICULUM_TARGET = V4_CURRICULUM_RHO * V4_EVAL_BALANCED_BUDGET


EXPERIMENTS = {
    "dense_ff_mappo_anneal": {
        "family": "ff",
        "algorithm": "mappo",
        "seed": 30,
        "exploratory": True,
    },
    "dense_gru_mappo_anneal": {
        "family": "gru",
        "algorithm": "mappo",
        "seed": 30,
        "exploratory": True,
    },
    "dense_gru_mappo_lagrangian": {
        "family": "gru",
        "algorithm": "lagrangian",
        "seed": 30,
        "exploratory": True,
    },
    "dense_gru_macpo": {
        "family": "gru",
        "algorithm": "macpo",
        "seed": 30,
        "exploratory": True,
    },
    "dense_gru_macpo_history_communication": {
        "family": "gru",
        "algorithm": "macpo",
        "seed": 30,
        "exploratory": True,
        "include_previous_action": True,
        "include_transaction_message": True,
    },
    "dense_gru_macpo_two_stage_intent_d_strict": {
        "family": "gru",
        "algorithm": "macpo",
        "seed": 30,
        "exploratory": True,
        "include_previous_action": True,
        "include_transaction_message": True,
        "two_stage_intent": True,
        "intent_dim": 3,
        "intent_broadcast_mode": "full",
        # Replaced by the empirical safety reference before a formal run.
        "cost_budget": 0.0,
        "budget_mode": "strict",
    },
    "dense_gru_macpo_two_stage_intent_d_balanced": {
        "family": "gru",
        "algorithm": "macpo",
        "seed": 30,
        "exploratory": True,
        "include_previous_action": True,
        "include_transaction_message": True,
        "two_stage_intent": True,
        "intent_dim": 3,
        "intent_broadcast_mode": "full",
        "cost_budget": 0.3,
        "budget_mode": "balanced",
    },
    "dense_gru_macpo_two_stage_intent_no_broadcast_d_balanced": {
        "family": "gru",
        "algorithm": "macpo",
        "seed": 30,
        "exploratory": True,
        "include_previous_action": True,
        "include_transaction_message": True,
        "two_stage_intent": True,
        "intent_dim": 3,
        "intent_broadcast_mode": "other_zero",
        "cost_budget": 0.3,
        "budget_mode": "balanced",
    },
    "v4_full_intent_curriculum": {
        "family": "gru",
        "algorithm": "macpo",
        "seed": 30,
        "exploratory": True,
        "include_previous_action": True,
        "include_transaction_message": True,
        "two_stage_intent": True,
        "intent_dim": 3,
        "intent_broadcast_mode": "full",
        "intent_residual_limit": 0.25,
        "intent_residual_coef": 0.01,
        "cost_budget": V4_CURRICULUM_TARGET,
        "curriculum_d_start": 3.5,
        "curriculum_d_target": V4_CURRICULUM_TARGET,
        "curriculum_updates": 200,
        "curriculum_log_std_start": -1.0,
        "curriculum_log_std_end": -2.3,
        "env_overrides": {
            "h2_day_ahead_forecast_enable": True,
            "h2_day_ahead_forecast_horizons": [4, 6, 10],
        },
    },
    "v4_no_broadcast_curriculum": {
        "family": "gru",
        "algorithm": "macpo",
        "seed": 30,
        "exploratory": True,
        "include_previous_action": True,
        "include_transaction_message": True,
        "two_stage_intent": True,
        "intent_dim": 3,
        "intent_broadcast_mode": "other_zero",
        "intent_residual_limit": 0.25,
        "intent_residual_coef": 0.01,
        "cost_budget": V4_CURRICULUM_TARGET,
        "curriculum_d_start": 3.5,
        "curriculum_d_target": V4_CURRICULUM_TARGET,
        "curriculum_updates": 200,
        "curriculum_log_std_start": -1.0,
        "curriculum_log_std_end": -2.3,
        "env_overrides": {
            "h2_day_ahead_forecast_enable": True,
            "h2_day_ahead_forecast_horizons": [4, 6, 10],
        },
    },
    "v5_supply_intent_full": {
        "family": "gru", "algorithm": "macpo", "seed": 30, "exploratory": True,
        "include_previous_action": True, "include_transaction_message": True,
        "two_stage_intent": True, "intent_dim": 3, "intent_broadcast_mode": "full",
        "intent_residual_limit": 0.25, "intent_residual_coef": 0.01,
        "h2_supply_intent_message_enable": True, "cost_budget": 0.0,
        "env_overrides": {"h2_supply_intent_message_enable": True},
    },
    "v5_supply_intent_no_broadcast": {
        "family": "gru", "algorithm": "macpo", "seed": 30, "exploratory": True,
        "include_previous_action": True, "include_transaction_message": True,
        "two_stage_intent": True, "intent_dim": 3, "intent_broadcast_mode": "other_zero",
        "intent_residual_limit": 0.25, "intent_residual_coef": 0.01,
        "h2_supply_intent_message_enable": True, "cost_budget": 0.0,
        "env_overrides": {"h2_supply_intent_message_enable": True},
    },
    "v52_full_gru_mappo": {
        "family": "gru", "algorithm": "mappo", "seed": 30, "exploratory": True,
        "include_previous_action": True, "include_transaction_message": True,
        "two_stage_intent": True, "intent_dim": 3,
        "intent_broadcast_mode": "full", "communication_scope": "full",
        "intent_residual_limit": 0.25, "intent_residual_coef": 0.01,
        "h2_supply_intent_message_enable": True,
        "curriculum_updates": 300,
        "curriculum_log_std_start": -1.0, "curriculum_log_std_end": -2.3,
        "env_overrides": {"h2_supply_intent_message_enable": True, "soc_init": 0.5},
    },
    "v52_full_gru_macpo": {
        "family": "gru", "algorithm": "macpo", "seed": 30, "exploratory": True,
        "include_previous_action": True, "include_transaction_message": True,
        "two_stage_intent": True, "intent_dim": 3,
        "intent_broadcast_mode": "full", "communication_scope": "full",
        "intent_residual_limit": 0.25, "intent_residual_coef": 0.01,
        "h2_supply_intent_message_enable": True,
        "cost_budget": 0.0, "budget_mode": "v52_nominal_curriculum",
        "curriculum_d_start": 0.0, "curriculum_d_target": 0.0,
        "curriculum_updates": 300,
        "curriculum_log_std_start": -1.0, "curriculum_log_std_end": -2.3,
        "env_overrides": {"h2_supply_intent_message_enable": True, "soc_init": 0.5},
    },
    "v52_self_only_gru_macpo": {
        "family": "gru", "algorithm": "macpo", "seed": 30, "exploratory": True,
        "include_previous_action": True, "include_transaction_message": True,
        "two_stage_intent": True, "intent_dim": 3,
        "intent_broadcast_mode": "self_only", "communication_scope": "self_only",
        "intent_residual_limit": 0.25, "intent_residual_coef": 0.01,
        "h2_supply_intent_message_enable": True,
        "cost_budget": 0.0, "budget_mode": "v52_nominal_curriculum",
        "curriculum_d_start": 0.0, "curriculum_d_target": 0.0,
        "curriculum_updates": 300,
        "curriculum_log_std_start": -1.0, "curriculum_log_std_end": -2.3,
        "env_overrides": {"h2_supply_intent_message_enable": True, "soc_init": 0.5},
    },
    "v6_nocomm_gru_mappo": {
        "family": "gru", "algorithm": "mappo", "seed": 30, "exploratory": True,
        "num_envs": 2, "env_parallel_backend": "process", "fused_rollout_kernel": True,
        "include_previous_action": True, "include_transaction_message": False,
        "two_stage_intent": False, "h2_supply_intent_message_enable": False,
        "cost_budget": 1.0, "voltage_cost_scale": 0.02,
        "curriculum_updates": 300,
        "curriculum_log_std_start": -1.0, "curriculum_log_std_end": -2.3,
        "env_overrides": {"h2_local_supply_facts_enable": True, "soc_init": 0.5},
    },
    "v6_nocomm_gru_mappo_penalty": {
        "family": "gru", "algorithm": "mappo_penalty", "seed": 30, "exploratory": True,
        "num_envs": 2, "env_parallel_backend": "process", "fused_rollout_kernel": True,
        "include_previous_action": True, "include_transaction_message": False,
        "two_stage_intent": False, "h2_supply_intent_message_enable": False,
        "cost_budget": 1.0, "voltage_cost_scale": 0.02,
        "fixed_cost_penalty_coef": 1.0,
        "curriculum_updates": 300,
        "curriculum_log_std_start": -1.0, "curriculum_log_std_end": -2.3,
        "env_overrides": {"h2_local_supply_facts_enable": True, "soc_init": 0.5},
    },
    "v6_nocomm_gru_macpo": {
        "family": "gru", "algorithm": "macpo", "seed": 30, "exploratory": True,
        "num_envs": 2, "env_parallel_backend": "process", "fused_rollout_kernel": True,
        "include_previous_action": True, "include_transaction_message": False,
        "two_stage_intent": False, "h2_supply_intent_message_enable": False,
        "cost_budget": 1.0, "voltage_cost_scale": 0.02,
        "curriculum_updates": 300,
        "curriculum_log_std_start": -1.0, "curriculum_log_std_end": -2.3,
        "env_overrides": {"h2_local_supply_facts_enable": True, "soc_init": 0.5},
    },
}


def build_gru_config(name: str, *, updates: int) -> dict[str, Any]:
    """Build the explicit config consumed by the independent GRU trainer."""
    spec = EXPERIMENTS[name]
    if spec["family"] != "gru":
        raise ValueError(f"{name} is an FF baseline, not a GRU variant")
    if updates < 1:
        raise ValueError("updates must be positive")
    env_overrides = env_v3_safe_overrides()
    env_overrides.update(dict(spec.get("env_overrides", {})))
    return {
        "seed": spec["seed"],
        "num_envs": int(spec.get("num_envs", 4)),
        "num_steps": 24,
        "total_updates": updates,
        "anneal_lr": True,
        "hidden_size": 128,
        "lr": 3e-4,
        "gamma": 1.0,
        "gae_lambda": 0.95,
        "clip_eps": 0.2,
        "entropy_coef": 0.01,
        "lagrange_lr": 0.05,
        "cost_budget": float(spec.get("cost_budget", 0.0)),
        "voltage_cost_scale": float(spec.get("voltage_cost_scale", 1.0)),
        "fixed_cost_penalty_coef": float(spec.get("fixed_cost_penalty_coef", 1.0)),
        "fused_rollout_kernel": bool(spec.get("fused_rollout_kernel", False)),
        "safety_budget_mode": spec.get("budget_mode", "fixed"),
        "macpo_max_kl": 0.01,
        "macpo_cg_iterations": 10,
        "macpo_damping": 1e-2,
        "include_previous_action": bool(spec.get("include_previous_action", False)),
        "include_transaction_message": bool(
            spec.get("include_transaction_message", False)
        ),
        "two_stage_intent": bool(spec.get("two_stage_intent", False)),
        "intent_dim": int(spec.get("intent_dim", 3)),
        "intent_broadcast_mode": str(spec.get("intent_broadcast_mode", "full")),
        "communication_scope": spec.get("communication_scope"),
        "intent_residual_limit": float(spec.get("intent_residual_limit", 0.25)),
        "intent_residual_coef": float(spec.get("intent_residual_coef", 0.0)),
        "h2_supply_intent_message_enable": bool(
            spec.get("h2_supply_intent_message_enable", False)
        ),
        "curriculum_d_start": spec.get("curriculum_d_start"),
        "curriculum_d_target": spec.get("curriculum_d_target"),
        "curriculum_updates": int(spec.get("curriculum_updates", 0)),
        "curriculum_log_std_start": spec.get("curriculum_log_std_start"),
        "curriculum_log_std_end": spec.get("curriculum_log_std_end"),
        "env_parallel_backend": str(spec.get("env_parallel_backend", "serial")),
        "env_overrides": env_overrides,
    }


def apply_env_v6_calibration(
    config: dict[str, Any], calibration: dict[str, Any]
) -> dict[str, Any]:
    """Apply one passing native Swiss MV calibration to a v6 trainer config."""
    if calibration.get("environment") != "env-v6-swiss":
        raise ValueError("Env-v6 training requires an env-v6-swiss calibration")
    selection = calibration.get("selection")
    if not calibration.get("feasible") or not isinstance(selection, dict):
        raise ValueError("Env-v6 training requires a passing physical gate")
    for key in ("pcc_injection_scale", "background_load_scale"):
        if not np.isclose(float(calibration.get(key, np.nan)), 1.0):
            raise ValueError(f"Env-v6 calibration {key} must be 1.0")
    economic_scale = float(calibration.get("economic_reward_scale_yuan", 0.0))
    cost_scale = float(calibration.get("training_cost_scale", 0.0))
    cost_budget = float(calibration.get("training_cost_budget", 0.0))
    if economic_scale <= 0.0 or cost_scale <= 0.0 or cost_budget <= 0.0:
        raise ValueError("Env-v6 calibration normalization scales must be positive")
    overrides = config["env_overrides"]
    overrides.update(
        {
            "power_flow_model": "swiss_mv",
            "power_flow_case_dir": str(selection["case_dir"]),
            "power_flow_pcc_bus_ids": [int(bus) for bus in selection["pcc_bus_ids"]],
            "power_flow_background_load_scale": 1.0,
            "power_flow_pcc_injection_scale": 1.0,
            "reward_scale": economic_scale,
        }
    )
    config["voltage_cost_scale"] = cost_scale
    config["cost_budget"] = cost_budget
    config["curriculum_d_start"] = None
    config["curriculum_d_target"] = None
    return config


def calibrated_safety_budgets(reference: dict[str, Any]) -> dict[str, float]:
    """Turn an empirical safe-reference report into daily MACPO budgets."""
    c_ref = float(reference["c_ref"])
    c_idle = float(reference["c_idle"])
    c_best = min(c_ref, c_idle)
    return {
        "strict": c_best + 0.10 * (c_idle - c_best),
        "balanced": c_best + 0.50 * (c_idle - c_best),
    }


def apply_safety_reference(
    config: dict[str, Any], reference: dict[str, Any]
) -> dict[str, Any]:
    """Apply legacy budgets or the Env-v5.2 nominal-to-zero curriculum."""
    mode = str(config["safety_budget_mode"])
    if mode in {"strict", "balanced"}:
        config["cost_budget"] = calibrated_safety_budgets(reference)[mode]
    elif mode == "v52_nominal_curriculum":
        selection = reference.get("selection")
        if not reference.get("feasible") or selection is None:
            raise ValueError("Env-v5.2 training requires a passing calibration")
        config["curriculum_d_start"] = float(
            selection["nominal_max_daily_cost"]
        )
        config["curriculum_d_target"] = 0.0
        config["cost_budget"] = 0.0
    return config


def reconcile_metrics_for_resume(
    metrics_path: str | Path,
    *,
    checkpoint_update: int,
) -> int:
    """Atomically discard metric rows newer than the restored checkpoint."""
    path = Path(metrics_path)
    if not path.exists():
        return 0
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line]
    rows = [json.loads(line) for line in lines]
    kept = [row for row in rows if int(row["update"]) <= int(checkpoint_update)]
    if checkpoint_update > 0 and not any(
        int(row["update"]) == int(checkpoint_update) for row in kept
    ):
        raise ValueError(
            f"metrics do not contain restored checkpoint update {checkpoint_update}"
        )
    removed = len(rows) - len(kept)
    if removed:
        temporary = path.with_name(path.name + ".resume-tmp")
        temporary.write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in kept),
            encoding="utf-8",
        )
        temporary.replace(path)
    return removed


def build_ff_command(name: str, *, updates: int) -> list[str]:
    """Return, without executing, the existing FF-MAPPO annealing command."""
    spec = EXPERIMENTS[name]
    if spec["family"] != "ff":
        raise ValueError(f"{name} is a GRU variant, not an FF baseline")
    root = Path(__file__).resolve().parents[1]
    total_timesteps = updates * 24 * 4
    return [
        sys.executable,
        "-m",
        "baselines.MAPPO.mappo_ff_shared_weights",
        "--config-name",
        "mappo_ff_independent_actors_microgrid",
        f"SEED={spec['seed']}",
        "NUM_SEEDS=1",
        f"TOTAL_TIMESTEPS={total_timesteps}",
        "ANNEAL_LR=true",
        "GAMMA=1.0",
        hydra_override_arg(env_v3_safe_overrides()),
    ]


def run(
    name: str,
    *,
    updates: int,
    dry_run: bool = False,
    run_dir: str | Path | None = None,
    checkpoint_interval: int = 25,
    validation_interval: int = 100,
    resume: str | Path | None = None,
    safety_reference: str | Path | None = None,
    env_v6_calibration: str | Path | None = None,
    env_parallel_backend: str | None = None,
    background_load_scale: float | None = None,
    pcc_injection_scale: float | None = None,
) -> dict[str, Any]:
    """Run one exploratory single-seed variant or return its launch description."""
    if name not in EXPERIMENTS:
        raise ValueError(f"Unknown experiment {name!r}")
    spec = EXPERIMENTS[name]
    output_root = Path(run_dir) if run_dir is not None else None
    if spec["family"] == "ff":
        if resume is not None:
            raise ValueError("FF resume is managed by its existing trainer, not this GRU runner")
        command = build_ff_command(name, updates=updates)
        if output_root is not None:
            checkpoint_root = output_root / "checkpoints"
            checkpoint_root.mkdir(parents=True, exist_ok=True)
            command.extend(
                [
                    f"CHECKPOINT_INTERVAL={checkpoint_interval * 24 * 4}",
                    f"+TRAINING_CHECKPOINT_PATH={checkpoint_root / (name + '.msgpack')}",
                ]
            )
        if not dry_run:
            subprocess.run(command, check=True)
        return {"variant": name, "exploratory": True, "command": command}

    if safety_reference is not None and env_v6_calibration is not None:
        raise ValueError("only one calibration interface may be provided")
    config = build_gru_config(name, updates=updates)
    if background_load_scale is not None:
        config["env_overrides"]["power_flow_background_load_scale"] = float(
            background_load_scale
        )
    if pcc_injection_scale is not None:
        config["env_overrides"]["power_flow_pcc_injection_scale"] = float(
            pcc_injection_scale
        )
    if safety_reference is not None:
        reference = json.loads(Path(safety_reference).read_text(encoding="utf-8"))
        apply_safety_reference(config, reference)
    if env_v6_calibration is not None:
        calibration = json.loads(
            Path(env_v6_calibration).read_text(encoding="utf-8")
        )
        apply_env_v6_calibration(config, calibration)
    if env_parallel_backend is not None:
        config["env_parallel_backend"] = str(env_parallel_backend)
    checkpoint_dir = (
        output_root / "checkpoints" / name if output_root is not None else None
    )
    metrics_path = (
        output_root / f"{name}.metrics.jsonl" if output_root is not None else None
    )
    if dry_run:
        return {
            "variant": name,
            "exploratory": True,
            "config": config,
            "target_updates": updates,
            "checkpoint_interval": checkpoint_interval,
            "validation_interval": validation_interval,
            "checkpoint_dir": str(checkpoint_dir) if checkpoint_dir is not None else None,
            "metrics_path": str(metrics_path) if metrics_path is not None else None,
            "resume": str(resume) if resume is not None else None,
        }
    trainer = SafeGRUMAPPOTrainer(config)
    try:
        start_update = 0
        if resume is not None:
            start_update = trainer.load_checkpoint(resume, algorithm=spec["algorithm"])
        if start_update > updates:
            raise ValueError("checkpoint update exceeds requested total updates")
        if resume is not None and metrics_path is not None:
            reconcile_metrics_for_resume(
                metrics_path,
                checkpoint_update=start_update,
            )
        rollout_root = output_root / "rollouts" / name if output_root is not None else None

        def save_validation(update: int) -> None:
            if rollout_root is None:
                return
            rollout_root.mkdir(parents=True, exist_ok=True)
            (rollout_root / f"update_{update:06d}.json").write_text(
                json.dumps(trainer.deterministic_rollout(seed=spec["seed"]), indent=2),
                encoding="utf-8",
            )

        metrics = trainer.train(
            updates - start_update,
            algorithm=spec["algorithm"],
            start_update=start_update,
            checkpoint_dir=checkpoint_dir,
            checkpoint_interval=checkpoint_interval,
            metrics_path=metrics_path,
            validation_interval=validation_interval if output_root is not None else 0,
            validation_callback=save_validation if output_root is not None else None,
        )
        report = trainer.deterministic_rollout(seed=spec["seed"])
    finally:
        trainer.close()
    result = {
        "variant": name,
        "exploratory": True,
        "target_updates": updates,
        "resumed_from_update": start_update,
        "dimensions": {
            "num_agents": trainer.num_agents,
            "action_dim": trainer.action_dim,
            "base_obs_dim": trainer.base_obs_dim,
            "actor_obs_dim": trainer.obs_dim,
        },
        "metrics": metrics,
        "deterministic_rollout": report,
    }
    if output_root is not None:
        output_root.mkdir(parents=True, exist_ok=True)
        (output_root / f"{name}.json").write_text(
            json.dumps(result, indent=2, default=str), encoding="utf-8"
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("variant", choices=sorted(EXPERIMENTS))
    parser.add_argument("--updates", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--checkpoint-interval", type=int, default=25)
    parser.add_argument("--validation-interval", type=int, default=100)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--safety-reference", type=Path)
    parser.add_argument("--env-v6-calibration", type=Path)
    parser.add_argument("--env-parallel-backend", choices=("serial", "process"))
    parser.add_argument("--background-load-scale", type=float)
    parser.add_argument("--pcc-injection-scale", type=float)
    args = parser.parse_args()
    print(
        json.dumps(
            run(
                args.variant,
                updates=args.updates,
                dry_run=args.dry_run,
                run_dir=args.run_dir,
                checkpoint_interval=args.checkpoint_interval,
                validation_interval=args.validation_interval,
                resume=args.resume,
                safety_reference=args.safety_reference,
                env_v6_calibration=args.env_v6_calibration,
                env_parallel_backend=args.env_parallel_backend,
                background_load_scale=args.background_load_scale,
                pcc_injection_scale=args.pcc_injection_scale,
            ),
            indent=2,
            default=str,
        )
    )


if __name__ == "__main__":
    main()
