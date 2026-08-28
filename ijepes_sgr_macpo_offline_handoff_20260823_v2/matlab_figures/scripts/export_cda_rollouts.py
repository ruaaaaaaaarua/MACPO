"""Replay the archived seed-32 checkpoints and export CDA diagnostics.

This is evaluation only: no training parameters or archived source files are
modified.  The script first checks each replay against the archived paper
summary, then writes full market diagnostics and a compact plotting CSV.
"""

from __future__ import annotations

import csv
from copy import deepcopy
import json
from pathlib import Path
import sys
from typing import Any

import jax.numpy as jnp
import numpy as np


PAPER_DIR = Path(__file__).resolve().parents[2]
WORKSPACE_DIR = PAPER_DIR.parent
ARCHIVE_DIR = WORKSPACE_DIR / "服务器归档_20260826" / "内容"
ARCHIVED_CODE_ROOT = ARCHIVE_DIR / "env_v6_1_dev" / "HyperMARL-main"
LOCAL_EVAL_CODE_ROOT = PAPER_DIR / "local_eval_code" / "HyperMARL-main"
CODE_ROOT = (
    LOCAL_EVAL_CODE_ROOT if LOCAL_EVAL_CODE_ROOT.is_dir() else ARCHIVED_CODE_ROOT
)
RUN_DIR = ARCHIVE_DIR / "env_v6_swiss_runs" / "long"
BEHAVIOR_REPORT_PATH = RUN_DIR / "env_v6_behavior_report.json"
CALIBRATION_PATH = ARCHIVE_DIR / "env_v6_swiss_runs" / "calibration.json"
SWISS_CASE_DIR = (
    ARCHIVE_DIR
    / "datasets"
    / "Swiss-PDGs"
    / "grids"
    / "matpower_data"
    / "MV"
    / "347_1"
)
DATA_DIR = PAPER_DIR / "matlab_figures" / "data"
DELIVERY_COUNTERFACTUAL_PATH = DATA_DIR / "seed32_delivery_counterfactual.json"

if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from baselines.MAPPO.safe_gru_trainer import SafeGRUMAPPOTrainer
from baselines.MAPPO.safe_recurrent import reset_actor_hidden
from baselines.utils.microgrid_vec_env import MicrogridVecEnv
from scripts.run_env_v3_safe_matrix import (
    EXPERIMENTS,
    apply_env_v6_calibration,
    build_gru_config,
)


VARIANTS = (
    ("v6_nocomm_gru_mappo", "MAPPO"),
    ("v6_nocomm_gru_mappo_penalty", "Fixed-penalty MAPPO"),
    ("v6_nocomm_gru_macpo", "SGR-MACPO"),
)

EXPECTED = {
    "v6_nocomm_gru_mappo": {
        "economic_cost": 872524.5943474465,
        "daily_voltage_cost": 0.9431825194005494,
        "voltage_min_pu": 0.930884488275922,
    },
    "v6_nocomm_gru_mappo_penalty": {
        "economic_cost": 7854344.25789804,
        "daily_voltage_cost": 0.0,
        "voltage_min_pu": 0.953348312534072,
    },
    "v6_nocomm_gru_macpo": {
        "economic_cost": 3817816.7437592787,
        "daily_voltage_cost": 0.0,
        "voltage_min_pu": 0.9572422358159942,
    },
}


def to_builtin(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): to_builtin(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_builtin(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def replay_with_market_info(
    trainer: SafeGRUMAPPOTrainer,
    seed: int,
    *,
    archived_actions: list[Any] | None = None,
) -> dict[str, Any]:
    """Mirror deterministic_rollout while retaining the environment market info."""
    overrides = dict(trainer.config["env_overrides"])
    env = MicrogridVecEnv(num_envs=1, auto_reset=False, config_overrides=overrides)
    records: list[dict[str, Any]] = []
    try:
        flat_obs, _ = env.reset(seed=int(seed))
        previous_actions = np.zeros(
            (1, trainer.num_agents, trainer.action_dim), dtype=np.float32
        )
        transaction_messages = np.zeros(
            (1, trainer.transaction_message_dim), dtype=np.float32
        )
        local_obs = trainer._reshape_local_obs(
            flat_obs,
            previous_actions=previous_actions,
            transaction_messages=transaction_messages,
            communication_scope=trainer.communication_scope,
        )
        actor_hidden = jnp.zeros(
            (1, trainer.num_agents, trainer.hidden_size), dtype=jnp.float32
        )
        done = jnp.ones(1, dtype=bool)
        episode_length = int(env.envs[0].env.T)

        for step in range(episode_length):
            actor_hidden = reset_actor_hidden(actor_hidden, done)
            means, _, next_hidden, _ = trainer.actor.apply(
                trainer.actor_state.params,
                local_obs,
                actor_hidden,
                return_intents=True,
                intent_broadcast_mode=None,
            )
            if archived_actions is None:
                action = np.asarray(jnp.tanh(means[0]), dtype=np.float32)
            else:
                action = np.asarray(archived_actions[step], dtype=np.float32)
            next_obs, _, termination, truncation, infos = env.step(
                action.reshape(-1, trainer.action_dim)
            )
            info = infos[0]
            keep = {
                "economic_cost",
                "total_cost",
                "terminal_settlement_cost",
                "voltage_cost",
                "voltage_min_pu",
                "voltage_max_pu",
                "pf_converged",
                "p_grid",
                "e_h2_ext",
                "pcc_p_kw",
                "net_electric_demand",
                "net_h2_demand",
                "elec_clearing_price",
                "h2_clearing_price",
                "elec_bid_price",
                "h2_bid_price",
                "elec_market_traded",
                "h2_market_traded",
                "cda_total_traded",
                "elec_market_buy_cost",
                "elec_market_sell_revenue",
                "h2_market_buy_cost",
                "h2_market_sell_revenue",
                "elec_cda_paid",
                "elec_cda_received",
                "h2_cda_paid",
                "h2_cda_received",
                "elec_market_trades",
                "h2_market_trades",
                "elec_market_agent_results",
                "h2_market_agent_results",
                "elec_open_buy_orders",
                "elec_open_sell_orders",
                "h2_open_buy_orders",
                "h2_open_sell_orders",
                "h2_buy_order_quantity_total",
                "h2_sell_order_quantity_total",
                "h2_planned_external_order_energy",
                "h2_emergency_buy_energy",
                "pending_h2_energy_total",
                "h2_level",
                "e_h2_load",
                "p_el",
            }
            record = {"step": step, "actions": action.tolist()}
            record.update({key: to_builtin(info[key]) for key in keep if key in info})
            records.append(record)

            done_np = np.logical_or(termination, truncation).reshape(
                1, trainer.num_agents
            ).any(axis=1)
            done = jnp.asarray(done_np)
            actor_hidden = reset_actor_hidden(next_hidden, done)
            previous_actions = np.where(
                done_np[:, None, None],
                np.zeros_like(action[None, :, :]),
                action[None, :, :],
            )
            transaction_messages = np.where(
                done_np[:, None],
                np.zeros_like(trainer._transaction_message_from_infos(infos)),
                trainer._transaction_message_from_infos(infos),
            )
            local_obs = trainer._reshape_local_obs(
                next_obs,
                previous_actions=previous_actions,
                transaction_messages=transaction_messages,
                communication_scope=trainer.communication_scope,
            )
            if bool(done_np[0]):
                break
    finally:
        env.close()

    finite_vmin = [
        float(row["voltage_min_pu"])
        for row in records
        if row.get("voltage_min_pu") is not None
    ]
    return {
        "summary": {
            "seed": seed,
            "steps": len(records),
            "economic_cost": sum(float(row["economic_cost"]) for row in records),
            "daily_voltage_cost": sum(float(row["voltage_cost"]) for row in records),
            "voltage_min_pu": min(finite_vmin),
            "electricity_cda_volume_mwh": sum(
                float(row["elec_market_traded"]) for row in records
            ) / 1000.0,
            "hydrogen_cda_volume_mwh": sum(
                float(row["h2_market_traded"]) for row in records
            ) / 1000.0,
            "planned_h2_order": sum(
                float(np.sum(row["h2_planned_external_order_energy"]))
                for row in records
            ),
            "emergency_h2_buy": sum(
                float(np.sum(row["h2_emergency_buy_energy"])) for row in records
            ),
            "pending_h2_energy": float(records[-1]["pending_h2_energy_total"]),
        },
        "steps": records,
    }


def assert_reference(variant: str, summary: dict[str, Any]) -> None:
    mismatches = {}
    for key, expected in EXPECTED[variant].items():
        actual = float(summary[key])
        tolerance = max(1.0e-7, abs(expected) * 1.0e-8)
        if abs(actual - expected) > tolerance:
            mismatches[key] = {"expected": expected, "actual": actual}
    if mismatches:
        raise RuntimeError(
            f"{variant} replay differs from the archived paper result: "
            + json.dumps(mismatches, sort_keys=True)
        )


def compact_row(label: str, record: dict[str, Any]) -> dict[str, Any]:
    p_grid = np.asarray(record.get("p_grid", []), dtype=float)
    h2_ext = np.asarray(record.get("e_h2_ext", []), dtype=float)
    return {
        "method": label,
        "hour": int(record["step"]) + 1,
        "electricity_cda_mwh": float(record["elec_market_traded"]) / 1000.0,
        "electricity_clearing_price": float(record["elec_clearing_price"]),
        "electricity_external_buy_mwh": float(np.maximum(p_grid, 0).sum()) / 1000.0,
        "electricity_external_sell_mwh": float(np.maximum(-p_grid, 0).sum()) / 1000.0,
        "hydrogen_cda_mwh": float(record["h2_market_traded"]) / 1000.0,
        "hydrogen_clearing_price": float(record["h2_clearing_price"]),
        "hydrogen_external_buy_mwh": float(np.maximum(h2_ext, 0).sum()) / 1000.0,
        "hydrogen_external_sell_mwh": float(np.maximum(-h2_ext, 0).sum()) / 1000.0,
        "minimum_voltage_pu": float(record["voltage_min_pu"]),
        "raw_voltage_cost": float(record["voltage_cost"]),
    }


def h2_delivery_row(scenario: str, record: dict[str, Any]) -> dict[str, Any]:
    return {
        "scenario": scenario,
        "hour": int(record["step"]) + 1,
        "h2_cda_mwh": float(record["h2_market_traded"]) / 1000.0,
        "h2_clearing_price": float(record["h2_clearing_price"]),
        "planned_external_order_mwh": float(
            np.sum(record["h2_planned_external_order_energy"])
        )
        / 1000.0,
        "emergency_supply_mwh": float(np.sum(record["h2_emergency_buy_energy"]))
        / 1000.0,
        "pending_energy_mwh": float(record["pending_h2_energy_total"]) / 1000.0,
        "tank_energy_mwh": float(np.sum(record["h2_level"])) * 33.33 / 1000.0,
        "electrolyzer_power_mw": float(np.sum(record["p_el"])) / 1000.0,
    }


def assert_counterfactual_match(
    result: dict[str, Any], authoritative: dict[str, Any]
) -> None:
    keys = (
        "economic_cost",
        "daily_voltage_cost",
        "voltage_min_pu",
        "planned_h2_order",
        "emergency_h2_buy",
        "pending_h2_energy",
    )
    mismatches = {}
    for key in keys:
        actual = float(result["summary"][key])
        expected = float(authoritative["summary"][key])
        tolerance = max(1.0e-7, abs(expected) * 1.0e-8)
        if abs(actual - expected) > tolerance:
            mismatches[key] = {"expected": expected, "actual": actual}
    if mismatches:
        raise RuntimeError(
            "local H2 delivery replay differs from the archived counterfactual: "
            + json.dumps(mismatches, sort_keys=True)
        )


def main() -> None:
    calibration = json.loads(CALIBRATION_PATH.read_text(encoding="utf-8"))
    behavior_report = json.loads(BEHAVIOR_REPORT_PATH.read_text(encoding="utf-8"))
    delivery_counterfactual = json.loads(
        DELIVERY_COUNTERFACTUAL_PATH.read_text(encoding="utf-8")
    )
    full_report: dict[str, Any] = {
        "seed": 32,
        "comparison": "same archived checkpoints and deterministic evaluation day",
        "variants": {},
    }
    csv_rows: list[dict[str, Any]] = []
    h2_delivery_report: dict[str, Any] = {
        "seed": 32,
        "variant": "v6_nocomm_gru_macpo",
        "comparison": "same checkpoint and seed; physical H2 delivery timing only",
        "scenarios": {},
    }
    h2_delivery_rows: list[dict[str, Any]] = []

    for variant, label in VARIANTS:
        # Keep the exact training configuration for checkpoint identity, while
        # relocating only execution details and the Swiss case path for this
        # Windows evaluation host.  These changes do not alter network shapes
        # or policy semantics.
        checkpoint_config = build_gru_config(variant, updates=1000)
        apply_env_v6_calibration(checkpoint_config, calibration)
        runtime_config = deepcopy(checkpoint_config)
        runtime_config["num_envs"] = 1
        runtime_config["env_parallel_backend"] = "serial"
        runtime_config["fused_rollout_kernel"] = False
        runtime_config["env_overrides"]["power_flow_case_dir"] = str(SWISS_CASE_DIR)
        trainer = SafeGRUMAPPOTrainer(runtime_config)
        checkpoint = RUN_DIR / "checkpoints" / variant / "update_001000.msgpack"
        try:
            runtime_merged_config = trainer.config
            checkpoint_merged_config = deepcopy(runtime_merged_config)
            checkpoint_merged_config.update(checkpoint_config)
            checkpoint_merged_config["env_overrides"] = deepcopy(
                checkpoint_config["env_overrides"]
            )
            trainer.config = checkpoint_merged_config
            restored = trainer.load_checkpoint(
                checkpoint, algorithm=str(EXPERIMENTS[variant]["algorithm"])
            )
            trainer.config = runtime_merged_config
            authoritative_rollout = behavior_report["variants"][variant]["rollouts"]["32"]
            archived_actions = [
                row["actions"] for row in authoritative_rollout["steps"]
            ]
            result = replay_with_market_info(
                trainer,
                seed=32,
                archived_actions=archived_actions,
            )
            if variant == "v6_nocomm_gru_macpo":
                assert_counterfactual_match(
                    result, delivery_counterfactual["delayed"]
                )
                instant_config = deepcopy(runtime_merged_config)
                instant_config["env_overrides"] = deepcopy(
                    runtime_merged_config["env_overrides"]
                )
                instant_config["env_overrides"][
                    "h2_instant_delivery_counterfactual"
                ] = True
                trainer.config = instant_config
                instant_actions = [
                    row["actions"]
                    for row in delivery_counterfactual["instant"]["steps"]
                ]
                instant_result = replay_with_market_info(
                    trainer,
                    seed=32,
                    archived_actions=instant_actions,
                )
                assert_counterfactual_match(
                    instant_result, delivery_counterfactual["instant"]
                )
                trainer.config = runtime_merged_config
        finally:
            trainer.config = runtime_config
            trainer.close()
        assert_reference(variant, result["summary"])
        result["label"] = label
        result["checkpoint"] = str(checkpoint)
        result["restored_update"] = int(restored)
        full_report["variants"][variant] = result
        csv_rows.extend(compact_row(label, row) for row in result["steps"])
        if variant == "v6_nocomm_gru_macpo":
            h2_delivery_report["scenarios"]["delayed"] = result
            h2_delivery_report["scenarios"]["instant"] = instant_result
            h2_delivery_rows.extend(
                h2_delivery_row("Delayed delivery", row) for row in result["steps"]
            )
            h2_delivery_rows.extend(
                h2_delivery_row("Instant delivery", row)
                for row in instant_result["steps"]
            )
        print(json.dumps({"variant": variant, **result["summary"]}, indent=2))

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    (DATA_DIR / "seed32_cda_market_rollouts.json").write_text(
        json.dumps(full_report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    fieldnames = list(csv_rows[0])
    with (DATA_DIR / "seed32_cda_market_rollouts.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)
    (DATA_DIR / "seed32_h2_cda_delivery_rollouts.json").write_text(
        json.dumps(h2_delivery_report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    h2_fieldnames = list(h2_delivery_rows[0])
    with (DATA_DIR / "seed32_h2_cda_delivery_rollouts.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=h2_fieldnames)
        writer.writeheader()
        writer.writerows(h2_delivery_rows)


if __name__ == "__main__":
    main()
