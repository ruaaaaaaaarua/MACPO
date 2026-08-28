#!/usr/bin/env python3
"""Roll out the best tuned STAS-MAPPO policy for H2 delay diagnostics."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import sys
from pathlib import Path
from typing import Any

import distrax
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training.train_state import TrainState
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from orbax.checkpoint import checkpointer
from orbax.checkpoint.pytree_checkpoint_handler import PyTreeCheckpointHandler

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    plt = None


HM_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = HM_ROOT.parent
if str(HM_ROOT) not in sys.path:
    sys.path.insert(0, str(HM_ROOT))

from baselines.MAPPO.mappo_ff_shared_weights import ActorCritic  # noqa: E402
from envs.microgrid.config import MICROGRID_CONFIG  # noqa: E402
from envs.microgrid.microgrid_continuous_env import MicrogridContinuousEnv  # noqa: E402
from scripts.microgrid_experiment_overrides import MICROGRID_EXPERIMENT_OVERRIDES  # noqa: E402


BEST_OVERRIDES = {
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

DEFAULT_CHECKPOINT = (
    "/tmp/models/microgrid__HPO-STAS-stage2_stas_candidate_1__seed30/"
    "hpo_mappo_stas_microgrid_240000_steps_2499_updates.agent_30_seed"
)

BASE_MICROGRID_CONFIG = copy.deepcopy(MICROGRID_CONFIG)


def set_nested(config: dict[str, Any], dotted_key: str, value: Any) -> None:
    parts = dotted_key.split(".")
    current = config
    for part in parts[:-1]:
        current = current.setdefault(part, {})
    current[parts[-1]] = value


def load_config() -> dict[str, Any]:
    config_dir = HM_ROOT / "baselines" / "STAS-MAPPO" / "config"
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        cfg = compose(config_name="stas_mappo_microgrid")
    config = OmegaConf.to_container(cfg, resolve=False)
    config["ENV_NAME"] = config["env"]["ENV_NAME"]
    config["EXP_NAME"] = "stas_mappo_microgrid"
    config["RUN_NAME"] = "rollout_best_stas_delay_analysis"
    config["GROUP"] = "microgrid_rollout_best_stas_delay_analysis"
    for key, value in BEST_OVERRIDES.items():
        set_nested(config, key, value)
    config["MICROGRID_CONFIG_OVERRIDES"] = dict(MICROGRID_EXPERIMENT_OVERRIDES)
    config["NUM_ENVS"] = 1
    config["SEED"] = 30
    config["ACTION_SPACE_TYPE"] = "continuous"
    return config


def price_overrides(mode: str) -> dict[str, float]:
    if mode == "kwh":
        return {}
    if mode == "kg30":
        lhv = float(BASE_MICROGRID_CONFIG.get("LHV_H2", 33.33))
        sell = 3.0 / lhv
        buy = 30.0 / lhv
        return {
            "lambda_h2": (sell + buy) / 2.0,
            "lambda_h2_buy": buy,
            "lambda_h2_sell": sell,
            "h2_price_min": sell,
            "h2_price_max": buy,
            "h2_price_init": (sell + buy) / 2.0,
        }
    raise ValueError(f"Unknown H2 price mode: {mode}")


def profile_overrides(profile: str) -> dict[str, Any]:
    if profile in {"", "none"}:
        return {}
    lhv = float(BASE_MICROGRID_CONFIG.get("LHV_H2", 33.33))
    group_a_overrides = {
        "lambda_h2": 16.5 / lhv,
        "lambda_h2_buy": 45.0 / lhv,
        "lambda_h2_sell": 3.0 / lhv,
        "h2_price_min": 3.0 / lhv,
        "h2_price_max": 30.0 / lhv,
        "h2_price_init": 16.5 / lhv,
        "external_h2_dependency_penalty_enable": True,
        "external_h2_dependency_penalty_kg": 15.0,
    }
    group_b_overrides = {
        "h2_learnable_rolling_order_enable": True,
        "h2_learnable_rolling_order_agent_indices": [2, 3],
        "h2_learnable_rolling_order_max_fraction": 0.25,
        "h2_buyer_reservation_demand_enable": False,
    }
    if profile == "group_a":
        return group_a_overrides
    if profile == "group_b":
        return group_b_overrides
    if profile == "group_ab":
        return {
            **group_a_overrides,
            **group_b_overrides,
        }
    if profile == "group_bc":
        return group_b_overrides
    if profile == "group_abc":
        return {
            **group_a_overrides,
            **group_b_overrides,
        }
    if profile == "group_c":
        return {}
    raise ValueError(f"Unknown experiment profile: {profile}")


def reset_microgrid_config(
    extra: dict[str, Any],
    price_mode: str = "kwh",
    experiment_profile: str = "none",
) -> None:
    MICROGRID_CONFIG.clear()
    MICROGRID_CONFIG.update(copy.deepcopy(BASE_MICROGRID_CONFIG))
    MICROGRID_CONFIG.update(copy.deepcopy(MICROGRID_EXPERIMENT_OVERRIDES))
    MICROGRID_CONFIG.update(price_overrides(price_mode))
    MICROGRID_CONFIG.update(profile_overrides(experiment_profile))
    MICROGRID_CONFIG.update(extra)


def append_agent_ids(obs: np.ndarray, num_agents: int) -> jnp.ndarray:
    obs_jnp = jnp.asarray(obs, dtype=jnp.float32)
    agent_ids = jnp.eye(num_agents, dtype=jnp.float32)
    return jnp.concatenate([obs_jnp, agent_ids], axis=-1)


def build_policy(
    config: dict[str, Any],
    checkpoint_dir: str,
    price_mode: str,
    experiment_profile: str,
):
    reset_microgrid_config(
        {}, price_mode=price_mode, experiment_profile=experiment_profile
    )
    probe_env = MicrogridContinuousEnv()
    num_agents = probe_env.num_agent
    obs_dim = probe_env.signal_obs_dim
    action_dim = probe_env.signal_action_dim
    probe_env.close()

    actor_obs_dim = obs_dim + num_agents
    critic_obs_dim = obs_dim * num_agents
    network = ActorCritic(
        action_dim,
        activation=config["ACTIVATION"],
        actor_layers=config.get("ACTOR_LAYERS"),
        critic_layers=config.get("CRITIC_LAYERS"),
        num_agents=num_agents,
        observation_dim=obs_dim,
        is_continuous=True,
        log_std_init=config.get("LOG_STD_INIT", 0.0),
    )
    rng = jax.random.PRNGKey(config["SEED"])
    params = network.init(
        rng,
        jnp.zeros((1, actor_obs_dim), dtype=jnp.float32),
        jnp.zeros((1, critic_obs_dim), dtype=jnp.float32),
    )
    tx = optax.set_to_zero()
    train_state = TrainState.create(apply_fn=network.apply, params=params, tx=tx)
    loaded = checkpointer.Checkpointer(
        PyTreeCheckpointHandler(aggregate_filename="checkpoints")
    ).restore(checkpoint_dir, item=train_state.params)

    @jax.jit
    def act(params, obs_for_policy):
        actor_obs = append_agent_ids(obs_for_policy, num_agents)
        dummy_critic = jnp.zeros((num_agents, critic_obs_dim), dtype=jnp.float32)
        actor_output, _ = network.apply(params, actor_obs, dummy_critic)
        actor_mean, actor_log_std = actor_output
        pi = distrax.MultivariateNormalDiag(
            loc=actor_mean,
            scale_diag=jnp.exp(actor_log_std),
        )
        return jnp.tanh(pi.mode())

    return loaded, act


def scenario_specs(learnable_rolling: bool = False) -> dict[str, dict[str, Any]]:
    target_ratios = [0.05, 0.05, 0.35, 0.45]
    rolling_on = {
        "h2_buyer_reservation_demand_enable": not learnable_rolling,
        "h2_delivery_reservation_enable": True,
    }
    return {
        "delay_no_buffer_no_rolling": {
            "initial_buffer": False,
            "hide_pending_obs": False,
            "overrides": {
                "h2_market_lag_enable": True,
                "h2_delivery_lag": 4,
                "h2_learnable_rolling_order_active": False,
                "h2_buyer_reservation_demand_enable": False,
                "h2_delivery_reservation_enable": False,
            },
        },
        "delay_initial_buffer_only": {
            "initial_buffer": True,
            "initial_buffer_ratios": target_ratios,
            "hide_pending_obs": False,
            "overrides": {
                "h2_market_lag_enable": True,
                "h2_delivery_lag": 4,
                "h2_learnable_rolling_order_active": False,
                "h2_buyer_reservation_demand_enable": False,
                "h2_delivery_reservation_enable": False,
            },
        },
        "delay_rolling_only": {
            "initial_buffer": False,
            "hide_pending_obs": False,
            "overrides": {
                "h2_market_lag_enable": True,
                "h2_delivery_lag": 4,
                "h2_learnable_rolling_order_active": True,
                **rolling_on,
            },
        },
        "delay_initial_buffer_plus_rolling": {
            "initial_buffer": True,
            "initial_buffer_ratios": target_ratios,
            "hide_pending_obs": False,
            "overrides": {
                "h2_market_lag_enable": True,
                "h2_delivery_lag": 4,
                "h2_learnable_rolling_order_active": True,
                **rolling_on,
            },
        },
        "delay_plus_rolling_pending_hidden": {
            "initial_buffer": True,
            "initial_buffer_ratios": target_ratios,
            "hide_pending_obs": True,
            "overrides": {
                "h2_market_lag_enable": True,
                "h2_delivery_lag": 4,
                "h2_learnable_rolling_order_active": True,
                **rolling_on,
            },
        },
        "instant_delivery_reference": {
            "initial_buffer": False,
            "hide_pending_obs": False,
            "overrides": {
                "h2_market_lag_enable": False,
                "h2_delivery_lag": 0,
                "h2_learnable_rolling_order_active": False,
                "h2_buyer_reservation_demand_enable": False,
                "h2_delivery_reservation_enable": False,
            },
        },
    }


def apply_initial_buffer(env: MicrogridContinuousEnv, ratios: list[float]) -> np.ndarray:
    raw_env = env.env
    target = raw_env.h2_tank_cap * np.asarray(ratios, dtype=np.float32)
    raw_env.h2_level = np.maximum(raw_env.h2_level, target).astype(np.float32)
    return np.stack(raw_env._get_obs()).astype(np.float32)


def policy_obs(obs: np.ndarray, hide_pending_obs: bool) -> np.ndarray:
    adjusted = np.asarray(obs, dtype=np.float32).copy()
    if hide_pending_obs and adjusted.shape[1] > 13:
        adjusted[:, 13:] = 0.0
    return adjusted


def run_episode(
    params,
    act_fn,
    scenario: str,
    spec: dict[str, Any],
    seed: int,
    price_mode: str,
    experiment_profile: str,
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    reset_microgrid_config(
        spec["overrides"], price_mode=price_mode, experiment_profile=experiment_profile
    )
    env = MicrogridContinuousEnv()
    env.seed(seed)
    obs = env.reset().astype(np.float32)
    if spec.get("initial_buffer"):
        obs = apply_initial_buffer(env, spec.get("initial_buffer_ratios", [0.05] * 4))

    rows: list[dict[str, Any]] = []
    done = np.zeros(env.num_agent, dtype=bool)
    step = 0
    while not np.any(done) and step < int(MICROGRID_CONFIG["episode_length"]):
        obs_for_policy = policy_obs(obs, bool(spec.get("hide_pending_obs", False)))
        action = np.asarray(act_fn(params, jnp.asarray(obs_for_policy)), dtype=np.float32)
        next_obs, reward, done, info = env.step(action)
        info0 = info[0]
        reward_value = float(np.asarray(reward, dtype=np.float32).reshape(-1).mean())
        for agent_id in range(env.num_agent):
            row = {
                "scenario": scenario,
                "seed": seed,
                "t": step,
                "agent": agent_id,
                "reward": reward_value,
                "action_p_el_norm": float(action[agent_id, 0]),
                "action_p_bat_norm": float(action[agent_id, 1]),
                "action_elec_price_norm": float(action[agent_id, 2]),
                "action_h2_price_norm": float(action[agent_id, 3]),
                "action_p_ht_norm": float(action[agent_id, 4]),
            }
            if action.shape[1] > 5:
                row["action_h2_rolling_order_norm"] = float(action[agent_id, 5])
            for key in [
                "p_el",
                "p_bat",
                "p_ht",
                "elec_bid_price",
                "h2_bid_price",
                "h2_level_ratio",
                "h2_level",
                "net_h2_demand",
                "h2_order_quantity_raw",
                "h2_order_quantity",
                "h2_learnable_rolling_order_extra",
                "h2_buyer_reservation_extra_order",
                "h2_buyer_reservation_shortfall",
                "h2_buy_clip_amount",
                "h2_delivery_received",
                "e_h2_ext",
                "pending_h2_energy_agent",
                "pending_adjusted_h2_headroom",
            ]:
                value = info0.get(key)
                if isinstance(value, list):
                    row[key] = float(value[agent_id])
            row.update(
                {
                    "h2_market_traded": float(info0.get("h2_market_traded", 0.0)),
                    "h2_pending_count": float(info0.get("h2_pending_count", 0)),
                    "pending_h2_energy_total": float(info0.get("pending_h2_energy_total", 0.0)),
                    "delivery_overflow_energy": float(info0.get("delivery_overflow_energy", 0.0)),
                }
            )
            buckets = info0.get("pending_h2_by_eta", [])
            if isinstance(buckets, list) and agent_id < len(buckets):
                for eta, value in enumerate(buckets[agent_id], start=1):
                    row[f"pending_eta_{eta}"] = float(value)
            rows.append(row)
        obs = np.asarray(next_obs, dtype=np.float32)
        step += 1

    env.close()
    metrics = summarize_rows(rows)
    return rows, metrics


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, float]:
    if not rows:
        return {}
    steps = sorted(set(int(row["t"]) for row in rows))
    agents = sorted(set(int(row["agent"]) for row in rows))
    per_step = []
    for t in steps:
        step_rows = [row for row in rows if int(row["t"]) == t]
        per_step.append(step_rows)
    reward = sum(float(step_rows[0]["reward"]) for step_rows in per_step)
    external_buy = sum(max(0.0, float(row.get("e_h2_ext", 0.0))) for row in rows)
    early_buy = sum(
        max(0.0, float(row.get("e_h2_ext", 0.0)))
        for row in rows
        if int(row["t"]) < 4
    )
    late_buy = sum(
        max(0.0, float(row.get("e_h2_ext", 0.0)))
        for row in rows
        if int(row["t"]) >= 4
    )
    orders = sum(float(row.get("h2_order_quantity", 0.0)) for row in rows)
    raw_orders = sum(float(row.get("h2_order_quantity_raw", 0.0)) for row in rows)
    traded = sum(
        float(step_rows[0].get("h2_market_traded", 0.0)) for step_rows in per_step
    )
    delivered = sum(float(row.get("h2_delivery_received", 0.0)) for row in rows)
    pending_mean = float(np.mean([row.get("pending_h2_energy_total", 0.0) for row in rows]))
    h2_ratio_consumer = [
        float(row.get("h2_level_ratio", 0.0))
        for row in rows
        if int(row["agent"]) in {2, 3}
    ]
    low_h2_hits = sum(1 for value in h2_ratio_consumer if value < 0.10)
    duplicate_pressure = 0
    under_order_pressure = 0
    for step_rows in per_step:
        for row in step_rows:
            if int(row["agent"]) not in {2, 3}:
                continue
            pending = float(row.get("pending_h2_energy_agent", 0.0))
            order = float(row.get("h2_order_quantity", 0.0))
            ext = max(0.0, float(row.get("e_h2_ext", 0.0)))
            if pending > 1000.0 and order > 1000.0:
                duplicate_pressure += 1
            if pending < 100.0 and ext > 1000.0:
                under_order_pressure += 1
    return {
        "episode_return": reward,
        "external_h2_buy_total": external_buy,
        "external_h2_buy_first_lag_window": early_buy,
        "external_h2_buy_after_lag": late_buy,
        "h2_order_total": orders,
        "h2_order_raw_total": raw_orders,
        "h2_internal_traded_total": traded,
        "h2_delivery_received_total": delivered,
        "pending_h2_energy_total_mean": pending_mean,
        "consumer_h2_ratio_mean": float(np.mean(h2_ratio_consumer)),
        "consumer_h2_ratio_min": float(np.min(h2_ratio_consumer)),
        "consumer_low_h2_hits": float(low_h2_hits),
        "duplicate_order_pressure_hits": float(duplicate_pressure),
        "under_order_pressure_hits": float(under_order_pressure),
        "num_steps": float(len(steps)),
        "num_agents": float(len(agents)),
    }


def aggregate_metrics(metrics: list[dict[str, float]]) -> dict[str, float]:
    keys = sorted({key for metric in metrics for key in metric})
    result = {}
    for key in keys:
        values = np.asarray([metric[key] for metric in metrics if key in metric], dtype=np.float64)
        result[f"{key}_mean"] = float(np.mean(values))
        result[f"{key}_std"] = float(np.std(values))
    return result


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_representative(rows: list[dict[str, Any]], out_dir: Path) -> None:
    if plt is None:
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    scenarios = sorted(set(row["scenario"] for row in rows))
    fig, axes = plt.subplots(4, 1, figsize=(12, 14), sharex=True)
    for scenario in scenarios:
        subset = [row for row in rows if row["scenario"] == scenario and row["seed"] == 1000]
        if not subset:
            continue
        times = sorted(set(int(row["t"]) for row in subset))
        external = [
            sum(max(0.0, float(row.get("e_h2_ext", 0.0))) for row in subset if int(row["t"]) == t)
            for t in times
        ]
        orders = [
            sum(float(row.get("h2_order_quantity", 0.0)) for row in subset if int(row["t"]) == t)
            for t in times
        ]
        pending = [
            max(float(row.get("pending_h2_energy_total", 0.0)) for row in subset if int(row["t"]) == t)
            for t in times
        ]
        h2_ratio = [
            np.mean([
                float(row.get("h2_level_ratio", 0.0))
                for row in subset
                if int(row["t"]) == t and int(row["agent"]) in {2, 3}
            ])
            for t in times
        ]
        axes[0].plot(times, external, marker="o", linewidth=1.5, label=scenario)
        axes[1].plot(times, orders, marker="o", linewidth=1.5, label=scenario)
        axes[2].plot(times, pending, marker="o", linewidth=1.5, label=scenario)
        axes[3].plot(times, h2_ratio, marker="o", linewidth=1.5, label=scenario)
    axes[0].set_ylabel("external H2 buy")
    axes[1].set_ylabel("H2 order")
    axes[2].set_ylabel("pending H2")
    axes[3].set_ylabel("consumer H2 ratio")
    axes[3].set_xlabel("hour")
    for ax in axes:
        ax.grid(alpha=0.25)
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out_dir / "representative_h2_delay_rollout.png", dpi=180)
    plt.close(fig)


def write_report(path: Path, summary: dict[str, Any]) -> None:
    best = summary["scenario_summary"]
    lines = [
        "# Best STAS Rollout Delay Analysis",
        "",
        "Checkpoint: `" + summary["checkpoint"] + "`",
        "",
        "## Scenario Means",
        "",
        "| scenario | return | ext H2 buy | first 4h buy | after-lag buy | order | delivered | pending mean | consumer H2 mean | low-H2 hits |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for scenario, metrics in best.items():
        lines.append(
            "| {scenario} | {ret:.3f} | {buy:.1f} | {early:.1f} | {late:.1f} | {order:.1f} | {deliv:.1f} | {pending:.1f} | {ratio:.3f} | {low:.1f} |".format(
                scenario=scenario,
                ret=metrics.get("episode_return_mean", float("nan")),
                buy=metrics.get("external_h2_buy_total_mean", float("nan")),
                early=metrics.get("external_h2_buy_first_lag_window_mean", float("nan")),
                late=metrics.get("external_h2_buy_after_lag_mean", float("nan")),
                order=metrics.get("h2_order_total_mean", float("nan")),
                deliv=metrics.get("h2_delivery_received_total_mean", float("nan")),
                pending=metrics.get("pending_h2_energy_total_mean_mean", float("nan")),
                ratio=metrics.get("consumer_h2_ratio_mean_mean", float("nan")),
                low=metrics.get("consumer_low_h2_hits_mean", float("nan")),
            )
        )
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- `delay_initial_buffer_plus_rolling` is the combined mechanism: higher initial consumer H2 inventory plus rolling reservation orders.",
            "- `delay_plus_rolling_pending_hidden` keeps the physical pending queue but zeros pending-related observation fields before policy inference.",
            "- `instant_delivery_reference` is a counterfactual reference without fixed delivery lag.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument(
        "--out-dir",
        default=str(REPO_ROOT / "result" / "rollout_best_stas_delay_analysis"),
    )
    parser.add_argument("--episodes", type=int, default=12)
    parser.add_argument("--seed-base", type=int, default=1000)
    parser.add_argument(
        "--h2-price-mode",
        choices=["kwh", "kg30"],
        default="kwh",
        help="kwh keeps 3-30 yuan/kWh_H2; kg30 converts 3-30 yuan/kg to yuan/kWh_H2.",
    )
    parser.add_argument(
        "--learnable-rolling",
        action="store_true",
        help="Use learnable rolling-order action instead of heuristic buyer reservation in rolling scenarios.",
    )
    parser.add_argument(
        "--experiment-profile",
        choices=["none", "group_a", "group_b", "group_c", "group_ab", "group_bc", "group_abc"],
        default="none",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    config = load_config()
    params, act_fn = build_policy(
        config, args.checkpoint, args.h2_price_mode, args.experiment_profile
    )
    rows: list[dict[str, Any]] = []
    summary: dict[str, Any] = {
        "checkpoint": args.checkpoint,
        "h2_price_mode": args.h2_price_mode,
        "h2_price_overrides": price_overrides(args.h2_price_mode),
        "experiment_profile": args.experiment_profile,
        "profile_overrides": profile_overrides(args.experiment_profile),
        "best_overrides": BEST_OVERRIDES,
        "scenario_summary": {},
        "episode_metrics": [],
    }
    learnable_rolling = args.learnable_rolling or args.experiment_profile in {
        "group_b",
        "group_ab",
        "group_bc",
        "group_abc",
    }
    for scenario, spec in scenario_specs(learnable_rolling).items():
        scenario_metrics = []
        for idx in range(args.episodes):
            seed = args.seed_base + idx
            episode_rows, metrics = run_episode(
                params,
                act_fn,
                scenario,
                spec,
                seed,
                args.h2_price_mode,
                args.experiment_profile,
            )
            rows.extend(episode_rows)
            scenario_metrics.append(metrics)
            summary["episode_metrics"].append(
                {"scenario": scenario, "seed": seed, **metrics}
            )
        summary["scenario_summary"][scenario] = aggregate_metrics(scenario_metrics)

    write_csv(out_dir / "rollout_steps.csv", rows)
    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    plot_representative(rows, out_dir / "plots")
    write_report(out_dir / "analysis_report.md", summary)
    print(json.dumps(summary["scenario_summary"], ensure_ascii=False, indent=2, sort_keys=True))
    print(f"wrote {out_dir}")


if __name__ == "__main__":
    main()
