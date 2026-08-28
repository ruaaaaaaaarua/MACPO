#!/usr/bin/env python3
"""Action rationality analysis: storage + internal market (FullCDA = inter-MG P2P)."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import distrax
import jax
import jax.numpy as jnp
import numpy as np
from orbax.checkpoint import checkpointer
from orbax.checkpoint.pytree_checkpoint_handler import PyTreeCheckpointHandler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from envs.microgrid.microgrid_continuous_env import MicrogridContinuousEnv
from scripts.microgrid_experiment_overrides import MICROGRID_EXPERIMENT_OVERRIDES

EPS = 1e-3


def _apply_overrides() -> None:
    from envs.microgrid.config import MICROGRID_CONFIG

    MICROGRID_CONFIG.update(dict(MICROGRID_EXPERIMENT_OVERRIDES))


def _find_checkpoints(root: Path, latest_only: bool) -> List[Path]:
    matches = sorted(p for p in root.glob("**/*_steps_*_updates.*") if p.is_dir())
    if not matches:
        return []
    return [matches[-1]] if latest_only else matches


def _load_params(checkpoint_dir: Path) -> Any:
    ckpt = checkpointer.Checkpointer(
        PyTreeCheckpointHandler(aggregate_filename="checkpoints")
    )
    return ckpt.restore(str(checkpoint_dir))


def _append_agent_ids(obs: np.ndarray, num_agents: int) -> jnp.ndarray:
    obs_j = jnp.asarray(obs, dtype=jnp.float32)
    agent_ids = jnp.tile(jnp.eye(num_agents, dtype=jnp.float32), (1, 1))
    return jnp.concatenate([obs_j, agent_ids], axis=-1)


def _actions_mappo_ia(params: Any, obs: np.ndarray, num_agents: int, action_dim: int) -> np.ndarray:
    from baselines.MAPPO.mappo_ff_shared_weights import ActorCritic

    obs_dim = obs.shape[-1]
    network = ActorCritic(
        action_dim=action_dim,
        num_agents=num_agents,
        observation_dim=obs_dim,
        activation="tanh",
        actor_layers=[128, 128],
        critic_layers=[128, 128],
        is_continuous=True,
        log_std_init=-1.0,
    )
    actor_obs = _append_agent_ids(obs, num_agents)
    dummy_critic = jnp.zeros((num_agents, num_agents * obs_dim), dtype=jnp.float32)
    actor_output, _ = network.apply(params, actor_obs, dummy_critic)
    actor_mean, _ = actor_output
    return np.asarray(jnp.tanh(actor_mean), dtype=np.float32)


def _actions_hypermarl(params: Any, obs: np.ndarray, num_agents: int, action_dim: int) -> np.ndarray:
    from baselines.MAPPO.mappo_ff_shared_weights_mlp_hypernets_microgrid import ActorCritic

    obs_dim = obs.shape[-1]
    network = ActorCritic(
        action_dim=action_dim,
        num_agents=num_agents,
        observation_dim=obs_dim,
        critic_obs_size=num_agents * obs_dim,
        actor_layers=[128, 128],
        critic_layers=[128, 128],
        embedding_dim=num_agents,
        init_scale=float(np.sqrt(2)),
        activation="tanh",
        use_agent_id_embeddings=False,
        use_bias_in_hypernet=True,
        hypernet_hidden_dims=[64],
        is_continuous=True,
        log_std_init=-1.0,
    )
    actor_obs = _append_agent_ids(obs, num_agents)
    dummy_critic = jnp.zeros((num_agents, num_agents * obs_dim), dtype=jnp.float32)
    actor_output, _ = network.apply(params, actor_obs, dummy_critic)
    actor_mean, _ = actor_output
    return np.asarray(jnp.tanh(actor_mean), dtype=np.float32)


def _renewable_ratio(inner, agent_idx: int, t: int) -> float:
    pv = float(inner.profiles["pv"][agent_idx, t])
    wt = float(inner.profiles["wt"][agent_idx, t])
    load_e = float(inner.profiles["load_e"][agent_idx, t])
    return (pv + wt) / max(load_e, 1e-6)


def _internal_trade_kwh(info: Dict[str, Any], agent_idx: int) -> Tuple[float, float]:
    """Internal market buy/sell quantity (kWh) from agent result summary."""
    bought = 0.0
    sold = 0.0
    for row in info.get("elec_market_agent_results", []):
        if int(row.get("agent_id", -1)) != agent_idx:
            continue
        qty = float(row.get("matched_quantity", 0.0))
        if row.get("side") == "buy":
            bought += qty
        elif row.get("side") == "sell":
            sold += qty
    return bought, sold


def _trade_direction_stats(info: Dict[str, Any], net_e: List[float]) -> Dict[str, int]:
    """Check internal CDA trades: surplus agent sells to deficit agent."""
    good = 0
    bad = 0
    for trade in info.get("elec_market_trades", []):
        buyer = int(trade.get("buyer_id", -1))
        seller = int(trade.get("seller_id", -1))
        qty = float(trade.get("quantity", 0.0))
        if qty <= EPS:
            continue
        buyer_deficit = net_e[buyer] > EPS
        seller_surplus = net_e[seller] < -EPS
        if buyer_deficit and seller_surplus:
            good += 1
        else:
            bad += 1
    return {"internal_trade_good_pairs": good, "internal_trade_bad_pairs": bad}


def rollout_policy(
    algorithm: str,
    checkpoint_dir: Path,
    seed: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    env = MicrogridContinuousEnv()
    env.seed(seed)
    inner = env.env
    num_agents = env.num_agent
    action_dim = env.signal_action_dim

    params = _load_params(checkpoint_dir)
    policy_fn = _actions_mappo_ia if algorithm == "mappo_ia" else _actions_hypermarl

    obs = env.reset()
    rows: List[Dict[str, Any]] = []
    total_reward = 0.0
    trade_good = 0
    trade_bad = 0

    for step in range(24):
        actions = np.clip(policy_fn(params, obs, num_agents, action_dim), -1.0, 1.0)
        next_obs, rewards, dones, infos = env.step(actions)
        info = infos[0] if isinstance(infos, list) else infos
        soc = info.get("soc", [0.0] * num_agents)
        p_bat = info.get("p_bat", [0.0] * num_agents)
        net_e = info.get("net_electric_demand", [0.0] * num_agents)
        p_grid = info.get("p_grid", [0.0] * num_agents)
        elec_traded = float(info.get("elec_market_traded", 0.0))

        tstats = _trade_direction_stats(info, net_e)
        trade_good += tstats["internal_trade_good_pairs"]
        trade_bad += tstats["internal_trade_bad_pairs"]

        for agent in range(num_agents):
            re_ratio = _renewable_ratio(inner, agent, step)
            internal_buy, internal_sell = _internal_trade_kwh(info, agent)
            surplus = float(net_e[agent]) < -EPS
            deficit = float(net_e[agent]) > EPS
            charging = float(p_bat[agent]) > 0.05
            discharging = float(p_bat[agent]) < -0.05
            high_re = re_ratio >= 1.0
            low_re = re_ratio < 0.8
            # Surplus not sold internally nor stored: possible curtail / inaction
            curtail_like = (
                surplus
                and high_re
                and internal_sell < EPS
                and not charging
                and float(p_grid[agent]) >= -EPS
            )

            rows.append(
                {
                    "step": step,
                    "agent": agent,
                    "re_ratio": re_ratio,
                    "soc": float(soc[agent]),
                    "p_bat": float(p_bat[agent]),
                    "net_electric_demand": float(net_e[agent]),
                    "p_grid": float(p_grid[agent]),
                    "internal_buy_kwh": internal_buy,
                    "internal_sell_kwh": internal_sell,
                    "elec_market_traded_step": elec_traded,
                    "surplus": surplus,
                    "deficit": deficit,
                    "charging": charging,
                    "discharging": discharging,
                    "high_re": high_re,
                    "low_re": low_re,
                    "curtail_like": curtail_like,
                }
            )
        total_reward += float(np.mean(rewards))
        obs = next_obs
        if np.all(dones):
            break

    summary = _summarize_rows(rows)
    summary.update(
        {
            "episode_return_mean": total_reward / max(len(rows) // num_agents, 1),
            "checkpoint": str(checkpoint_dir),
            "internal_trade_good_pairs_total": trade_good,
            "internal_trade_bad_pairs_total": trade_bad,
            "market_note": (
                "内部互济通过 FullCDA 实现；业务上即微电网间 P2P 交易，"
                "非 elec_p2p_enable 双边撮合器。"
            ),
        }
    )
    return rows, summary


def _summarize_rows(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    high_re = [r for r in rows if r["high_re"]]
    low_re = [r for r in rows if r["low_re"]]
    high_re_surplus = [r for r in high_re if r["surplus"]]

    def rate(subset: List[Dict[str, Any]], pred) -> float:
        if not subset:
            return 0.0
        return float(sum(1 for r in subset if pred(r)) / len(subset))

    return {
        "high_re_charge_rate": rate(high_re, lambda r: r["charging"]),
        "high_re_surplus_internal_sell_rate": rate(
            high_re_surplus, lambda r: r["internal_sell_kwh"] > EPS
        ),
        "low_re_discharge_rate": rate(low_re, lambda r: r["discharging"]),
        "deficit_agent_wrong_internal_sell_count": int(
            sum(1 for r in rows if r["deficit"] and r["internal_sell_kwh"] > EPS)
        ),
        "surplus_agent_wrong_internal_buy_count": int(
            sum(1 for r in rows if r["surplus"] and r["internal_buy_kwh"] > EPS)
        ),
        "high_re_surplus_no_charge_no_trade_count": int(
            sum(
                1
                for r in high_re_surplus
                if not r["charging"] and r["internal_sell_kwh"] <= EPS
            )
        ),
        "curtail_like_count": int(sum(1 for r in rows if r["curtail_like"])),
        "high_re_samples": len(high_re),
        "high_re_surplus_samples": len(high_re_surplus),
        "low_re_samples": len(low_re),
    }


def _write_csv(rows: List[Dict[str, Any]], path: Path) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _checkpoint_tag(path: Path) -> str:
    for part in path.name.split("."):
        if "_steps_" in part:
            return part
    return path.name


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=30)
    parser.add_argument("--all-checkpoints", action="store_true")
    args = parser.parse_args()

    _apply_overrides()
    out_root = args.out_root.resolve()
    action_dir = out_root / "action_analysis"
    periodic_dir = action_dir / "periodic"
    action_dir.mkdir(parents=True, exist_ok=True)
    periodic_dir.mkdir(parents=True, exist_ok=True)

    specs = {
        "MAPPO-IA-CTDE": {"algorithm": "mappo_ia", "search": out_root / "mappo_independent"},
        "HyperMARL-MAPPO": {"algorithm": "hypermarl_mappo", "search": out_root / "hypermarl_mappo"},
    }

    all_summaries: Dict[str, Any] = {}
    periodic_summaries: Dict[str, List[Dict[str, Any]]] = {}
    report_lines = [
        "# 动作合理性分析报告",
        "",
        "内部市场：FullCDA（微电网间 P2P 互济的业务实现）",
        "检查项：高 RE 充电 / 盈余内部售电 / 低 RE 放电 / 买卖方向 / 弃电与不交易",
        "",
    ]

    for name, spec in specs.items():
        ckpts = _find_checkpoints(spec["search"], latest_only=not args.all_checkpoints)
        if not ckpts:
            report_lines.append(f"## {name}\n- 未找到 checkpoint\n")
            continue

        periodic_summaries[name] = []
        final_rows: List[Dict[str, Any]] = []
        final_summary: Dict[str, Any] = {}

        for ckpt in ckpts:
            rows, summary = rollout_policy(spec["algorithm"], ckpt, args.seed)
            periodic_summaries[name].append({"checkpoint": _checkpoint_tag(ckpt), **summary})
            final_rows, final_summary = rows, summary

        slug = name.lower().replace("-", "_").replace(" ", "_")
        _write_csv(final_rows, action_dir / f"{slug}_step_actions.csv")
        all_summaries[name] = final_summary

        s = final_summary
        report_lines.extend(
            [
                f"## {name}",
                "",
                f"- checkpoint: `{s.get('checkpoint', '')}`",
                f"- 高 RE 时段充电比例: {s.get('high_re_charge_rate', 0):.2%} "
                f"(样本 {s.get('high_re_samples', 0)})",
                f"- 高 RE 且盈余时内部售电比例: {s.get('high_re_surplus_internal_sell_rate', 0):.2%} "
                f"(样本 {s.get('high_re_surplus_samples', 0)})",
                f"- 低 RE 时段放电比例: {s.get('low_re_discharge_rate', 0):.2%} "
                f"(样本 {s.get('low_re_samples', 0)})",
                f"- 内部交易方向正确配对: {s.get('internal_trade_good_pairs_total', 0)}",
                f"- 内部交易方向错误配对: {s.get('internal_trade_bad_pairs_total', 0)}",
                f"- 缺电 agent 仍内部售电次数: {s.get('deficit_agent_wrong_internal_sell_count', 0)}",
                f"- 盈余 agent 仍内部购电次数: {s.get('surplus_agent_wrong_internal_buy_count', 0)}",
                f"- 高 RE 盈余但不充电且不内部售电: {s.get('high_re_surplus_no_charge_no_trade_count', 0)}",
                f"- 疑似弃电/不交易: {s.get('curtail_like_count', 0)}",
                f"- 说明: {s.get('market_note', '')}",
                "",
            ]
        )

    (action_dir / "action_analysis_summary.json").write_text(
        json.dumps(all_summaries, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (periodic_dir / "periodic_eval_summary.json").write_text(
        json.dumps(periodic_summaries, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (action_dir / "action_analysis_report.md").write_text(
        "\n".join(report_lines), encoding="utf-8"
    )
    print(f"Wrote {action_dir / 'action_analysis_report.md'}")


if __name__ == "__main__":
    main()
