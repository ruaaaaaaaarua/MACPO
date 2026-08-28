#!/usr/bin/env python3
"""Rollout analysis for instant vs delayed H2 delivery MAPPO policies."""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from orbax.checkpoint import checkpointer

plt.rcParams["font.sans-serif"] = [
    "DejaVu Sans",
    "WenQuanYi Micro Hei",
    "Noto Sans CJK SC",
    "SimHei",
    "Arial Unicode MS",
]
plt.rcParams["axes.unicode_minus"] = False
from orbax.checkpoint.pytree_checkpoint_handler import PyTreeCheckpointHandler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from envs.microgrid.microgrid_continuous_env import MicrogridContinuousEnv
from scripts.microgrid_experiment_overrides_h2_delivery_lag import build_env_overrides

EPS = 1e-6


def _apply_overrides(overrides: Dict[str, Any]) -> None:
    import envs.microgrid.microgrid_continuous_env as mce_mod
    import envs.microgrid.microgrid_env as me_mod
    from envs.microgrid import config as cfg_mod

    importlib.reload(cfg_mod)
    cfg_mod.MICROGRID_CONFIG.update(dict(overrides))
    importlib.reload(me_mod)
    importlib.reload(mce_mod)


def _checkpoint_step(path: Path) -> int:
    name = path.name
    marker = "_steps_"
    if marker not in name:
        return -1
    try:
        return int(name.split(marker, 1)[1].split("_", 1)[0])
    except ValueError:
        return -1


def _find_latest_checkpoint(root: Path) -> Optional[Path]:
    matches = sorted(
        (p for p in root.glob("**/*_steps_*_updates.*") if p.is_dir()),
        key=lambda p: (_checkpoint_step(p), str(p)),
    )
    if matches:
        return matches[-1]

    # Fallback for legacy runs that saved under /tmp/models.
    tmp_models = Path("/tmp/models")
    if not tmp_models.exists():
        return None
    tag = root.name.lower()
    legacy = sorted(
        (
            p
            for p in tmp_models.glob("**/*_steps_*_updates.*")
            if p.is_dir() and tag in str(p).lower()
        ),
        key=lambda p: (_checkpoint_step(p), str(p)),
    )
    return legacy[-1] if legacy else None


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


def _h2_level_ratio(inner, agent_id: int) -> float:
    cap = float(inner.h2_tank_cap[agent_id])
    if cap <= 0:
        return 0.0
    return float(inner.h2_level[agent_id] / cap)


def rollout_policy(
    env_name: str,
    checkpoint_dir: Path,
    seed: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    overrides = build_env_overrides("instant" if env_name == "instant" else "lag4h")
    _apply_overrides(overrides)

    env = MicrogridContinuousEnv()
    env.seed(seed)
    inner = env.env
    num_agents = env.num_agent
    action_dim = env.signal_action_dim
    params = _load_params(checkpoint_dir)

    obs = env.reset()
    rows: List[Dict[str, Any]] = []
    total_reward = 0.0
    charge_events = 0
    discharge_events = 0
    ext_h2_buy_total = 0.0
    h2_internal_total = 0.0

    for step in range(24):
        actions = np.clip(_actions_mappo_ia(params, obs, num_agents, action_dim), -1.0, 1.0)
        next_obs, rewards, dones, infos = env.step(actions)
        info = infos[0] if isinstance(infos, list) else infos

        p_ht = info.get("p_ht", [0.0] * num_agents)
        e_h2_ext = info.get("e_h2_ext", [0.0] * num_agents)
        h2_traded = float(info.get("h2_market_traded", 0.0))
        pending_total = float(info.get("pending_h2_energy_total", 0.0))
        h2_clear = float(info.get("h2_clearing_price", 0.0))

        for agent in range(num_agents):
            pht = float(p_ht[agent])
            if pht > 0.05:
                charge_events += 1
            elif pht < -0.05:
                discharge_events += 1
            ext_h2_buy_total += max(0.0, float(e_h2_ext[agent]))

            rows.append(
                {
                    "env": env_name,
                    "step": step,
                    "agent": agent,
                    "h2_level_ratio": _h2_level_ratio(inner, agent),
                    "h2_level_kg": float(inner.h2_level[agent]),
                    "p_ht": pht,
                    "e_h2_ext": float(e_h2_ext[agent]),
                    "h2_market_traded_step": h2_traded,
                    "h2_clearing_price": h2_clear,
                    "pending_h2_energy_total": pending_total,
                    "reward_step": float(np.mean(rewards)),
                }
            )

        total_reward += float(np.mean(rewards))
        h2_internal_total += h2_traded
        obs = next_obs

    summary = {
        "env": env_name,
        "checkpoint": str(checkpoint_dir),
        "obs_dim": int(env.signal_obs_dim),
        "episode_return": total_reward,
        "charge_events": charge_events,
        "discharge_events": discharge_events,
        "ext_h2_buy_total_kwh": ext_h2_buy_total,
        "h2_internal_traded_total_kwh": h2_internal_total,
        "mean_h2_level_ratio": float(
            np.mean([r["h2_level_ratio"] for r in rows if r["agent"] == 0])
            if rows
            else 0.0
        ),
        "mean_h2_level_ratio_all_agents": float(
            np.mean([r["h2_level_ratio"] for r in rows]) if rows else 0.0
        ),
        "max_pending_h2_energy": float(
            max((r["pending_h2_energy_total"] for r in rows), default=0.0)
        ),
    }
    return rows, summary


def _plot_series(
    out_root: Path,
    rows_by_env: Dict[str, List[Dict[str, Any]]],
    num_agents: int = 4,
) -> None:
    figures_dir = out_root / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    # H2 tank inventory ratio (mean across agents)
    plt.figure(figsize=(10, 5))
    for env_name, rows in rows_by_env.items():
        steps = sorted({r["step"] for r in rows})
        means = []
        for s in steps:
            vals = [r["h2_level_ratio"] for r in rows if r["step"] == s]
            means.append(float(np.mean(vals)))
        label = "即时交付" if env_name == "instant" else "延迟4h交付"
        plt.plot(steps, means, marker="o", label=label)
    plt.xlabel("Hour (step)")
    plt.ylabel("Mean H2 tank level ratio")
    plt.title("储氢罐库存比例（全 agent 均值）")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figures_dir / "h2_tank_inventory_ratio.png", dpi=150)
    plt.close()

    # Per-agent H2 inventory
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    for agent, ax in enumerate(axes.flatten()):
        for env_name, rows in rows_by_env.items():
            steps = [r["step"] for r in rows if r["agent"] == agent]
            ratios = [r["h2_level_ratio"] for r in rows if r["agent"] == agent]
            label = "即时" if env_name == "instant" else "延迟4h"
            ax.plot(steps, ratios, marker=".", label=label)
        ax.set_title(f"Agent {agent}")
        ax.set_ylabel("H2 level ratio")
        ax.legend(fontsize=8)
    for ax in axes[-1]:
        ax.set_xlabel("Hour")
    fig.suptitle("储氢罐库存比例（分 agent）")
    fig.tight_layout()
    plt.savefig(figures_dir / "h2_tank_inventory_by_agent.png", dpi=150)
    plt.close()

    # External H2 purchase per step
    plt.figure(figsize=(10, 5))
    for env_name, rows in rows_by_env.items():
        steps = sorted({r["step"] for r in rows})
        buys = []
        for s in steps:
            vals = [max(0.0, r["e_h2_ext"]) for r in rows if r["step"] == s]
            buys.append(float(np.sum(vals)))
        label = "即时交付" if env_name == "instant" else "延迟4h交付"
        plt.plot(steps, buys, marker="o", label=label)
    plt.xlabel("Hour (step)")
    plt.ylabel("External H2 purchase (kWh_H2, sum over agents)")
    plt.title("外部购氢量逐步对比")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figures_dir / "external_h2_purchase.png", dpi=150)
    plt.close()

    # Pending H2 delivery
    plt.figure(figsize=(10, 5))
    for env_name, rows in rows_by_env.items():
        steps = sorted({r["step"] for r in rows})
        pending = [
            float(
                np.mean([r["pending_h2_energy_total"] for r in rows if r["step"] == s])
            )
            for s in steps
        ]
        label = "即时交付" if env_name == "instant" else "延迟4h交付"
        plt.plot(steps, pending, marker="o", label=label)
    plt.xlabel("Hour (step)")
    plt.ylabel("Pending H2 energy (kWh_H2)")
    plt.title("Pending 氢交付量变化")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figures_dir / "pending_h2_delivery.png", dpi=150)
    plt.close()

    # H2 internal market volume
    plt.figure(figsize=(10, 5))
    for env_name, rows in rows_by_env.items():
        steps = sorted({r["step"] for r in rows})
        traded = [
            float(
                np.mean([r["h2_market_traded_step"] for r in rows if r["step"] == s])
            )
            for s in steps
        ]
        label = "即时交付" if env_name == "instant" else "延迟4h交付"
        plt.plot(steps, traded, marker="o", label=label)
    plt.xlabel("Hour (step)")
    plt.ylabel("H2 internal CDA volume (kWh_H2)")
    plt.title("氢市场内部成交量对比")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figures_dir / "h2_internal_market_volume.png", dpi=150)
    plt.close()


def _write_report(
    out_root: Path,
    summaries: Dict[str, Dict[str, Any]],
    compare_stats_path: Path,
    progress_instant: List[dict],
    progress_lag: List[dict],
) -> None:
    from baselines.utils.experiment_progress import load_progress_jsonl

    compare_stats = {}
    if compare_stats_path.exists():
        compare_stats = json.loads(compare_stats_path.read_text(encoding="utf-8"))

    stats = compare_stats.get("stats", {})
    instant_stats = stats.get("即时交付", {})
    lag_stats = stats.get("延迟4h交付", {})
    best = compare_stats.get("best_final_stage", "N/A")

    def _progress_table(rows: List[dict]) -> str:
        if not rows:
            return "- 无数据"
        lines = []
        for row in rows:
            lines.append(
                f"| {row['episode']} | {row['recent_mean_reward']:.2f} | "
                f"{row['cumulative_mean_reward']:.2f} | {row['best_recent_mean_reward']:.2f} | "
                f"{row.get('stability_note', '')} |"
            )
        return "\n".join(lines)

    instant_sum = summaries.get("instant", {})
    lag_sum = summaries.get("lag4h", {})

    charge_instant = instant_sum.get("charge_events", 0)
    charge_lag = lag_sum.get("charge_events", 0)
    discharge_instant = instant_sum.get("discharge_events", 0)
    discharge_lag = lag_sum.get("discharge_events", 0)

    conservative_note = ""
    if lag_sum.get("mean_h2_level_ratio_all_agents", 0) > instant_sum.get(
        "mean_h2_level_ratio_all_agents", 0
    ):
        conservative_note = (
            "延迟交付环境下 rollout 平均储氢库存比例更高，"
            "与“交付时滞约束诱导储氢罐前瞻性补能行为、提高库存安全裕度”的假设一致。"
        )
    else:
        conservative_note = (
            "本 seed 下延迟环境的平均储氢库存比例未显著高于即时环境，"
            "需结合多 seed 与更长训练进一步验证保守储氢策略假设。"
        )

    report = f"""# 氢能交付延迟对 MAPPO 微电网调度性能影响的对比实验报告

## 1. 实验目的

本实验旨在比较**氢能交易即时交付**与**氢能交易固定延迟交付（4h）**两种微电网环境配置下，普通 MAPPO（独立 actor + 集中式 critic）的策略学习差异，重点分析交付时滞约束对 reward 收敛、储氢罐充放行为、氢能库存安全性及交易策略的影响。

## 2. 对比环境设置

两组实验共享 FullCDA-ReserveDemand + Price30 基线配置（Italian train split、24h episode），**仅修改氢能交付时滞相关参数**：

| 参数 | 环境 A：即时交付 | 环境 B：延迟 4h 交付 |
|------|------------------|----------------------|
| h2_market_lag_enable | false | true |
| h2_delivery_lag | 0 | 4 |
| h2_pending_obs_enable | false | true |
| h2_delivery_reservation_enable | false | true |
| h2_cap_aware_buy_enable | false | true |

- 环境 A：内部氢 CDA 成交后**立即**计入可用氢库存。
- 环境 B：成交后进入 pending 队列，经**交付时滞约束**（4h）后方可交付，形成**跨时段氢能库存耦合**与**可用氢库存滞后**。

观测维度：即时约 {instant_sum.get('obs_dim', 13)} 维；延迟约 {lag_sum.get('obs_dim', 19)} 维（含 pending 信息结构）。

## 3. 训练配置

- 算法：MAPPO-IA-CTDE（`mappo_ff_shared_weights.py` + `mappo_ff_independent_actors_microgrid.yaml`）
- SEED=30，NUM_ENVS=6，NUM_STEPS=24，CPU 线程=3
- TOTAL_TIMESTEPS=120000 → NUM_UPDATES=833，实际 env steps≈119952，约 4998 episodes
- 串行训练：先即时交付，后延迟交付

## 4. Reward 对比结果

| 环境 | 全程均值 | 末 10% 均值 | 末 10% 波动 | 全程波动幅度 |
|------|----------|-------------|-------------|--------------|
| 即时交付 | {instant_stats.get('overall_mean', 0):.4f} | {instant_stats.get('final_mean', 0):.4f} | {instant_stats.get('final_std', 0):.4f} | {instant_stats.get('volatility', 0):.4f} |
| 延迟 4h 交付 | {lag_stats.get('overall_mean', 0):.4f} | {lag_stats.get('final_mean', 0):.4f} | {lag_stats.get('final_std', 0):.4f} | {lag_stats.get('volatility', 0):.4f} |

- 末段训练效果更好：**{best}**
- Reward 曲线见 `figures/episode_reward_curves.png`、`figures/reward_moving_average.png`

### 即时交付进度（每 500 episode）

| Episode | Recent mean | Cumulative | Best recent | Stability |
|---------|-------------|------------|-------------|-----------|
{_progress_table(progress_instant)}

### 延迟交付进度（每 500 episode）

| Episode | Recent mean | Cumulative | Best recent | Stability |
|---------|-------------|------------|-------------|-----------|
{_progress_table(progress_lag)}

## 5. 氢能交付延迟对策略学习的影响

延迟交付引入**交付时滞约束**，使策略无法依赖当期成交氢能即时满足负荷，需在观测中跟踪 pending 队列并提前规划。相较即时环境，延迟环境下的策略学习面临**短期交易收益与长期供氢安全之间的权衡**：过早售出氢能或库存不足将导致后续时段外部购氢成本上升。

## 6. 氢能交付延迟对储氢充放行为的影响（deterministic rollout）

| 指标 | 即时交付 | 延迟 4h 交付 |
|------|----------|--------------|
| 充氢事件数（|p_ht|>0.05） | {charge_instant} | {charge_lag} |
| 放氢事件数 | {discharge_instant} | {discharge_lag} |
| 平均储氢库存比例 | {instant_sum.get('mean_h2_level_ratio_all_agents', 0):.3f} | {lag_sum.get('mean_h2_level_ratio_all_agents', 0):.3f} |
| 外部购氢总量 (kWh_H2) | {instant_sum.get('ext_h2_buy_total_kwh', 0):.2f} | {lag_sum.get('ext_h2_buy_total_kwh', 0):.2f} |
| 内部氢成交量 (kWh_H2) | {instant_sum.get('h2_internal_traded_total_kwh', 0):.2f} | {lag_sum.get('h2_internal_traded_total_kwh', 0):.2f} |
| 最大 pending 氢能量 | {instant_sum.get('max_pending_h2_energy', 0):.2f} | {lag_sum.get('max_pending_h2_energy', 0):.2f} |

行为曲线见 `figures/h2_tank_inventory_ratio.png`、`figures/external_h2_purchase.png`、`figures/pending_h2_delivery.png`、`figures/h2_internal_market_volume.png`。

## 7. 关于“延迟诱导更强库存前瞻性与保守储氢策略”的结论

{conservative_note}

若延迟环境下外部购氢更少、库存比例更高、充氢更前置，则支持该假设；反之则需谨慎表述为“单 seed 下证据不足”。

## 8. 结论可靠性与局限性

- 单随机种子（SEED=30），统计显著性有限；
- 两组观测维度因 pending 信息结构不同而存在差异，属交付时滞建模的固有特征；
- 训练预算约 5000 episodes，策略可能尚未完全收敛；
- rollout 为 deterministic 策略，未反映训练期探索噪声。

## 9. 输出目录

`{out_root}`
"""
    (out_root / "compare_report_cn.md").write_text(report, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=30)
    args = parser.parse_args()

    out_root = args.out_root.resolve()
    rollouts_dir = out_root / "rollouts"
    rollouts_dir.mkdir(parents=True, exist_ok=True)

    specs = {
        "instant": out_root / "instant",
        "lag4h": out_root / "lag4h",
    }

    rows_by_env: Dict[str, List[Dict[str, Any]]] = {}
    summaries: Dict[str, Dict[str, Any]] = {}

    for env_name, search_root in specs.items():
        ckpt = _find_latest_checkpoint(search_root)
        if ckpt is None:
            raise FileNotFoundError(f"No checkpoint under {search_root}")
        rows, summary = rollout_policy(env_name, ckpt, args.seed)
        rows_by_env[env_name] = rows
        summaries[env_name] = summary

        csv_path = rollouts_dir / f"rollout_{env_name}.csv"
        if rows:
            with csv_path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                writer.writeheader()
                writer.writerows(rows)

    with (rollouts_dir / "behavior_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summaries, f, ensure_ascii=False, indent=2)

    _plot_series(out_root, rows_by_env)

    from baselines.utils.experiment_progress import load_progress_jsonl

    _write_report(
        out_root,
        summaries,
        out_root / "compare_stats.json",
        load_progress_jsonl(out_root / "logs/progress_instant.jsonl"),
        load_progress_jsonl(out_root / "logs/progress_lag4h.jsonl"),
    )

    print(json.dumps(summaries, ensure_ascii=False, indent=2))
    print(f"Wrote {out_root / 'compare_report_cn.md'}")


if __name__ == "__main__":
    main()
