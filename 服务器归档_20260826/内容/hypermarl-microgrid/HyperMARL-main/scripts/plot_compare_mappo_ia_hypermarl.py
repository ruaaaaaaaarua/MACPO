#!/usr/bin/env python3
"""Plot MAPPO-IA-CTDE vs HyperMARL-MAPPO reward curves and write Chinese report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

from baselines.utils.experiment_progress import load_progress_jsonl


def _find_returns_npy(root: Path, alg_tag: str) -> Optional[Path]:
    matches = sorted(root.glob(f"**/returns_microgrid_{alg_tag}.npy"))
    return matches[-1] if matches else None


def _find_metrics_npz(root: Path, alg_tag: str) -> Optional[Path]:
    matches = sorted(root.glob(f"**/metrics_microgrid_{alg_tag}.npz"))
    return matches[-1] if matches else None


def _per_episode_series(arr: np.ndarray, episodes_per_point: int) -> np.ndarray:
    if episodes_per_point <= 1:
        return arr
    return np.repeat(arr, episodes_per_point)


def _moving_average(x: np.ndarray, window: int) -> Tuple[np.ndarray, np.ndarray]:
    if len(x) < window:
        return np.arange(len(x)), x
    kernel = np.ones(window, dtype=np.float64) / window
    ma = np.convolve(x, kernel, mode="valid")
    xs = np.arange(window - 1, window - 1 + len(ma))
    return xs, ma


def _stats(series: np.ndarray) -> Dict[str, float]:
    final_k = max(1, len(series) // 10)
    return {
        "overall_mean": float(np.mean(series)),
        "final_mean": float(np.mean(series[-final_k:])),
        "final_std": float(np.std(series[-final_k:])),
    }


def _loss_summary(metrics_path: Optional[Path]) -> str:
    if metrics_path is None or not metrics_path.exists():
        return "- 无本地 loss npz（请确认 WANDB_MODE=offline/disabled 且训练已完成）"
    data = np.load(metrics_path)
    lines = []
    for key in ("total_loss", "actor_loss", "critic_loss", "entropy"):
        if key not in data:
            continue
        arr = np.asarray(data[key], dtype=np.float64)
        lines.append(f"- {key}: final={arr[-1]:.4f}, mean={arr.mean():.4f}")
    return "\n".join(lines) if lines else "- metrics npz 为空"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out-root",
        type=Path,
        default=Path("result/compare_mappo_ia_hypermarl_10env_5kep"),
    )
    parser.add_argument("--ma-window", type=int, default=500)
    parser.add_argument("--num-envs", type=int, default=10)
    parser.add_argument("--ia-tag", type=str, default="MAPPO-IA-CTDE-5kEp")
    parser.add_argument("--hyper-tag", type=str, default="HyperMARL-MAPPO-5kEp")
    parser.add_argument(
        "--env-note",
        type=str,
        default="FullCDA-ReserveDemand + Price30；内部 P2P 互济经 CDA 撮合",
    )
    parser.add_argument("--total-timesteps", type=int, default=120000)
    parser.add_argument("--episode-length", type=int, default=24)
    args = parser.parse_args()

    out_root = args.out_root.resolve()
    num_episodes = args.total_timesteps // args.episode_length
    num_updates = args.total_timesteps // args.episode_length // args.num_envs
    specs = {
        "MAPPO-IA-CTDE": {
            "tag": args.ia_tag,
            "dir": out_root / "mappo_independent",
            "progress": out_root / "logs/progress_mappo_ia.jsonl",
            "episodes_per_point": args.num_envs,
        },
        "HyperMARL-MAPPO": {
            "tag": args.hyper_tag,
            "dir": out_root / "hypermarl_mappo",
            "progress": out_root / "logs/progress_hypermarl_mappo.jsonl",
            "episodes_per_point": args.num_envs,
        },
    }

    series_map: Dict[str, np.ndarray] = {}
    stats_map: Dict[str, Dict[str, float]] = {}
    paths_map: Dict[str, str] = {}
    metrics_map: Dict[str, Optional[Path]] = {}

    for name, spec in specs.items():
        npy = _find_returns_npy(spec["dir"], spec["tag"])
        if npy is None:
            raise FileNotFoundError(f"Missing returns for {name} ({spec['tag']})")
        raw = np.load(npy).astype(np.float64)
        episode_series = _per_episode_series(raw, spec["episodes_per_point"])
        series_map[name] = episode_series
        stats_map[name] = _stats(episode_series)
        paths_map[name] = str(npy)
        metrics_map[name] = _find_metrics_npz(spec["dir"], spec["tag"])

    plt.figure(figsize=(11, 6))
    for name, series in series_map.items():
        x = np.arange(len(series))
        plt.plot(x, series, alpha=0.2, label=f"{name} raw")
        ma_x, ma_y = _moving_average(series, args.ma_window)
        plt.plot(ma_x, ma_y, linewidth=2, label=f"{name} MA({args.ma_window})")

    plt.xlabel("Episode")
    plt.ylabel("Episode return")
    plt.title("MAPPO-IA-CTDE vs HyperMARL-MAPPO (10 parallel envs)")
    plt.legend()
    plt.tight_layout()
    fig_path = out_root / "compare_reward_curves.png"
    plt.savefig(fig_path, dpi=150)
    plt.close()

    # Loss / entropy curves (value loss = critic_loss in MAPPO).
    loss_keys = ["actor_loss", "critic_loss", "entropy", "total_loss"]
    fig_loss, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()
    for ax, key in zip(axes, loss_keys):
        for name, spec in specs.items():
            mp = metrics_map[name]
            if mp is None or not mp.exists():
                continue
            arr = np.asarray(np.load(mp)[key], dtype=np.float64)
            ax.plot(arr, label=name, alpha=0.85)
        ax.set_title(key)
        ax.set_xlabel("PPO update")
        ax.legend(fontsize=8)
    fig_loss.suptitle("Training metrics (actor/critic/entropy/value)")
    fig_loss.tight_layout()
    loss_fig_path = out_root / "compare_training_metrics.png"
    fig_loss.savefig(loss_fig_path, dpi=150)
    plt.close(fig_loss)

    best_final = max(stats_map.items(), key=lambda kv: kv[1]["final_mean"])[0]
    compare_stats = {
        "budget": {
            "total_timesteps": args.total_timesteps,
            "episodes": num_episodes,
            "episode_length": args.episode_length,
            "num_envs": args.num_envs,
            "num_updates": num_updates,
        },
        "stats": stats_map,
        "paths": paths_map,
        "best_final_stage": best_final,
    }
    with (out_root / "compare_stats.json").open("w", encoding="utf-8") as f:
        json.dump(compare_stats, f, ensure_ascii=False, indent=2)

    progress_sections: List[str] = []
    for name, spec in specs.items():
        rows = load_progress_jsonl(spec["progress"])
        if not rows:
            progress_sections.append(f"### {name}\n- 无进度日志\n")
            continue
        lines = [f"### {name}", ""]
        for row in rows:
            lines.append(
                f"- ep={row['episode']}: recent_mean={row['recent_mean_reward']:.2f}, "
                f"cumulative={row['cumulative_mean_reward']:.2f}, "
                f"best_recent={row['best_recent_mean_reward']:.2f}, "
                f"stability={row['stability_note']}"
            )
        progress_sections.append("\n".join(lines))

    loss_sections = [
        f"### {name}\n{_loss_summary(metrics_map[name])}" for name in specs
    ]

    report = f"""# MAPPO-IA-CTDE vs HyperMARL-MAPPO 对比实验报告

## 脚本与配置

| 算法 | 脚本 | 配置 |
|------|------|------|
| MAPPO-IA-CTDE | `baselines/MAPPO/mappo_ff_shared_weights.py` | `mappo_ff_independent_actors_microgrid` |
| HyperMARL-MAPPO | `baselines/MAPPO/mappo_ff_shared_weights_mlp_hypernets_microgrid.py` | `mappo_ff_shared_weights_mlp_hypernets_microgrid` |

- 环境 override：`MICROGRID_EXPERIMENT_OVERRIDES`（{args.env_note}）
- SEED=30，NUM_ENVS={args.num_envs}，CPU 线程=8，NUM_STEPS=24
- 训练预算：`TOTAL_TIMESTEPS={args.total_timesteps}` = {num_episodes} episode × {args.episode_length} 步（不乘 NUM_ENVS）
- NUM_UPDATES = {num_updates}
- 串行训练：先 MAPPO-IA-CTDE，后 HyperMARL-MAPPO

## Reward 对比（episode 级）

| 算法 | 全程均值 | 末 10% 均值 | 末 10% std |
|------|----------|-------------|------------|
| MAPPO-IA-CTDE | {stats_map['MAPPO-IA-CTDE']['overall_mean']:.4f} | {stats_map['MAPPO-IA-CTDE']['final_mean']:.4f} | {stats_map['MAPPO-IA-CTDE']['final_std']:.4f} |
| HyperMARL-MAPPO | {stats_map['HyperMARL-MAPPO']['overall_mean']:.4f} | {stats_map['HyperMARL-MAPPO']['final_mean']:.4f} | {stats_map['HyperMARL-MAPPO']['final_std']:.4f} |

- Reward 曲线：`compare_reward_curves.png`
- 训练指标曲线：`compare_training_metrics.png`

## 训练 loss / entropy / value loss 摘要

{chr(10).join(loss_sections)}

## 每 500 episode 进度摘要

{chr(10).join(progress_sections)}

## 动作合理性（内部 CDA / 微电网 P2P 互济 + 储能）

详见 `action_analysis/action_analysis_report.md` 与 `action_analysis/periodic/periodic_eval_summary.json`。

## 结论

- 末段平均 return 最高：**{best_final}**
- 本实验为单 seed；电力内部互济经 **FullCDA** 实现（即微电网间 P2P）

## 输出目录

`{out_root}`
"""
    report_path = out_root / "compare_report.md"
    report_path.write_text(report, encoding="utf-8")
    print(f"Wrote {fig_path}")
    print(f"Wrote {loss_fig_path}")
    print(f"Wrote {report_path}")
    print(json.dumps(compare_stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
