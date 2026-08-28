#!/usr/bin/env python3
"""Plot MAPPO-IA / MATRPO / HyperMARL-MAPPO comparison and write Chinese report."""

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


def _per_episode_series(arr: np.ndarray, episodes_per_point: int) -> np.ndarray:
    if episodes_per_point <= 1:
        return arr
    expanded = np.repeat(arr, episodes_per_point)
    return expanded


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out-root",
        type=Path,
        default=Path("result/compare_mappo_matrpo_hypermarl"),
    )
    parser.add_argument("--ma-window", type=int, default=500)
    args = parser.parse_args()

    out_root = args.out_root.resolve()
    specs = {
        "MAPPO-IA": {
            "tag": "MAPPO-IA-CTDE-5kEp",
            "dir": out_root / "mappo_independent",
            "progress": out_root / "logs/progress_mappo_ia.jsonl",
            "episodes_per_point": 4,
        },
        "MATRPO": {
            "tag": "MATRPO-24h-FullCDA-ReserveDemand-5kEp",
            "dir": out_root / "matrpo",
            "progress": out_root / "logs/progress_matrpo.jsonl",
            "episodes_per_point": 1,
        },
        "HyperMARL-MAPPO": {
            "tag": "HyperMARL-MAPPO-5kEp",
            "dir": out_root / "hypermarl_mappo",
            "progress": out_root / "logs/progress_hypermarl_mappo.jsonl",
            "episodes_per_point": 4,
        },
    }

    series_map: Dict[str, np.ndarray] = {}
    stats_map: Dict[str, Dict[str, float]] = {}
    paths_map: Dict[str, str] = {}

    for name, spec in specs.items():
        npy = _find_returns_npy(spec["dir"], spec["tag"])
        if npy is None:
            raise FileNotFoundError(f"Missing returns for {name} ({spec['tag']})")
        raw = np.load(npy).astype(np.float64)
        episode_series = _per_episode_series(raw, spec["episodes_per_point"])
        series_map[name] = episode_series
        stats_map[name] = _stats(episode_series)
        paths_map[name] = str(npy)

    plt.figure(figsize=(11, 6))
    for name, series in series_map.items():
        x = np.arange(len(series))
        plt.plot(x, series, alpha=0.2, label=f"{name} raw")
        ma_x, ma_y = _moving_average(series, args.ma_window)
        plt.plot(ma_x, ma_y, linewidth=2, label=f"{name} MA({args.ma_window})")

    plt.xlabel("Episode")
    plt.ylabel("Episode return")
    plt.title("MAPPO-IA vs MATRPO vs HyperMARL-MAPPO")
    plt.legend()
    plt.tight_layout()
    fig_path = out_root / "compare_reward_curves.png"
    plt.savefig(fig_path, dpi=150)
    plt.close()

    best_final = max(stats_map.items(), key=lambda kv: kv[1]["final_mean"])[0]
    compare_stats = {
        "budget": {
            "total_timesteps": 120000,
            "episodes": 5000,
            "episode_length": 24,
            "num_envs_jax": 4,
        },
        "stats": stats_map,
        "paths": paths_map,
        "best_final_stage": best_final,
    }
    with (out_root / "compare_stats.json").open("w", encoding="utf-8") as f:
        json.dump(compare_stats, f, ensure_ascii=False, indent=2)

    progress_sections = []
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

    report = f"""# MAPPO / MATRPO / HyperMARL-MAPPO 三算法对比实验报告

## 脚本与配置

| 算法 | 脚本 | 配置 |
|------|------|------|
| MAPPO（独立 actor + CTDE critic） | `baselines/MAPPO/mappo_ff_shared_weights.py` | `mappo_ff_independent_actors_microgrid` |
| MATRPO（独立 actor + 共享 critic） | `baselines/MATRPO/train_matrpo_microgrid.py` | PyTorch TRPO, hidden=128 |
| HyperMARL-MAPPO | `baselines/MAPPO/mappo_ff_shared_weights_mlp_hypernets_microgrid.py` | `mappo_ff_shared_weights_mlp_hypernets_microgrid` |

- 环境 override：与 `run_ctde_40k.sh` 相同（`scripts/microgrid_experiment_overrides.py`）
- SEED=30，NUM_ENVS=4（JAX），NUM_STEPS=24，LR=1e-4，ACTOR/CRITIC=[128,128]
- 训练预算：`TOTAL_TIMESTEPS=120000` = 5000 episode × 24 步（老师定义，不乘 NUM_ENVS）

## 公平性说明

- **相同**：环境配置、随机种子、episode 长度、总环境步数预算、JAX 侧 PPO 超参
- **不同**：
  - MAPPO：4 个独立 actor MLP + 集中式 critic
  - HyperMARL-MAPPO：共享 hypernetwork 生成各 agent actor 参数 + 集中式 critic
  - MATRPO：PyTorch TRPO，4 独立 actor + 共享 critic；`NUM_ENVS=1` 串行采样（总步数仍为 120000）
- Actor 参数无法逐项相同初始化（结构不同）

## Reward 对比（episode 级）

| 算法 | 全程均值 | 末 10% 均值 | 末 10% std |
|------|----------|-------------|------------|
| MAPPO-IA | {stats_map['MAPPO-IA']['overall_mean']:.4f} | {stats_map['MAPPO-IA']['final_mean']:.4f} | {stats_map['MAPPO-IA']['final_std']:.4f} |
| MATRPO | {stats_map['MATRPO']['overall_mean']:.4f} | {stats_map['MATRPO']['final_mean']:.4f} | {stats_map['MATRPO']['final_std']:.4f} |
| HyperMARL-MAPPO | {stats_map['HyperMARL-MAPPO']['overall_mean']:.4f} | {stats_map['HyperMARL-MAPPO']['final_mean']:.4f} | {stats_map['HyperMARL-MAPPO']['final_std']:.4f} |

- 对比图：`compare_reward_curves.png`（含 500-episode 滑动平均）

## 每 500 episode 进度摘要

{chr(10).join(progress_sections)}

## 结论

- 末段平均 return 最高：**{best_final}**
- 本实验为单 seed，且 MATRPO 与 JAX 算法采样并行度不同，结论仅供初步对比

## 输出目录

`{out_root}`
"""
    report_path = out_root / "实验报告.md"
    report_path.write_text(report, encoding="utf-8")
    print(f"Wrote {fig_path}")
    print(f"Wrote {report_path}")
    print(json.dumps(compare_stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
