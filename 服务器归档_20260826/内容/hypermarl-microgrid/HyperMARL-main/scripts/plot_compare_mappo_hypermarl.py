#!/usr/bin/env python3
"""Plot MAPPO vs HyperMARL-MAPPO reward curves and write a short Chinese report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _find_returns_npy(root: Path, alg_tag: str) -> Path | None:
    patterns = [
        f"returns_microgrid_{alg_tag}.npy",
        f"**/returns_microgrid_{alg_tag}.npy",
    ]
    for pat in patterns:
        matches = sorted(root.glob(pat))
        if matches:
            return matches[-1]
    return None


def _load_series(path: Path) -> np.ndarray:
    arr = np.load(path)
    arr = np.asarray(arr, dtype=np.float64)
    if arr.ndim == 0:
        return arr.reshape(1)
    return arr


def _moving_average(x: np.ndarray, window: int) -> np.ndarray:
    if len(x) < window:
        return x.copy()
    kernel = np.ones(window, dtype=np.float64) / window
    return np.convolve(x, kernel, mode="valid")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out-root",
        type=Path,
        default=Path("result/compare_mappo_hypermarl"),
        help="Experiment output root",
    )
    parser.add_argument("--window", type=int, default=50)
    args = parser.parse_args()

    out_root = args.out_root.resolve()
    mappo_tag = "MAPPO-baseline-24h-FullCDA-ReserveDemand-5kEp"
    hyper_tag = "HyperMARL-MAPPO-24h-FullCDA-ReserveDemand-5kEp"

    mappo_npy = _find_returns_npy(out_root / "mappo_baseline", mappo_tag)
    hyper_npy = _find_returns_npy(out_root / "hypermarl_mappo", hyper_tag)
    if mappo_npy is None:
        mappo_npy = _find_returns_npy(out_root, mappo_tag)
    if hyper_npy is None:
        hyper_npy = _find_returns_npy(out_root, hyper_tag)

    if mappo_npy is None or hyper_npy is None:
        raise FileNotFoundError(
            f"Missing returns npy. mappo={mappo_npy} hypermarl={hyper_npy}"
        )

    mappo = _load_series(mappo_npy)
    hyper = _load_series(hyper_npy)
    n = min(len(mappo), len(hyper))
    mappo = mappo[:n]
    hyper = hyper[:n]
    x = np.arange(n)

    mappo_ma = _moving_average(mappo, args.window)
    hyper_ma = _moving_average(hyper, args.window)
    ma_x = np.arange(args.window - 1, n)

    final_k = max(1, n // 10)
    stats = {
        "updates": int(n),
        "mappo_final_mean": float(np.mean(mappo[-final_k:])),
        "hypermarl_final_mean": float(np.mean(hyper[-final_k:])),
        "mappo_overall_mean": float(np.mean(mappo)),
        "hypermarl_overall_mean": float(np.mean(hyper)),
        "mappo_std_last10pct": float(np.std(mappo[-final_k:])),
        "hypermarl_std_last10pct": float(np.std(hyper[-final_k:])),
        "mappo_npy": str(mappo_npy),
        "hypermarl_npy": str(hyper_npy),
    }

    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "returns").mkdir(exist_ok=True)

    plt.figure(figsize=(10, 5))
    plt.plot(x, mappo, alpha=0.25, label="MAPPO (per-update)")
    plt.plot(x, hyper, alpha=0.25, label="HyperMARL-MAPPO (per-update)")
    if len(mappo_ma):
        plt.plot(ma_x, mappo_ma, linewidth=2, label=f"MAPPO MA({args.window})")
    if len(hyper_ma):
        plt.plot(ma_x, hyper_ma, linewidth=2, label=f"HyperMARL-MAPPO MA({args.window})")
    plt.xlabel("Update")
    plt.ylabel("Mean episode return")
    plt.title("MAPPO vs HyperMARL-MAPPO on Microgrid")
    plt.legend()
    plt.tight_layout()
    fig_path = out_root / "compare_reward_curves.png"
    plt.savefig(fig_path, dpi=150)
    plt.close()

    with open(out_root / "compare_stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    better = (
        "HyperMARL-MAPPO"
        if stats["hypermarl_final_mean"] > stats["mappo_final_mean"]
        else "MAPPO-baseline"
    )
    report = f"""# MAPPO vs HyperMARL-MAPPO 对比实验报告

## 实验脚本与配置

| 算法 | 脚本 | Hydra 配置 |
|------|------|------------|
| 普通 MAPPO | `baselines/MAPPO/mappo_ff_shared_weights_microgrid.py` | `mappo_ff_shared_weights_microgrid` |
| HyperMARL-MAPPO | `baselines/MAPPO/mappo_ff_shared_weights_mlp_hypernets_microgrid.py` | `mappo_ff_shared_weights_mlp_hypernets_microgrid` |

- 环境 override：与 `run_ctde_40k.sh` 相同（FullCDA-RD + Price30），未修改老师参数
- SEED=30，NUM_ENVS=4，NUM_STEPS=24，LR=1e-4，ACTOR/CRITIC=[128,128]
- 训练预算：TOTAL_TIMESTEPS={stats['updates'] * 24 * 4}（约 {stats['updates']} 次 update）

## 公平性说明

- **相同**：随机种子、微电网环境配置、并行环境数、PPO 超参数、优化器与学习率
- **不同**：普通 MAPPO 使用固定共享 MLP Actor；HyperMARL-MAPPO 使用 MLP hypernetwork 为每个 agent 生成 Actor 权重（hypernet hidden=[64]）。两者网络参数量与初始化路径不同，**无法严格做到参数级完全一致**。

## Reward 对比结果

| 指标 | MAPPO-baseline | HyperMARL-MAPPO |
|------|----------------|-----------------|
| 全程均值 | {stats['mappo_overall_mean']:.4f} | {stats['hypermarl_overall_mean']:.4f} |
| 末 10% 阶段均值 | {stats['mappo_final_mean']:.4f} | {stats['hypermarl_final_mean']:.4f} |
| 末 10% 标准差 | {stats['mappo_std_last10pct']:.4f} | {stats['hypermarl_std_last10pct']:.4f} |

- 对比曲线：{fig_path.name}
- 原始数据：{mappo_npy.name} / {hyper_npy.name}

## 结论

- 末段平均 return 更高的是：**{better}**
- 稳定性（末段 std 更小者更稳）：MAPPO={stats['mappo_std_last10pct']:.4f}，HyperMARL={stats['hypermarl_std_last10pct']:.4f}
- 可靠性：本实验为**单 seed**；且 Actor 结构不同，结论仅供初步对比，建议多 seed 复现后再下最终结论。

## 输出目录

`{out_root}`
"""
    report_path = out_root / "实验报告.md"
    report_path.write_text(report, encoding="utf-8")
    print(f"Wrote {fig_path}")
    print(f"Wrote {report_path}")
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
