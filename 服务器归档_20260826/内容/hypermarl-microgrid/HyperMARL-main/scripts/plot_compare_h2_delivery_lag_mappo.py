#!/usr/bin/env python3
"""Plot instant vs delayed H2 delivery MAPPO reward curves and progress summary."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.sans-serif"] = [
    "DejaVu Sans",
    "WenQuanYi Micro Hei",
    "Noto Sans CJK SC",
    "SimHei",
    "Arial Unicode MS",
]
plt.rcParams["axes.unicode_minus"] = False

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from baselines.utils.experiment_progress import load_progress_jsonl


def _find_returns_npy(root: Path, alg_tag: str) -> Optional[Path]:
    matches = sorted(root.glob(f"**/returns_microgrid_{alg_tag}.npy"))
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
        "volatility": float(np.std(series)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--ma-window", type=int, default=500)
    parser.add_argument("--num-envs", type=int, default=6)
    parser.add_argument("--instant-tag", type=str, default="MAPPO-IA-InstantH2-5kEp")
    parser.add_argument("--lag-tag", type=str, default="MAPPO-IA-Lag4hH2-5kEp")
    parser.add_argument("--total-timesteps", type=int, default=120000)
    parser.add_argument("--episode-length", type=int, default=24)
    args = parser.parse_args()

    out_root = args.out_root.resolve()
    figures_dir = out_root / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    num_updates = args.total_timesteps // args.episode_length // args.num_envs
    specs = {
        "即时交付": {
            "tag": args.instant_tag,
            "dir": out_root / "instant",
            "progress": out_root / "logs/progress_instant.jsonl",
            "episodes_per_point": args.num_envs,
        },
        "延迟4h交付": {
            "tag": args.lag_tag,
            "dir": out_root / "lag4h",
            "progress": out_root / "logs/progress_lag4h.jsonl",
            "episodes_per_point": args.num_envs,
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

    # Episode reward curves (raw + MA)
    plt.figure(figsize=(11, 6))
    for name, series in series_map.items():
        x = np.arange(len(series))
        plt.plot(x, series, alpha=0.15, label=f"{name} raw")
        ma_x, ma_y = _moving_average(series, args.ma_window)
        plt.plot(ma_x, ma_y, linewidth=2, label=f"{name} MA({args.ma_window})")
    plt.xlabel("Episode")
    plt.ylabel("Episode return")
    plt.title("MAPPO-IA: Instant vs Delayed H2 Delivery")
    plt.legend()
    plt.tight_layout()
    reward_path = figures_dir / "episode_reward_curves.png"
    plt.savefig(reward_path, dpi=150)
    plt.close()

    # MA-only figure
    plt.figure(figsize=(11, 6))
    for name, series in series_map.items():
        ma_x, ma_y = _moving_average(series, args.ma_window)
        plt.plot(ma_x, ma_y, linewidth=2, label=f"{name} MA({args.ma_window})")
    plt.xlabel("Episode")
    plt.ylabel("Moving average return")
    plt.title(f"Reward moving average (window={args.ma_window})")
    plt.legend()
    plt.tight_layout()
    ma_path = figures_dir / "reward_moving_average.png"
    plt.savefig(ma_path, dpi=150)
    plt.close()

    best_final = max(stats_map.items(), key=lambda kv: kv[1]["final_mean"])[0]
    compare_stats = {
        "budget": {
            "total_timesteps": args.total_timesteps,
            "num_envs": args.num_envs,
            "episode_length": args.episode_length,
            "num_updates": num_updates,
            "actual_env_steps": num_updates * args.episode_length * args.num_envs,
            "approx_episodes": num_updates * args.num_envs,
        },
        "stats": stats_map,
        "paths": paths_map,
        "best_final_stage": best_final,
    }
    with (out_root / "compare_stats.json").open("w", encoding="utf-8") as f:
        json.dump(compare_stats, f, ensure_ascii=False, indent=2)

    progress_rows: Dict[str, List[dict]] = {}
    progress_sections: List[str] = []
    for name, spec in specs.items():
        rows = load_progress_jsonl(spec["progress"])
        progress_rows[name] = rows
        if not rows:
            progress_sections.append(f"### {name}\n- 无进度日志\n")
            continue
        lines = [f"### {name}", ""]
        for row in rows:
            lines.append(
                f"- ep={row['episode']}: recent_mean={row['recent_mean_reward']:.2f}, "
                f"cumulative={row['cumulative_mean_reward']:.2f}, "
                f"best_recent={row['best_recent_mean_reward']:.2f}, "
                f"std={row.get('recent_std', 0):.2f}, "
                f"stability={row['stability_note']}"
            )
        progress_sections.append("\n".join(lines))

    with (out_root / "progress_summary.md").open("w", encoding="utf-8") as f:
        f.write("# 训练进度摘要（每 500 episode）\n\n")
        f.write("\n\n".join(progress_sections))

    print(f"Wrote {reward_path}")
    print(f"Wrote {ma_path}")
    print(json.dumps(compare_stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
