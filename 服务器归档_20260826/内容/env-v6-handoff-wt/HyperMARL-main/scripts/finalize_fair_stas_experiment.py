#!/usr/bin/env python3
"""One-shot test evaluation and Chinese report after all 30k runs finish."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT.parent / "fair-stas-results-20260710"


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _wait_for_training(root: Path) -> dict[str, Any]:
    path = root / "training_complete.json"
    while not path.exists():
        time.sleep(30)
    return json.loads(path.read_text())


def _jax_action_fn(checkpoint: Path, width: int, activation: str) -> Callable:
    import jax
    import jax.numpy as jnp
    import optax
    from flax.training.train_state import TrainState

    from baselines.MAPPO.mappo_ff_shared_weights import ActorCritic
    from baselines.utils.microgrid_vec_env import MicrogridVecEnv
    from baselines.utils.training_checkpoint import load_jax_training_checkpoint
    from scripts.run_abc_multialg_parallel import group_abc_spec

    env = MicrogridVecEnv(
        num_envs=1,
        auto_reset=True,
        config_overrides=group_abc_spec().env_overrides,
    )
    num_agents = env.num_agents
    obs_dim = env.obs_dim
    action_dim = env.action_dim
    env.close()
    network = ActorCritic(
        action_dim,
        activation=activation,
        actor_layers=[width, width],
        critic_layers=[width, width],
        num_agents=num_agents,
        observation_dim=obs_dim,
        is_continuous=True,
        log_std_init=-1.0,
        log_std_min=-2.5,
        log_std_max=-0.5,
    )
    actor_template = jnp.zeros((num_agents, obs_dim + num_agents), dtype=jnp.float32)
    critic_template = jnp.zeros((num_agents, obs_dim * num_agents), dtype=jnp.float32)
    params = network.init(jax.random.PRNGKey(0), actor_template, critic_template)
    tx = optax.chain(
        optax.clip_by_global_norm(5.0),
        optax.adam(3e-4, eps=1e-5),
    )
    template = TrainState.create(apply_fn=network.apply, params=params, tx=tx)
    restored = load_jax_training_checkpoint(checkpoint, template)
    trained_params = restored.train_state.params
    ids = jnp.eye(num_agents, dtype=jnp.float32)
    dummy_critic = jnp.zeros((num_agents, obs_dim * num_agents), dtype=jnp.float32)

    @jax.jit
    def act(obs):
        actor_obs = jnp.concatenate([obs, ids], axis=-1)
        actor_output, _ = network.apply(trained_params, actor_obs, dummy_critic)
        mean, _ = actor_output
        return jnp.tanh(mean)

    def action(observations):
        return np.asarray(act(jnp.asarray(observations, dtype=jnp.float32)))

    return action


def _matd3_action_fn(checkpoint: Path) -> Callable:
    import torch

    from baselines.MATD3.matd3 import MATD3, MATD3Config

    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = MATD3(MATD3Config(**state["config"]), torch.device("cpu"))
    model.load_checkpoint_state(state)
    return lambda observations: model.select_action(observations, noise_std=0.0)


def _write_validation_csv(
    output: Path, paths: dict[str, Path]
) -> dict[str, list[dict[str, Any]]]:
    curves = {name: _read_jsonl(path) for name, path in paths.items()}
    fields = [
        "algorithm",
        "training_episode",
        "return_mean",
        "return_std",
        "base_cost_mean",
        "external_h2_buy_mean",
        "internal_h2_trade_mean",
        "low_h2_hits_mean",
        "action_saturation_rate",
    ]
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for name, records in curves.items():
            for record in records:
                summary = record["summary"]
                writer.writerow(
                    {
                        "algorithm": name,
                        "training_episode": record["training_episode"],
                        **{field: summary.get(field) for field in fields[2:]},
                    }
                )
    return curves


def _plot_validation(output: Path, curves: dict[str, list[dict[str, Any]]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 5.5))
    for name, records in curves.items():
        episodes = [row["training_episode"] for row in records]
        means = [row["summary"]["return_mean"] for row in records]
        stds = [row["summary"]["return_std"] for row in records]
        ax.plot(episodes, means, linewidth=2, label=name)
        ax.fill_between(
            episodes,
            np.asarray(means) - np.asarray(stds),
            np.asarray(means) + np.asarray(stds),
            alpha=0.12,
        )
    ax.set_xlabel("Training episodes")
    ax.set_ylabel("Fixed validation return")
    ax.set_title("ABC microgrid deterministic validation (seed=30)")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def _plot_training(output: Path, paths: dict[str, Path]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 5.5))
    for name, path in paths.items():
        values = np.load(path).reshape(-1).astype(np.float64)
        stride = 1 if name == "MATD3" else 4
        episodes = np.arange(1, values.size + 1) * stride
        valid = np.isfinite(values) & (values != 0.0)
        ax.plot(episodes[valid], values[valid], linewidth=0.8, alpha=0.8, label=name)
    ax.set_xlabel("Training episodes")
    ax.set_ylabel("Raw training return")
    ax.set_title("Training return (diagnostic only)")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def run(root: Path) -> None:
    completion = _wait_for_training(root)
    selected_mappo = completion["selected_mappo"]
    selected_stas = completion["selected_stas"]
    width = 128 if selected_mappo.endswith("128") else 256
    activation = "tanh" if width == 128 else "relu"
    mappo_algorithm = "Stable-MAPPO-128" if width == 128 else "Stable-MAPPO-256-ReLU"
    stas_algorithm = (
        "Conserved-Bidirectional-STAS"
        if selected_stas == "stas_bidirectional"
        else "Conserved-Causal-STAS"
    )
    mappo_out = root / "stage1_10k" / selected_mappo / "output"
    matd3_out = root / "stage1_10k" / "matd3_256" / "output"
    stas_out = root / "stage2_stas" / selected_stas / "output"
    report_dir = root / "final"
    report_dir.mkdir(parents=True, exist_ok=True)
    marker = report_dir / "FINAL_TEST_COMPLETE"
    if marker.exists():
        return

    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    import jax

    _ = jax.devices()
    from baselines.utils.experiment_reporting import compute_curve_metrics
    from baselines.utils.final_comparison import run_final_comparison
    from scripts.run_abc_multialg_parallel import group_abc_spec

    checkpoints = {
        "MAPPO": mappo_out / "checkpoints" / "training_state.msgpack",
        "STAS": stas_out / "checkpoints" / "training_state.msgpack",
        "MATD3": matd3_out / "checkpoints" / "MATD3-Fair-256" / "matd3_episode_30000.pt",
    }
    actions = {
        "MAPPO": _jax_action_fn(checkpoints["MAPPO"], width, activation),
        "STAS": _jax_action_fn(checkpoints["STAS"], width, activation),
        "MATD3": _matd3_action_fn(checkpoints["MATD3"]),
    }
    algorithms = {
        "MAPPO": mappo_algorithm,
        "STAS": stas_algorithm,
        "MATD3": "MATD3-Fair-256",
    }
    test_jsonl = report_dir / "final_test_eval.jsonl"
    if test_jsonl.exists():
        test_jsonl.unlink()
    comparison = run_final_comparison(
        actions,
        group_abc_spec().env_overrides,
        report_dir,
        algorithm_names=algorithms,
        training_episode=30000,
    )
    test_results = {
        name: comparison["results"][name]["metrics"]
        for name in ("MAPPO", "STAS", "MATD3")
    }
    # Retain the historical normal-only JSONL artifact without re-evaluating.
    comparison_rows = _read_jsonl(report_dir / "final_comparison.jsonl")
    with test_jsonl.open("w", encoding="utf-8") as stream:
        for row in comparison_rows:
            if row.get("result_name") in {"MAPPO", "STAS", "MATD3"}:
                stream.write(
                    json.dumps(
                        row,
                        ensure_ascii=False,
                        sort_keys=True,
                        allow_nan=False,
                    )
                    + "\n"
                )

    validation_paths = {
        "MAPPO": mappo_out / "validation_eval.jsonl",
        "STAS": stas_out / "validation_eval.jsonl",
        "MATD3": matd3_out / "validation_eval.jsonl",
    }
    curves = _write_validation_csv(report_dir / "validation_curves.csv", validation_paths)
    _plot_validation(report_dir / "validation_return.png", curves)
    training_paths = {
        "MAPPO": mappo_out / "returns" / f"returns_microgrid_{mappo_algorithm}.npy",
        "STAS": stas_out / "returns" / f"returns_microgrid_{stas_algorithm}.npy",
        "MATD3": matd3_out / "returns" / "returns_microgrid_MATD3-Fair-256.npy",
    }
    _plot_training(report_dir / "training_return_appendix.png", training_paths)
    metrics = {name: compute_curve_metrics(records) for name, records in curves.items()}
    stas_wins = (
        metrics["STAS"]["final_score"] > metrics["MAPPO"]["final_score"]
        and metrics["STAS"]["normalized_auc"] > metrics["MAPPO"]["normalized_auc"]
        and metrics["STAS"]["final_score"] > metrics["MATD3"]["final_score"]
    )
    commit_hash = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    summary = {
        "seed": 30,
        "selected_mappo": selected_mappo,
        "selected_stas": selected_stas,
        "validation_metrics": metrics,
        "final_test": test_results,
        "final_comparison": comparison,
        "stas_wins_this_seed": stas_wins,
        "commit_hash": commit_hash,
        "checkpoints": {name: str(path) for name, path in checkpoints.items()},
    }
    (report_dir / "final_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    verdict = "胜出" if stas_wins else "未满足胜出条件"
    report = f"""# MAPPO–MATD3–STAS 公平实验结论（seed=30）

本轮只使用 seed=30，属于探索性结果，不声明统计显著性。

## 模型选择

- 最佳修复后 MAPPO：{selected_mappo}
- 最佳 STAS：{selected_stas}
- 本 seed 判定：STAS {verdict}

## 固定 validation 指标

| 算法 | 29k/29.5k/30k 均值 | 0–30k 归一化 AUC |
|---|---:|---:|
| STAS | {metrics['STAS']['final_score']:.3f} | {metrics['STAS']['normalized_auc']:.3f} |
| MAPPO | {metrics['MAPPO']['final_score']:.3f} | {metrics['MAPPO']['normalized_auc']:.3f} |
| MATD3 | {metrics['MATD3']['final_score']:.3f} | {metrics['MATD3']['normalized_auc']:.3f} |

## 一次性 test split

| 算法 | test return mean | test return std |
|---|---:|---:|
| STAS | {test_results['STAS']['return_mean']:.3f} | {test_results['STAS']['return_std']:.3f} |
| MAPPO | {test_results['MAPPO']['return_mean']:.3f} | {test_results['MAPPO']['return_std']:.3f} |
| MATD3 | {test_results['MATD3']['return_mean']:.3f} | {test_results['MATD3']['return_std']:.3f} |

主图使用固定 deterministic validation return；原始 training return 仅作为附图诊断。完整数值、场景明细、checkpoint 与配置见同目录文件。
"""
    (report_dir / "结论报告.md").write_text(report, encoding="utf-8")
    marker.write_text("complete\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    run(args.root.resolve())


if __name__ == "__main__":
    main()
