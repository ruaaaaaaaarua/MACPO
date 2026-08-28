#!/usr/bin/env python3
"""Staged HPO runner for MAPPO-IA and STAS-MAPPO on the microgrid task."""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover - plotting is verified on the training host.
    plt = None

HM_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = HM_ROOT.parent
DEFAULT_ROOT = REPO_ROOT / "result" / "hpo_mappo_stas_20260708"
COMPARE_ROOT = REPO_ROOT / "result" / "compare_mappo_stas_matd3_10k"
STAGE1_TIMESTEPS = 48_000
LONGRUN_TIMESTEPS = 240_000
SEED = 30
EPISODE_STRIDE_JAX = 4


@dataclass(frozen=True)
class TrialSpec:
    stage: str
    algorithm: str
    trial_id: str
    total_timesteps: int
    overrides: Dict[str, Any]
    episode_stride: int = EPISODE_STRIDE_JAX


def _timestamp() -> str:
    return datetime.utcnow().isoformat() + "Z"


def _safe_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _hydra_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, list):
        return "[" + ",".join(_hydra_value(item) for item in value) + "]"
    return str(value)


def build_planned_trials() -> List[TrialSpec]:
    """Build deterministic Stage 0 and Stage 1 trials."""
    trials: List[TrialSpec] = [
        TrialSpec(
            "stage0",
            "stas",
            "stage0_stas_mix0p2_default",
            STAGE1_TIMESTEPS,
            {"STAS.MIX_COEF": 0.2},
        ),
    ]

    # MAPPO is kept as a fixed baseline only; HPO trials tune STAS-MAPPO.
    stas_space = {
        "LR": [1e-4, 2e-4, 3e-4, 5e-4],
        "ANNEAL_LR": [False, True],
        "UPDATE_EPOCHS": [4, 6, 8],
        "NUM_MINIBATCHES": [1, 2, 4],
        "CLIP_EPS": [0.10, 0.15, 0.20],
        "ENT_COEF": [0.001, 0.003, 0.01],
        "GAE_LAMBDA": [0.90, 0.95, 0.98],
        "LOG_STD_INIT": [-1.5, -1.0, -0.5],
        "MAX_GRAD_NORM": [1.0, 5.0, 10.0],
        "STAS.MIX_COEF": [0.05, 0.10, 0.20, 0.30, 0.40],
        "STAS.LR": [1e-4, 3e-4, 1e-3],
        "STAS.BATCH_SIZE": [8, 16, 32],
        "STAS.UPDATE_FREQ": [1, 2, 4],
        "STAS.UPDATES_PER_STEP": [1, 2],
        "STAS.WARMUP_ROLLOUTS": [2, 4, 8],
        "STAS.DROPOUT": [0.0, 0.1, 0.2],
    }
    for idx in range(32):
        overrides = {
            key: values[(idx * (step + 1) + step) % len(values)]
            for step, (key, values) in enumerate(stas_space.items(), start=1)
        }
        trials.append(
            TrialSpec(
                "stage1",
                "stas",
                f"stage1_stas_{idx:02d}",
                STAGE1_TIMESTEPS,
                overrides,
            )
        )
    return trials


def score_returns(returns: np.ndarray, trial: TrialSpec) -> Dict[str, float]:
    arr = np.asarray(returns, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        return {
            "score": float("-inf"),
            "final_return": float("nan"),
            "mean_return": float("nan"),
            "best_rolling_500": float("-inf"),
            "final_smoothed": float("-inf"),
            "window_points": max(1, int(round(500 / trial.episode_stride))),
            "num_points": 0,
        }
    window = min(arr.size, max(1, int(round(500 / trial.episode_stride))))
    rolling = np.convolve(arr, np.ones(window, dtype=np.float64) / window, mode="valid")
    score = float(np.mean(arr[-window:]))
    return {
        "score": score,
        "final_return": float(arr[-1]),
        "mean_return": float(np.mean(arr)),
        "best_rolling_500": float(np.max(rolling)) if rolling.size else score,
        "final_smoothed": score,
        "window_points": int(window),
        "num_points": int(arr.size),
    }


def moving_average(values: np.ndarray, window: int) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    if arr.size == 0 or window <= 1:
        return arr
    window = min(window, arr.size)
    prefix = np.asarray([np.mean(arr[:idx]) for idx in range(1, window)])
    valid = np.convolve(arr, np.ones(window) / window, mode="valid")
    return np.concatenate([prefix, valid])


def root_dirs(root: Path) -> None:
    for rel in ["logs", "summaries", "plots", "trials", "longruns"]:
        (root / rel).mkdir(parents=True, exist_ok=True)


def load_microgrid_override_arg() -> str:
    sys.path.insert(0, str(HM_ROOT))
    from scripts.microgrid_experiment_overrides import HYDRA_OVERRIDE_ARGS

    return HYDRA_OVERRIDE_ARGS


def trial_dir(root: Path, trial: TrialSpec) -> Path:
    parent = "longruns" if trial.stage == "stage2" else "trials"
    return root / parent / trial.stage / trial.trial_id


def command_for_trial(trial: TrialSpec, out_dir: Path) -> List[str]:
    override_arg = load_microgrid_override_arg()
    alg_label = f"HPO-{trial.algorithm.upper()}-{trial.trial_id}"
    run_name = f"microgrid__{alg_label}__seed{SEED}"
    common = [
        f"ALG={alg_label}",
        "EXP_NAME=hpo_mappo_stas_microgrid",
        f"RUN_NAME={run_name}",
        f"SEED={SEED}",
        f"TOTAL_TIMESTEPS={trial.total_timesteps}",
        "WANDB_MODE=disabled",
        "EVAL_INTERVAL=100000000",
        "CAPTURE_VIDEO_INTERVAL=null",
        override_arg,
    ]
    checkpoint = trial.stage == "stage2"
    if checkpoint:
        common.extend(["CHECKPOINT=True", "CHECKPOINT_INTERVAL=96000"])
    else:
        common.extend(["CHECKPOINT=False", "CHECKPOINT_INTERVAL=100000000"])
    for key, value in trial.overrides.items():
        common.append(f"{key}={_hydra_value(value)}")
    if trial.algorithm == "mappo":
        return [
            sys.executable,
            "baselines/MAPPO/mappo_ff_shared_weights.py",
            "--config-name=mappo_ff_independent_actors_microgrid",
            *common,
        ]
    if trial.algorithm == "stas":
        return [
            sys.executable,
            "baselines/STAS-MAPPO/mappo_stas.py",
            "--config-name=stas_mappo_microgrid",
            *common,
        ]
    raise ValueError(f"Unknown algorithm: {trial.algorithm}")


def expected_returns_file(out_dir: Path, trial: TrialSpec) -> Path:
    alg_label = f"HPO-{trial.algorithm.upper()}-{trial.trial_id}"
    return out_dir / "returns" / f"returns_microgrid_{alg_label}.npy"


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, sort_keys=True)


def append_jsonl(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")


def run_trial(root: Path, trial: TrialSpec, *, dry_run: bool = False, force: bool = False) -> None:
    tdir = trial_dir(root, trial)
    result_path = tdir / "trial_result.json"
    if result_path.exists() and not force:
        existing = json.loads(result_path.read_text(encoding="utf-8"))
        if existing.get("status") == "success":
            print(f"[skip] {trial.trial_id} already succeeded")
            return
    tdir.mkdir(parents=True, exist_ok=True)
    out_dir = tdir / "output"
    progress_log = tdir / "progress.jsonl"
    command = command_for_trial(trial, out_dir)
    config_payload = {
        **asdict(trial),
        "seed": SEED,
        "command": command,
        "output_dir": str(out_dir),
        "progress_log": str(progress_log),
        "created_at": _timestamp(),
    }
    write_json(tdir / "trial_config.json", config_payload)
    append_jsonl(root / "hpo_manifest.jsonl", {"event": "planned", **config_payload})
    if dry_run:
        print(" ".join(command))
        return

    env = os.environ.copy()
    env["PYTHONPATH"] = f"{HM_ROOT}:{env.get('PYTHONPATH', '')}"
    env["WANDB_MODE"] = "disabled"
    env["HYPERMARL_OUTPUT_DIR"] = str(out_dir)
    env["HYPERMARL_PROGRESS_LOG"] = str(progress_log)
    env["XLA_PYTHON_CLIENT_MEM_FRACTION"] = env.get("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.70")
    start = time.time()
    log_path = tdir / "train.log"
    print(f"[run] {trial.trial_id}")
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.run(command, cwd=HM_ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
    elapsed = time.time() - start
    returns_file = expected_returns_file(out_dir, trial)
    status = "success" if proc.returncode == 0 and returns_file.exists() else "failed"
    result: Dict[str, Any] = {
        "trial_id": trial.trial_id,
        "stage": trial.stage,
        "algorithm": trial.algorithm,
        "status": status,
        "returncode": int(proc.returncode),
        "elapsed_seconds": elapsed,
        "returns_file": str(returns_file),
        "train_log": str(log_path),
        "finished_at": _timestamp(),
        "overrides": trial.overrides,
        "total_timesteps": trial.total_timesteps,
    }
    if returns_file.exists():
        returns = np.load(returns_file)
        result.update(score_returns(returns, trial))
        shutil.copy2(returns_file, tdir / "returns.npy")
    write_json(result_path, result)
    append_jsonl(root / "hpo_manifest.jsonl", {"event": "finished", **result})
    if status != "success":
        print(f"[failed] {trial.trial_id} rc={proc.returncode} log={log_path}")
    else:
        print(f"[done] {trial.trial_id} score={result['score']:.3f}")


def iter_results(root: Path, stage: str | None = None, algorithm: str | None = None) -> List[Dict[str, Any]]:
    matches = sorted(root.glob("**/trial_result.json"))
    rows: List[Dict[str, Any]] = []
    for path in matches:
        row = json.loads(path.read_text(encoding="utf-8"))
        if stage is not None and row.get("stage") != stage:
            continue
        if algorithm is not None and row.get("algorithm") != algorithm:
            continue
        row["result_path"] = str(path)
        rows.append(row)
    rows.sort(
        key=lambda row: (
            row.get("status") != "success",
            -float(row.get("score", float("-inf"))),
            -float(row.get("best_rolling_500", float("-inf"))),
            -float(row.get("final_smoothed", float("-inf"))),
        )
    )
    return rows


def local_variants(anchor: Dict[str, Any], index: int) -> List[TrialSpec]:
    overrides = dict(anchor.get("overrides", {}))
    alg = str(anchor["algorithm"])
    variants: List[TrialSpec] = []
    lr = float(overrides.get("LR", 3e-4))
    ent = float(overrides.get("ENT_COEF", 0.01))
    clip = float(overrides.get("CLIP_EPS", 0.2))
    base_id = str(anchor["trial_id"]).replace("stage1_", "")
    for suffix, factor in [("low_lr", 0.7), ("high_lr", 1.3)]:
        tuned = dict(overrides)
        tuned["LR"] = max(5e-5, min(8e-4, lr * factor))
        tuned["ENT_COEF"] = max(1e-4, min(0.05, ent * (1.2 if factor < 1 else 0.8)))
        tuned["CLIP_EPS"] = max(0.08, min(0.30, clip + (-0.03 if factor < 1 else 0.03)))
        if alg == "stas" and "STAS.MIX_COEF" in tuned:
            mix = float(tuned["STAS.MIX_COEF"])
            tuned["STAS.MIX_COEF"] = max(0.02, min(0.45, mix + (-0.05 if factor < 1 else 0.05)))
        variants.append(
            TrialSpec(
                "stage1b",
                alg,
                f"stage1b_{alg}_{index:02d}_{base_id}_{suffix}",
                STAGE1_TIMESTEPS,
                tuned,
            )
        )
    return variants


def build_stage1b_trials(root: Path) -> List[TrialSpec]:
    anchors = iter_results(root, "stage1", "stas")[:3]
    trials: List[TrialSpec] = []
    for idx, anchor in enumerate(anchors):
        if anchor.get("status") == "success":
            trials.extend(local_variants(anchor, idx))
    return trials


def build_stage2_trials(root: Path) -> List[TrialSpec]:
    candidates = [
        row
        for row in iter_results(root, algorithm="stas")
        if row.get("stage") in {"stage1", "stage1b"} and row.get("status") == "success"
    ]
    trials: List[TrialSpec] = []
    for idx, row in enumerate(candidates[:3], start=1):
        trials.append(
            TrialSpec(
                "stage2",
                "stas",
                f"stage2_stas_candidate_{idx}",
                LONGRUN_TIMESTEPS,
                row["overrides"],
            )
        )
    if not trials:
        trials.append(
            TrialSpec(
                "stage2",
                "stas",
                "stage2_stas_mix0p2_default",
                LONGRUN_TIMESTEPS,
                {"STAS.MIX_COEF": 0.2},
            )
        )
    return trials[:3]


def write_rows_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "rank",
        "trial_id",
        "stage",
        "algorithm",
        "status",
        "score",
        "best_rolling_500",
        "final_smoothed",
        "final_return",
        "mean_return",
        "num_points",
        "window_points",
        "total_timesteps",
        "elapsed_seconds",
        "overrides",
        "returns_file",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for rank, row in enumerate(rows, start=1):
            payload = {key: row.get(key, "") for key in fieldnames}
            payload["rank"] = rank
            payload["overrides"] = json.dumps(row.get("overrides", {}), sort_keys=True)
            writer.writerow(payload)


def summarize(root: Path) -> None:
    root_dirs(root)
    planned = build_planned_trials()
    write_json(
        root / "search_space.json",
        {
            "stage1_timesteps": STAGE1_TIMESTEPS,
            "longrun_timesteps": LONGRUN_TIMESTEPS,
            "seed": SEED,
            "tuning_scope": "STAS-MAPPO only; MAPPO and MATD3 are fixed baselines",
            "planned_trial_count": len(planned),
            "planned_trials": [asdict(trial) for trial in planned],
        },
    )
    all_rows = iter_results(root)
    stage1_rows = [row for row in all_rows if row.get("stage") == "stage1"]
    stage1b_rows = [row for row in all_rows if row.get("stage") == "stage1b"]
    longrun_rows = [row for row in all_rows if row.get("stage") == "stage2"]
    write_rows_csv(root / "summaries" / "stage1_trials.csv", stage1_rows)
    write_rows_csv(root / "summaries" / "stage1b_trials.csv", stage1b_rows)
    write_rows_csv(root / "summaries" / "longrun_trials.csv", longrun_rows)
    write_rows_csv(root / "summaries" / "all_trials.csv", all_rows)
    make_plots(root, stage1_rows, longrun_rows)
    write_report(root, all_rows, longrun_rows)
    print(f"[summary] wrote summaries under {root}")


def make_plots(root: Path, stage1_rows: Sequence[Dict[str, Any]], longrun_rows: Sequence[Dict[str, Any]]) -> None:
    if plt is None:
        return
    success = [row for row in stage1_rows if row.get("status") == "success"]
    if success:
        scores = [float(row["score"]) for row in success]
        labels = [row["trial_id"] for row in success]
        colors = ["#1f77b4" if row["algorithm"] == "mappo" else "#2ca02c" for row in success]
        fig, ax = plt.subplots(figsize=(12, 5), dpi=160)
        ax.bar(range(len(scores)), scores, color=colors)
        ax.set_title("Stage 1 HPO Scores")
        ax.set_xlabel("Trial rank")
        ax.set_ylabel("Mean return over last ~500 episodes")
        ax.set_xticks(range(len(scores)))
        ax.set_xticklabels(labels, rotation=90, fontsize=6)
        fig.tight_layout()
        fig.savefig(root / "plots" / "stage1_optimization_history.png")
        plt.close(fig)

        params = ["LR", "ENT_COEF", "CLIP_EPS", "UPDATE_EPOCHS", "NUM_MINIBATCHES"]
        fig, axes = plt.subplots(len(params), 1, figsize=(10, 9), dpi=160, sharex=True)
        for ax, param in zip(axes, params):
            xs = [row.get("overrides", {}).get(param, np.nan) for row in success]
            ax.scatter(range(len(xs)), xs, c=colors, s=18)
            ax.set_ylabel(param)
        axes[-1].set_xlabel("Trial rank")
        fig.suptitle("Stage 1 Parallel-Coordinate Proxy")
        fig.tight_layout()
        fig.savefig(root / "plots" / "stage1_parallel_coordinates.png")
        plt.close(fig)

    comparison_rows = list(longrun_rows)
    compare_returns = [
        ("MATD3 baseline", COMPARE_ROOT / "matd3/returns/returns_microgrid_MATD3-10kEp.npy", 1, "#9467bd"),
        ("MAPPO old", COMPARE_ROOT / "mappo_ia/returns/returns_microgrid_MAPPO-IA-CTDE-10kEp.npy", 4, "#7f7f7f"),
    ]
    palette = {"mappo": "#1f77b4", "stas": "#2ca02c"}
    for row in comparison_rows:
        path = Path(str(row.get("returns_file", "")))
        if path.exists():
            compare_returns.append((row["trial_id"], path, EPISODE_STRIDE_JAX, palette.get(row["algorithm"], "#d62728")))
    if compare_returns:
        fig, ax = plt.subplots(figsize=(12, 7), dpi=180)
        for label, path, stride, color in compare_returns:
            if not path.exists():
                continue
            arr = np.load(path).astype(np.float64).reshape(-1)
            x = np.arange(1, arr.size + 1) * stride
            window = max(1, int(round(500 / stride)))
            smooth = moving_average(arr, window)
            ax.plot(x, smooth, label=label, color=color, linewidth=2)
        ax.set_xlim(1, 10_000)
        ax.set_xlabel("Episode")
        ax.set_ylabel("Smoothed return")
        ax.set_title("HPO Longrun vs Baselines")
        ax.legend()
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(root / "plots" / "longrun_smoothed_comparison.png")
        plt.close(fig)


def write_report(root: Path, all_rows: Sequence[Dict[str, Any]], longrun_rows: Sequence[Dict[str, Any]]) -> None:
    best_mappo = next((row for row in all_rows if row.get("algorithm") == "mappo" and row.get("status") == "success"), None)
    best_stas = next((row for row in all_rows if row.get("algorithm") == "stas" and row.get("status") == "success"), None)
    lines = [
        "# MAPPO / STAS-MAPPO HPO Report",
        "",
        "## A. Direct Conclusion",
        "- Status: candidate best only; this run intentionally does not include multi-seed validation.",
        "- Tuning scope: STAS-MAPPO only. MAPPO and MATD3 are fixed baselines for comparison.",
        f"- Best STAS config: `{json.dumps(best_stas.get('overrides', {}) if best_stas else {}, sort_keys=True)}`",
        "",
        "## B. Evidence",
        f"- Total successful trials: {sum(1 for row in all_rows if row.get('status') == 'success')}",
        f"- Failed trials: {sum(1 for row in all_rows if row.get('status') != 'success')}",
        "- Search space: `search_space.json`",
        "- Trial tables: `summaries/stage1_trials.csv`, `summaries/stage1b_trials.csv`, `summaries/longrun_trials.csv`",
        "- Plots: `plots/stage1_optimization_history.png`, `plots/stage1_parallel_coordinates.png`, `plots/longrun_smoothed_comparison.png`",
        "",
        "### Longrun Trials",
    ]
    if not longrun_rows:
        lines.append("- No longrun trials completed yet.")
    for row in longrun_rows:
        lines.append(
            f"- {row['trial_id']} ({row['algorithm']}): status={row['status']}, "
            f"score={float(row.get('score', float('nan'))):.3f}, "
            f"best_rolling_500={float(row.get('best_rolling_500', float('nan'))):.3f}"
        )
    lines.extend(
        [
            "",
            "## C. Risks And Next Actions",
            "- No multi-seed gate has been run, so do not call these validated best configs.",
            "- If a PPO candidate beats MATD3 in longrun, rerun it with at least 3 seeds before paper-level claims.",
            "- If all PPO candidates remain below MATD3, report that tuning did not close the gap under this environment and implementation.",
        ]
    )
    (root / "final_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_trials(root: Path, trials: Iterable[TrialSpec], *, dry_run: bool, force: bool) -> None:
    root_dirs(root)
    for trial in trials:
        run_trial(root, trial, dry_run=dry_run, force=force)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["dry-run", "stage0", "stage1", "stage1b", "stage2", "all", "summarize"])
    parser.add_argument("--root", type=Path, default=Path(os.environ.get("OUT_ROOT", DEFAULT_ROOT)))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    root_dirs(root)
    if args.mode == "dry-run":
        run_trials(root, build_planned_trials(), dry_run=True, force=args.force)
        summarize(root)
    elif args.mode == "stage0":
        run_trials(root, [trial for trial in build_planned_trials() if trial.stage == "stage0"], dry_run=False, force=args.force)
        summarize(root)
    elif args.mode == "stage1":
        run_trials(root, [trial for trial in build_planned_trials() if trial.stage == "stage1"], dry_run=False, force=args.force)
        summarize(root)
    elif args.mode == "stage1b":
        run_trials(root, build_stage1b_trials(root), dry_run=False, force=args.force)
        summarize(root)
    elif args.mode == "stage2":
        run_trials(root, build_stage2_trials(root), dry_run=False, force=args.force)
        summarize(root)
    elif args.mode == "all":
        run_trials(root, [trial for trial in build_planned_trials() if trial.stage == "stage0"], dry_run=False, force=args.force)
        run_trials(root, [trial for trial in build_planned_trials() if trial.stage == "stage1"], dry_run=False, force=args.force)
        summarize(root)
        run_trials(root, build_stage1b_trials(root), dry_run=False, force=args.force)
        summarize(root)
        run_trials(root, build_stage2_trials(root), dry_run=False, force=args.force)
        summarize(root)
        (root / "logs" / "all_done.txt").write_text(f"ALL_DONE {_timestamp()}\n", encoding="utf-8")
    elif args.mode == "summarize":
        summarize(root)


if __name__ == "__main__":
    main()
