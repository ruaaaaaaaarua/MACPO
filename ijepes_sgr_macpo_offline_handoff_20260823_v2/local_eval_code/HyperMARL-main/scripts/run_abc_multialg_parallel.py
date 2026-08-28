#!/usr/bin/env python3
"""Run STAS-MAPPO, MAPPO, and MATD3 on the A+B+C microgrid mechanism."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

try:
    from run_stas_mechanism_ablation import (
        HM_ROOT,
        LONGRUN_TIMESTEPS,
        SEED,
        SMOKE_TIMESTEPS,
        microgrid_override_arg,
        planned_experiments,
        score_returns,
        stas_override_args,
        write_json,
    )
except ModuleNotFoundError:
    from scripts.run_stas_mechanism_ablation import (
        HM_ROOT,
        LONGRUN_TIMESTEPS,
        SEED,
        SMOKE_TIMESTEPS,
        microgrid_override_arg,
        planned_experiments,
        score_returns,
        stas_override_args,
        write_json,
    )


REPO_ROOT = HM_ROOT.parent
DEFAULT_ROOT = REPO_ROOT / "result" / "abc_multialg_parallel_kg30_20260709"


def timestamp() -> str:
    return datetime.utcnow().isoformat() + "Z"


def group_abc_spec():
    for spec in planned_experiments():
        if spec.group == "group_abc":
            return spec
    raise RuntimeError("group_abc experiment spec is not available")


def safe_tag(value: str) -> str:
    return value.replace("/", "_").replace(" ", "_")


def alg_dir(root: Path, phase: str, alg: str) -> Path:
    return root / phase / alg


def command_specs(root: Path, smoke: bool) -> dict[str, dict[str, Any]]:
    phase = "smoke" if smoke else "longrun"
    spec = group_abc_spec()
    total_timesteps = SMOKE_TIMESTEPS if smoke else LONGRUN_TIMESTEPS
    matd3_episodes = 2 if smoke else 10000
    matd3_checkpoint_interval = 1 if smoke else 1000

    common_ppo = {
        "LR": 0.0002,
        "ANNEAL_LR": False,
        "UPDATE_EPOCHS": 8,
        "NUM_MINIBATCHES": 4,
        "CLIP_EPS": 0.2,
        "ENT_COEF": 0.01,
        "GAE_LAMBDA": 0.98,
        "LOG_STD_INIT": -0.5,
        "MAX_GRAD_NORM": 10.0,
    }

    stas_out = alg_dir(root, phase, "stas_mappo_abc") / "output"
    mappo_out = alg_dir(root, phase, "mappo_abc") / "output"
    matd3_out = alg_dir(root, phase, "matd3_abc") / "output"

    stas_alg = "STAS-MAPPO-ABC-smoke" if smoke else "STAS-MAPPO-ABC-10k"
    mappo_alg = "MAPPO-ABC-smoke" if smoke else "MAPPO-ABC-10k"
    matd3_alg = "MATD3-ABC-smoke" if smoke else "MATD3-ABC-10k"

    stas_cmd = [
        sys.executable,
        "baselines/STAS-MAPPO/mappo_stas.py",
        "--config-name=stas_mappo_microgrid",
        f"ALG={stas_alg}",
        "EXP_NAME=stas_mappo_abc_multialg",
        f"RUN_NAME=microgrid__{stas_alg}__seed{SEED}",
        f"SEED={SEED}",
        f"TOTAL_TIMESTEPS={total_timesteps}",
        "WANDB_MODE=disabled",
        "EVAL_INTERVAL=100000000",
        "CAPTURE_VIDEO_INTERVAL=null",
        "CHECKPOINT=True",
        f"CHECKPOINT_INTERVAL={96 if smoke else 96000}",
        microgrid_override_arg(spec.env_overrides),
        *stas_override_args(spec.stas_overrides),
    ]

    mappo_overrides = {**common_ppo, "VF_COEF": 1.0}
    mappo_cmd = [
        sys.executable,
        "baselines/MAPPO/mappo_ff_shared_weights.py",
        "--config-name=mappo_ff_independent_actors_microgrid",
        f"ALG={mappo_alg}",
        "EXP_NAME=mappo_abc_multialg",
        f"RUN_NAME=microgrid__{mappo_alg}__seed{SEED}",
        f"SEED={SEED}",
        f"TOTAL_TIMESTEPS={total_timesteps}",
        "WANDB_MODE=disabled",
        "EVAL_INTERVAL=100000000",
        "CAPTURE_VIDEO_INTERVAL=null",
        "CHECKPOINT=True",
        f"CHECKPOINT_INTERVAL={96 if smoke else 96000}",
        microgrid_override_arg(spec.env_overrides),
        *[f"{key}={str(value).lower() if isinstance(value, bool) else value}" for key, value in mappo_overrides.items()],
    ]

    matd3_cmd = [
        sys.executable,
        "baselines/MATD3/train_matd3_microgrid.py",
        "--seed",
        str(SEED),
        "--episodes",
        str(matd3_episodes),
        "--episode-length",
        "24",
        "--alg",
        matd3_alg,
        "--checkpoint-interval",
        str(matd3_checkpoint_interval),
        "--microgrid-overrides-json",
        json.dumps(spec.env_overrides, separators=(",", ":")),
    ]
    if smoke:
        matd3_cmd.extend(
            [
                "--batch-size",
                "16",
                "--start-steps",
                "0",
                "--update-after",
                "0",
                "--hidden-dim",
                "64",
                "--log-interval",
                "1",
            ]
        )

    return {
        "stas_mappo_abc": {
            "command": stas_cmd,
            "output_dir": stas_out,
            "progress_log": alg_dir(root, phase, "stas_mappo_abc") / "progress.jsonl",
            "returns_file": stas_out / "returns" / f"returns_microgrid_{stas_alg}.npy",
            "checkpoint_dir": Path(f"/tmp/models/microgrid__{stas_alg}__seed{SEED}"),
            "kind": "jax",
        },
        "mappo_abc": {
            "command": mappo_cmd,
            "output_dir": mappo_out,
            "progress_log": alg_dir(root, phase, "mappo_abc") / "progress.jsonl",
            "returns_file": mappo_out / "returns" / f"returns_microgrid_{mappo_alg}.npy",
            "checkpoint_dir": Path(f"/tmp/models/microgrid__{mappo_alg}__seed{SEED}"),
            "kind": "jax",
        },
        "matd3_abc": {
            "command": matd3_cmd,
            "output_dir": matd3_out,
            "progress_log": alg_dir(root, phase, "matd3_abc") / "progress.jsonl",
            "returns_file": matd3_out / "returns" / f"returns_microgrid_{safe_tag(matd3_alg)}.npy",
            "checkpoint_dir": matd3_out / "checkpoints" / safe_tag(matd3_alg),
            "kind": "torch",
        },
    }


def process_env(base_env: dict[str, str], item: dict[str, Any]) -> dict[str, str]:
    env = dict(base_env)
    env["PYTHONPATH"] = f"{HM_ROOT}:{env.get('PYTHONPATH', '')}"
    env["WANDB_MODE"] = "disabled"
    env["HYPERMARL_OUTPUT_DIR"] = str(Path(item["output_dir"]).resolve())
    env["HYPERMARL_PROGRESS_LOG"] = str(Path(item["progress_log"]).resolve())
    env["WANDB_DIR"] = str((Path(item["output_dir"]).resolve() / "wandb").resolve())
    Path(env["WANDB_DIR"]).mkdir(parents=True, exist_ok=True)
    if item["kind"] == "jax":
        env["XLA_PYTHON_CLIENT_MEM_FRACTION"] = env.get("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.30")
        env["XLA_PYTHON_CLIENT_PREALLOCATE"] = env.get("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    return env


def write_manifest(root: Path, phase: str, commands: dict[str, dict[str, Any]]) -> None:
    spec = group_abc_spec()
    write_json(
        root / phase / "manifest.json",
        {
            "created_at": timestamp(),
            "phase": phase,
            "seed": SEED,
            "environment": "group_abc",
            "group_abc_spec": asdict(spec),
            "algorithms": {
                name: {
                    "command": item["command"],
                    "output_dir": str(item["output_dir"]),
                    "progress_log": str(item["progress_log"]),
                    "returns_file": str(item["returns_file"]),
                    "checkpoint_dir": str(item["checkpoint_dir"]),
                    "kind": item["kind"],
                }
                for name, item in commands.items()
            },
        },
    )


def launch(root: Path, smoke: bool, parallel: bool) -> None:
    phase = "smoke" if smoke else "longrun"
    commands = command_specs(root, smoke)
    write_manifest(root, phase, commands)
    logs_dir = root / phase / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    procs: dict[str, subprocess.Popen[Any]] = {}
    for name, item in commands.items():
        Path(item["output_dir"]).mkdir(parents=True, exist_ok=True)
        Path(item["progress_log"]).parent.mkdir(parents=True, exist_ok=True)
        log_path = logs_dir / f"{name}.log"
        result_path = alg_dir(root, phase, name) / "run_result.json"
        if result_path.exists():
            existing = json.loads(result_path.read_text(encoding="utf-8"))
            if existing.get("status") == "success":
                print(f"[skip] {name} {phase} already succeeded")
                continue
        print(f"[launch] {name}: {' '.join(map(str, item['command']))}")
        log = log_path.open("w", encoding="utf-8")
        proc = subprocess.Popen(
            item["command"],
            cwd=HM_ROOT,
            env=process_env(os.environ, item),
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        procs[name] = proc
        (alg_dir(root, phase, name) / "pid.txt").write_text(str(proc.pid), encoding="utf-8")
        if not parallel:
            rc = proc.wait()
            log.close()
            write_run_result(root, phase, name, item, rc)

    if parallel:
        for name, proc in procs.items():
            rc = proc.wait()
            write_run_result(root, phase, name, commands[name], rc)
    summarize(root, phase)


def write_run_result(root: Path, phase: str, name: str, item: dict[str, Any], rc: int) -> None:
    returns_file = Path(item["returns_file"])
    status = "success" if rc == 0 and returns_file.exists() else "failed"
    result = {
        "algorithm": name,
        "phase": phase,
        "status": status,
        "returncode": int(rc),
        "returns_file": str(returns_file),
        "finished_at": timestamp(),
    }
    if returns_file.exists():
        if name == "matd3_abc":
            arr = np.load(returns_file).reshape(-1).astype(np.float64)
            result.update(
                {
                    "num_points": int(arr.size),
                    "final_return": float(arr[-1]),
                    "mean_return": float(np.mean(arr)),
                    "final_500_mean": float(np.mean(arr[-min(arr.size, 500) :])),
                    "best_rolling_500": float(
                        np.max(
                            np.convolve(
                                arr,
                                np.ones(min(arr.size, 500)) / min(arr.size, 500),
                                mode="valid",
                            )
                        )
                    ),
                }
            )
        else:
            result.update(score_returns(returns_file))
    result["checkpoints"] = checkpoint_paths_for(item, name)
    write_json(alg_dir(root, phase, name) / "checkpoint_manifest.json", {
        "algorithm": name,
        "phase": phase,
        "created_at": timestamp(),
        "checkpoints": result["checkpoints"],
    })
    write_json(alg_dir(root, phase, name) / "run_result.json", result)
    print(f"[{status}] {name} {phase}")


def checkpoint_paths_for(item: dict[str, Any], name: str) -> list[str]:
    checkpoint_dir = Path(item["checkpoint_dir"])
    if name == "matd3_abc":
        paths = sorted(checkpoint_dir.glob("*.pt"))
    else:
        paths = sorted(checkpoint_dir.glob("*"))
        paths = [path for path in paths if path.is_dir() or path.suffix == ".pt"]
    return [str(path) for path in paths]


def status(root: Path, phase: str) -> None:
    for name in command_specs(root, phase == "smoke"):
        adir = alg_dir(root, phase, name)
        print(f"\n== {name} ==")
        result_path = adir / "run_result.json"
        if result_path.exists():
            print(result_path.read_text(encoding="utf-8"))
        pid_path = adir / "pid.txt"
        if pid_path.exists():
            print(f"pid={pid_path.read_text(encoding='utf-8').strip()}")
        progress = adir / "progress.jsonl"
        if progress.exists():
            print("progress tail:")
            print("\n".join(progress.read_text(encoding="utf-8").splitlines()[-5:]))


def summarize(root: Path, phase: str) -> None:
    rows = []
    for name in command_specs(root, phase == "smoke"):
        result_path = alg_dir(root, phase, name) / "run_result.json"
        if result_path.exists():
            rows.append(json.loads(result_path.read_text(encoding="utf-8")))
    write_json(root / phase / "summary.json", {"created_at": timestamp(), "rows": rows})
    lines = [f"# ABC Multialg {phase} Summary", ""]
    lines.append("| algorithm | status | final_500 | best_500 | final_return | returns |")
    lines.append("|---|---|---:|---:|---:|---|")
    for row in rows:
        lines.append(
            "| {algorithm} | {status} | {final_500:.3f} | {best:.3f} | {final:.3f} | `{returns}` |".format(
                algorithm=row["algorithm"],
                status=row.get("status"),
                final_500=row.get("final_500_mean", float("nan")),
                best=row.get("best_rolling_500", float("nan")),
                final=row.get("final_return", float("nan")),
                returns=row.get("returns_file", ""),
            )
        )
    (root / phase / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["dry-run", "smoke", "launch", "status", "summarize"])
    parser.add_argument("--root", default=str(DEFAULT_ROOT))
    parser.add_argument("--phase", choices=["smoke", "longrun"], default="longrun")
    parser.add_argument("--sequential", action="store_true")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    if args.mode == "dry-run":
        for name, item in command_specs(root, args.phase == "smoke").items():
            print(name, " ".join(map(str, item["command"])))
        return
    if args.mode == "smoke":
        launch(root, smoke=True, parallel=not args.sequential)
        return
    if args.mode == "launch":
        launch(root, smoke=False, parallel=not args.sequential)
        return
    if args.mode == "status":
        status(root, args.phase)
        return
    if args.mode == "summarize":
        summarize(root, args.phase)


if __name__ == "__main__":
    main()
