#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

try:
    from run_abc_multialg_resume10k import write_run_result_resume
    from run_abc_multialg_parallel import alg_dir, group_abc_spec, process_env, timestamp
    from run_stas_mechanism_ablation import HM_ROOT, LONGRUN_TIMESTEPS, SEED, microgrid_override_arg, write_json
except ModuleNotFoundError:
    from scripts.run_abc_multialg_resume10k import write_run_result_resume
    from scripts.run_abc_multialg_parallel import alg_dir, group_abc_spec, process_env, timestamp
    from scripts.run_stas_mechanism_ablation import HM_ROOT, LONGRUN_TIMESTEPS, SEED, microgrid_override_arg, write_json

REPO_ROOT = HM_ROOT.parent
DEFAULT_ROOT = REPO_ROOT / "result" / "abc_ppo_resume20to30k_kg30_20260709"

SRC = {
    "stas_policy": "/tmp/models/microgrid__STAS-MAPPO-ABC-resume10k__seed30/stas_mappo_abc_resume10k_240000_steps_2499_updates.agent_30_seed",
    "stas_reward_model": "/tmp/models/microgrid__STAS-MAPPO-ABC-resume10k__seed30/stas_reward_model_240000_steps.pt",
    "mappo_policy": "/tmp/models/microgrid__MAPPO-ABC-resume10k__seed30/mappo_abc_resume10k_240000_steps_2499_updates.agent_30_seed",
}


def validate_sources() -> None:
    missing = [path for path in SRC.values() if not Path(path).exists()]
    if missing:
        raise FileNotFoundError("Missing resume checkpoint(s): " + ", ".join(missing))


def command_specs(root: Path) -> dict[str, dict[str, Any]]:
    spec = group_abc_spec()
    common = {
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
    stas = {
        **common,
        "STAS.MIX_COEF": 0.2,
        "STAS.LR": 0.001,
        "STAS.BATCH_SIZE": 32,
        "STAS.UPDATE_FREQ": 4,
        "STAS.UPDATES_PER_STEP": 1,
        "STAS.WARMUP_ROLLOUTS": 8,
        "STAS.DROPOUT": 0.2,
    }
    mappo = {**common, "VF_COEF": 1.0}
    stas_alg = "STAS-MAPPO-ABC-resume20to30k"
    mappo_alg = "MAPPO-ABC-resume20to30k"
    stas_out = alg_dir(root, "longrun", "stas_mappo_abc_resume20to30k") / "output"
    mappo_out = alg_dir(root, "longrun", "mappo_abc_resume20to30k") / "output"
    stas_cmd = [
        sys.executable,
        "baselines/STAS-MAPPO/mappo_stas.py",
        "--config-name=stas_mappo_microgrid",
        f"ALG={stas_alg}",
        "EXP_NAME=stas_mappo_abc_resume20to30k",
        f"RUN_NAME=microgrid__{stas_alg}__seed{SEED}",
        f"SEED={SEED}",
        f"TOTAL_TIMESTEPS={LONGRUN_TIMESTEPS}",
        "WANDB_MODE=disabled",
        "EVAL_INTERVAL=100000000",
        "CAPTURE_VIDEO_INTERVAL=null",
        "CHECKPOINT=True",
        "CHECKPOINT_INTERVAL=96000",
        f"+CHECKPOINT_LOAD_DIR={SRC['stas_policy']}",
        f"+STAS.REWARD_MODEL_LOAD_PATH={SRC['stas_reward_model']}",
        microgrid_override_arg(spec.env_overrides),
        *[f"{k}={str(v).lower() if isinstance(v, bool) else v}" for k, v in stas.items()],
    ]
    mappo_cmd = [
        sys.executable,
        "baselines/MAPPO/mappo_ff_shared_weights.py",
        "--config-name=mappo_ff_independent_actors_microgrid",
        f"ALG={mappo_alg}",
        "EXP_NAME=mappo_abc_resume20to30k",
        f"RUN_NAME=microgrid__{mappo_alg}__seed{SEED}",
        f"SEED={SEED}",
        f"TOTAL_TIMESTEPS={LONGRUN_TIMESTEPS}",
        "WANDB_MODE=disabled",
        "EVAL_INTERVAL=100000000",
        "CAPTURE_VIDEO_INTERVAL=null",
        "CHECKPOINT=True",
        "CHECKPOINT_INTERVAL=96000",
        f"+CHECKPOINT_LOAD_DIR={SRC['mappo_policy']}",
        microgrid_override_arg(spec.env_overrides),
        *[f"{k}={str(v).lower() if isinstance(v, bool) else v}" for k, v in mappo.items()],
    ]
    return {
        "stas_mappo_abc_resume20to30k": {
            "command": stas_cmd,
            "output_dir": stas_out,
            "progress_log": alg_dir(root, "longrun", "stas_mappo_abc_resume20to30k") / "progress.jsonl",
            "returns_file": stas_out / "returns" / f"returns_microgrid_{stas_alg}.npy",
            "checkpoint_dir": Path(f"/tmp/models/microgrid__{stas_alg}__seed{SEED}"),
            "kind": "jax",
            "source_checkpoint": SRC["stas_policy"],
            "source_stas_reward_model": SRC["stas_reward_model"],
        },
        "mappo_abc_resume20to30k": {
            "command": mappo_cmd,
            "output_dir": mappo_out,
            "progress_log": alg_dir(root, "longrun", "mappo_abc_resume20to30k") / "progress.jsonl",
            "returns_file": mappo_out / "returns" / f"returns_microgrid_{mappo_alg}.npy",
            "checkpoint_dir": Path(f"/tmp/models/microgrid__{mappo_alg}__seed{SEED}"),
            "kind": "jax",
            "source_checkpoint": SRC["mappo_policy"],
        },
    }


def write_manifest(root: Path, commands: dict[str, dict[str, Any]]) -> None:
    write_json(root / "longrun" / "manifest.json", {
        "created_at": timestamp(),
        "phase": "longrun",
        "seed": SEED,
        "environment": "group_abc",
        "note": "PPO-only continuation from 20k to 30k while MATD3 resume10k continues.",
        "source_checkpoints": SRC,
        "algorithms": {
            name: {
                "command": item["command"],
                "output_dir": str(item["output_dir"]),
                "progress_log": str(item["progress_log"]),
                "returns_file": str(item["returns_file"]),
                "checkpoint_dir": str(item["checkpoint_dir"]),
                "source_checkpoint": item.get("source_checkpoint"),
                "source_stas_reward_model": item.get("source_stas_reward_model"),
            } for name, item in commands.items()
        },
    })


def summarize(root: Path) -> None:
    rows = []
    for name in command_specs(root):
        path = alg_dir(root, "longrun", name) / "run_result.json"
        if path.exists():
            rows.append(json.loads(path.read_text(encoding="utf-8")))
    write_json(root / "longrun" / "summary.json", {"created_at": timestamp(), "rows": rows})
    lines = ["# ABC PPO Resume 20k to 30k Summary", ""]
    lines.append("| algorithm | status | final_500 | best_500 | final_return | returns |")
    lines.append("|---|---|---:|---:|---:|---|")
    for row in rows:
        lines.append("| {algorithm} | {status} | {final_500:.3f} | {best:.3f} | {final:.3f} | `{returns}` |".format(
            algorithm=row["algorithm"], status=row.get("status"),
            final_500=row.get("final_500_mean", float("nan")), best=row.get("best_rolling_500", float("nan")),
            final=row.get("final_return", float("nan")), returns=row.get("returns_file", ""),
        ))
    (root / "longrun" / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def launch(root: Path, parallel: bool) -> None:
    validate_sources()
    commands = command_specs(root)
    write_manifest(root, commands)
    logs_dir = root / "longrun" / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    procs = {}
    for name, item in commands.items():
        result_path = alg_dir(root, "longrun", name) / "run_result.json"
        if result_path.exists() and json.loads(result_path.read_text(encoding="utf-8")).get("status") == "success":
            print(f"[skip] {name} already succeeded")
            continue
        Path(item["output_dir"]).mkdir(parents=True, exist_ok=True)
        Path(item["progress_log"]).parent.mkdir(parents=True, exist_ok=True)
        log = (logs_dir / f"{name}.log").open("w", encoding="utf-8")
        print(f"[launch] {name}: {' '.join(map(str, item['command']))}")
        proc = subprocess.Popen(item["command"], cwd=HM_ROOT, env=process_env(os.environ, item), stdout=log, stderr=subprocess.STDOUT)
        procs[name] = proc
        (alg_dir(root, "longrun", name) / "pid.txt").write_text(str(proc.pid), encoding="utf-8")
        if not parallel:
            rc = proc.wait()
            log.close()
            write_run_result_resume(root, "longrun", name, item, rc)
    if parallel:
        for name, proc in procs.items():
            rc = proc.wait()
            write_run_result_resume(root, "longrun", name, commands[name], rc)
    summarize(root)


def status(root: Path) -> None:
    for name in command_specs(root):
        adir = alg_dir(root, "longrun", name)
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["dry-run", "launch", "status", "summarize"])
    parser.add_argument("--root", default=str(DEFAULT_ROOT))
    parser.add_argument("--sequential", action="store_true")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    if args.mode == "dry-run":
        validate_sources()
        for name, item in command_specs(root).items():
            print(name, " ".join(map(str, item["command"])))
    elif args.mode == "launch":
        launch(root, parallel=not args.sequential)
    elif args.mode == "status":
        status(root)
    elif args.mode == "summarize":
        summarize(root)

if __name__ == "__main__":
    main()
