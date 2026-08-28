#!/usr/bin/env python
"""Phase 2 matrix wave 2: STAS-blend / STAS-uniform / MAPPO+LR-anneal on env v2.

blend 调度按 2026-07-17 曲线诊断定案: 早开 (warmup 500)、低顶 (max mix
0.05)、缓坡 (ramp 2000)、因果单向、EV 门控 + negative patience。uniform
与 blend 完全同调度, 仅 credit 换均匀摊 (照妖镜)。mappo_anneal 与 wave-1
MAPPO 仅差 ANNEAL_LR=True, 用于隔离后程退化问题。

用法: PYTHONPATH=. python scripts/run_env_v2_matrix_wave2.py <output_root>
"""

from __future__ import annotations

import datetime
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.env_v2_overrides import env_v2_overrides, hydra_override_arg  # noqa: E402

SRC_MANIFEST = Path(
    "/root/autodl-tmp/traffic-stas-paper-sparse-20260716-10k-v1/train_10k_20k/manifest.json"
)
OLD_ROOT = "/root/autodl-tmp/traffic-stas-paper-sparse-20260716-10k-v1/train_10k_20k"
JOB_TIMEOUT_S = 5 * 3600

PAPER_ONLY_PREFIXES = (
    "STAS.MODE=",
    "+STAS.POLICY_WARMUP_EPISODES=",
    "+STAS.REWARD_MODEL_UPDATE_INTERVAL_EPISODES=",
    "+STAS.REWARD_MODEL_UPDATES_PER_INTERVAL=",
    "STAS.BUFFER_SIZE=",
    "STAS.BATCH_SIZE=",
)
BLEND_ARGS = [
    "+STAS.CONSERVE_DISCOUNTED=true",
    "+STAS.QUALITY_GATE_ENABLE=true",
    "+STAS.BIDIRECTIONAL=false",
    "+STAS.WARMUP_EPISODES=500",
    "+STAS.RAMP_EPISODES=2000",
    "+STAS.MAX_MIX_COEF=0.05",
    "+STAS.EXPLAINED_VARIANCE_THRESHOLD=0.2",
    "+STAS.NEGATIVE_PATIENCE=3",
    "STAS.MIX_COEF=0.0",
]


def log(message):
    print("[%s] %s" % (datetime.datetime.now().strftime("%H:%M:%S"), message), flush=True)


def source_jobs() -> dict:
    manifest = json.loads(SRC_MANIFEST.read_text())
    for value in manifest.values():
        if isinstance(value, dict) and value and all(
            isinstance(job, dict) and "command" in job for job in value.values()
        ):
            return value
    raise RuntimeError("jobs not found in paper-sparse manifest")


def build_commands(output_root: Path) -> dict:
    jobs = source_jobs()
    hydra_arg = hydra_override_arg(env_v2_overrides(sparse=True))

    def rebase(cmd, old_job, new_job):
        out = []
        for arg in cmd:
            arg = str(arg)
            if f"{OLD_ROOT}/{old_job}" in arg:
                arg = arg.replace(f"{OLD_ROOT}/{old_job}", str(output_root / new_job))
            if arg.startswith("+MICROGRID_CONFIG_OVERRIDES="):
                arg = hydra_arg
            out.append(arg)
        return out

    def rename(cmd, alg, name):
        return [
            f"ALG={alg}" if a.startswith("ALG=") else
            f"EXP_NAME={name}" if a.startswith("EXP_NAME=") else
            f"RUN_NAME={name}__seed30" if a.startswith("RUN_NAME=") else a
            for a in cmd
        ]

    def stas_variant(name, alg, extra):
        cmd = rebase(jobs["stas"]["command"], "stas", name)
        cmd = [a for a in cmd if not a.startswith(PAPER_ONLY_PREFIXES)]
        cmd = rename(cmd, alg, name)
        return cmd + BLEND_ARGS + extra

    commands = {
        "stas_blend": stas_variant(
            "stas_blend", "EnvV2-Blend-STAS", ["STAS.MODE=conserved"]
        ),
        "stas_uniform": stas_variant(
            "stas_uniform", "EnvV2-Uniform-STAS", ["STAS.MODE=uniform"]
        ),
        "mappo_anneal": rename(
            rebase(jobs["mappo"]["command"], "mappo", "mappo_anneal"),
            "EnvV2-MAPPO-Anneal", "mappo_anneal",
        ) + ["ANNEAL_LR=True"],
    }
    return commands


def main():
    output_root = Path(sys.argv[1])
    (output_root / "logs").mkdir(parents=True, exist_ok=True)
    commands = build_commands(output_root)
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True
    ).stdout.strip()
    (output_root / "manifest.json").write_text(json.dumps({
        "purpose": "Phase2 wave2: STAS-blend/STAS-uniform/MAPPO-anneal on frozen env v2",
        "created_at": datetime.datetime.now().isoformat(),
        "commit": commit,
        "seed": 30,
        "jobs": {name: {"command": cmd} for name, cmd in commands.items()},
    }, indent=2, ensure_ascii=False))

    env = dict(os.environ)
    env["PYTHONPATH"] = f"{ROOT}:{env.get('PYTHONPATH', '')}"
    env["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.25"
    env["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    env["WANDB_MODE"] = "disabled"

    pool = []
    for name, cmd in commands.items():
        (output_root / name / "output").mkdir(parents=True, exist_ok=True)
        logf = open(output_root / "logs" / f"{name}.log", "w")
        process = subprocess.Popen(
            cmd, cwd=ROOT, stdout=logf, stderr=subprocess.STDOUT, env=env
        )
        pool.append({
            "name": name, "proc": process, "logf": logf,
            "deadline": time.time() + JOB_TIMEOUT_S,
        })
        log(f"launched {name} pid={process.pid}")
        time.sleep(25)

    while pool:
        time.sleep(30)
        for job in list(pool):
            code = job["proc"].poll()
            if code is None and time.time() > job["deadline"]:
                job["proc"].kill()
                code = "timeout"
            if code is not None:
                job["logf"].close()
                (output_root / "logs" / f"{job['name']}.result.json").write_text(
                    json.dumps({"name": job["name"], "returncode": str(code),
                                "status": "success" if code == 0 else "failed"})
                )
                log(f"job {job['name']} finished rc={code}")
                pool.remove(job)

    (output_root / "WAVE2_DONE").write_text(datetime.datetime.now().isoformat())
    log("WAVE2 ALL DONE")


if __name__ == "__main__":
    main()
