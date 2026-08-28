#!/usr/bin/env python
"""Phase 2 matrix launcher (wave 1): MAPPO / MATD3 / STAS-paper on env v2.

复用 paper-sparse 清单中的已验证命令, 仅替换环境覆盖 (v2 规范配置)、
路径与命名。wave 2 (uniform-IRCR / STAS-blend + 新门控) 另行实现。

用法: PYTHONPATH=. python scripts/run_env_v2_matrix.py <output_root>
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
JOB_TIMEOUT_S = {"mappo": 3 * 3600, "matd3": 5 * 3600, "stas_paper": 3 * 3600}


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
    overrides = env_v2_overrides(sparse=True)
    hydra_arg = hydra_override_arg(overrides)
    json_arg = json.dumps(overrides, separators=(",", ":"))
    commands = {}

    def rewrite(cmd, old_job, new_job, alg):
        out = []
        for arg in cmd:
            arg = str(arg)
            if f"{OLD_ROOT}/{old_job}" in arg:
                arg = arg.replace(
                    f"{OLD_ROOT}/{old_job}", str(output_root / new_job)
                )
            out.append(arg)
        return out

    mappo = rewrite(jobs["mappo"]["command"], "mappo", "mappo", None)
    mappo = [
        hydra_arg if a.startswith("+MICROGRID_CONFIG_OVERRIDES=") else a
        for a in mappo
    ]
    mappo = [
        a.replace("ALG=", "ALG=EnvV2-") if a.startswith("ALG=") else a
        for a in mappo
    ]
    commands["mappo"] = mappo

    matd3 = rewrite(jobs["matd3"]["command"], "matd3", "matd3", None)
    patched = []
    replace_next = None
    for arg in matd3:
        if replace_next is not None:
            patched.append(replace_next)
            replace_next = None
            continue
        if arg == "--microgrid-overrides-json":
            patched.append(arg)
            replace_next = json_arg
            continue
        if arg == "--alg":
            patched.append(arg)
            replace_next = "EnvV2-MATD3"
            continue
        patched.append(arg)
    commands["matd3"] = patched

    stas = rewrite(jobs["stas"]["command"], "stas", "stas_paper", None)
    stas = [
        hydra_arg if a.startswith("+MICROGRID_CONFIG_OVERRIDES=") else a
        for a in stas
    ]
    stas = [
        "ALG=EnvV2-Paper-STAS" if a.startswith("ALG=") else
        "EXP_NAME=stas_paper" if a.startswith("EXP_NAME=") else
        "RUN_NAME=stas_paper__seed30" if a.startswith("RUN_NAME=") else a
        for a in stas
    ]
    commands["stas_paper"] = stas
    return commands


def main():
    output_root = Path(sys.argv[1])
    (output_root / "logs").mkdir(parents=True, exist_ok=True)
    commands = build_commands(output_root)
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True
    ).stdout.strip()
    (output_root / "manifest.json").write_text(json.dumps({
        "purpose": "Phase2 wave1: MAPPO/MATD3/STAS-paper on frozen env v2 (sparse)",
        "created_at": datetime.datetime.now().isoformat(),
        "commit": commit,
        "seed": 30,
        "episodes": 10000,
        "jobs": {name: {"command": cmd} for name, cmd in commands.items()},
    }, indent=2, ensure_ascii=False))

    env = dict(os.environ)
    env["PYTHONPATH"] = f"{ROOT}:{env.get('PYTHONPATH', '')}"
    env["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.28"
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
            "deadline": time.time() + JOB_TIMEOUT_S[name],
        })
        log(f"launched {name} pid={process.pid}")
        time.sleep(20)

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

    (output_root / "WAVE1_DONE").write_text(datetime.datetime.now().isoformat())
    log("WAVE1 ALL DONE")


if __name__ == "__main__":
    main()
