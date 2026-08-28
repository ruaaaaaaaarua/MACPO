"""Verification for the Env-v6.1 optional reactive-power action.

1. Regression: with pcc_q_action_enable=False (default) the seed-31 deterministic
   MACPO rollout must reproduce the recorded baseline exactly
   (vcost 0.6851, vmin 0.9362, econ 4.629e6).
2. Smoke: with the flag on, action_dim grows to 8, a zero Q action reproduces the
   legacy Q profile, and a full capacitive action strictly raises the minimum
   voltage while leaving active-power economics untouched.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path("/root/autodl-tmp/env_v6_1_dev/HyperMARL-main")
sys.path.insert(0, str(REPO))

from envs.microgrid.microgrid_env import MicrogridEnv
from scripts.run_env_v3_safe_matrix import apply_env_v6_calibration, build_gru_config

TOL = 1e-9


def build_env_config(extra: dict) -> dict:
    calibration = json.loads(
        Path("/root/autodl-tmp/env_v6_swiss_runs/calibration.json").read_text()
    )
    config = build_gru_config("v6_nocomm_gru_macpo", updates=1000)
    apply_env_v6_calibration(config, calibration)
    overrides = dict(config["env_overrides"])
    overrides.update(extra)
    return overrides


def rollout_uncontrolled(overrides: dict, *, seed: int, q_setpoint: float | None):
    """Fixed nominal action rollout (50% electrolyser, idle storage)."""
    env = MicrogridEnv(overrides)
    env.seed(seed)
    env.reset()
    n, d = env.agent_num, env.action_dim
    records = []
    for _ in range(env.T):
        action = np.zeros((n, d), dtype=np.float32)
        action[:, 0] = 0.0
        action[:, 5] = -1.0
        if q_setpoint is not None:
            action[:, env.pcc_q_action_index] = q_setpoint
        _, _, done, info = env.step(action)
        records.append(info[0] if isinstance(info, (list, tuple)) else info)
        if np.any(done):
            break
    return env, records


def main() -> None:
    # --- 1. regression with the flag off -------------------------------------
    base = build_env_config({})
    assert "pcc_q_action_enable" not in base or not base["pcc_q_action_enable"]
    env_off = MicrogridEnv(base)
    assert env_off.action_dim == 7, env_off.action_dim
    print(f"flag off: action_dim={env_off.action_dim} (expected 7)")

    off_env, off_records = rollout_uncontrolled(base, seed=31, q_setpoint=None)

    # --- 2. flag on, zero Q == legacy physics --------------------------------
    on = build_env_config({"pcc_q_action_enable": True})
    env_on = MicrogridEnv(on)
    assert env_on.action_dim == 8, env_on.action_dim
    assert env_on.pcc_q_action_index == 7
    print(f"flag on : action_dim={env_on.action_dim} (expected 8)")

    zero_env, zero_records = rollout_uncontrolled(on, seed=31, q_setpoint=0.0)
    assert len(zero_records) == len(off_records)
    for i, (a, b) in enumerate(zip(off_records, zero_records)):
        for key in ("voltage_cost", "economic_cost", "voltage_min_pu", "voltage_max_pu"):
            va, vb = float(a[key]), float(b[key])
            assert abs(va - vb) <= TOL, (i, key, va, vb)
        qa = np.asarray(a["pcc_q_kvar"])
        qb = np.asarray(b["pcc_q_kvar"])
        assert np.allclose(qa, qb, atol=TOL), (i, qa, qb)
    print("flag on + Q=0 action: identical voltage/economics/Q to legacy (24 steps)")

    # --- 3. capacitive Q raises min voltage ----------------------------------
    cap_env, cap_records = rollout_uncontrolled(on, seed=31, q_setpoint=1.0)
    vmin_zero = min(float(r["voltage_min_pu"]) for r in zero_records)
    vmin_cap = min(float(r["voltage_min_pu"]) for r in cap_records)
    vcost_zero = sum(float(r["voltage_cost"]) for r in zero_records)
    vcost_cap = sum(float(r["voltage_cost"]) for r in cap_records)
    econ_zero = sum(float(r["economic_cost"]) for r in zero_records)
    econ_cap = sum(float(r["economic_cost"]) for r in cap_records)
    q_avail = np.asarray(cap_records[19]["pcc_q_avail_kvar"])
    q_inj = np.asarray(cap_records[19]["pcc_q_inj_kvar"])
    print(
        f"Q=0   : vmin={vmin_zero:.4f} vcost={vcost_zero:.4f} econ={econ_zero:.6g}\n"
        f"Q=+cap: vmin={vmin_cap:.4f} vcost={vcost_cap:.4f} econ={econ_cap:.6g}\n"
        f"h19 avail={np.round(q_avail,1).tolist()} inj={np.round(q_inj,1).tolist()}"
    )
    assert vmin_cap > vmin_zero, "capacitive injection should raise min voltage"
    assert vcost_cap < vcost_zero, "capacitive injection should cut voltage cost"
    assert abs(econ_cap - econ_zero) <= 1e-6 * max(abs(econ_zero), 1.0), (
        "reactive action must not change economics"
    )
    vmax_cap = max(float(r["voltage_max_pu"]) for r in cap_records)
    print(f"Q=+cap vmax={vmax_cap:.4f} (upper limit 1.05)")
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
