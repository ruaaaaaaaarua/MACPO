"""Gate Env-v6 fused JIT + process rollout performance against the legacy path."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from baselines.MAPPO.safe_gru_trainer import SafeGRUMAPPOTrainer, SafeRollout
from baselines.utils.microgrid_vec_env import MicrogridVecEnv
from scripts.run_env_v3_safe_matrix import apply_env_v6_calibration, build_gru_config


MINIMUM_SPEEDUP = 1.25
KERNEL_PARITY_TOLERANCE = 5e-4
TRAJECTORY_PARITY_TOLERANCE = 1e-6


def assess_performance_gate(
    *,
    legacy_seconds: float,
    fused_process_seconds: float,
    semantic_parity: bool,
) -> dict[str, Any]:
    if legacy_seconds <= 0.0 or fused_process_seconds <= 0.0:
        raise ValueError("benchmark durations must be positive")
    speedup = float(legacy_seconds) / float(fused_process_seconds)
    return {
        "passed": bool(semantic_parity and speedup + 1e-12 >= MINIMUM_SPEEDUP),
        "semantic_parity": bool(semantic_parity),
        "speedup": speedup,
        "minimum_speedup": MINIMUM_SPEEDUP,
        "legacy_seconds": float(legacy_seconds),
        "fused_process_seconds": float(fused_process_seconds),
    }


def rollout_difference(first: SafeRollout, second: SafeRollout) -> dict[str, Any]:
    fields = (
        "local_obs",
        "global_obs",
        "dones_before",
        "dones",
        "actions",
        "log_probs",
        "rewards",
        "costs",
        "raw_costs",
        "reward_values",
        "cost_values",
        "intents",
    )
    differences = {}
    for field in fields:
        left = np.asarray(getattr(first, field))
        right = np.asarray(getattr(second, field))
        if not left.size:
            differences[field] = 0.0
        elif left.dtype == np.bool_ or right.dtype == np.bool_:
            differences[field] = float(np.max(np.not_equal(left, right)))
        else:
            differences[field] = float(np.max(np.abs(left - right)))
    maximum = max(differences.values(), default=0.0)
    return {
        "fields": differences,
        "max_abs_difference": maximum,
        "passed": bool(maximum <= 1e-6),
    }


def _timed_rollouts(trainer: SafeGRUMAPPOTrainer, count: int) -> float:
    start = time.perf_counter()
    for index in range(int(count)):
        trainer.collect_rollout(update_index=index + 2)
    return time.perf_counter() - start


def process_serial_trajectory_difference(
    config: dict[str, Any], *, seed: int = 30
) -> dict[str, Any]:
    """Compare serial/process physics under the exact same 24h action sequence."""
    serial = MicrogridVecEnv(
        num_envs=2,
        auto_reset=False,
        config_overrides=config["env_overrides"],
        parallel_backend="serial",
    )
    process = MicrogridVecEnv(
        num_envs=2,
        auto_reset=False,
        config_overrides=config["env_overrides"],
        parallel_backend="process",
    )
    differences: dict[str, float] = {}

    def record(name: str, left: Any, right: Any) -> None:
        first = np.asarray(left)
        second = np.asarray(right)
        if first.dtype == np.bool_ or second.dtype == np.bool_:
            value = float(np.max(np.not_equal(first, second))) if first.size else 0.0
        else:
            value = float(np.max(np.abs(first - second))) if first.size else 0.0
        differences[name] = max(differences.get(name, 0.0), value)

    try:
        serial_obs, _ = serial.reset(seed=int(seed))
        process_obs, _ = process.reset(seed=int(seed))
        record("reset_obs", serial_obs, process_obs)
        action = np.zeros(
            (2 * serial.num_agents, serial.action_dim), dtype=np.float32
        )
        action[:, 5] = -1.0
        for _ in range(24):
            serial_step = serial.step(action)
            process_step = process.step(action)
            for name, left, right in zip(
                ("obs", "rewards", "terminations", "truncations"),
                serial_step[:4],
                process_step[:4],
            ):
                record(name, left, right)
            for env_index in range(2):
                offset = env_index * serial.num_agents
                left_info = serial_step[4][offset]
                right_info = process_step[4][offset]
                for field in (
                    "voltage_cost", "voltage_min_pu", "voltage_max_pu",
                    "voltages_pu", "pcc_p_kw", "pcc_q_kvar", "pf_converged",
                ):
                    record(f"info.{field}", left_info[field], right_info[field])
    finally:
        serial.close()
        process.close()
    maximum = max(differences.values(), default=0.0)
    return {
        "fields": differences,
        "max_abs_difference": maximum,
        "passed": bool(maximum <= TRAJECTORY_PARITY_TOLERANCE),
        "action_policy": "nominal_uncontrolled_shared_actions",
        "seeds": [int(seed), int(seed) + 1],
    }


def benchmark(calibration: dict[str, Any], *, rollout_count: int = 5) -> dict[str, Any]:
    if rollout_count < 1:
        raise ValueError("rollout_count must be positive")
    legacy_config = build_gru_config("v6_nocomm_gru_mappo", updates=rollout_count + 1)
    candidate_config = build_gru_config("v6_nocomm_gru_mappo", updates=rollout_count + 1)
    apply_env_v6_calibration(legacy_config, calibration)
    apply_env_v6_calibration(candidate_config, calibration)
    legacy_config.update(
        {"env_parallel_backend": "serial", "fused_rollout_kernel": False}
    )
    candidate_config.update(
        {"env_parallel_backend": "process", "fused_rollout_kernel": True}
    )
    legacy = SafeGRUMAPPOTrainer(legacy_config)
    candidate = SafeGRUMAPPOTrainer(candidate_config)
    try:
        kernel_parity = candidate.rollout_kernel_parity(update_index=1)
        trajectory_parity = process_serial_trajectory_difference(
            candidate_config, seed=int(candidate_config["seed"])
        )
        # Compile and advance both paths once before timing stable throughput.
        legacy.collect_rollout(update_index=1)
        candidate.collect_rollout(update_index=1)
        legacy_seconds = _timed_rollouts(legacy, rollout_count)
        candidate_seconds = _timed_rollouts(candidate, rollout_count)
    finally:
        legacy.close()
        candidate.close()
    semantic_parity = bool(
        float(kernel_parity["max_abs_difference"]) <= KERNEL_PARITY_TOLERANCE
        and trajectory_parity["passed"]
    )
    gate = assess_performance_gate(
        legacy_seconds=legacy_seconds,
        fused_process_seconds=candidate_seconds,
        semantic_parity=semantic_parity,
    )
    gate.update(
        {
            "rollout_count": int(rollout_count),
            "num_envs": 2,
            "num_steps": 24,
            "system_transitions_per_rollout": 48,
            "kernel_parity": kernel_parity,
            "kernel_parity_tolerance": KERNEL_PARITY_TOLERANCE,
            "trajectory_parity": trajectory_parity,
            "trajectory_parity_tolerance": TRAJECTORY_PARITY_TOLERANCE,
            "legacy_transitions_per_second": (
                rollout_count * 48 / legacy_seconds
            ),
            "fused_process_transitions_per_second": (
                rollout_count * 48 / candidate_seconds
            ),
        }
    )
    return gate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rollouts", type=int, default=5)
    args = parser.parse_args()
    calibration = json.loads(args.calibration.read_text(encoding="utf-8"))
    report = benchmark(calibration, rollout_count=int(args.rollouts))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
