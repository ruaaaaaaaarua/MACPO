#!/usr/bin/env python3
"""Build and run the isolated dynamic-H2-traffic smoke experiment."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from run_fair_stas_pipeline import (
        ROOT,
        RunSpec,
        TRAIN_SEED,
        VALIDATION_NOISE_SEED,
        TEST_NOISE_SEED,
        _ensure_fresh_output_root,
        _fair_group_abc_overrides,
        _git_metadata,
        _matd3_command,
        _ppo_command,
        _run,
        _write_json,
    )
except ModuleNotFoundError:
    from scripts.run_fair_stas_pipeline import (
        ROOT,
        RunSpec,
        TRAIN_SEED,
        VALIDATION_NOISE_SEED,
        TEST_NOISE_SEED,
        _ensure_fresh_output_root,
        _fair_group_abc_overrides,
        _git_metadata,
        _matd3_command,
        _ppo_command,
        _run,
        _write_json,
    )

from envs.microgrid.microgrid_env import MicrogridEnv


DEFAULT_OUTPUT = ROOT.parents[1] / "traffic-stas-20260716"


def traffic_group_abc_overrides() -> dict[str, Any]:
    overrides = _fair_group_abc_overrides()
    overrides.update(
        {
            "h2_traffic_enable": True,
            "h2_route_action_enable": True,
            "h2_traffic_min_eta": 4,
            "h2_traffic_max_eta": 6,
            "h2_traffic_truck_capacity_kg": 500.0,
            "h2_traffic_edge_capacity": 8.0,
            "h2_traffic_bpr_alpha": 0.15,
            "h2_traffic_bpr_beta": 4.0,
            "h2_traffic_background_base_min": 0.25,
            "h2_traffic_background_base_max": 0.45,
            "h2_traffic_morning_peak_amplitude": 1.0,
            "h2_traffic_evening_peak_amplitude": 1.1,
            "h2_traffic_peak_width_hours": 2.0,
            "h2_traffic_directional_phase_hours": 4.0,
            "h2_traffic_seed": 20260716,
            "h2_pending_obs_horizon": 6,
            "h2_pending_summary_obs_enable": True,
            "h2_delivery_reservation_horizon": 6,
            "h2_buyer_reservation_demand_enable": False,
        }
    )
    return overrides


def build_traffic_specs(root: Path, episodes: int) -> dict[str, RunSpec]:
    overrides = traffic_group_abc_overrides()

    def out(name: str) -> Path:
        return Path(root) / name / "output"

    return {
        "mappo_256": _ppo_command(
            name="mappo_256",
            algorithm="Traffic-MAPPO-256",
            output_dir=out("mappo_256"),
            episodes=episodes,
            stable=True,
            width=256,
            activation="relu",
            env_overrides=overrides,
        ),
        "matd3_256": _matd3_command(
            out("matd3_256"), episodes, env_overrides=overrides
        ),
        "stas_mix010": _ppo_command(
            name="stas_mix010",
            algorithm="Traffic-STAS-Mix010",
            output_dir=out("stas_mix010"),
            episodes=episodes,
            stable=True,
            width=256,
            activation="relu",
            stas=True,
            env_overrides=overrides,
            stas_max_mix_coef=0.1,
        ),
        "stas_mix020": _ppo_command(
            name="stas_mix020",
            algorithm="Traffic-STAS-Mix020",
            output_dir=out("stas_mix020"),
            episodes=episodes,
            stable=True,
            width=256,
            activation="relu",
            stas=True,
            env_overrides=overrides,
            stas_max_mix_coef=0.2,
        ),
    }


def write_traffic_manifest(root: Path, episodes: int, specs: dict[str, RunSpec]):
    overrides = traffic_group_abc_overrides()
    env = MicrogridEnv(overrides)
    try:
        dimensions = {
            "agents": int(env.agent_num),
            "episode_steps": int(env.T),
            "obs_dim": int(env.obs_dim),
            "action_dim": int(env.action_dim),
        }
    finally:
        env.close()
    try:
        branch, commit = _git_metadata()
    except Exception:
        branch, commit = "unavailable", "unavailable"
    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "branch": branch,
        "commit": commit,
        "output_root": str(Path(root)),
        "episodes": int(episodes),
        "seeds": {
            "training": TRAIN_SEED,
            "validation": VALIDATION_NOISE_SEED,
            "test": TEST_NOISE_SEED,
        },
        "dimensions": dimensions,
        "traffic": {
            "topology": "four_node_complete_directed",
            "edge_count": 12,
            "route_count_per_od": 3,
            "eta_range": [4, 6],
            "random_incidents": False,
            "transport_cost_in_reward": False,
        },
        "stas_mix_candidates": [0.1, 0.2],
        "abc_overrides": overrides,
        "runs": {
            name: {
                "command": spec.command,
                "output_dir": str(spec.output_dir),
                "eval_jsonl": str(spec.eval_jsonl),
                "checkpoints": [str(path) for path in spec.checkpoints],
            }
            for name, spec in specs.items()
        },
    }
    _write_json(Path(root) / "manifest.json", payload)
    return payload


def run_smoke(root: Path, episodes: int = 100):
    _ensure_fresh_output_root(root)
    specs = build_traffic_specs(root, episodes)
    write_traffic_manifest(root, episodes, specs)
    results = []
    for name, spec in specs.items():
        print(f"[traffic smoke launch] {name}", flush=True)
        result = _run(spec, Path(root) / "logs" / f"{name}.log")
        results.append(result)
        print(f"[traffic smoke {result['status']}] {name}: {result['errors']}", flush=True)
        if result["status"] != "success":
            raise RuntimeError(f"traffic smoke failed: {name}: {result['errors']}")
    _write_json(Path(root) / "gate_results.json", results)
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--spec-only", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    if args.spec_only:
        _ensure_fresh_output_root(root)
        specs = build_traffic_specs(root, args.episodes)
        write_traffic_manifest(root, args.episodes, specs)
        return
    run_smoke(root, episodes=args.episodes)


if __name__ == "__main__":
    main()
