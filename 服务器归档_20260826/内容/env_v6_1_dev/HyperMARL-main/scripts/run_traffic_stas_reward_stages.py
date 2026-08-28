#!/usr/bin/env python3
"""Run the three lightweight STAS traffic reward stages."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from run_fair_stas_pipeline import (
        ROOT,
        TEST_NOISE_SEED,
        TRAIN_SEED,
        VALIDATION_NOISE_SEED,
        RunSpec,
        _ensure_fresh_output_root,
        _git_metadata,
        _ppo_command,
        _run,
        _write_json,
    )
    from run_traffic_experiment import traffic_group_abc_overrides
except ModuleNotFoundError:
    from scripts.run_fair_stas_pipeline import (
        ROOT,
        TEST_NOISE_SEED,
        TRAIN_SEED,
        VALIDATION_NOISE_SEED,
        RunSpec,
        _ensure_fresh_output_root,
        _git_metadata,
        _ppo_command,
        _run,
        _write_json,
    )
    from scripts.run_traffic_experiment import traffic_group_abc_overrides


DEFAULT_OUTPUT = ROOT.parents[1] / "traffic-stas-reward-stages-20260716"
LHV_H2 = 33.33
STAGES = (
    "stage60_no_terminal",
    "stage45_no_terminal",
    "stage45_terminal20",
)


def stage_overrides(name: str) -> dict[str, Any]:
    if name not in STAGES:
        raise ValueError(f"unknown reward stage: {name}")
    overrides = traffic_group_abc_overrides()
    overrides.update(
        {
            "terminal_h2_shortfall_value_enable": False,
            "terminal_h2_shortfall_value_targets": [0.20] * 4,
            "terminal_h2_shortfall_value_coef": 0.0,
            "terminal_h2_shortfall_value_agent_indices": [0, 1, 2, 3],
            "terminal_h2_settlement_in_reward_enable": False,
        }
    )
    if name != "stage60_no_terminal":
        overrides["external_h2_dependency_penalty_enable"] = False
    if name == "stage45_terminal20":
        overrides["terminal_h2_shortfall_value_enable"] = True
        overrides["terminal_h2_settlement_in_reward_enable"] = True
    return overrides


def build_specs(root: Path, episodes: int) -> dict[str, RunSpec]:
    specs = {}
    for name in STAGES:
        output = Path(root) / name / "output"
        specs[name] = _ppo_command(
            name=name,
            algorithm=f"Traffic-STAS-005-{name}",
            output_dir=output,
            episodes=episodes,
            stable=True,
            width=256,
            activation="relu",
            stas=True,
            env_overrides=stage_overrides(name),
            stas_max_mix_coef=0.05,
            stas_ramp_episodes=4000,
        )
    return specs


def _price_metadata(overrides: dict[str, Any]) -> dict[str, float]:
    nominal = float(overrides["lambda_h2_buy"]) * LHV_H2
    dependency = (
        float(overrides.get("external_h2_dependency_penalty_kg", 0.0))
        if overrides.get("external_h2_dependency_penalty_enable", False)
        else 0.0
    )
    return {
        "nominal_external_buy_yuan_per_kg": nominal,
        "dependency_penalty_yuan_per_kg": dependency,
        "effective_external_cost_yuan_per_kg": nominal + dependency,
    }


def write_manifest(root: Path, episodes: int, specs: dict[str, RunSpec]) -> dict:
    branch, commit = _git_metadata()
    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "branch": branch,
        "commit": commit,
        "exploratory_single_seed": True,
        "episodes": int(episodes),
        "seeds": {
            "training": TRAIN_SEED,
            "validation": VALIDATION_NOISE_SEED,
            "test": TEST_NOISE_SEED,
        },
        "stas": {
            "warmup_episodes": 2000,
            "ramp_episodes": 4000,
            "max_mix_coef": 0.05,
        },
        "stages": {
            name: {
                "prices": _price_metadata(stage_overrides(name)),
                "terminal_h2_target_ratios": (
                    [0.20] * 4 if name == "stage45_terminal20" else None
                ),
                "terminal_settlement_in_reward": (
                    name == "stage45_terminal20"
                ),
                "command": specs[name].command,
                "output_dir": str(specs[name].output_dir),
            }
            for name in STAGES
        },
    }
    _write_json(Path(root) / "manifest.json", payload)
    return payload


def run(root: Path, episodes: int) -> list[dict]:
    _ensure_fresh_output_root(root)
    specs = build_specs(root, episodes)
    write_manifest(root, episodes, specs)
    results = []
    for name in STAGES:
        print(f"[reward stage launch] {name}", flush=True)
        result = _run(specs[name], Path(root) / "logs" / f"{name}.log")
        results.append(result)
        print(
            f"[reward stage {result['status']}] {name}: {result['errors']}",
            flush=True,
        )
        if result["status"] != "success":
            raise RuntimeError(f"reward stage failed: {name}: {result['errors']}")
    _write_json(Path(root) / "gate_results.json", results)
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--spec-only", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    if args.spec_only:
        _ensure_fresh_output_root(root)
        specs = build_specs(root, args.episodes)
        write_manifest(root, args.episodes, specs)
        return
    run(root, args.episodes)


if __name__ == "__main__":
    main()
