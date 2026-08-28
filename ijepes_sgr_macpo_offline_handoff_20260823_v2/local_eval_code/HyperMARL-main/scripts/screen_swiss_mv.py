"""Deterministically screen Swiss-PDGs MV cases for Env-v6 PCC placement."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import heapq
import json
import os
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from pypower.idx_brch import BR_R, BR_STATUS, BR_X, F_BUS, T_BUS
from pypower.idx_bus import BUS_I, BUS_TYPE, PD, QD, REF


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from envs.microgrid.microgrid_env import MicrogridEnv
from envs.microgrid.power_flow import SwissMVPowerFlow
from scripts.calibrate_env_v5_feasibility import (
    NOMINAL,
    REFERENCE,
    nominal_uncontrolled_action,
    voltage_support_reference_action,
)
from scripts.env_v3_safe_overrides import env_v3_safe_overrides


RAW_DAILY_VOLTAGE_BUDGET = 0.02
EVALUATION_SEEDS = (30, 31, 32)


def discover_case_dirs(dataset_root: str | Path) -> list[Path]:
    """Return grid directories containing exactly one complete MATPOWER CSV triplet."""
    root = Path(dataset_root)
    if not root.is_dir():
        raise ValueError(f"Swiss-PDGs MV dataset directory does not exist: {root}")
    cases: list[Path] = []
    for case_dir in sorted((path for path in root.iterdir() if path.is_dir()), key=lambda p: p.name):
        counts = [
            len(list(case_dir.glob(pattern)))
            for pattern in (
                "*_bus_data.csv",
                "*_branch_data.csv",
                "*_generator_data.csv",
            )
        ]
        if counts == [1, 1, 1]:
            cases.append(case_dir)
    return cases


def _lookup_distance(
    distances: Mapping[tuple[int, int], float], first: int, second: int
) -> float:
    if (int(first), int(second)) in distances:
        return float(distances[(int(first), int(second))])
    return float(distances[(int(second), int(first))])


def _farthest_four(
    bus_ids: Sequence[int],
    sensitivity: Mapping[int, float],
    native_pd: Mapping[int, float],
    distances: Mapping[tuple[int, int], float],
) -> tuple[int, int, int, int] | None:
    ordered = sorted(int(bus_id) for bus_id in bus_ids)
    if len(ordered) < 4:
        return None
    first = min(ordered, key=lambda bus: (-float(native_pd[bus]), bus))
    selected = [first]
    remaining = set(ordered)
    remaining.remove(first)
    while len(selected) < 4:
        chosen = min(
            remaining,
            key=lambda bus: (
                -min(_lookup_distance(distances, bus, other) for other in selected),
                float(sensitivity[bus]),
                bus,
            ),
        )
        selected.append(chosen)
        remaining.remove(chosen)
    return tuple(selected)  # type: ignore[return-value]


def select_pcc_sets(
    eligible_bus_ids: Iterable[int],
    sensitivity: Mapping[int, float],
    native_pd: Mapping[int, float],
    distances: Mapping[tuple[int, int], float],
) -> list[tuple[int, int, int, int]]:
    """Create up to three deterministic sensitivity-band PCC sets."""
    ordered = sorted(
        {int(bus_id) for bus_id in eligible_bus_ids},
        key=lambda bus: (float(sensitivity[bus]), bus),
    )
    sets: list[tuple[int, int, int, int]] = []
    band_count = min(3, len(ordered) // 4)
    if band_count < 1:
        return []
    for band in np.array_split(np.asarray(ordered, dtype=np.int64), band_count):
        selected = _farthest_four(
            [int(bus_id) for bus_id in band], sensitivity, native_pd, distances
        )
        if selected is not None:
            sets.append(selected)
    return sets


def assign_agents_to_pcc(
    *,
    pcc_bus_ids: Sequence[int],
    sensitivity: Mapping[int, float],
    agent_import_p95: Sequence[float],
) -> tuple[int, int, int, int]:
    """Pair the largest-import agent with the electrically strongest PCC."""
    if len(pcc_bus_ids) != 4 or len(agent_import_p95) != 4:
        raise ValueError("Env-v6 requires four PCC buses and four agent import values")
    agents = sorted(
        range(4), key=lambda agent: (-float(agent_import_p95[agent]), agent)
    )
    buses = sorted(
        (int(bus_id) for bus_id in pcc_bus_ids),
        key=lambda bus: (float(sensitivity[bus]), bus),
    )
    assigned = [0, 0, 0, 0]
    for agent, bus in zip(agents, buses):
        assigned[agent] = bus
    return tuple(assigned)  # type: ignore[return-value]


def _complete_day(day: Mapping[str, Any]) -> bool:
    return bool(day.get("pf_converged", False)) and int(day.get("steps", 0)) == 24


def dynamic_gate(
    nominal_days: Sequence[Mapping[str, Any]],
    reference_days: Sequence[Mapping[str, Any]],
    *,
    budget: float = RAW_DAILY_VOLTAGE_BUDGET,
) -> dict[str, Any]:
    """Evaluate the fixed Env-v6 raw-voltage physical gate."""
    nominal_risk_days = sum(
        _complete_day(day) and float(day.get("daily_voltage_cost", 0.0)) > float(budget)
        for day in nominal_days
    )
    reference_safe_days = sum(
        _complete_day(day) and float(day.get("daily_voltage_cost", np.inf)) <= float(budget)
        for day in reference_days
    )
    all_converged = (
        len(nominal_days) == len(EVALUATION_SEEDS)
        and len(reference_days) == len(EVALUATION_SEEDS)
        and all(_complete_day(day) for day in [*nominal_days, *reference_days])
    )
    return {
        "passed": bool(
            all_converged
            and nominal_risk_days >= 2
            and reference_safe_days == len(EVALUATION_SEEDS)
        ),
        "nominal_risk_days": int(nominal_risk_days),
        "reference_safe_days": int(reference_safe_days),
        "all_power_flows_converged": bool(all_converged),
        "raw_daily_voltage_budget": float(budget),
    }


def rank_static_candidates(
    candidates: Sequence[Mapping[str, Any]], *, limit: int = 30
) -> list[dict[str, Any]]:
    """Rank the deterministic static shortlist without mutating input rows."""
    ranked = sorted(
        (dict(candidate) for candidate in candidates),
        key=lambda row: (
            -float(row["proxy_gap"]),
            -float(row["native_total_mw"]),
            str(row["grid_id"]),
            tuple(int(bus) for bus in row["pcc_bus_ids"]),
        ),
    )
    return ranked[: max(0, int(limit))]


def select_passing_candidate(
    candidates: Sequence[Mapping[str, Any]],
) -> dict[str, Any] | None:
    """Select the passing dynamic candidate with the largest controllability gap."""
    passing: list[dict[str, Any]] = []
    for candidate in candidates:
        row = dict(candidate)
        gate = dynamic_gate(row.get("nominal_days", []), row.get("reference_days", []))
        if not gate["passed"]:
            continue
        nominal_total = sum(
            float(day["daily_voltage_cost"]) for day in row["nominal_days"]
        )
        reference_total = sum(
            float(day["daily_voltage_cost"]) for day in row["reference_days"]
        )
        row["gate"] = gate
        row["controllability_gap"] = float(nominal_total - reference_total)
        passing.append(row)
    if not passing:
        return None
    return min(
        passing,
        key=lambda row: (
            -float(row["controllability_gap"]),
            -float(row["native_total_mw"]),
            str(row["grid_id"]),
            tuple(int(bus) for bus in row["pcc_bus_ids"]),
        ),
    )


def _case_grid_id(case_dir: Path) -> str:
    return case_dir.name


def _provisional_pcc_bus_ids(case_dir: Path) -> list[int]:
    paths = list(case_dir.glob("*_bus_data.csv"))
    if len(paths) != 1:
        raise ValueError(f"invalid Swiss MV bus table set: {case_dir}")
    bus = np.genfromtxt(paths[0], delimiter=",", skip_header=1, ndmin=2)
    eligible = [
        int(row[BUS_I])
        for row in bus
        if int(row[BUS_TYPE]) != REF and float(row[PD]) > 0.0
    ]
    if len(eligible) < 4:
        raise ValueError("Swiss MV case has fewer than four non-slack load buses")
    return eligible[:4]


def _flow_for_case(case_dir: Path, pcc_bus_ids: Sequence[int] | None = None) -> SwissMVPowerFlow:
    return SwissMVPowerFlow(
        {
            "power_flow_model": "swiss_mv",
            "power_flow_case_dir": str(case_dir),
            "power_flow_pcc_bus_ids": list(
                _provisional_pcc_bus_ids(case_dir)
                if pcc_bus_ids is None
                else pcc_bus_ids
            ),
            "power_flow_background_load_scale": 1.0,
            "power_flow_pcc_injection_scale": 1.0,
            "power_flow_vmin_pu": 0.95,
            "power_flow_vmax_pu": 1.05,
            "power_flow_failure_cost": 1.0,
        }
    )


def electrical_distances(flow: SwissMVPowerFlow) -> dict[tuple[int, int], float]:
    """Return all-pairs shortest electrical distances for connected native buses."""
    bus_ids = [int(value) for value in flow._case["bus"][:, BUS_I]]
    graph: dict[int, list[tuple[int, float]]] = {bus_id: [] for bus_id in bus_ids}
    for row in flow._case["branch"]:
        if int(row[BR_STATUS]) == 0:
            continue
        first, second = int(row[F_BUS]), int(row[T_BUS])
        weight = max(1e-12, abs(float(row[BR_R])) + abs(float(row[BR_X])))
        graph[first].append((second, weight))
        graph[second].append((first, weight))
    distances: dict[tuple[int, int], float] = {}
    for source in bus_ids:
        best = {source: 0.0}
        queue = [(0.0, source)]
        while queue:
            distance, bus = heapq.heappop(queue)
            if distance > best[bus]:
                continue
            for neighbor, weight in graph[bus]:
                candidate = distance + weight
                if candidate < best.get(neighbor, np.inf):
                    best[neighbor] = candidate
                    heapq.heappush(queue, (candidate, neighbor))
        for target, distance in best.items():
            distances[(source, target)] = float(distance)
    return distances


def voltage_sensitivities(
    flow: SwissMVPowerFlow, eligible_bus_ids: Sequence[int]
) -> tuple[dict[int, float], dict[str, Any]]:
    """Measure native voltage response to a 1 MW, 0.95-pf load at each bus."""
    native = flow.solve_bus_injections([], [], [])
    if not native["pf_converged"]:
        return {}, native
    native_voltages = np.asarray(native["voltages_pu"], dtype=np.float64)
    q_kvar = float(1000.0 * np.tan(np.arccos(0.95)))
    sensitivity: dict[int, float] = {}
    for bus_id in eligible_bus_ids:
        loaded = flow.solve_bus_injections([bus_id], [1000.0], [q_kvar])
        if not loaded["pf_converged"]:
            sensitivity[int(bus_id)] = float("inf")
            continue
        drop = np.maximum(
            0.0,
            native_voltages - np.asarray(loaded["voltages_pu"], dtype=np.float64),
        )
        sensitivity[int(bus_id)] = float(np.sum(drop))
    return sensitivity, native


def _fixed_support_action(agent_count: int, action_dim: int) -> np.ndarray:
    action = nominal_uncontrolled_action(agent_count, action_dim)
    action[:, 0] = -1.0
    action[:, 1] = 1.0
    return action


def collect_microgrid_proxy_profiles() -> dict[str, Any]:
    """Collect grid-independent nominal/support PCC traces for static screening."""
    policies = {
        NOMINAL: nominal_uncontrolled_action,
        "fixed_voltage_support": lambda agents, dim: _fixed_support_action(agents, dim),
    }
    traces: dict[str, list[dict[str, Any]]] = {name: [] for name in policies}
    for policy_name, action_builder in policies.items():
        for seed in EVALUATION_SEEDS:
            overrides = env_v3_safe_overrides()
            overrides.update(
                {
                    "soc_init": 0.5,
                    "power_flow_enable": False,
                    "power_flow_background_load_scale": 1.0,
                    "power_flow_pcc_injection_scale": 1.0,
                }
            )
            env = MicrogridEnv(overrides)
            env.seed(int(seed))
            env.reset()
            p_trace: list[list[float]] = []
            q_trace: list[list[float]] = []
            final_economic_cost = 0.0
            for _ in range(int(env.T)):
                action = action_builder(env.agent_num, env.action_dim)
                _, _, done, infos = env.step(action)
                info = infos[0]
                p_trace.append(list(info["pcc_p_kw"]))
                q_trace.append(list(info["pcc_q_kvar"]))
                final_economic_cost = float(info["episode_total_cost"])
                if bool(np.any(done)):
                    break
            traces[policy_name].append(
                {
                    "seed": int(seed),
                    "pcc_p_kw": p_trace,
                    "pcc_q_kvar": q_trace,
                    "economic_cost": final_economic_cost,
                }
            )
    nominal_p = np.concatenate(
        [np.asarray(day["pcc_p_kw"], dtype=np.float64) for day in traces[NOMINAL]],
        axis=0,
    )
    return {
        "traces": traces,
        "agent_import_p95_kw": np.percentile(np.maximum(nominal_p, 0.0), 95, axis=0).tolist(),
    }


def _profile_voltage_cost(
    flow: SwissMVPowerFlow,
    pcc_bus_ids: Sequence[int],
    days: Sequence[Mapping[str, Any]],
) -> tuple[float, bool]:
    total = 0.0
    for day in days:
        for p_kw, q_kvar in zip(day["pcc_p_kw"], day["pcc_q_kvar"]):
            result = flow.solve_bus_injections(pcc_bus_ids, p_kw, q_kvar)
            if not result["pf_converged"]:
                return float("inf"), False
            total += float(result["voltage_cost"])
    return float(total), True


def _static_screen_case(task: tuple[str, dict[str, Any]]) -> list[dict[str, Any]]:
    case_dir_text, profiles = task
    case_dir = Path(case_dir_text)
    try:
        flow = _flow_for_case(case_dir)
        bus = flow._case["bus"]
        eligible = [
            int(row[BUS_I])
            for row in bus
            if int(row[BUS_TYPE]) != REF and float(row[PD]) > 0.0
        ]
        if len(eligible) < 4:
            return []
        sensitivity, native = voltage_sensitivities(flow, eligible)
        if (
            not native.get("pf_converged", False)
            or float(native["voltage_cost"]) > 1e-12
            or any(not np.isfinite(value) for value in sensitivity.values())
        ):
            return []
        native_p = float(np.sum(bus[:, PD]))
        native_q = float(np.sum(bus[:, QD]))
        nominal_days = profiles["traces"][NOMINAL]
        max_transformer_mva = max(
            float(
                np.hypot(
                    native_p + np.sum(np.asarray(p_kw, dtype=np.float64)) / 1000.0,
                    native_q + np.sum(np.asarray(q_kvar, dtype=np.float64)) / 1000.0,
                )
            )
            for day in nominal_days
            for p_kw, q_kvar in zip(day["pcc_p_kw"], day["pcc_q_kvar"])
        )
        if max_transformer_mva > 25.0:
            return []
        distances = electrical_distances(flow)
        if any((first, second) not in distances for first in eligible for second in eligible):
            return []
        native_pd = {int(row[BUS_I]): float(row[PD]) for row in bus}
        rows: list[dict[str, Any]] = []
        for raw_set in select_pcc_sets(eligible, sensitivity, native_pd, distances):
            assigned = assign_agents_to_pcc(
                pcc_bus_ids=raw_set,
                sensitivity=sensitivity,
                agent_import_p95=profiles["agent_import_p95_kw"],
            )
            nominal_cost, nominal_converged = _profile_voltage_cost(
                flow, assigned, nominal_days
            )
            support_cost, support_converged = _profile_voltage_cost(
                flow, assigned, profiles["traces"]["fixed_voltage_support"]
            )
            if not nominal_converged or not support_converged:
                continue
            rows.append(
                {
                    "grid_id": _case_grid_id(case_dir),
                    "case_dir": str(case_dir.resolve()),
                    "pcc_bus_ids": list(assigned),
                    "native_total_mw": native_p,
                    "native_total_mvar": native_q,
                    "max_transformer_mva": max_transformer_mva,
                    "base_voltage_min_pu": float(native["voltage_min_pu"]),
                    "base_voltage_max_pu": float(native["voltage_max_pu"]),
                    "proxy_nominal_cost": float(nominal_cost),
                    "proxy_support_cost": float(support_cost),
                    "proxy_gap": float(nominal_cost - support_cost),
                    "pcc_sensitivity": [float(sensitivity[bus]) for bus in assigned],
                }
            )
        return rows
    except (ArithmeticError, OSError, ValueError):
        return []


def static_screen(
    dataset_root: Path,
    profiles: dict[str, Any],
    *,
    workers: int,
    shortlist_limit: int = 30,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    cases = discover_case_dirs(dataset_root)
    tasks = [(str(case_dir), profiles) for case_dir in cases]
    if workers <= 1:
        evaluated = map(_static_screen_case, tasks)
    else:
        pool = ProcessPoolExecutor(max_workers=int(workers))
        evaluated = pool.map(_static_screen_case, tasks, chunksize=1)
    candidates: list[dict[str, Any]] = []
    try:
        for rows in evaluated:
            candidates.extend(rows)
    finally:
        if workers > 1:
            pool.shutdown(wait=True)
    shortlist = rank_static_candidates(candidates, limit=shortlist_limit)
    return shortlist, {
        "dataset_case_count": len(cases),
        "static_candidate_count": len(candidates),
        "shortlist_count": len(shortlist),
    }


def swiss_env_overrides(case_dir: str, pcc_bus_ids: Sequence[int]) -> dict[str, Any]:
    overrides = env_v3_safe_overrides()
    overrides.update(
        {
            "soc_init": 0.5,
            "power_flow_model": "swiss_mv",
            "power_flow_case_dir": str(case_dir),
            "power_flow_pcc_bus_ids": [int(bus) for bus in pcc_bus_ids],
            "power_flow_background_load_scale": 1.0,
            "power_flow_pcc_injection_scale": 1.0,
        }
    )
    return overrides


def evaluate_dynamic_day(
    candidate: Mapping[str, Any],
    *,
    seed: int,
    policy: str,
    preview_workers: int,
) -> dict[str, Any]:
    env = MicrogridEnv(
        swiss_env_overrides(candidate["case_dir"], candidate["pcc_bus_ids"])
    )
    env.seed(int(seed))
    env.reset()
    costs: list[float] = []
    converged: list[bool] = []
    minimums: list[float] = []
    maximums: list[float] = []
    actions: list[list[list[float]]] = []
    final_economic_cost = 0.0
    executor = (
        ProcessPoolExecutor(max_workers=int(preview_workers))
        if policy == REFERENCE and int(preview_workers) > 1
        else None
    )
    try:
        for _ in range(int(env.T)):
            if policy == NOMINAL:
                action = nominal_uncontrolled_action(env.agent_num, env.action_dim)
            elif policy == REFERENCE:
                action = voltage_support_reference_action(
                    env, preview_executor=executor
                )
            else:
                raise ValueError(f"unknown Swiss calibration policy: {policy}")
            _, _, done, infos = env.step(action)
            info = infos[0]
            costs.append(float(info["voltage_cost"]))
            converged.append(bool(info["pf_converged"]))
            if info["voltage_min_pu"] is not None:
                minimums.append(float(info["voltage_min_pu"]))
            if info["voltage_max_pu"] is not None:
                maximums.append(float(info["voltage_max_pu"]))
            actions.append(np.asarray(action, dtype=float).tolist())
            final_economic_cost = float(info["episode_total_cost"])
            if policy == REFERENCE and (
                not bool(info["pf_converged"])
                or sum(costs) > RAW_DAILY_VOLTAGE_BUDGET
            ):
                break
            if bool(np.any(done)):
                break
    finally:
        if executor is not None:
            executor.shutdown(wait=True)
    return {
        "seed": int(seed),
        "policy": str(policy),
        "daily_voltage_cost": float(sum(costs)),
        "step_voltage_cost": costs,
        "pf_converged": bool(len(costs) == 24 and all(converged)),
        "voltage_min_pu": min(minimums) if minimums else None,
        "voltage_max_pu": max(maximums) if maximums else None,
        "steps": len(costs),
        "economic_cost": final_economic_cost,
        "actions": actions,
    }


def dynamic_screen(
    shortlist: Sequence[Mapping[str, Any]], *, preview_workers: int
) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    evaluated: list[dict[str, Any]] = []
    for candidate in shortlist:
        row = dict(candidate)
        row["nominal_days"] = [
            evaluate_dynamic_day(
                candidate, seed=seed, policy=NOMINAL, preview_workers=1
            )
            for seed in EVALUATION_SEEDS
        ]
        row["reference_days"] = [
            evaluate_dynamic_day(
                candidate,
                seed=seed,
                policy=REFERENCE,
                preview_workers=preview_workers,
            )
            for seed in EVALUATION_SEEDS
        ]
        row["gate"] = dynamic_gate(row["nominal_days"], row["reference_days"])
        evaluated.append(row)
    return evaluated, select_passing_candidate(evaluated)


def calibrate_swiss_mv(
    dataset_root: Path,
    *,
    workers: int,
    preview_workers: int,
    shortlist_limit: int = 30,
) -> dict[str, Any]:
    profiles = collect_microgrid_proxy_profiles()
    shortlist, static_summary = static_screen(
        dataset_root,
        profiles,
        workers=workers,
        shortlist_limit=shortlist_limit,
    )
    evaluated, selection = dynamic_screen(
        shortlist, preview_workers=preview_workers
    )
    economic_scale = None
    if selection is not None:
        economic_scale = float(
            np.median(
                [
                    abs(float(day["economic_cost"]))
                    for day in selection["nominal_days"]
                ]
            )
        )
        economic_scale = max(1.0, economic_scale)
    return {
        "environment": "env-v6-swiss",
        "dataset": "aeonetos/Swiss-PDGs",
        "dataset_root": str(dataset_root.resolve()),
        "seeds": list(EVALUATION_SEEDS),
        "voltage_limits_pu": [0.95, 1.05],
        "raw_daily_voltage_budget": RAW_DAILY_VOLTAGE_BUDGET,
        "pcc_injection_scale": 1.0,
        "background_load_scale": 1.0,
        "base_mva": 100.0,
        "base_kv": 20.0,
        "static_summary": static_summary,
        "shortlist": shortlist,
        "dynamic_candidates": evaluated,
        "feasible": selection is not None,
        "selection": selection,
        "economic_reward_scale_yuan": economic_scale,
        "training_cost_scale": RAW_DAILY_VOLTAGE_BUDGET,
        "training_cost_budget": 1.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=min(12, os.cpu_count() or 1))
    parser.add_argument("--preview-workers", type=int, default=4)
    parser.add_argument("--shortlist-limit", type=int, default=30)
    args = parser.parse_args()
    report = calibrate_swiss_mv(
        args.dataset_root,
        workers=max(1, int(args.workers)),
        preview_workers=max(1, int(args.preview_workers)),
        shortlist_limit=max(1, int(args.shortlist_limit)),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "feasible": report["feasible"],
                "static_summary": report["static_summary"],
                "selection": (
                    {
                        key: report["selection"][key]
                        for key in (
                            "grid_id",
                            "pcc_bus_ids",
                            "controllability_gap",
                        )
                    }
                    if report["selection"] is not None
                    else None
                ),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
