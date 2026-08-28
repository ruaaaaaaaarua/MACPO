"""Calibrate the Env-v5.2 IEEE-33 feasible-but-nontrivial operating point."""

from __future__ import annotations

import argparse
import copy
from concurrent.futures import ProcessPoolExecutor
import json
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from envs.microgrid.microgrid_env import MicrogridEnv
from scripts.env_v3_safe_overrides import env_v3_safe_overrides


SEEDS = (30, 31, 32)
BACKGROUND_SCALES = tuple(round(index / 10.0, 1) for index in range(1, 11))
PCC_SCALES = tuple(round(index / 20.0, 2) for index in range(1, 21))
PCC_REFINEMENT_ORDER = (4, 12, 23, 32)
CONTROL_GRID = tuple(
    (p_el, p_bat)
    for p_el in (-1.0, 0.0, 1.0)
    for p_bat in (-1.0, -0.5, 0.0, 0.5, 1.0)
)
NOMINAL = "nominal_uncontrolled"
REFERENCE = "voltage_support_reference"
MATERIAL_RISK = 0.02
FINE_PCC_STEP = 0.005
ENVIRONMENT_VERSION = "env-v5.2-safe"
VOLTAGE_COST_DEFINITION = "sum_bus_limit_violation"
PREVIEW_WORKERS = max(1, int(os.environ.get("ENV_V52_PREVIEW_WORKERS", "4")))


def nominal_uncontrolled_action(agent_count: int, action_dim: int) -> np.ndarray:
    """50% electrolyzer, idle battery/storage, neutral bids/routes, no H2 order."""
    action = np.zeros((int(agent_count), int(action_dim)), dtype=np.float32)
    if action_dim > 5:
        action[:, 5] = -1.0
    return action


def _preview_info(env: MicrogridEnv, action: np.ndarray) -> dict[str, Any]:
    """Evaluate one step on a clone, leaving every real rollout state untouched."""
    preview = copy.deepcopy(env)
    _, _, _, infos = preview.step(np.asarray(action, dtype=np.float32))
    return infos[0]


def _preview_info_task(task: tuple[MicrogridEnv, np.ndarray]) -> dict[str, Any]:
    """Step a process-deserialized environment clone for one candidate."""
    env, action = task
    _, _, _, infos = env.step(np.asarray(action, dtype=np.float32))
    return infos[0]


def _candidate_score(
    info: dict[str, Any],
    action: np.ndarray,
    nominal: np.ndarray,
    order_index: int,
) -> tuple[float, float, int]:
    return (
        float(info["voltage_cost"]),
        float(np.sum(np.abs(np.asarray(action) - nominal))),
        int(order_index),
    )


def _best_candidate(
    env: MicrogridEnv,
    candidates: list[np.ndarray],
    nominal: np.ndarray,
    executor: ProcessPoolExecutor | None,
) -> np.ndarray:
    if executor is None:
        infos = [_preview_info(env, action) for action in candidates]
    else:
        infos = list(
            executor.map(
                _preview_info_task,
                ((env, action) for action in candidates),
            )
        )
    scored = [
        (_candidate_score(info, action, nominal, order_index), action)
        for order_index, (info, action) in enumerate(zip(infos, candidates))
    ]
    return min(scored, key=lambda item: item[0])[1].copy()


def _agent_indices_in_pcc_order(env: MicrogridEnv) -> tuple[int, ...]:
    buses = tuple(
        int(bus)
        for bus in (
            env.power_flow.agent_buses
            if env.power_flow is not None
            else env.cfg.get("elec_lmp_agent_bus_indices", PCC_REFINEMENT_ORDER)
        )
    )
    ordered = [buses.index(bus) for bus in PCC_REFINEMENT_ORDER if bus in buses]
    ordered.extend(index for index in range(env.agent_num) if index not in ordered)
    return tuple(ordered)


def voltage_support_reference_action(
    env: MicrogridEnv,
    *,
    preview_executor: ProcessPoolExecutor | None = None,
) -> np.ndarray:
    """Choose the deterministic global candidate, then one PCC-ordered sweep."""
    nominal = nominal_uncontrolled_action(env.agent_num, env.action_dim)
    global_candidates: list[np.ndarray] = []
    for order_index, (p_el, p_bat) in enumerate(CONTROL_GRID):
        candidate = nominal.copy()
        candidate[:, 0] = p_el
        candidate[:, 1] = p_bat
        global_candidates.append(candidate)
    action = _best_candidate(env, global_candidates, nominal, preview_executor)
    for agent_index in _agent_indices_in_pcc_order(env):
        local_candidates: list[np.ndarray] = []
        for p_el, p_bat in CONTROL_GRID:
            candidate = action.copy()
            candidate[agent_index, 0] = p_el
            candidate[agent_index, 1] = p_bat
            local_candidates.append(candidate)
        action = _best_candidate(env, local_candidates, nominal, preview_executor)
    return action


def _environment_overrides(background_scale: float, pcc_scale: float) -> dict[str, Any]:
    overrides = env_v3_safe_overrides()
    overrides.update(
        {
            "soc_init": 0.5,
            "power_flow_background_load_scale": float(background_scale),
            "power_flow_pcc_injection_scale": float(pcc_scale),
        }
    )
    return overrides


def evaluate_day(
    *,
    background_scale: float,
    pcc_scale: float,
    seed: int,
    policy: str,
    preview_workers: int = PREVIEW_WORKERS,
) -> dict[str, Any]:
    env = MicrogridEnv(_environment_overrides(background_scale, pcc_scale))
    env.seed(int(seed))
    env.reset()
    costs: list[float] = []
    converged: list[bool] = []
    mins: list[float] = []
    maxs: list[float] = []
    action_trace: list[list[list[float]]] = []
    executor = (
        ProcessPoolExecutor(max_workers=max(1, int(preview_workers)))
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
                raise ValueError(f"unknown calibration policy {policy!r}")
            _, _, done, infos = env.step(action)
            info = infos[0]
            costs.append(float(info["voltage_cost"]))
            converged.append(bool(info["pf_converged"]))
            if info["voltage_min_pu"] is not None:
                mins.append(float(info["voltage_min_pu"]))
            if info["voltage_max_pu"] is not None:
                maxs.append(float(info["voltage_max_pu"]))
            action_trace.append(np.asarray(action, dtype=float).tolist())
            # A single violation or PF failure irreversibly fails the zero-cost
            # reference gate, so rejected scan points need no further previews.
            if policy == REFERENCE and (
                float(info["voltage_cost"]) > 1e-12
                or not bool(info["pf_converged"])
            ):
                break
            if bool(np.any(done)):
                break
    finally:
        if executor is not None:
            executor.shutdown(wait=True)
    daily_cost = float(sum(costs))
    all_converged = len(costs) == int(env.T) and all(converged)
    safe = all_converged and daily_cost <= 1e-12
    return {
        "seed": int(seed),
        "policy": policy,
        "daily_voltage_cost": daily_cost,
        "step_voltage_cost": costs,
        "pf_converged": bool(all_converged),
        "voltage_min_pu": min(mins) if mins else None,
        "voltage_max_pu": max(maxs) if maxs else None,
        "steps": len(costs),
        "safe": bool(safe),
        "early_rejection": bool(policy == REFERENCE and len(costs) < int(env.T)),
        "actions": action_trace,
    }


def _evaluate_seed_task(
    task: tuple[float, float, int]
) -> tuple[dict[str, Any], dict[str, Any]]:
    background_scale, pcc_scale, seed = task
    nominal = evaluate_day(
        background_scale=background_scale,
        pcc_scale=pcc_scale,
        seed=seed,
        policy=NOMINAL,
        preview_workers=1,
    )
    reference = evaluate_day(
        background_scale=background_scale,
        pcc_scale=pcc_scale,
        seed=seed,
        policy=REFERENCE,
        preview_workers=PREVIEW_WORKERS,
    )
    return nominal, reference


def _evaluate_nominal_task(task: tuple[float, float, int]) -> dict[str, Any]:
    return evaluate_day(
        background_scale=task[0],
        pcc_scale=task[1],
        seed=task[2],
        policy=NOMINAL,
        preview_workers=1,
    )


def _evaluate_combo(
    background_scale: float,
    pcc_scale: float,
    workers: int,
) -> dict[str, list[dict[str, Any]]]:
    tasks = [(background_scale, pcc_scale, seed) for seed in SEEDS]
    if workers <= 1:
        evaluated = map(_evaluate_seed_task, tasks)
    else:
        with ProcessPoolExecutor(max_workers=min(int(workers), len(tasks))) as pool:
            evaluated = pool.map(_evaluate_seed_task, tasks)
    result = {NOMINAL: [], REFERENCE: []}
    for nominal, reference in evaluated:
        result[NOMINAL].append(nominal)
        result[REFERENCE].append(reference)
    return result


def _rounded_scale(value: Any) -> float:
    return round(float(value), 6)


def _normalize_results(
    results: dict[Any, dict[Any, dict[str, list[dict[str, Any]]]]]
) -> dict[float, dict[float, dict[str, list[dict[str, Any]]]]]:
    normalized: dict[float, dict[float, dict[str, list[dict[str, Any]]]]] = {}
    for background_scale, pcc_results in results.items():
        background = _rounded_scale(background_scale)
        normalized[background] = {
            _rounded_scale(pcc_scale): copy.deepcopy(per_policy)
            for pcc_scale, per_policy in pcc_results.items()
        }
    return normalized


def validate_coarse_reference(
    report: dict[str, Any],
) -> dict[float, dict[float, dict[str, list[dict[str, Any]]]]]:
    """Validate and normalize the exact Env-v5.2 200-point coarse scan."""
    expected = {
        "environment": ENVIRONMENT_VERSION,
        "seeds": list(SEEDS),
        "background_scales": list(BACKGROUND_SCALES),
        "pcc_scales": list(PCC_SCALES),
        "voltage_cost_definition": VOLTAGE_COST_DEFINITION,
    }
    for field, expected_value in expected.items():
        actual = report.get(field)
        if field in {"background_scales", "pcc_scales"}:
            actual = [_rounded_scale(value) for value in (actual or [])]
            expected_value = [_rounded_scale(value) for value in expected_value]
        elif field == "seeds":
            actual = [int(value) for value in (actual or [])]
        if actual != expected_value:
            raise ValueError(
                f"coarse reference {field} mismatch: "
                f"expected {expected_value!r}, got {actual!r}"
            )
    results = _normalize_results(report.get("results", {}))
    expected_backgrounds = set(BACKGROUND_SCALES)
    if set(results) != expected_backgrounds:
        raise ValueError("coarse reference background grid is incomplete")
    expected_pccs = set(PCC_SCALES)
    for background, pcc_results in results.items():
        if set(pcc_results) != expected_pccs:
            raise ValueError(
                f"coarse reference PCC grid is incomplete at background {background}"
            )
    return results


def _gate_status(per_policy: dict[str, list[dict[str, Any]]]) -> dict[str, bool]:
    nominal_days = per_policy.get(NOMINAL, [])
    reference_days = per_policy.get(REFERENCE, [])
    nominal_material_risk = sum(
        float(day.get("daily_voltage_cost", 0.0)) >= MATERIAL_RISK
        for day in nominal_days
    ) >= 2
    reference_safe = len(reference_days) == len(SEEDS) and all(
        bool(day.get("safe", False))
        and bool(day.get("pf_converged", False))
        and float(day.get("daily_voltage_cost", 0.0)) <= 1e-12
        for day in reference_days
    )
    return {
        "reference_safe": bool(reference_safe),
        "nominal_material_risk": bool(nominal_material_risk),
    }


def find_transition_intervals(
    results: dict[Any, dict[Any, dict[str, list[dict[str, Any]]]]]
) -> list[dict[str, Any]]:
    """Find adjacent coarse intervals where either physical gate predicate flips."""
    normalized = _normalize_results(results)
    intervals: list[dict[str, Any]] = []
    for background in sorted(normalized):
        pcc_results = normalized[background]
        pcc_scales = sorted(pcc_results)
        for lower, upper in zip(pcc_scales, pcc_scales[1:]):
            lower_status = _gate_status(pcc_results[lower])
            upper_status = _gate_status(pcc_results[upper])
            changed = [
                name
                for name in ("reference_safe", "nominal_material_risk")
                if lower_status[name] != upper_status[name]
            ]
            if changed:
                intervals.append(
                    {
                        "background_load_scale": background,
                        "lower_pcc_scale": lower,
                        "upper_pcc_scale": upper,
                        "changed": changed,
                    }
                )
    return intervals


def fine_pcc_points(
    intervals: list[dict[str, Any]], *, step: float = FINE_PCC_STEP
) -> list[tuple[float, float]]:
    """Generate unique interior fine-grid points for transition intervals."""
    fine: set[tuple[float, float]] = set()
    for interval in intervals:
        background = _rounded_scale(interval["background_load_scale"])
        lower = _rounded_scale(interval["lower_pcc_scale"])
        upper = _rounded_scale(interval["upper_pcc_scale"])
        candidate = _rounded_scale(lower + step)
        while candidate < upper - 1e-9:
            fine.add((background, candidate))
            candidate = _rounded_scale(candidate + step)
    return sorted(fine)


def merge_calibration_results(
    coarse: dict[Any, dict[Any, dict[str, list[dict[str, Any]]]]],
    fine: dict[Any, dict[Any, dict[str, list[dict[str, Any]]]]],
) -> dict[float, dict[float, dict[str, list[dict[str, Any]]]]]:
    merged = _normalize_results(coarse)
    for background, pcc_results in _normalize_results(fine).items():
        target = merged.setdefault(background, {})
        for pcc_scale, per_policy in pcc_results.items():
            if pcc_scale in target:
                raise ValueError(
                    f"fine result would overwrite coarse point {background}/{pcc_scale}"
                )
            target[pcc_scale] = per_policy
    return merged


def find_feasible_windows(
    results: dict[Any, dict[Any, dict[str, list[dict[str, Any]]]]],
    *,
    step: float = FINE_PCC_STEP,
) -> list[dict[str, Any]]:
    """Collect contiguous passing windows on the refined PCC grid."""
    normalized = _normalize_results(results)
    windows: list[dict[str, Any]] = []
    for background in sorted(normalized):
        passing = sorted(
            pcc
            for pcc, per_policy in normalized[background].items()
            if all(_gate_status(per_policy).values())
        )
        current: list[float] = []
        for pcc in passing:
            if current and abs(pcc - current[-1] - step) > 1e-9:
                windows.append(
                    {
                        "background_load_scale": background,
                        "pcc_scales": current,
                        "window_bounds": [current[0], current[-1]],
                        "point_count": len(current),
                    }
                )
                current = []
            current.append(pcc)
        if current:
            windows.append(
                {
                    "background_load_scale": background,
                    "pcc_scales": current,
                    "window_bounds": [current[0], current[-1]],
                    "point_count": len(current),
                }
            )
    return windows


def select_robust_feasible_window(
    results: dict[Any, dict[Any, dict[str, list[dict[str, Any]]]]]
) -> dict[str, Any] | None:
    """Select largest upper bound, then background, then lower midpoint."""
    normalized = _normalize_results(results)
    windows = sorted(
        find_feasible_windows(normalized),
        key=lambda window: (
            float(window["window_bounds"][1]),
            float(window["background_load_scale"]),
        ),
        reverse=True,
    )
    for window in windows:
        points = list(window["pcc_scales"])
        if len(points) < 3:
            continue
        midpoint_index = (len(points) - 1) // 2
        selected_pcc = points[midpoint_index]
        left = _rounded_scale(selected_pcc - FINE_PCC_STEP)
        right = _rounded_scale(selected_pcc + FINE_PCC_STEP)
        if left not in points or right not in points:
            continue
        background = float(window["background_load_scale"])
        per_policy = normalized[background][selected_pcc]
        nominal_days = per_policy.get(NOMINAL, [])
        reference_days = per_policy.get(REFERENCE, [])
        return {
            "background_load_scale": background,
            "pcc_injection_scale": selected_pcc,
            "idle_material_risk_days": sum(
                float(day.get("daily_voltage_cost", 0.0)) >= MATERIAL_RISK
                for day in nominal_days
            ),
            "reference_safe_days": sum(
                bool(day.get("safe", False))
                and bool(day.get("pf_converged", False))
                and float(day.get("daily_voltage_cost", 0.0)) <= 1e-12
                for day in reference_days
            ),
            "nominal_max_daily_cost": max(
                (float(day["daily_voltage_cost"]) for day in nominal_days),
                default=0.0,
            ),
            "nominal_days": nominal_days,
            "reference_days": reference_days,
            "window_bounds": list(window["window_bounds"]),
            "window_point_count": len(points),
            "neighbor_pcc_scales": [left, right],
        }
    return None


def select_largest_feasible_scale(
    results: dict[float, dict[float, dict[str, list[dict[str, Any]]]]]
) -> dict[str, Any] | None:
    """Select maximum PCC scale first, then maximum background load scale."""
    pairs = [
        (float(pcc_scale), float(background_scale), per_policy)
        for background_scale, pcc_results in results.items()
        for pcc_scale, per_policy in pcc_results.items()
    ]
    for pcc_scale, background_scale, per_policy in sorted(
        pairs, key=lambda item: (item[0], item[1]), reverse=True
    ):
        nominal_days = per_policy.get(NOMINAL, [])
        reference_days = per_policy.get(REFERENCE, [])
        material_days = sum(
            float(day.get("daily_voltage_cost", 0.0)) >= MATERIAL_RISK
            for day in nominal_days
        )
        reference_safe = (
            len(reference_days) == len(SEEDS)
            and all(
                bool(day.get("safe", False))
                and bool(day.get("pf_converged", False))
                for day in reference_days
            )
        )
        if material_days >= 2 and reference_safe:
            return {
                "background_load_scale": background_scale,
                "pcc_injection_scale": pcc_scale,
                "idle_material_risk_days": int(material_days),
                "nominal_max_daily_cost": max(
                    float(day["daily_voltage_cost"]) for day in nominal_days
                ),
                "nominal_days": nominal_days,
                "reference_days": reference_days,
            }
    return None


def parallel_determinism_check() -> dict[str, Any]:
    task = (0.1, 0.05, 30)
    serial = _evaluate_nominal_task(task)
    with ProcessPoolExecutor(max_workers=1) as pool:
        parallel = next(pool.map(_evaluate_nominal_task, (task,)))
    if serial != parallel:
        raise RuntimeError("serial and process calibration diagnostics differ")
    return {"task": list(task), "passed": True}


def _report(
    *,
    results: dict[float, dict[float, dict[str, list[dict[str, Any]]]]],
    workers: int,
    determinism: dict[str, Any],
    selection: dict[str, Any] | None,
    coarse_source: str | None = None,
    transitions: list[dict[str, Any]] | None = None,
    fine_points: list[tuple[float, float]] | None = None,
) -> dict[str, Any]:
    windows = find_feasible_windows(results) if coarse_source is not None else []
    report = {
        "environment": ENVIRONMENT_VERSION,
        "exploratory": True,
        "seeds": list(SEEDS),
        "background_scales": list(BACKGROUND_SCALES),
        "pcc_scales": list(PCC_SCALES),
        "search_priority": ["pcc_injection_scale", "background_load_scale"],
        "workers": int(workers),
        "preview_workers_per_reference_day": int(PREVIEW_WORKERS),
        "parallel_determinism": determinism,
        "voltage_limits_pu": [0.95, 1.05],
        "voltage_cost_definition": VOLTAGE_COST_DEFINITION,
        "nominal_material_risk_threshold": MATERIAL_RISK,
        "results": results,
        "feasible": selection is not None,
        "selection": selection,
    }
    if coarse_source is not None:
        report.update(
            {
                "coarse_source": coarse_source,
                "fine_pcc_step": FINE_PCC_STEP,
                "transition_intervals": transitions or [],
                "fine_points": [list(point) for point in (fine_points or [])],
                "feasible_windows": windows,
                "selection_rule": [
                    "maximum_window_upper_pcc",
                    "maximum_background_load_scale",
                    "lower_midpoint_for_even_candidate_count",
                ],
                "window_bounds": (
                    list(selection["window_bounds"])
                    if selection is not None
                    else None
                ),
                "selected_point_trajectories": (
                    {
                        NOMINAL: selection["nominal_days"],
                        REFERENCE: selection["reference_days"],
                    }
                    if selection is not None
                    else None
                ),
            }
        )
    return report


def calibrate(
    *,
    workers: int | None = None,
    coarse_reference: Path | dict[str, Any] | None = None,
) -> dict[str, Any]:
    worker_count = (
        min(6, os.cpu_count() or 1)
        if workers is None
        else max(1, min(6, int(workers)))
    )
    if coarse_reference is not None:
        if isinstance(coarse_reference, Path):
            coarse_report = json.loads(coarse_reference.read_text(encoding="utf-8"))
            coarse_source = str(coarse_reference.resolve())
        else:
            coarse_report = coarse_reference
            coarse_source = "in-memory"
        coarse_results = validate_coarse_reference(coarse_report)
        transitions = find_transition_intervals(coarse_results)
        fine_points = fine_pcc_points(transitions)
        fine_results: dict[
            float, dict[float, dict[str, list[dict[str, Any]]]]
        ] = {}
        for background_scale, pcc_scale in fine_points:
            fine_results.setdefault(background_scale, {})[pcc_scale] = _evaluate_combo(
                background_scale, pcc_scale, worker_count
            )
        results = merge_calibration_results(coarse_results, fine_results)
        selection = select_robust_feasible_window(results)
        return _report(
            results=results,
            workers=worker_count,
            determinism=coarse_report.get("parallel_determinism", {}),
            selection=selection,
            coarse_source=coarse_source,
            transitions=transitions,
            fine_points=fine_points,
        )

    determinism = parallel_determinism_check()
    results: dict[float, dict[float, dict[str, list[dict[str, Any]]]]] = {}
    # The loop order is the selection order, so the first passing point is final.
    for pcc_scale in sorted(PCC_SCALES, reverse=True):
        for background_scale in sorted(BACKGROUND_SCALES, reverse=True):
            results.setdefault(background_scale, {})[pcc_scale] = _evaluate_combo(
                background_scale, pcc_scale, worker_count
            )
            selection = select_largest_feasible_scale(results)
            if selection is not None:
                return _report(
                    results=results,
                    workers=worker_count,
                    determinism=determinism,
                    selection=selection,
                )
    return _report(
        results=results,
        workers=worker_count,
        determinism=determinism,
        selection=None,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--coarse-reference", type=Path)
    args = parser.parse_args()
    report = calibrate(
        workers=args.workers,
        coarse_reference=args.coarse_reference,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    selection = report["selection"]
    selection_summary = (
        {
            key: selection[key]
            for key in (
                "background_load_scale",
                "pcc_injection_scale",
                "idle_material_risk_days",
                "nominal_max_daily_cost",
            )
        }
        if selection is not None
        else None
    )
    print(
        json.dumps(
            {"feasible": report["feasible"], "selection": selection_summary},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
