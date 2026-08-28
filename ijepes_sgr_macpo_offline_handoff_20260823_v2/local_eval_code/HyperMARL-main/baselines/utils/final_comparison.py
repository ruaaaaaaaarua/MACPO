"""Locked-test comparison for learned policies, counterfactuals, and rules."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Callable, Mapping

import numpy as np

from baselines.utils.fixed_scenario_eval import (
    TEST_DAYS,
    TEST_NOISE_SEEDS,
    append_evaluation_record,
    build_scenarios,
    evaluate_contextual_policy,
    evaluate_policy,
)
from baselines.utils.rule_baselines import RULE_BASELINES


REQUIRED_METRICS = (
    "return_mean",
    "total_cost_mean",
    "base_cost_mean",
    "external_h2_buy_mean",
    "internal_h2_trade_mean",
    "action_requested_buy_mean",
    "effective_buy_mean",
    "clipped_buy_mean",
    "pending_mean",
    "delivered_mean",
    "terminal_soc_ratio_mean",
    "terminal_h2_ratio_mean",
    "order_vs_t4_load_correlation_mean",
    "arrival_vs_h2_load_correlation_mean",
    "terminal_h2_shortfall_kg_mean",
    "terminal_h2_settlement_cost_mean",
    "effective_external_h2_cost_yuan_per_kg",
    "action_saturation_rate",
    "transport_shipment_count_mean",
    "transport_gross_mean",
    "transport_loss_mean",
    "transport_eta_mean",
    "transport_delayed_rate_mean",
    "route_entropy_mean",
    "horizon_clipped_buy_mean",
    "edge_utilization_max_mean",
)


def force_no_order(action_fn: Callable[[np.ndarray], np.ndarray]):
    """Wrap a learned policy and overwrite only its complete-order action a5."""

    def counterfactual(obs):
        normal = np.asarray(action_fn(obs))
        if normal.ndim != 2 or normal.shape[1] < 6:
            raise ValueError(
                "learned-order counterfactual requires actions shaped (agents, >=6)"
            )
        forced = normal.copy()
        forced[:, 5] = -1
        return forced

    counterfactual.original_action_fn = action_fn
    counterfactual.counterfactual = "forced_no_order"
    return counterfactual


def force_direct_route(action_fn: Callable[[np.ndarray], np.ndarray]):
    """Keep a0-a5 unchanged and force every route preference to direct."""

    def counterfactual(obs):
        normal = np.asarray(action_fn(obs))
        if normal.ndim != 2 or normal.shape[1] < 7:
            raise ValueError("route counterfactual requires actions shaped (agents, >=7)")
        forced = normal.copy()
        forced[:, 6] = -1.0
        return forced

    counterfactual.original_action_fn = action_fn
    counterfactual.counterfactual = "forced_direct_route"
    return counterfactual


def permute_route_actions(action_fn: Callable[[np.ndarray], np.ndarray]):
    """Break buyer-route alignment while preserving each step's a6 multiset."""

    def counterfactual(obs):
        normal = np.asarray(action_fn(obs))
        if normal.ndim != 2 or normal.shape[1] < 7:
            raise ValueError("route counterfactual requires actions shaped (agents, >=7)")
        forced = normal.copy()
        forced[:, 6] = np.roll(normal[:, 6], 1)
        return forced

    counterfactual.original_action_fn = action_fn
    counterfactual.counterfactual = "permuted_route"
    return counterfactual


def validate_summary_metrics(summary):
    """Require finite physical metrics while null-normalizing undefined correlation."""
    missing = [key for key in REQUIRED_METRICS if key not in summary]
    if missing:
        raise ValueError(f"missing required metrics: {missing}")
    validated = dict(summary)
    correlation_keys = {
        "order_vs_t4_load_correlation_mean",
        "arrival_vs_h2_load_correlation_mean",
    }
    for key in REQUIRED_METRICS:
        value = validated[key]
        if key in correlation_keys:
            if value is None:
                continue
            try:
                value = float(value)
            except (TypeError, ValueError) as error:
                raise ValueError(f"metric {key!r} must be numeric or null") from error
            validated[key] = value if math.isfinite(value) else None
            continue
        try:
            value = float(value)
        except (TypeError, ValueError) as error:
            raise ValueError(f"metric {key!r} must be finite") from error
        if not math.isfinite(value):
            raise ValueError(f"metric {key!r} must be finite")
        validated[key] = value
    return validated


def _metric_delta(normal, counterfactual):
    delta = {}
    for key in REQUIRED_METRICS:
        left = normal[key]
        right = counterfactual[key]
        delta[key] = None if left is None or right is None else float(left - right)
    return delta


def _format_metric(value):
    return "null" if value is None else f"{float(value):.6g}"


def render_comparison_markdown(summary):
    columns = ("result", "category", *REQUIRED_METRICS)
    lines = [
        "# Locked test comparison",
        "",
        "All rows use only the strict test split (days 1, 7, 14, 24; seed 5200).",
        "The privileged_t4_rule row is a privileged diagnostic and is excluded from model selection and ranking claims.",
        "",
        "| " + " | ".join(columns) + " |",
        "|" + "|".join(["---"] * len(columns)) + "|",
    ]
    for label, row in summary["results"].items():
        metrics = row["metrics"]
        lines.append(
            "| "
            + " | ".join(
                [label, row["category"]]
                + [_format_metric(metrics[key]) for key in REQUIRED_METRICS]
            )
            + " |"
        )

    lines.extend(["", "## Normal-minus-no-order deltas", ""])
    for algorithm, delta in summary["normal_minus_no_order"].items():
        lines.append(
            f"- {algorithm}: "
            + ", ".join(
                f"{key}={_format_metric(delta[key])}" for key in REQUIRED_METRICS
            )
        )

    current_return = summary["results"]["current_deficit_rule"]["metrics"]["return_mean"]
    privileged_return = summary["results"]["privileged_t4_rule"]["metrics"]["return_mean"]
    lines.extend(["", "## Interpretation guardrails", ""])
    for algorithm in summary["model_selection"]["learned_normal"]:
        normal = summary["results"][algorithm]["metrics"]["return_mean"]
        forced = summary["results"][f"{algorithm}__forced_no_order"]["metrics"]["return_mean"]
        lines.append(
            f"- {algorithm} beats current rule: {'yes' if normal > current_return else 'no'} "
            f"(return delta {normal - current_return:.6g})."
        )
        lines.append(
            f"- {algorithm} approaches privileged diagnostic: return gap "
            f"{normal - privileged_return:.6g}; this is diagnostic, not a ranking claim."
        )
        lines.append(
            f"- {algorithm} benefits from learned ordering: "
            f"{'yes' if normal > forced else 'no'} (normal-minus-no-order return "
            f"{normal - forced:.6g})."
        )
    lines.append(
        "- Planning inference guardrail: inventory/pending growth alone is not evidence "
        "of learned planning; use the forced-no-order delta and order-vs-t+4-load correlation together."
    )
    return "\n".join(lines) + "\n"


def run_final_comparison(
    learned_policies: Mapping[str, Callable],
    base_overrides: Mapping[str, object],
    output_root,
    *,
    algorithm_names: Mapping[str, str] | None = None,
    training_episode: int = 30000,
):
    """Run every final comparison exactly once on the locked strict test split."""
    required_learned = {"MAPPO", "MATD3", "STAS"}
    if set(learned_policies) != required_learned:
        raise ValueError(
            f"learned_policies must contain exactly {sorted(required_learned)}"
        )
    algorithm_names = dict(algorithm_names or {})
    route_counterfactuals_enabled = bool(
        base_overrides.get("h2_traffic_enable", False)
        and base_overrides.get("h2_route_action_enable", False)
    )
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    records_path = output_root / "final_comparison.jsonl"
    summary_path = output_root / "final_comparison_summary.json"
    report_path = output_root / "final_comparison.md"
    existing = [path for path in (records_path, summary_path, report_path) if path.exists()]
    if existing:
        raise FileExistsError(f"final comparison outputs already exist: {existing}")

    scenarios = build_scenarios(TEST_DAYS, TEST_NOISE_SEEDS)
    results = {}
    raw_records = []
    deltas = {}
    direct_deltas = {}
    permuted_deltas = {}
    for name in ("MAPPO", "MATD3", "STAS"):
        policy = learned_policies[name]
        display_algorithm = algorithm_names.get(name, name)
        normal = evaluate_policy(
            policy,
            base_overrides,
            scenarios,
            algorithm=display_algorithm,
            split_name="test",
        )
        normal_metrics = validate_summary_metrics(normal["summary"])
        normal["summary"] = normal_metrics
        normal.update(
            {
                "result_name": name,
                "category": "learned_normal",
                "variant": "normal",
                "selection_eligible": True,
            }
        )
        raw_records.append(normal)
        results[name] = {
            "algorithm": display_algorithm,
            "category": "learned_normal",
            "variant": "normal",
            "privileged_diagnostic": False,
            "metrics": normal_metrics,
        }

        counterfactual_name = f"{name}__forced_no_order"
        no_order = evaluate_policy(
            force_no_order(policy),
            base_overrides,
            scenarios,
            algorithm=counterfactual_name,
            split_name="test",
        )
        no_order_metrics = validate_summary_metrics(no_order["summary"])
        no_order["summary"] = no_order_metrics
        no_order.update(
            {
                "result_name": counterfactual_name,
                "category": "learned_counterfactual",
                "variant": "forced_no_order",
                "selection_eligible": False,
            }
        )
        raw_records.append(no_order)
        results[counterfactual_name] = {
            "algorithm": display_algorithm,
            "category": "learned_counterfactual",
            "variant": "forced_no_order",
            "privileged_diagnostic": False,
            "metrics": no_order_metrics,
        }
        deltas[name] = _metric_delta(normal_metrics, no_order_metrics)

        route_variants = (
            (
                ("forced_direct_route", force_direct_route, direct_deltas),
                ("permuted_route", permute_route_actions, permuted_deltas),
            )
            if route_counterfactuals_enabled else ()
        )
        for suffix, wrapper, destination in route_variants:
            counterfactual_name = f"{name}__{suffix}"
            route_counterfactual = evaluate_policy(
                wrapper(policy),
                base_overrides,
                scenarios,
                algorithm=counterfactual_name,
                split_name="test",
            )
            route_metrics = validate_summary_metrics(route_counterfactual["summary"])
            route_counterfactual["summary"] = route_metrics
            route_counterfactual.update(
                {
                    "result_name": counterfactual_name,
                    "category": "learned_counterfactual",
                    "variant": suffix,
                    "selection_eligible": False,
                }
            )
            raw_records.append(route_counterfactual)
            results[counterfactual_name] = {
                "algorithm": display_algorithm,
                "category": "learned_counterfactual",
                "variant": suffix,
                "privileged_diagnostic": False,
                "metrics": route_metrics,
            }
            destination[name] = _metric_delta(normal_metrics, route_metrics)

    for name, policy in RULE_BASELINES.items():
        rule = evaluate_contextual_policy(
            policy,
            base_overrides,
            scenarios,
            algorithm=name,
            split_name="test",
        )
        rule_metrics = validate_summary_metrics(rule["summary"])
        rule["summary"] = rule_metrics
        privileged = bool(rule.get("privileged_diagnostic", False))
        rule.update(
            {
                "result_name": name,
                "category": "privileged_diagnostic" if privileged else "rule_baseline",
                "variant": "rule",
                "selection_eligible": False,
            }
        )
        raw_records.append(rule)
        results[name] = {
            "algorithm": name,
            "category": "privileged_diagnostic" if privileged else "rule_baseline",
            "variant": "rule",
            "privileged_diagnostic": privileged,
            "metrics": rule_metrics,
        }

    for record in raw_records:
        append_evaluation_record(
            records_path,
            record,
            training_episode=training_episode,
        )

    summary = {
        "split_name": "test",
        "scenario_days": list(TEST_DAYS),
        "scenario_seed": TEST_NOISE_SEEDS[0],
        "results": results,
        "normal_minus_no_order": deltas,
        # Privileged diagnostic is intentionally absent from all selection fields.
        "model_selection": {
            "learned_normal": ["MAPPO", "MATD3", "STAS"],
            "non_privileged_reference_baselines": [
                "physical_idle",
                "current_deficit_rule",
            ],
        },
    }
    if route_counterfactuals_enabled:
        summary["normal_minus_forced_direct_route"] = direct_deltas
        summary["normal_minus_permuted_route"] = permuted_deltas
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    report_path.write_text(render_comparison_markdown(summary), encoding="utf-8")
    return summary
