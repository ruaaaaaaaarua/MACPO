import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from baselines.utils.final_comparison import (
    REQUIRED_METRICS,
    force_direct_route,
    force_no_order,
    permute_route_actions,
    run_final_comparison,
    validate_summary_metrics,
)
from baselines.utils.fixed_scenario_eval import TEST_DAYS


def metric_summary(return_mean=-10.0):
    summary = {key: 1.0 for key in REQUIRED_METRICS}
    summary["return_mean"] = return_mean
    summary["order_vs_t4_load_correlation_mean"] = None
    if "arrival_vs_h2_load_correlation_mean" in summary:
        summary["arrival_vs_h2_load_correlation_mean"] = None
    return summary


class FinalComparisonReportingTest(unittest.TestCase):
    def test_counterfactual_changes_only_action_column_five(self):
        original = np.arange(24, dtype=np.float32).reshape(4, 6) / 24.0

        def learned(_obs):
            return original.copy()

        forced = force_no_order(learned)(np.zeros((4, 13), dtype=np.float32))
        np.testing.assert_array_equal(forced[:, :5], original[:, :5])
        np.testing.assert_array_equal(forced[:, 5], -1.0)

    def test_route_counterfactuals_preserve_a0_to_a5(self):
        original = np.arange(28, dtype=np.float32).reshape(4, 7) / 28.0

        def learned(_obs):
            return original.copy()

        direct = force_direct_route(learned)(np.zeros((4, 24), dtype=np.float32))
        permuted = permute_route_actions(learned)(np.zeros((4, 24), dtype=np.float32))
        np.testing.assert_array_equal(direct[:, :6], original[:, :6])
        np.testing.assert_array_equal(direct[:, 6], -1.0)
        np.testing.assert_array_equal(permuted[:, :6], original[:, :6])
        np.testing.assert_array_equal(np.sort(permuted[:, 6]), np.sort(original[:, 6]))
        self.assertFalse(np.array_equal(permuted[:, 6], original[:, 6]))

    def test_missing_or_nonfinite_required_metric_fails_but_correlation_null_is_valid(self):
        valid = metric_summary()
        self.assertIsNone(
            validate_summary_metrics(valid)["order_vs_t4_load_correlation_mean"]
        )
        self.assertIsNone(
            validate_summary_metrics(valid)["arrival_vs_h2_load_correlation_mean"]
        )
        missing = dict(valid)
        missing.pop("total_cost_mean")
        with self.assertRaisesRegex(ValueError, "missing required metrics"):
            validate_summary_metrics(missing)
        nonfinite = dict(valid)
        nonfinite["base_cost_mean"] = float("inf")
        with self.assertRaisesRegex(ValueError, "must be finite"):
            validate_summary_metrics(nonfinite)
        nullified = dict(valid)
        nullified["order_vs_t4_load_correlation_mean"] = float("nan")
        self.assertIsNone(
            validate_summary_metrics(nullified)["order_vs_t4_load_correlation_mean"]
        )

    def test_final_runner_uses_only_strict_test_and_writes_complete_outputs(self):
        learned_calls = []
        rule_calls = []

        def fake_learned(policy, _overrides, scenarios, *, algorithm, split_name):
            learned_calls.append((policy, split_name, tuple(item.day for item in scenarios)))
            return {
                "algorithm": algorithm,
                "split_name": split_name,
                "privileged_diagnostic": False,
                "summary": metric_summary(-10.0),
                "episodes": [],
            }

        def fake_rule(policy, _overrides, scenarios, *, algorithm, split_name):
            rule_calls.append((split_name, tuple(item.day for item in scenarios)))
            return {
                "algorithm": algorithm,
                "split_name": split_name,
                "privileged_diagnostic": algorithm == "privileged_t4_rule",
                "summary": metric_summary(-9.0),
                "episodes": [],
            }

        policies = {name: (lambda obs: np.zeros((obs.shape[0], 6), dtype=np.float32)) for name in ("MAPPO", "MATD3", "STAS")}
        with tempfile.TemporaryDirectory() as tmp, patch(
            "baselines.utils.final_comparison.evaluate_policy", side_effect=fake_learned
        ), patch(
            "baselines.utils.final_comparison.evaluate_contextual_policy",
            side_effect=fake_rule,
        ):
            output = Path(tmp) / "comparison"
            summary = run_final_comparison(
                policies,
                {"h2_traffic_enable": True, "h2_route_action_enable": True},
                output,
            )
            rows = [
                json.loads(line)
                for line in (output / "final_comparison.jsonl").read_text().splitlines()
            ]
            markdown = (output / "final_comparison.md").read_text()

        self.assertEqual(len(learned_calls), 12)
        self.assertEqual(len(rule_calls), 5)
        self.assertTrue(all(call[1] == "test" for call in learned_calls))
        self.assertTrue(all(call[0] == "test" for call in rule_calls))
        self.assertTrue(
            all(call[-1] == TEST_DAYS for call in learned_calls + rule_calls)
        )
        self.assertEqual(len(rows), 17)
        expected = {
            "MAPPO", "MATD3", "STAS",
            "MAPPO__forced_no_order", "MATD3__forced_no_order", "STAS__forced_no_order",
            "MAPPO__forced_direct_route", "MATD3__forced_direct_route", "STAS__forced_direct_route",
            "MAPPO__permuted_route", "MATD3__permuted_route", "STAS__permuted_route",
            "physical_idle", "current_deficit_rule", "privileged_t4_rule",
            "base_stock_rule", "base_stock_privileged",
        }
        self.assertEqual(set(summary["results"]), expected)
        self.assertNotIn(
            "privileged_t4_rule",
            json.dumps(summary["model_selection"], sort_keys=True),
        )
        for label in expected:
            self.assertIn(label, markdown)
        for metric in REQUIRED_METRICS:
            self.assertIn(metric, markdown)
        self.assertIn("inventory/pending growth alone", markdown)

    def test_nontraffic_final_runner_keeps_legacy_counterfactual_set(self):
        learned_calls = []

        def fake_learned(policy, _overrides, _scenarios, *, algorithm, split_name):
            learned_calls.append((algorithm, split_name))
            return {
                "algorithm": algorithm,
                "split_name": split_name,
                "privileged_diagnostic": False,
                "summary": metric_summary(-10.0),
                "episodes": [],
            }

        def fake_rule(_policy, _overrides, _scenarios, *, algorithm, split_name):
            return {
                "algorithm": algorithm,
                "split_name": split_name,
                "privileged_diagnostic": algorithm == "privileged_t4_rule",
                "summary": metric_summary(-9.0),
                "episodes": [],
            }

        policies = {
            name: (lambda obs: np.zeros((obs.shape[0], 6), dtype=np.float32))
            for name in ("MAPPO", "MATD3", "STAS")
        }
        with tempfile.TemporaryDirectory() as tmp, patch(
            "baselines.utils.final_comparison.evaluate_policy", side_effect=fake_learned
        ), patch(
            "baselines.utils.final_comparison.evaluate_contextual_policy",
            side_effect=fake_rule,
        ):
            summary = run_final_comparison(policies, {}, Path(tmp) / "comparison")
        self.assertEqual(len(learned_calls), 6)
        self.assertFalse(any("route" in key for key in summary["results"]))


if __name__ == "__main__":
    unittest.main()
