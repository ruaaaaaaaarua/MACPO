import unittest
import json
import tempfile
from pathlib import Path

import numpy as np

from baselines.utils.fixed_scenario_eval import (
    DEFAULT_NOISE_SEEDS,
    TEST_DAYS,
    TEST_NOISE_SEEDS,
    VALIDATION_DAYS,
    build_scenarios,
    evaluate_policy,
    append_evaluation_record,
)


ABC_MINIMAL = {
    "episode_length": 24,
    "italian_split_enable": True,
    "h2_pending_obs_enable": True,
    "h2_pending_obs_horizon": 4,
    "h2_pending_summary_obs_enable": True,
    "h2_learnable_rolling_order_enable": True,
}


class FixedScenarioEvaluatorTest(unittest.TestCase):
    def test_validation_and_test_use_four_single_seed_scenarios(self):
        scenarios = build_scenarios(VALIDATION_DAYS, DEFAULT_NOISE_SEEDS)
        test_scenarios = build_scenarios(TEST_DAYS, TEST_NOISE_SEEDS)
        self.assertEqual(len(scenarios), 4)
        self.assertEqual(len({(item.day, item.seed) for item in scenarios}), 4)
        self.assertEqual({item.seed for item in scenarios}, {4200})
        self.assertEqual(len(test_scenarios), 4)
        self.assertEqual({item.seed for item in test_scenarios}, {5200})

    def test_zero_policy_evaluation_is_reproducible_and_complete(self):
        scenarios = build_scenarios([8], [4200])

        def zero_policy(obs):
            return np.zeros((obs.shape[0], 6), dtype=np.float32)

        first = evaluate_policy(
            zero_policy,
            ABC_MINIMAL,
            scenarios,
            algorithm="zero",
            split_name="validation",
        )
        second = evaluate_policy(
            zero_policy,
            ABC_MINIMAL,
            scenarios,
            algorithm="zero",
            split_name="validation",
        )
        self.assertEqual(first["summary"], second["summary"])
        self.assertEqual(len(first["episodes"]), 1)
        expected = {
            "return_mean",
            "return_std",
            "base_cost_mean",
            "external_h2_buy_mean",
            "internal_h2_trade_mean",
            "low_h2_hits_mean",
            "action_saturation_rate",
        }
        self.assertTrue(expected.issubset(first["summary"]))
        finite_values = [
            value for value in first["summary"].values() if value is not None
        ]
        self.assertTrue(np.isfinite(finite_values).all())

    def test_out_of_bounds_policy_is_clipped_and_counted(self):
        def saturated_policy(obs):
            return np.full((obs.shape[0], 6), 2.0, dtype=np.float32)

        result = evaluate_policy(
            saturated_policy,
            ABC_MINIMAL,
            build_scenarios([8], [4200]),
            algorithm="saturated",
            split_name="validation",
        )
        self.assertEqual(result["summary"]["action_saturation_rate"], 1.0)
        self.assertLessEqual(result["episodes"][0]["action_max_abs"], 1.0)

    def test_nonfinite_correlation_is_serialized_as_null(self):
        payload = {
            "algorithm": "zero",
            "summary": {"order_vs_t4_load_correlation_mean": float("nan")},
            "episodes": [{"order_vs_t4_load_correlation": float("inf")}],
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "eval.jsonl"
            append_evaluation_record(path, payload, training_episode=0)
            row = json.loads(path.read_text())
        self.assertIsNone(row["summary"]["order_vs_t4_load_correlation_mean"])
        self.assertIsNone(row["episodes"][0]["order_vs_t4_load_correlation"])

    def test_evaluation_record_is_appended_as_jsonl(self):
        payload = {
            "algorithm": "zero",
            "summary": {"return_mean": -1.0},
            "episodes": [],
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "eval.jsonl"
            append_evaluation_record(path, payload, training_episode=500)
            append_evaluation_record(path, payload, training_episode=1000)
            rows = [json.loads(line) for line in path.read_text().splitlines()]
        self.assertEqual([row["training_episode"] for row in rows], [500, 1000])
        self.assertEqual(rows[0]["summary"]["return_mean"], -1.0)



if __name__ == "__main__":
    unittest.main()
