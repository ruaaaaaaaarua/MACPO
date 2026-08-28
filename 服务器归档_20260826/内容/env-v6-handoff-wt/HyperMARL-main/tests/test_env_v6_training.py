import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from scripts.benchmark_env_v6_rollout import assess_performance_gate, rollout_difference

from scripts.run_env_v3_safe_matrix import (
    EXPERIMENTS,
    apply_env_v6_calibration,
    build_gru_config,
    run,
)


V6_VARIANTS = (
    "v6_nocomm_gru_mappo",
    "v6_nocomm_gru_mappo_penalty",
    "v6_nocomm_gru_macpo",
)


def calibration_report():
    return {
        "environment": "env-v6-swiss",
        "feasible": True,
        "pcc_injection_scale": 1.0,
        "background_load_scale": 1.0,
        "economic_reward_scale_yuan": 2_500_000.0,
        "training_cost_scale": 0.02,
        "training_cost_budget": 1.0,
        "selection": {
            "grid_id": "10_2",
            "case_dir": "/datasets/Swiss-PDGs/MV/10_2",
            "pcc_bus_ids": [11, 22, 33, 44],
        },
    }


class EnvV6TrainingConfigTest(unittest.TestCase):
    def test_rollout_difference_handles_boolean_done_arrays(self):
        fields = (
            "local_obs", "global_obs", "dones_before", "dones", "actions",
            "log_probs", "rewards", "costs", "raw_costs", "reward_values",
            "cost_values", "intents",
        )
        first = SimpleNamespace(**{
            field: np.asarray([False, True]) if field.startswith("done")
            else np.asarray([0.0, 1.0])
            for field in fields
        })
        second = SimpleNamespace(**{
            field: np.asarray([False, False]) if field == "dones"
            else np.asarray(getattr(first, field))
            for field in fields
        })

        difference = rollout_difference(first, second)

        self.assertEqual(difference["fields"]["dones"], 1.0)
        self.assertFalse(difference["passed"])

    def test_performance_gate_requires_parity_and_twenty_five_percent_speedup(self):
        passed = assess_performance_gate(
            legacy_seconds=10.0,
            fused_process_seconds=8.0,
            semantic_parity=True,
        )
        slow = assess_performance_gate(
            legacy_seconds=10.0,
            fused_process_seconds=8.1,
            semantic_parity=True,
        )
        mismatch = assess_performance_gate(
            legacy_seconds=10.0,
            fused_process_seconds=7.0,
            semantic_parity=False,
        )

        self.assertTrue(passed["passed"])
        self.assertAlmostEqual(passed["speedup"], 1.25)
        self.assertFalse(slow["passed"])
        self.assertFalse(mismatch["passed"])

    def test_three_variants_are_no_communication_two_environment_gru_runs(self):
        self.assertEqual(EXPERIMENTS[V6_VARIANTS[0]]["algorithm"], "mappo")
        self.assertEqual(
            EXPERIMENTS[V6_VARIANTS[1]]["algorithm"], "mappo_penalty"
        )
        self.assertEqual(EXPERIMENTS[V6_VARIANTS[2]]["algorithm"], "macpo")

        for variant in V6_VARIANTS:
            with self.subTest(variant=variant):
                config = build_gru_config(variant, updates=1000)
                self.assertEqual(config["num_envs"], 2)
                self.assertEqual(config["num_steps"], 24)
                self.assertEqual(config["env_parallel_backend"], "process")
                self.assertTrue(config["fused_rollout_kernel"])
                self.assertTrue(config["include_previous_action"])
                self.assertFalse(config["include_transaction_message"])
                self.assertFalse(config["two_stage_intent"])
                self.assertFalse(config["h2_supply_intent_message_enable"])
                self.assertTrue(
                    config["env_overrides"]["h2_local_supply_facts_enable"]
                )

    def test_calibration_sets_native_swiss_case_and_dimensionless_scales(self):
        config = build_gru_config(V6_VARIANTS[2], updates=1000)

        apply_env_v6_calibration(config, calibration_report())

        overrides = config["env_overrides"]
        self.assertEqual(overrides["power_flow_model"], "swiss_mv")
        self.assertEqual(overrides["power_flow_case_dir"], "/datasets/Swiss-PDGs/MV/10_2")
        self.assertEqual(overrides["power_flow_pcc_bus_ids"], [11, 22, 33, 44])
        self.assertEqual(overrides["power_flow_pcc_injection_scale"], 1.0)
        self.assertEqual(overrides["power_flow_background_load_scale"], 1.0)
        self.assertEqual(overrides["reward_scale"], 2_500_000.0)
        self.assertEqual(config["voltage_cost_scale"], 0.02)
        self.assertEqual(config["cost_budget"], 1.0)
        self.assertIsNone(config["curriculum_d_start"])
        self.assertIsNone(config["curriculum_d_target"])

    def test_calibration_rejects_missing_gate_or_non_native_scaling(self):
        for change in (
            {"feasible": False},
            {"pcc_injection_scale": 0.9},
            {"background_load_scale": 0.9},
        ):
            report = calibration_report()
            report.update(change)
            config = build_gru_config(V6_VARIANTS[2], updates=10)
            with self.subTest(change=change), self.assertRaises(ValueError):
                apply_env_v6_calibration(config, report)

    def test_v6_runner_loads_calibration_without_overriding_process_backend(self):
        with tempfile.TemporaryDirectory() as directory:
            calibration_path = Path(directory) / "calibration.json"
            calibration_path.write_text(
                json.dumps(calibration_report()), encoding="utf-8"
            )

            description = run(
                V6_VARIANTS[2],
                updates=100,
                dry_run=True,
                env_v6_calibration=calibration_path,
            )

        config = description["config"]
        self.assertEqual(config["env_parallel_backend"], "process")
        self.assertEqual(config["env_overrides"]["power_flow_model"], "swiss_mv")
        self.assertEqual(config["voltage_cost_scale"], 0.02)
        self.assertEqual(config["cost_budget"], 1.0)

    def test_v6_runner_allows_an_explicit_backend_override(self):
        with tempfile.TemporaryDirectory() as directory:
            calibration_path = Path(directory) / "calibration.json"
            calibration_path.write_text(
                json.dumps(calibration_report()), encoding="utf-8"
            )

            description = run(
                V6_VARIANTS[0],
                updates=1,
                dry_run=True,
                env_v6_calibration=calibration_path,
                env_parallel_backend="serial",
            )

        self.assertEqual(description["config"]["env_parallel_backend"], "serial")

    def test_runner_rejects_two_calibration_interfaces_at_once(self):
        with tempfile.TemporaryDirectory() as directory:
            calibration_path = Path(directory) / "calibration.json"
            calibration_path.write_text(
                json.dumps(calibration_report()), encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "only one calibration"):
                run(
                    V6_VARIANTS[2],
                    updates=1,
                    dry_run=True,
                    safety_reference=calibration_path,
                    env_v6_calibration=calibration_path,
                )


if __name__ == "__main__":
    unittest.main()
