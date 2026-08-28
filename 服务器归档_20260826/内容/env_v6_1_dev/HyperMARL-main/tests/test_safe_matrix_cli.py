import subprocess
import sys
import unittest
from pathlib import Path

from scripts.run_env_v3_safe_matrix import build_gru_config


class EnvV3SafeMatrixCliTest(unittest.TestCase):
    def test_two_stage_variants_keep_history_and_expose_distinct_budget_modes(self):
        strict = build_gru_config("dense_gru_macpo_two_stage_intent_d_strict", updates=3)
        balanced = build_gru_config("dense_gru_macpo_two_stage_intent_d_balanced", updates=3)
        no_broadcast = build_gru_config(
            "dense_gru_macpo_two_stage_intent_no_broadcast_d_balanced", updates=3
        )

        self.assertTrue(strict["two_stage_intent"])
        self.assertTrue(strict["include_previous_action"])
        self.assertTrue(strict["include_transaction_message"])
        self.assertEqual(strict["intent_dim"], 3)
        self.assertLess(strict["cost_budget"], balanced["cost_budget"])
        self.assertEqual(no_broadcast["intent_broadcast_mode"], "other_zero")

    def test_v4_intent_variants_bind_forecasts_and_the_balanced_course(self):
        full = build_gru_config("v4_full_intent_curriculum", updates=1000)
        no_broadcast = build_gru_config("v4_no_broadcast_curriculum", updates=1000)

        self.assertTrue(full["two_stage_intent"])
        self.assertTrue(full["env_overrides"]["h2_day_ahead_forecast_enable"])
        self.assertEqual(full["env_overrides"]["h2_day_ahead_forecast_horizons"], [4, 6, 10])
        self.assertEqual(full["intent_residual_limit"], 0.25)
        self.assertEqual(full["intent_residual_coef"], 0.01)
        self.assertEqual(full["curriculum_d_start"], 3.5)
        self.assertEqual(full["curriculum_updates"], 200)
        self.assertAlmostEqual(full["curriculum_log_std_start"], -1.0)
        self.assertAlmostEqual(full["curriculum_log_std_end"], -2.3)
        self.assertEqual(no_broadcast["intent_broadcast_mode"], "other_zero")
    def test_direct_script_dry_run_resolves_project_imports(self):
        root = Path(__file__).resolve().parents[1]
        completed = subprocess.run(
            [
                sys.executable,
                str(root / "scripts" / "run_env_v3_safe_matrix.py"),
                "dense_gru_mappo_anneal",
                "--updates",
                "1",
                "--dry-run",
            ],
            cwd=root,
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_direct_safety_reference_script_resolves_project_imports(self):
        root = Path(__file__).resolve().parents[1]
        completed = subprocess.run(
            [
                sys.executable,
                str(root / "scripts" / "build_env_v3_safe_reference.py"),
                "--help",
            ],
            cwd=root,
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)


if __name__ == "__main__":
    unittest.main()
