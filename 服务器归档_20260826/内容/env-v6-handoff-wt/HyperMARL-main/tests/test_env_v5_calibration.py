import unittest
from unittest import mock

import numpy as np

from envs.microgrid.microgrid_env import MicrogridEnv
from scripts import calibrate_env_v5_feasibility as calibration
from scripts.calibrate_env_v5_feasibility import (
    BACKGROUND_SCALES,
    PCC_SCALES,
    nominal_uncontrolled_action,
    select_largest_feasible_scale,
    voltage_support_reference_action,
)


def days(*, safe=False, nominal=0.03):
    return [
        {
            "safe": safe,
            "daily_voltage_cost": nominal,
            "pf_converged": True,
        }
    ] * 3


def point(*, reference_safe: bool, material_days: int):
    nominal = days(nominal=0.0)
    nominal = [
        {**day, "daily_voltage_cost": 0.03 if index < material_days else 0.0}
        for index, day in enumerate(nominal)
    ]
    return {
        "nominal_uncontrolled": nominal,
        "voltage_support_reference": days(safe=reference_safe, nominal=0.0),
    }


def full_coarse_report():
    results = {
        str(background): {
            str(pcc): point(reference_safe=True, material_days=0)
            for pcc in PCC_SCALES
        }
        for background in BACKGROUND_SCALES
    }
    return {
        "environment": "env-v5.2-safe",
        "seeds": [30, 31, 32],
        "background_scales": list(BACKGROUND_SCALES),
        "pcc_scales": list(PCC_SCALES),
        "voltage_cost_definition": "sum_bus_limit_violation",
        "parallel_determinism": {"passed": True},
        "results": results,
    }


class EnvV52CalibrationTest(unittest.TestCase):
    def test_coarse_reference_validation_normalizes_json_numeric_keys(self):
        report = full_coarse_report()

        normalized = calibration.validate_coarse_reference(report)

        self.assertIn(0.1, normalized)
        self.assertIn(0.05, normalized[0.1])
        self.assertEqual(sum(len(row) for row in normalized.values()), 200)

    def test_coarse_reference_validation_rejects_metadata_mismatch(self):
        report = full_coarse_report()
        report["seeds"] = [30, 31]

        with self.assertRaisesRegex(ValueError, "seeds"):
            calibration.validate_coarse_reference(report)

    def test_transition_intervals_include_only_changed_adjacent_coarse_points(self):
        results = {
            0.1: {
                0.25: point(reference_safe=True, material_days=0),
                0.30: point(reference_safe=True, material_days=1),
                0.35: point(reference_safe=False, material_days=2),
                0.40: point(reference_safe=False, material_days=3),
            }
        }

        intervals = calibration.find_transition_intervals(results)

        self.assertEqual(
            intervals,
            [
                {
                    "background_load_scale": 0.1,
                    "lower_pcc_scale": 0.3,
                    "upper_pcc_scale": 0.35,
                    "changed": ["reference_safe", "nominal_material_risk"],
                }
            ],
        )

    def test_fine_points_cover_only_interval_interior_at_0005(self):
        intervals = [
            {
                "background_load_scale": 0.1,
                "lower_pcc_scale": 0.30,
                "upper_pcc_scale": 0.35,
                "changed": ["reference_safe"],
            }
        ]

        self.assertEqual(
            calibration.fine_pcc_points(intervals),
            [(0.1, value) for value in (0.305, 0.31, 0.315, 0.32, 0.325, 0.33, 0.335, 0.34, 0.345)],
        )

    def test_merge_accepts_json_keys_and_preserves_coarse_points(self):
        coarse = {"0.1": {"0.3": point(reference_safe=True, material_days=1)}}
        fine = {0.1: {0.305: point(reference_safe=True, material_days=2)}}

        merged = calibration.merge_calibration_results(coarse, fine)

        self.assertEqual(sorted(merged[0.1]), [0.3, 0.305])
        self.assertEqual(
            merged[0.1][0.3]["nominal_uncontrolled"][0]["daily_voltage_cost"],
            0.03,
        )

    def test_feasible_windows_merge_contiguous_passing_fine_points(self):
        results = {0.1: {}}
        for pcc in (0.30, 0.305, 0.31, 0.315, 0.32, 0.325, 0.33):
            passed = 0.305 <= pcc <= 0.325
            results[0.1][pcc] = point(
                reference_safe=passed,
                material_days=2 if passed else 1,
            )

        windows = calibration.find_feasible_windows(results)

        self.assertEqual(len(windows), 1)
        self.assertEqual(windows[0]["background_load_scale"], 0.1)
        self.assertEqual(windows[0]["pcc_scales"], [0.305, 0.31, 0.315, 0.32, 0.325])
        self.assertEqual(windows[0]["window_bounds"], [0.305, 0.325])

    def test_robust_selector_uses_lower_middle_and_requires_neighbors(self):
        results = {0.1: {}}
        for pcc in (0.305, 0.31, 0.315, 0.32, 0.325):
            results[0.1][pcc] = point(reference_safe=True, material_days=2)

        selected = calibration.select_robust_feasible_window(results)

        self.assertEqual(selected["pcc_injection_scale"], 0.315)
        self.assertEqual(selected["window_bounds"], [0.305, 0.325])
        self.assertEqual(selected["neighbor_pcc_scales"], [0.31, 0.32])

        del results[0.1][0.31]
        del results[0.1][0.32]
        self.assertIsNone(calibration.select_robust_feasible_window(results))

    def test_robust_selector_prioritizes_window_upper_bound_then_background(self):
        results = {
            0.9: {
                pcc: point(reference_safe=True, material_days=2)
                for pcc in (0.305, 0.31, 0.315)
            },
            0.1: {
                pcc: point(reference_safe=True, material_days=2)
                for pcc in (0.39, 0.395, 0.4)
            },
        }

        selected = calibration.select_robust_feasible_window(results)

        self.assertEqual(selected["background_load_scale"], 0.1)
        self.assertEqual(selected["pcc_injection_scale"], 0.395)

        results[0.9] = {
            pcc: point(reference_safe=True, material_days=2)
            for pcc in (0.39, 0.395, 0.4)
        }
        selected = calibration.select_robust_feasible_window(results)
        self.assertEqual(selected["background_load_scale"], 0.9)

    def test_calibrate_reuses_coarse_grid_and_evaluates_only_fine_points(self):
        report = full_coarse_report()
        report["results"]["0.1"]["0.3"] = point(
            reference_safe=True, material_days=1
        )
        for pcc in PCC_SCALES:
            if pcc >= 0.35:
                report["results"]["0.1"][str(pcc)] = point(
                    reference_safe=False, material_days=2
                )
        evaluated = []

        def evaluate_fine(background, pcc, workers):
            evaluated.append((background, pcc))
            passed = background == 0.1 and 0.305 <= pcc <= 0.325
            return point(
                reference_safe=passed,
                material_days=2 if passed else 1,
            )

        with mock.patch.object(calibration, "_evaluate_combo", side_effect=evaluate_fine):
            refined = calibration.calibrate(workers=1, coarse_reference=report)

        self.assertEqual(
            evaluated,
            [(0.1, value) for value in (0.305, 0.31, 0.315, 0.32, 0.325, 0.33, 0.335, 0.34, 0.345)],
        )
        self.assertEqual(refined["selection"]["pcc_injection_scale"], 0.315)
        self.assertEqual(refined["coarse_source"], "in-memory")
        self.assertEqual(refined["fine_pcc_step"], 0.005)

    def test_nominal_action_has_half_electrolyzer_and_no_planned_order(self):
        action = nominal_uncontrolled_action(agent_count=4, action_dim=7)
        np.testing.assert_allclose(action[:, 0], 0.0)
        np.testing.assert_allclose(action[:, 1], 0.0)
        np.testing.assert_allclose(action[:, 4], 0.0)
        np.testing.assert_allclose(action[:, 5], -1.0)
        np.testing.assert_allclose(action[:, 2:5], 0.0)
        np.testing.assert_allclose(action[:, 6], 0.0)

    def test_selector_prioritizes_pcc_then_background_and_requires_risk(self):
        report = {
            0.6: {
                0.4: {
                    "nominal_uncontrolled": days(),
                    "voltage_support_reference": days(safe=True),
                },
                0.5: {
                    "nominal_uncontrolled": days(),
                    "voltage_support_reference": days(safe=False),
                },
            },
            0.5: {
                0.9: {
                    "nominal_uncontrolled": days(),
                    "voltage_support_reference": days(safe=True),
                },
            },
        }
        selected = select_largest_feasible_scale(report)
        self.assertEqual(selected["pcc_injection_scale"], 0.9)
        self.assertEqual(selected["background_load_scale"], 0.5)
        self.assertAlmostEqual(selected["nominal_max_daily_cost"], 0.03)

    def test_selector_rejects_nominal_without_two_material_risk_days(self):
        nominal = days()
        nominal[0] = {**nominal[0], "daily_voltage_cost": 0.019}
        nominal[1] = {**nominal[1], "daily_voltage_cost": 0.0}
        report = {
            0.1: {
                1.0: {
                    "nominal_uncontrolled": nominal,
                    "voltage_support_reference": days(safe=True),
                }
            }
        }
        self.assertIsNone(select_largest_feasible_scale(report))

    def test_reference_preview_is_deterministic_and_does_not_mutate_environment(self):
        env = MicrogridEnv(
            {
                "profile_source": "synthetic",
                "italian_split_enable": False,
                "episode_length": 4,
                "power_flow_enable": False,
                "h2_learnable_rolling_order_enable": True,
                "h2_learnable_rolling_order_active": True,
                "soc_init": 0.5,
            }
        )
        env.seed(30)
        env.reset()
        before = {
            "t": env.t,
            "soc": env.soc.copy(),
            "h2_level": env.h2_level.copy(),
            "pending": list(env.pending_h2_deliveries),
            "episode_total_cost": env.episode_total_cost,
        }

        first = voltage_support_reference_action(env)
        second = voltage_support_reference_action(env)

        np.testing.assert_allclose(first, second)
        self.assertEqual(env.t, before["t"])
        np.testing.assert_allclose(env.soc, before["soc"])
        np.testing.assert_allclose(env.h2_level, before["h2_level"])
        self.assertEqual(env.pending_h2_deliveries, before["pending"])
        self.assertEqual(env.episode_total_cost, before["episode_total_cost"])


if __name__ == "__main__":
    unittest.main()
