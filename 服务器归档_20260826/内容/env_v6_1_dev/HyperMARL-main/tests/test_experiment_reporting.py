import unittest

from baselines.utils.experiment_reporting import compute_curve_metrics


class ExperimentReportingTest(unittest.TestCase):
    def test_final_score_uses_exact_last_three_gate_points(self):
        records = [
            {"training_episode": episode, "summary": {"return_mean": value}}
            for episode, value in (
                (500, -500.0),
                (29000, -300.0),
                (29500, -290.0),
                (30000, -280.0),
            )
        ]
        metrics = compute_curve_metrics(records, final_episode=30000)
        self.assertEqual(metrics["final_score"], -290.0)
        self.assertEqual(metrics["final_points"], [29000, 29500, 30000])

    def test_auc_is_normalized_over_zero_to_final_episode(self):
        records = [
            {"training_episode": 0, "summary": {"return_mean": 0.0}},
            {"training_episode": 15000, "summary": {"return_mean": 1.0}},
            {"training_episode": 29000, "summary": {"return_mean": 2.0}},
            {"training_episode": 29500, "summary": {"return_mean": 2.0}},
            {"training_episode": 30000, "summary": {"return_mean": 2.0}},
        ]
        metrics = compute_curve_metrics(records, final_episode=30000)
        expected = (0.5 * 15000 + 1.5 * 14000 + 2.0 * 1000) / 30000
        self.assertAlmostEqual(metrics["normalized_auc"], expected)

    def test_missing_gate_point_is_rejected(self):
        records = [
            {"training_episode": 29000, "summary": {"return_mean": 1.0}},
            {"training_episode": 30000, "summary": {"return_mean": 2.0}},
        ]
        with self.assertRaisesRegex(ValueError, "29500"):
            compute_curve_metrics(records, final_episode=30000)


if __name__ == "__main__":
    unittest.main()
