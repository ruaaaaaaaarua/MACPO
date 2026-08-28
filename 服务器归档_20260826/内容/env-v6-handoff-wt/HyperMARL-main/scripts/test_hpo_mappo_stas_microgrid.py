import unittest

import numpy as np

from hpo_mappo_stas_microgrid import (
    STAGE1_TIMESTEPS,
    TrialSpec,
    build_planned_trials,
    score_returns,
)


class HpoMappoStasMicrogridTest(unittest.TestCase):
    def test_planned_trial_counts_match_staged_protocol(self):
        trials = build_planned_trials()
        counts = {}
        for trial in trials:
            counts[(trial.stage, trial.algorithm)] = (
                counts.get((trial.stage, trial.algorithm), 0) + 1
            )

        self.assertNotIn(("stage0", "mappo"), counts)
        self.assertEqual(counts[("stage0", "stas")], 1)
        self.assertNotIn(("stage1", "mappo"), counts)
        self.assertEqual(counts[("stage1", "stas")], 32)
        self.assertTrue(
            all(trial.total_timesteps == STAGE1_TIMESTEPS for trial in trials)
        )

    def test_score_returns_uses_episode_equivalent_window(self):
        returns = np.arange(2500, dtype=np.float64)
        mappo = TrialSpec(
            stage="stage1",
            algorithm="mappo",
            trial_id="mappo_test",
            total_timesteps=48_000,
            overrides={},
        )
        matd3_style = TrialSpec(
            stage="stage1",
            algorithm="matd3",
            trial_id="matd3_test",
            total_timesteps=48_000,
            overrides={},
            episode_stride=1,
        )

        mappo_score = score_returns(returns, mappo)
        matd3_score = score_returns(returns, matd3_style)

        self.assertEqual(mappo_score["window_points"], 125)
        self.assertEqual(matd3_score["window_points"], 500)
        self.assertAlmostEqual(mappo_score["score"], np.mean(returns[-125:]))
        self.assertAlmostEqual(matd3_score["score"], np.mean(returns[-500:]))


if __name__ == "__main__":
    unittest.main()
