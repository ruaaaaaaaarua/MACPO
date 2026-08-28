import sys
import unittest
from pathlib import Path

import numpy as np
import torch


STAS_ROOT = Path(__file__).resolve().parents[1] / "baselines" / "STAS-MAPPO"
if str(STAS_ROOT) not in sys.path:
    sys.path.insert(0, str(STAS_ROOT))

from stas_mappo.credit import (  # noqa: E402
    CreditQualityGate,
    discounted_team_return,
    project_discounted_credits,
)
from stas_mappo.reward_model import STASRewardModel  # noqa: E402
from stas_mappo.conserved_credit import ConservedSTASCreditAssigner  # noqa: E402
from mappo_stas import _credit_diagnostics, _make_credit_assigner  # noqa: E402


class DiscountedCreditConservationTest(unittest.TestCase):
    def test_training_entry_selects_conserved_bidirectional_assigner(self):
        config = {
            "NUM_STEPS": 6,
            "GAMMA": 0.99,
            "STAS": {
                "CONSERVE_DISCOUNTED": True,
                "QUALITY_GATE_ENABLE": True,
                "BIDIRECTIONAL": True,
                "WARMUP_EPISODES": 11,
                "RAMP_EPISODES": 23,
                "MAX_MIX_COEF": 0.1,
                "EXPLAINED_VARIANCE_THRESHOLD": 0.2,
                "NEGATIVE_PATIENCE": 3,
            },
        }
        assigner = _make_credit_assigner(
            config, raw_obs_dim=5, action_dim=2, num_agents=4
        )
        self.assertIsInstance(assigner, ConservedSTASCreditAssigner)
        self.assertFalse(assigner.config.causal)
        self.assertTrue(assigner.config.quality_gate_enable)
        self.assertEqual(assigner.config.warmup_episodes, 11)

    def test_conserved_diagnostics_report_effective_gate_values(self):
        class Assigner:
            last_mix_coef = 0.075
            last_explained_variance = 0.35
            last_conservation_error = 8e-6
            episodes_seen = 1200

            class gate:
                disabled = False

            class config:
                mix_coef = 1.0

        diagnostics = _credit_diagnostics(Assigner())
        self.assertEqual(
            diagnostics,
            {
                "stas_mix_coef": 0.075,
                "stas_explained_variance": 0.35,
                "stas_conservation_error": 8e-6,
                "stas_episodes_seen": 1200,
                "stas_gate_disabled": False,
            },
        )

    def test_projection_exactly_preserves_discounted_team_return(self):
        rng = np.random.default_rng(4)
        rewards = rng.normal(size=(3, 4, 24)).astype(np.float32)
        raw_credit = rng.normal(size=(3, 4, 24)).astype(np.float32)
        targets = discounted_team_return(rewards, gamma=0.99)
        projected, errors = project_discounted_credits(
            raw_credit, targets, gamma=0.99
        )
        actual = discounted_team_return(projected, gamma=0.99)
        np.testing.assert_allclose(actual, targets, rtol=0.0, atol=1e-4)
        self.assertLess(float(np.max(np.abs(errors))), 1e-4)

    def test_quality_gate_warms_up_then_ramps_to_point_one(self):
        gate = CreditQualityGate(
            warmup_episodes=2000,
            ramp_episodes=8000,
            max_mix_coef=0.1,
            explained_variance_threshold=0.2,
            negative_patience=3,
        )
        self.assertEqual(gate.mix_coef(1999, explained_variance=1.0), 0.0)
        self.assertEqual(gate.mix_coef(6000, explained_variance=0.1), 0.0)
        self.assertAlmostEqual(
            gate.mix_coef(6000, explained_variance=0.3), 0.05, places=6
        )
        self.assertAlmostEqual(
            gate.mix_coef(10000, explained_variance=0.3), 0.1, places=6
        )

    def test_negative_quality_suppresses_credit_without_permanent_disable(self):
        gate = CreditQualityGate(2000, 8000, 0.1, 0.2, 3)
        for _ in range(3):
            gate.mix_coef(10000, explained_variance=-0.1)
        self.assertFalse(gate.disabled)
        self.assertEqual(gate.mix_coef(10000, explained_variance=-0.1), 0.0)
        self.assertAlmostEqual(
            gate.mix_coef(10000, explained_variance=1.0), 0.1, places=6
        )
        self.assertEqual(gate.negative_streak, 0)

    def test_negative_quality_during_warmup_and_ramp_does_not_disable_credit(self):
        gate = CreditQualityGate(200, 800, 0.1, 0.2, 3)
        for episode in range(4, 1000, 4):
            gate.mix_coef(episode, explained_variance=-0.5)
        self.assertFalse(gate.disabled)
        self.assertEqual(gate.negative_streak, 0)

    def test_negative_quality_after_ramp_tracks_streak_without_disabling(self):
        gate = CreditQualityGate(200, 800, 0.1, 0.2, 3)
        gate.mix_coef(1000, explained_variance=-0.5)
        gate.mix_coef(1004, explained_variance=-0.5)
        self.assertFalse(gate.disabled)
        gate.mix_coef(1008, explained_variance=-0.5)
        self.assertFalse(gate.disabled)
        self.assertEqual(gate.negative_streak, 3)
        self.assertAlmostEqual(
            gate.mix_coef(1012, explained_variance=0.3), 0.1, places=6
        )
        self.assertEqual(gate.negative_streak, 0)

    def test_bidirectional_reward_model_runs_without_causal_mask(self):
        model = STASRewardModel(
            obs_dim=5,
            action_dim=2,
            n_agents=4,
            seq_length=6,
            emb_dim=16,
            n_heads=4,
            n_layers=1,
            sample_num=1,
            dropout=0.0,
            causal=False,
        )
        output = model(
            torch.zeros(2, 4, 6, 5),
            torch.zeros(2, 4, 6, 2),
            torch.zeros(2, 4, 6),
            torch.zeros(2, 4, 6),
        )
        self.assertEqual(tuple(output.shape), (2, 4, 6))


if __name__ == "__main__":
    unittest.main()
