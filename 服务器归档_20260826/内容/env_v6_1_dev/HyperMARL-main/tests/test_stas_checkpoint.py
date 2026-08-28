import sys
import unittest
from pathlib import Path

import numpy as np
import torch


STAS_ROOT = Path(__file__).resolve().parents[1] / "baselines" / "STAS-MAPPO"
if str(STAS_ROOT) not in sys.path:
    sys.path.insert(0, str(STAS_ROOT))

from stas_mappo.checkpoint import (  # noqa: E402
    credit_assigner_state_dict,
    load_credit_assigner_state,
)
from stas_mappo.conserved_credit import ConservedSTASCreditAssigner  # noqa: E402
from stas_mappo.credit import STASCreditConfig  # noqa: E402


class STASTrainingCheckpointTest(unittest.TestCase):
    def test_round_trip_restores_buffers_normalizer_gate_and_optimizer(self):
        config = STASCreditConfig(
            obs_dim=4,
            action_dim=1,
            n_agents=2,
            seq_length=3,
            emb_dim=8,
            n_heads=2,
            n_layers=1,
            sample_num=1,
            dropout=0.0,
            buffer_size=10,
            batch_size=1,
            warmup_rollouts=1,
            conserve_discounted=True,
            warmup_episodes=0,
            ramp_episodes=1,
            negative_patience=3,
        )
        original = ConservedSTASCreditAssigner(config)
        rng = np.random.default_rng(5)
        obs = rng.normal(size=(5, 2, 3, 4)).astype(np.float32)
        actions = rng.normal(size=(5, 2, 3, 1)).astype(np.float32)
        rewards = rng.normal(size=(5, 2, 3)).astype(np.float32)
        dones = np.zeros((5, 2, 3), dtype=np.float32)
        original.add_rollout(obs, actions, rewards, dones)
        original.train_if_ready()
        for _ in range(3):
            original.gate.mix_coef(original.episodes_seen, -0.1)
        original.last_explained_variance = -0.1
        original.last_conservation_error = 9e-6
        original.last_mix_coef = 0.0

        state = credit_assigner_state_dict(original)
        restored = ConservedSTASCreditAssigner(config)
        load_credit_assigner_state(restored, state)

        self.assertEqual(restored.rollouts_seen, original.rollouts_seen)
        self.assertEqual(restored.episodes_seen, original.episodes_seen)
        self.assertEqual(len(restored.buffer), len(original.buffer))
        self.assertEqual(len(restored.holdout_buffer), len(original.holdout_buffer))
        self.assertEqual(restored.normalizer.count, original.normalizer.count)
        self.assertAlmostEqual(restored.normalizer.mean, original.normalizer.mean)
        self.assertEqual(restored.gate.disabled, original.gate.disabled)
        self.assertFalse(restored.gate.disabled)
        self.assertEqual(restored.gate.negative_streak, 3)
        np.testing.assert_allclose(
            restored.buffer.storage[0][0], original.buffer.storage[0][0]
        )
        for actual, expected in zip(
            restored.model.parameters(), original.model.parameters()
        ):
            torch.testing.assert_close(actual, expected)


if __name__ == "__main__":
    unittest.main()
