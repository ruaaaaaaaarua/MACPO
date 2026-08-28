import sys
import unittest
from pathlib import Path

import numpy as np


STAS_ROOT = Path(__file__).resolve().parents[1] / "baselines" / "STAS-MAPPO"
if str(STAS_ROOT) not in sys.path:
    sys.path.insert(0, str(STAS_ROOT))

from stas_mappo.credit import EpisodeCreditBuffer  # noqa: E402
from stas_mappo.credit_conservation import discounted_team_return  # noqa: E402


class EpisodeCreditBufferOwnershipTest(unittest.TestCase):
    def test_add_takes_ownership_of_all_array_inputs(self):
        obs = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
        actions = np.arange(12, dtype=np.int64).reshape(2, 3, 2)
        rewards = np.arange(6, dtype=np.float64).reshape(2, 3)
        dones = np.array(
            [[False, False, True], [False, True, True]], dtype=np.bool_
        )
        original = tuple(array.copy() for array in (obs, actions, rewards, dones))
        buffer = EpisodeCreditBuffer(capacity=2)

        buffer.add(obs, actions, rewards, dones, np.float32(7.5))
        obs.fill(-100.0)
        actions.fill(-101)
        rewards.fill(-102.0)
        dones[...] = ~dones

        stored = buffer.storage[0]
        for stored_array, original_array in zip(stored[:4], original):
            np.testing.assert_array_equal(stored_array, original_array)
            self.assertEqual(stored_array.dtype, original_array.dtype)
            self.assertEqual(stored_array.shape, original_array.shape)
        self.assertIs(type(stored[4]), float)

    def test_stored_target_remains_consistent_after_caller_mutation(self):
        gamma = 0.97
        obs = np.zeros((2, 3, 4), dtype=np.float32)
        actions = np.zeros((2, 3, 2), dtype=np.float32)
        rewards = np.array(
            [[1.0, 2.0, 3.0], [-0.5, 0.25, 1.5]], dtype=np.float32
        )
        dones = np.zeros((2, 3), dtype=np.float32)
        target = discounted_team_return(rewards[None, ...], gamma)[0]
        buffer = EpisodeCreditBuffer(capacity=1)

        buffer.add(obs, actions, rewards, dones, target)
        obs.fill(11.0)
        actions.fill(12.0)
        rewards.fill(13.0)
        dones.fill(1.0)

        stored_rewards = buffer.storage[0][2]
        stored_target = buffer.storage[0][4]
        recomputed_target = discounted_team_return(
            stored_rewards[None, ...], gamma
        )[0]
        self.assertAlmostEqual(stored_target, recomputed_target, delta=1e-12)


if __name__ == "__main__":
    unittest.main()
