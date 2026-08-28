import unittest

import numpy as np
import torch

from baselines.MATD3.matd3 import ReplayBuffer
from baselines.MATD3.train_matd3_microgrid import should_update


class MATD3ReplayCheckpointTest(unittest.TestCase):
    def test_update_every_six_matches_actor_budget_schedule(self):
        selected = [
            step
            for step in range(1000, 1061)
            if should_update(step, replay_size=256, batch_size=256,
                             update_after=1000, update_every=6)
        ]
        self.assertEqual(selected, list(range(1002, 1061, 6)))

    def test_update_every_one_preserves_existing_schedule(self):
        selected = [
            step
            for step in range(998, 1003)
            if should_update(step, replay_size=256, batch_size=256,
                             update_after=1000, update_every=1)
        ]
        self.assertEqual(selected, [1000, 1001, 1002])

    def test_update_every_must_be_positive(self):
        with self.assertRaisesRegex(ValueError, "update_every"):
            should_update(1000, replay_size=256, batch_size=256,
                          update_after=1000, update_every=0)

    @staticmethod
    def _transition(value):
        obs = np.full((2, 3), value, dtype=np.float32)
        state = obs.reshape(-1)
        action = np.full((2, 1), value / 10.0, dtype=np.float32)
        reward = float(value)
        next_obs = obs + 1.0
        return obs, state, action, reward, next_obs, next_obs.reshape(-1), False

    def test_round_trip_restores_ring_contents_pointer_and_rng(self):
        original = ReplayBuffer(
            obs_dim=3,
            state_dim=6,
            action_dim=1,
            num_agents=2,
            capacity=3,
            seed=19,
        )
        for value in range(5):
            original.add(*self._transition(value))

        state = original.state_dict()
        restored = ReplayBuffer(
            obs_dim=3,
            state_dim=6,
            action_dim=1,
            num_agents=2,
            capacity=3,
            seed=999,
        )
        restored.load_state_dict(state)

        self.assertEqual(restored.size, original.size)
        self.assertEqual(restored.ptr, original.ptr)
        for name in (
            "obs",
            "states",
            "actions",
            "rewards",
            "next_obs",
            "next_states",
            "dones",
        ):
            np.testing.assert_array_equal(getattr(restored, name), getattr(original, name))
        expected = original.sample(8, torch.device("cpu"))
        actual = restored.sample(8, torch.device("cpu"))
        for name in expected:
            torch.testing.assert_close(actual[name], expected[name])

    def test_capacity_mismatch_is_rejected(self):
        original = ReplayBuffer(3, 6, 1, 2, capacity=3, seed=1)
        state = original.state_dict()
        incompatible = ReplayBuffer(3, 6, 1, 2, capacity=4, seed=1)
        with self.assertRaisesRegex(ValueError, "capacity"):
            incompatible.load_state_dict(state)


if __name__ == "__main__":
    unittest.main()
