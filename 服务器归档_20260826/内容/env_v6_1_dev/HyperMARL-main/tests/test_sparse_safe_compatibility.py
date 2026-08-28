import unittest

import numpy as np

from envs.microgrid.microgrid_env import MicrogridEnv
from scripts.env_v2_overrides import env_v2_overrides


class SparseSafeCompatibilityTest(unittest.TestCase):
    def test_frozen_v2_sparse_accounting_and_interface_are_unchanged(self):
        sparse = MicrogridEnv(env_v2_overrides(sparse=True))
        dense = MicrogridEnv(env_v2_overrides(sparse=False))
        sparse.seed(30)
        dense.seed(30)
        sparse.reset()
        dense.reset()
        self.assertFalse(sparse.power_flow_enable)
        self.assertFalse(sparse.terminal_economic_settlement_enable)
        self.assertEqual(sparse.action_dim, dense.action_dim)

        actions = np.full((sparse.agent_num, sparse.action_dim), -1.0, dtype=np.float32)
        sparse_rewards = []
        dense_total_cost = 0.0
        for step in range(sparse.T):
            _, sparse_step_rewards, sparse_dones, sparse_infos = sparse.step(actions)
            _, _, dense_dones, dense_infos = dense.step(actions)
            sparse_rewards.append(float(np.mean(sparse_step_rewards)))
            dense_total_cost += dense_infos[0]["step_total_cost"]
            self.assertEqual(bool(sparse_dones[0]), bool(dense_dones[0]))
            self.assertEqual(sparse_infos[0]["voltage_cost"], 0.0)
            self.assertFalse(sparse_infos[0]["pf_converged"])

        np.testing.assert_allclose(sparse_rewards[:-1], 0.0, rtol=0.0, atol=0.0)
        self.assertAlmostEqual(
            sparse_rewards[-1], -dense_total_cost / sparse.reward_scale, places=5
        )


if __name__ == "__main__":
    unittest.main()
