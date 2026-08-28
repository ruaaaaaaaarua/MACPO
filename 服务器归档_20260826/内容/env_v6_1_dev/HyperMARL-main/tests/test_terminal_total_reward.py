import unittest

import numpy as np

from envs.microgrid.microgrid_env import MicrogridEnv


class TerminalTotalRewardTest(unittest.TestCase):
    @staticmethod
    def _make_env(mode):
        env = MicrogridEnv(
            {
                "profile_source": "synthetic",
                "italian_split_enable": False,
                "episode_length": 4,
                "reward_emission_mode": mode,
                "penalty_enable": False,
                "low_inventory_penalty_enable": False,
                "terminal_h2_floor_penalty_enable": False,
                "terminal_h2_shortfall_value_enable": False,
                "terminal_h2_settlement_in_reward_enable": False,
                "terminal_soc_floor_penalty_enable": False,
                "terminal_battery_salvage_enable": False,
                "stepwise_h2_floor_penalty_enable": False,
                "action_reg_enable": False,
            }
        )
        env.seed(123)
        env.reset()
        return env

    @staticmethod
    def _step(env, actions):
        _, rewards, dones, infos = env.step(actions)
        return float(np.mean(rewards)), bool(dones[0]), infos[0]

    def test_dense_sum_equals_single_terminal_total_reward(self):
        dense = self._make_env("dense")
        sparse = self._make_env("terminal_total")
        actions = np.zeros((dense.agent_num, dense.action_dim), dtype=np.float32)
        dense_rewards = []
        sparse_rewards = []
        cumulative_cost = 0.0

        for step in range(dense.T):
            dense_reward, dense_done, dense_info = self._step(dense, actions)
            sparse_reward, sparse_done, sparse_info = self._step(sparse, actions)
            dense_rewards.append(dense_reward)
            sparse_rewards.append(sparse_reward)
            cumulative_cost += dense_info["step_total_cost"]

            self.assertEqual(dense_done, sparse_done)
            self.assertAlmostEqual(
                dense_info["step_total_cost"], sparse_info["step_total_cost"], places=6
            )
            self.assertAlmostEqual(
                sparse_info["episode_total_cost"], cumulative_cost, places=6
            )
            self.assertTrue(dense_info["reward_emitted"])
            self.assertEqual(sparse_info["reward_emitted"], step == dense.T - 1)

        np.testing.assert_allclose(sparse_rewards[:-1], 0.0, rtol=0.0, atol=0.0)
        self.assertAlmostEqual(sum(dense_rewards), sparse_rewards[-1], places=6)
        self.assertAlmostEqual(sum(sparse_rewards), sparse_rewards[-1], places=6)
        self.assertAlmostEqual(
            sparse_rewards[-1], -cumulative_cost / sparse.reward_scale, places=6
        )

    def test_reset_clears_episode_cost_accumulator(self):
        env = self._make_env("terminal_total")
        actions = np.zeros((env.agent_num, env.action_dim), dtype=np.float32)
        reward, done, first = self._step(env, actions)
        self.assertEqual(reward, 0.0)
        self.assertFalse(done)
        self.assertAlmostEqual(
            first["episode_total_cost"], first["step_total_cost"], places=6
        )

        self._step(env, actions)
        env.reset()
        reward, done, after_reset = self._step(env, actions)
        self.assertEqual(reward, 0.0)
        self.assertFalse(done)
        self.assertAlmostEqual(
            after_reset["episode_total_cost"],
            after_reset["step_total_cost"],
            places=6,
        )

    def test_unknown_reward_emission_mode_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "reward_emission_mode"):
            MicrogridEnv({"reward_emission_mode": "hourly_but_weird"})


if __name__ == "__main__":
    unittest.main()
