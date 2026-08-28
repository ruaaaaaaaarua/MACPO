import unittest

import numpy as np

from envs.microgrid.microgrid_env import MicrogridEnv
from scripts.env_v3_safe_overrides import env_v3_safe_overrides


class EnvV3SafeOverridesTest(unittest.TestCase):
    def test_safe_overrides_define_a_dense_single_day_safety_line(self):
        overrides = env_v3_safe_overrides()

        self.assertEqual(overrides["episode_length"], 24)
        self.assertEqual(overrides["reward_emission_mode"], "dense")
        self.assertEqual(overrides["gamma"], 1.0)
        self.assertTrue(overrides["power_flow_enable"])
        self.assertTrue(overrides["terminal_economic_settlement_enable"])
        self.assertFalse(overrides["penalty_enable"])
        self.assertFalse(overrides["action_reg_enable"])
        self.assertFalse(overrides["external_h2_dependency_penalty_enable"])

    def test_safe_overrides_construct_the_power_flow_environment(self):
        env = MicrogridEnv(env_v3_safe_overrides())
        env.seed(30)
        env.reset()
        actions = np.zeros((env.agent_num, env.action_dim), dtype=np.float32)

        _, rewards, dones, infos = env.step(actions)
        info = infos[0]
        self.assertFalse(dones[0])
        self.assertTrue(info["pf_converged"])
        self.assertEqual(len(info["voltages_pu"]), 33)
        self.assertAlmostEqual(
            float(np.mean(rewards)), -info["total_cost"] / env.reward_scale
        )


if __name__ == "__main__":
    unittest.main()
