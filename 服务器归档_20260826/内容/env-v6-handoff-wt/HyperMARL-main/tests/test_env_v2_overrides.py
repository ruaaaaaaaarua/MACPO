import unittest

import numpy as np

from envs.microgrid.microgrid_env import MicrogridEnv
from scripts.env_v2_overrides import env_v2_overrides, hydra_override_arg


class EnvV2OverridesTest(unittest.TestCase):
    """v2 规范配置能构造环境、v2 语义全部生效、Hydra 串机械生成正确。"""

    def test_env_constructs_with_v2_semantics_and_steps(self):
        overrides = env_v2_overrides(sparse=True)
        env = MicrogridEnv(overrides)
        env.seed(3)
        env.reset()
        # v2 语义逐条生效。
        self.assertEqual(env.h2_emergency_price_multiplier, 2.0)
        self.assertEqual(env.h2_order_horizon_clip_mode, "pay_and_lose")
        self.assertFalse(env.h2_cap_aware_buy_enable)
        self.assertTrue(env.h2_planned_external_order_enable)
        self.assertEqual(env.h2_delivery_reservation_ratio, 0.5)
        net = env.h2_transport_network
        self.assertTrue(net.external_node_enable)
        self.assertEqual((net.min_eta, net.max_eta), (4, 10))
        self.assertEqual(net.truck_capacity_kg, 100.0)
        self.assertEqual(net.edge_capacity, 2.5)
        self.assertAlmostEqual(net.transit_loss_per_hour, 0.008)
        # 能完整走一个 episode 不炸, 且稀疏模式只在终端给奖励。
        rewards_seen = []
        done = False
        while not done:
            actions = np.full((env.agent_num, env.action_dim), -1.0, dtype=np.float32)
            _, rewards, dones, _ = env.step(actions)
            rewards_seen.append(float(np.asarray(rewards).reshape(-1)[0]))
            done = bool(np.asarray(dones).reshape(-1)[0])
        self.assertEqual(len(rewards_seen), 24)
        self.assertTrue(all(r == 0.0 for r in rewards_seen[:-1]))
        self.assertLess(rewards_seen[-1], 0.0)

    def test_dense_variant_drops_terminal_emission(self):
        overrides = env_v2_overrides(sparse=False)
        self.assertNotIn("reward_emission_mode", overrides)
        self.assertNotIn("gamma", overrides)

    def test_hydra_arg_round_trips_key_values(self):
        overrides = env_v2_overrides(sparse=True)
        arg = hydra_override_arg(overrides)
        self.assertTrue(arg.startswith("+MICROGRID_CONFIG_OVERRIDES={"))
        self.assertTrue(arg.endswith("}"))
        self.assertIn("h2_emergency_price_multiplier:2.0", arg)
        self.assertIn("h2_order_horizon_clip_mode:pay_and_lose", arg)
        self.assertIn("h2_traffic_external_node_enable:true", arg)
        self.assertIn("pv_cap:[7500.0,1500.0,500.0,2000.0]", arg)
        self.assertEqual(arg.count("{"), 1)
        self.assertEqual(arg.count("}"), 1)


if __name__ == "__main__":
    unittest.main()
