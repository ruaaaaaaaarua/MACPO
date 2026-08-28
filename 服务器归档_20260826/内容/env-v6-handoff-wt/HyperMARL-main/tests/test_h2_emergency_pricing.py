import unittest

import numpy as np

from envs.microgrid.microgrid_env import MicrogridEnv


class H2EmergencyPricingTest(unittest.TestCase):
    """设计规格 2c: 应急外购乘数只作用于当小时未满足负荷的瞬时平衡采购。"""

    def _make_env(self, **overrides):
        config = {
            "profile_source": "synthetic",
            "italian_split_enable": False,
            "h2_learnable_rolling_order_enable": True,
            "h2_learnable_rolling_order_active": True,
            "h2_buyer_reservation_demand_enable": False,
            "h2_market_schedule_enable": False,
            "h2_market_lag_enable": True,
            "h2_delivery_lag": 4,
            "h2_pending_obs_enable": True,
            "h2_pending_obs_horizon": 4,
            "h2_pending_summary_obs_enable": True,
            "h2_cap_aware_buy_enable": True,
            "h2_delivery_reservation_enable": True,
            "h2_delivery_reservation_horizon": 4,
            "h2_traffic_enable": False,
            "h2_route_action_enable": False,
            "penalty_enable": False,
            "low_inventory_penalty_enable": False,
            "terminal_h2_floor_penalty_enable": False,
            "terminal_h2_shortfall_value_enable": False,
            "terminal_soc_floor_penalty_enable": False,
            "terminal_battery_salvage_enable": False,
            "stepwise_h2_floor_penalty_enable": False,
            "action_reg_enable": False,
        }
        config.update(overrides)
        env = MicrogridEnv(config)
        env.seed(7)
        env.reset()
        for key in ("pv", "wt", "load_e"):
            env.profiles[key] = np.zeros((env.agent_num, env.T), dtype=np.float32)
        # 恒定氢负荷、零产出 -> 每小时必然出现未满足负荷的应急外购。
        env.profiles["load_h"] = np.full(
            (env.agent_num, env.T), 50.0, dtype=np.float32
        )
        return env

    @staticmethod
    def _idle_actions(env):
        actions = np.full((env.agent_num, env.action_dim), -1.0, dtype=np.float32)
        return actions

    def test_default_multiplier_is_one_and_reported(self):
        env = self._make_env()
        self.assertEqual(env.h2_emergency_price_multiplier, 1.0)
        info = env.step(self._idle_actions(env))[3][0]
        self.assertEqual(info["h2_emergency_price_multiplier"], 1.0)
        self.assertGreater(info["h2_emergency_buy_cost"], 0.0)
        np.testing.assert_allclose(
            np.asarray(info["h2_emergency_buy_energy"]),
            np.maximum(np.asarray(info["e_h2_ext"]), 0.0),
            rtol=0.0,
            atol=1e-6,
        )

    def test_multiplier_scales_emergency_cost_without_touching_physics(self):
        env_base = self._make_env()
        env_double = self._make_env(h2_emergency_price_multiplier=2.0)
        for _ in range(6):
            actions = self._idle_actions(env_base)
            ob1, r1, d1, i1 = env_base.step(actions)
            ob2, r2, d2, i2 = env_double.step(actions)
            info1, info2 = i1[0], i2[0]
            # 物理量完全一致: 应急采购能量、库存轨迹不受计价影响。
            np.testing.assert_allclose(
                info1["e_h2_ext"], info2["e_h2_ext"], rtol=0.0, atol=1e-5
            )
            np.testing.assert_allclose(
                info1["h2_level"], info2["h2_level"], rtol=0.0, atol=1e-5
            )
            self.assertGreater(info1["h2_emergency_buy_cost"], 0.0)
            # 应急成本严格按乘数放大。
            self.assertAlmostEqual(
                info2["h2_emergency_buy_cost"],
                2.0 * info1["h2_emergency_buy_cost"],
                places=3,
            )
            # 共享奖励的差额 == 多付的应急成本 / reward_scale。
            reward_scale = float(env_base.cfg.get("reward_scale", 200.0))
            self.assertAlmostEqual(
                float(np.asarray(r1).reshape(-1)[0])
                - float(np.asarray(r2).reshape(-1)[0]),
                info1["h2_emergency_buy_cost"] / reward_scale,
                places=4,
            )

    def test_invalid_multiplier_rejected(self):
        with self.assertRaisesRegex(ValueError, "emergency_price_multiplier"):
            self._make_env(h2_emergency_price_multiplier=0.0)


if __name__ == "__main__":
    unittest.main()
