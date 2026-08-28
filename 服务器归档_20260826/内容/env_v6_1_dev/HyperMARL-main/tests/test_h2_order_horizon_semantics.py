import unittest

import numpy as np

from envs.microgrid.microgrid_env import MicrogridEnv


class H2OrderHorizonSemanticsTest(unittest.TestCase):
    """设计规格 2a: 订单越界语义 free_cancel(历史) vs pay_and_lose(真实承诺)。"""

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
            "h2_cap_aware_buy_enable": False,
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
        env.seed(11)
        env.reset()
        for key in ("pv", "wt", "load_e", "load_h"):
            env.profiles[key] = np.zeros((env.agent_num, env.T), dtype=np.float32)
        return env

    @staticmethod
    def _producer_actions(env):
        actions = np.zeros((env.agent_num, env.action_dim), dtype=np.float32)
        actions[:, 0] = -1.0
        actions[:, 5] = -1.0
        actions[0, 0] = 1.0   # 0 号电解产氢, 形成可售出的物理盈余
        actions[0, 3] = -1.0  # 低价挂卖
        return actions

    def _trade_actions(self, env):
        actions = self._producer_actions(env)
        actions[1, 3] = 1.0   # 1 号高价买
        actions[1, 5] = 1.0   # 满额下单
        return actions

    def _advance_to(self, env, target_t):
        while env.t < target_t:
            env.step(self._producer_actions(env))
        self.assertEqual(env.t, target_t)

    def test_free_cancel_blocks_late_orders(self):
        env = self._make_env()  # 默认 free_cancel
        self.assertEqual(env.h2_order_horizon_clip_mode, "free_cancel")
        self._advance_to(env, 21)  # 21+4=25 > T=24: 永不可交付
        info = env.step(self._trade_actions(env))[3][0]
        self.assertEqual(info["h2_market_traded"], 0.0)
        self.assertGreater(info["h2_buy_horizon_clip_amount"][1], 0.0)
        self.assertEqual(info["h2_late_order_energy"][1], 0.0)
        self.assertEqual(len(env.pending_h2_deliveries), 0)

    def test_pay_and_lose_charges_buyer_and_never_delivers(self):
        env = self._make_env(h2_order_horizon_clip_mode="pay_and_lose")
        self._advance_to(env, 21)
        info = env.step(self._trade_actions(env))[3][0]
        # 订单照常入市并成交: 买方付款、无免费取消。
        self.assertGreater(info["h2_market_traded"], 0.0)
        self.assertGreater(info["h2_cda_paid"][1], 0.0)
        self.assertGreater(info["h2_late_order_energy"][1], 0.0)
        self.assertEqual(info["h2_buy_horizon_clip_amount"][1], 0.0)
        # 交付永不发生: pending 留有 deliver_at > T 的记录直到 episode 结束。
        self.assertEqual(len(env.pending_h2_deliveries), 1)
        self.assertEqual(int(env.pending_h2_deliveries[0]["deliver_at"]), 25)
        buyer_level_before_end = float(env.h2_level[1])
        done = False
        while not done:
            step_out = env.step(self._producer_actions(env))
            done = bool(np.asarray(step_out[2]).reshape(-1)[0])
        self.assertEqual(len(env.pending_h2_deliveries), 1)
        self.assertAlmostEqual(float(env.h2_level[1]), buyer_level_before_end, places=5)

    def test_invalid_mode_rejected(self):
        with self.assertRaisesRegex(ValueError, "h2_order_horizon_clip_mode"):
            self._make_env(h2_order_horizon_clip_mode="cancel_with_fee")


if __name__ == "__main__":
    unittest.main()
