import unittest

import numpy as np

from baselines.utils.rule_baselines import (
    RULE_BASELINES,
    make_base_stock_rule,
)


def _config(**overrides):
    config = {
        "num_agents": 4,
        "episode_length": 24,
        "dt": 1.0,
        "boiler_eff": 1.0,
        "pv_cap": [1.0, 1.0, 1.0, 1.0],
        "wt_cap": [1.0, 1.0, 1.0, 1.0],
        "load_e_peak": [1.0, 1.0, 1.0, 1.0],
        "load_h_peak": [100.0, 50.0, 50.0, 50.0],
        "el_cap": [10.0, 10.0, 10.0, 10.0],
        "el_eff": [0.7, 0.7, 0.7, 0.7],
        "h2_tank_cap": [30.0, 30.0, 30.0, 30.0],
        "LHV_H2": 33.33,
        "h2_delivery_lag": 4,
        "h2_traffic_enable": False,
        "h2_route_action_enable": False,
        "h2_action_order_max_peak_hours": 1.0,
    }
    config.update(overrides)
    return config


def _obs(load_h_norm=(1.0, 0.0, 0.0, 0.0), tank_ratio=(0.0, 0.0, 0.0, 0.0)):
    obs = np.zeros((4, 13), dtype=np.float32)
    obs[:, 3] = np.asarray(load_h_norm, dtype=np.float32)
    obs[:, 5] = np.asarray(tank_ratio, dtype=np.float32)
    return obs


def _decode_order(action, qmax):
    return (np.asarray(action)[:, 5] + 1.0) / 2.0 * np.asarray(qmax)


class BaseStockRuleTest(unittest.TestCase):
    """设计规格 3a/3b: 订单台账 + 目标库存位置, 修复重复订货病根。"""

    def test_full_inventory_means_no_order(self):
        rule = make_base_stock_rule(safety_hours=2.0, target_mult=1.0)
        config = _config()
        obs = _obs(tank_ratio=(1.0, 1.0, 1.0, 1.0))  # 满罐 30kg*33.33 ≈ 1000 kWh
        action = rule(obs, {"config": config, "episode_step": 0})
        np.testing.assert_allclose(action[:, 5], -1.0, atol=1e-6)

    def test_ledger_stops_reordering_once_pipeline_covers_target(self):
        # 缺口 100/h, 窗口 = lag4+safety0 = 4h, mult 0.375 -> 目标 150。
        rule = make_base_stock_rule(safety_hours=0.0, target_mult=0.375)
        config = _config()
        obs = _obs()  # 空罐、恒定缺口 (obs 冻结以隔离台账效应)
        qmax = np.array([100.0, 50.0, 50.0, 50.0])
        expected_agent0 = [100.0, 50.0, 0.0, 0.0]
        for step, expected in enumerate(expected_agent0):
            action = rule(obs, {"config": config, "episode_step": step})
            orders = _decode_order(action, qmax)
            self.assertAlmostEqual(float(orders[0]), expected, places=4)
        # 无缺口的 agent 全程不订货。
        self.assertAlmostEqual(float(orders[1]), 0.0, places=4)

    def test_new_episode_resets_ledger(self):
        rule = make_base_stock_rule(safety_hours=0.0, target_mult=0.375)
        config = _config()
        obs = _obs()
        qmax = np.array([100.0, 50.0, 50.0, 50.0])
        for step in range(3):
            rule(obs, {"config": config, "episode_step": step})
        # 新 episode 从零计步: 台账清空, 首单回到满额。
        action = rule(obs, {"config": config, "episode_step": 0})
        self.assertAlmostEqual(float(_decode_order(action, qmax)[0]), 100.0, places=4)

    def test_privileged_forecast_sees_future_load_only(self):
        config = _config()
        profiles = {
            "pv": np.zeros((4, 24), dtype=np.float32),
            "wt": np.zeros((4, 24), dtype=np.float32),
            "load_e": np.zeros((4, 24), dtype=np.float32),
            "load_h": np.zeros((4, 24), dtype=np.float32),
        }
        profiles["load_h"][0, 3:9] = 80.0  # 只有未来几小时有负荷
        obs = _obs(load_h_norm=(0.0, 0.0, 0.0, 0.0))  # 当前零缺口
        context = {"config": config, "episode_step": 0, "profiles": profiles}
        plain = make_base_stock_rule(safety_hours=2.0)(obs, dict(context))
        privileged = make_base_stock_rule(
            safety_hours=2.0, privileged_forecast=True
        )(obs, dict(context))
        qmax = np.array([100.0, 50.0, 50.0, 50.0])
        self.assertAlmostEqual(float(_decode_order(plain, qmax)[0]), 0.0, places=4)
        self.assertGreater(float(_decode_order(privileged, qmax)[0]), 0.0)

    def test_registry_exposes_rules_with_privilege_flags(self):
        self.assertIn("base_stock_rule", RULE_BASELINES)
        self.assertIn("base_stock_privileged", RULE_BASELINES)
        self.assertFalse(RULE_BASELINES["base_stock_rule"].privileged_diagnostic)
        self.assertTrue(RULE_BASELINES["base_stock_privileged"].privileged_diagnostic)


if __name__ == "__main__":
    unittest.main()
