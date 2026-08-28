import unittest

import numpy as np

from envs.microgrid.h2_transport import H2TransportNetwork
from envs.microgrid.microgrid_env import MicrogridEnv


def _v2_network(**overrides):
    config = {
        "num_agents": 4,
        "h2_traffic_external_node_enable": True,
        "h2_traffic_eta_min": 4,
        "h2_traffic_eta_max": 10,
        "h2_traffic_seed": 20260717,
    }
    config.update(overrides)
    return H2TransportNetwork(config)


class H2TransportV2NetworkTest(unittest.TestCase):
    """设计规格 1a/1c/1d: EXT 节点、放宽 ETA、按小时在途损耗。"""

    def test_external_node_extends_graph_without_touching_mg_routes(self):
        net = _v2_network()
        self.assertEqual(net.num_route_nodes, 5)
        self.assertEqual(net.external_node_id, 4)
        self.assertIn((4, 1), net.edge_ids)
        # EXT -> buyer 的三条候选: 直达 + 两条微电网节点绕行。
        self.assertEqual(
            net.route_options(4, 1), ((4, 1), (4, 0, 1), (4, 2, 1))
        )
        # 微电网对之间的路由与 v1 完全一致, EXT 永不作为中间节点。
        v1 = H2TransportNetwork({"num_agents": 4, "h2_traffic_seed": 20260717})
        for seller in range(4):
            for buyer in range(4):
                if seller == buyer:
                    continue
                self.assertEqual(
                    net.route_options(seller, buyer),
                    v1.route_options(seller, buyer),
                )

    def test_v2_eta_bounds_validated(self):
        net = _v2_network()
        self.assertEqual((net.min_eta, net.max_eta), (4, 10))
        with self.assertRaisesRegex(ValueError, "v2 ETA"):
            _v2_network(h2_traffic_eta_min=0)
        with self.assertRaisesRegex(ValueError, "v2 ETA"):
            _v2_network(h2_traffic_eta_min=6, h2_traffic_eta_max=6)

    def test_transit_loss_accumulates_per_hour_of_eta(self):
        net = _v2_network(h2_traffic_transit_loss_per_hour=0.01)
        shipments = net.assign_shipments(
            [{"seller_id": 4, "buyer_id": 1, "quantity": 1000.0, "price": 1.0}],
            route_actions=[-1.0, -1.0, -1.0, -1.0],
            dispatch_t=3,
        )
        self.assertEqual(len(shipments), 1)
        shipment = shipments[0]
        eta = int(shipment["eta"])
        self.assertGreaterEqual(eta, 4)
        self.assertLessEqual(eta, 10)
        expected_loss = 1000.0 * min(1.0, 0.01 * eta)
        self.assertAlmostEqual(shipment["loss_quantity"], expected_loss, places=5)
        self.assertAlmostEqual(
            shipment["net_quantity"], 1000.0 - expected_loss, places=5
        )

    def test_invalid_transit_loss_rejected(self):
        with self.assertRaisesRegex(ValueError, "transit_loss_per_hour"):
            _v2_network(h2_traffic_transit_loss_per_hour=1.0)


class H2PlannedExternalOrderTest(unittest.TestCase):
    """设计规格 1a: 计划性外购通道 —— 未撮合剩余量按计划价付款、延迟交付。"""

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
            "h2_planned_external_order_enable": True,
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
        env.seed(13)
        env.reset()
        for key in ("pv", "wt", "load_e", "load_h"):
            env.profiles[key] = np.zeros((env.agent_num, env.T), dtype=np.float32)
        return env

    @staticmethod
    def _idle_actions(env):
        return np.full((env.agent_num, env.action_dim), -1.0, dtype=np.float32)

    def _buy_actions(self, env, route=-1.0):
        actions = self._idle_actions(env)
        actions[1, 3] = 1.0
        actions[1, 5] = 1.0
        if env.action_dim >= 7:
            actions[1, 6] = route
        return actions

    def test_unmatched_buy_becomes_planned_external_order_fixed_lag(self):
        env = self._make_env()
        env.step(self._idle_actions(env))
        env.step(self._idle_actions(env))
        info = env.step(self._buy_actions(env))[3][0]
        qty = info["h2_planned_external_order_energy"][1]
        self.assertGreater(qty, 0.0)
        self.assertAlmostEqual(
            info["h2_planned_external_order_cost"],
            qty * float(env.h2_external_buy_prices[1]),
            places=3,
        )
        self.assertEqual(len(env.pending_h2_deliveries), 1)
        record = env.pending_h2_deliveries[0]
        self.assertEqual(int(record["buyer_id"]), 1)
        self.assertEqual(int(record["seller_id"]), -1)
        self.assertEqual(int(record["deliver_at"]), 2 + env.h2_delivery_lag)
        level_before = float(env.h2_level[1])
        for _ in range(env.h2_delivery_lag):
            env.step(self._idle_actions(env))
        self.assertGreater(float(env.h2_level[1]), level_before)
        self.assertEqual(len(env.pending_h2_deliveries), 0)

    def test_disabled_channel_preserves_v1_behavior(self):
        env = self._make_env(h2_planned_external_order_enable=False)
        env.step(self._idle_actions(env))
        env.step(self._idle_actions(env))
        info = env.step(self._buy_actions(env))[3][0]
        self.assertEqual(info["h2_planned_external_order_energy"][1], 0.0)
        self.assertEqual(info["h2_planned_external_order_cost"], 0.0)
        self.assertEqual(len(env.pending_h2_deliveries), 0)

    def test_planned_orders_ride_the_network_from_ext_node(self):
        env = self._make_env(
            h2_traffic_enable=True,
            h2_route_action_enable=True,
            h2_traffic_external_node_enable=True,
            h2_traffic_eta_min=4,
            h2_traffic_eta_max=10,
            h2_traffic_seed=20260717,
        )
        env.step(self._idle_actions(env))
        env.step(self._idle_actions(env))
        info = env.step(self._buy_actions(env, route=-1.0))[3][0]
        self.assertGreater(info["h2_planned_external_order_energy"][1], 0.0)
        self.assertEqual(len(env.pending_h2_deliveries), 1)
        record = env.pending_h2_deliveries[0]
        self.assertEqual(int(record["seller_id"]), 4)
        self.assertEqual(record["path"], [4, 1])
        self.assertGreaterEqual(int(record["eta"]), 4)
        self.assertLessEqual(int(record["eta"]), 10)

    def test_traffic_without_external_node_rejected(self):
        with self.assertRaisesRegex(ValueError, "external_node_enable"):
            self._make_env(
                h2_traffic_enable=True,
                h2_route_action_enable=True,
            )

    def test_planned_channel_requires_lagged_delivery(self):
        with self.assertRaisesRegex(ValueError, "lagged H2 delivery"):
            self._make_env(h2_market_lag_enable=False, h2_delivery_lag=0)


if __name__ == "__main__":
    unittest.main()
