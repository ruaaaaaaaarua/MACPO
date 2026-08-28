import json
import unittest

import numpy as np

from envs.microgrid.microgrid_env import MicrogridEnv


class H2TrafficIntegrationTest(unittest.TestCase):
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
            "h2_traffic_enable": True,
            "h2_route_action_enable": True,
            "h2_traffic_min_eta": 4,
            "h2_traffic_max_eta": 6,
            "h2_traffic_seed": 20260716,
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
        for key in ("pv", "wt", "load_e", "load_h"):
            env.profiles[key] = np.zeros((env.agent_num, env.T), dtype=np.float32)
        return env

    @staticmethod
    def _actions(env, buyer_route=-1.0, seller_route=-1.0):
        actions = np.zeros((env.agent_num, env.action_dim), dtype=np.float32)
        actions[:, 0] = -1.0
        actions[:, 5] = -1.0
        actions[0, 0] = 1.0
        actions[0, 3] = -1.0
        actions[0, 6] = seller_route
        actions[1, 3] = 1.0
        actions[1, 5] = 1.0
        actions[1, 6] = buyer_route
        return actions

    @staticmethod
    def _info(env, actions):
        return env.step(actions)[3][0]

    def test_traffic_mode_has_locked_24_observations_and_7_actions(self):
        env = self._make_env()
        self.assertEqual((env.obs_dim, env.action_dim), (24, 7))
        self.assertEqual(env.h2_pending_obs_horizon, 6)
        self.assertEqual(env.h2_delivery_reservation_horizon, 6)
        obs = env._get_obs()
        self.assertEqual(np.asarray(obs).shape, (4, 24))
        for agent_id in range(env.agent_num):
            np.testing.assert_allclose(
                obs[agent_id][-3:],
                env.h2_transport_network.route_features(agent_id, env.t),
                rtol=0.0,
                atol=1e-7,
            )

    def test_traffic_off_preserves_group_shape_and_fixed_lag_contract(self):
        env = self._make_env(
            h2_traffic_enable=False,
            h2_route_action_enable=False,
            h2_pending_obs_horizon=4,
            h2_delivery_reservation_horizon=4,
        )
        self.assertEqual((env.obs_dim, env.action_dim), (19, 6))
        self.assertEqual(env.h2_delivery_lag, 4)

    def test_route_action_requires_action_controlled_h2_ordering(self):
        with self.assertRaisesRegex(ValueError, "route action.*action-controlled"):
            self._make_env(h2_learnable_rolling_order_enable=False)

    def test_route_action_disabled_defaults_to_direct_path(self):
        env = self._make_env(h2_route_action_enable=False)
        self.assertEqual(env.action_dim, 6)
        env.profiles["load_h"][1, 0] = 90.0
        actions = np.zeros((env.agent_num, env.action_dim), dtype=np.float32)
        actions[:, 0] = -1.0
        actions[:, 5] = -1.0
        actions[0, 0] = 1.0
        actions[0, 3] = -1.0
        actions[1, 3] = 1.0
        actions[1, 5] = 1.0
        info = self._info(env, actions)
        self.assertEqual(info["h2_transport_shipments"][0]["route_rank"], 0)

    def test_buyer_a6_selects_route_and_seller_a6_is_ignored(self):
        route_records = []
        for buyer_route, seller_route in ((-1.0, -1.0), (1.0, -1.0), (1.0, 1.0)):
            env = self._make_env()
            env.profiles["load_h"][1, 0] = 90.0
            info = self._info(env, self._actions(env, buyer_route, seller_route))
            shipment = info["h2_transport_shipments"][0]
            route_records.append(shipment)
            self.assertEqual(shipment["buyer_id"], 1)
            self.assertEqual(shipment["seller_id"], 0)
            self.assertEqual(shipment["deliver_at"], shipment["dispatch_t"] + shipment["eta"])
            self.assertGreaterEqual(shipment["eta"], 4)
            self.assertLessEqual(shipment["eta"], 6)
        self.assertEqual(route_records[0]["route_rank"], 0)
        self.assertEqual(route_records[1]["route_rank"], 2)
        self.assertEqual(route_records[1]["path"], route_records[2]["path"])

    def test_traffic_horizon_uses_max_eta_t17_allowed_t18_clipped(self):
        env17 = self._make_env()
        env17.t = 17
        env17.profiles["load_h"][1, 17] = 90.0
        info17 = self._info(env17, self._actions(env17))
        self.assertGreater(info17["h2_action_effective_buy_quantity"][1], 0.0)
        self.assertAlmostEqual(info17["h2_buy_horizon_clip_amount"][1], 0.0)

        env18 = self._make_env()
        env18.t = 18
        env18.profiles["load_h"][1, 18] = 90.0
        info18 = self._info(env18, self._actions(env18))
        self.assertEqual(info18["h2_action_effective_buy_quantity"][1], 0.0)
        self.assertAlmostEqual(
            info18["h2_buy_horizon_clip_amount"][1],
            info18["h2_action_requested_buy_quantity"][1],
            delta=1e-4,
        )
        self.assertEqual(info18["h2_transport_shipments"], [])

    def test_dynamic_pending_arrives_at_recorded_state_and_conserves_loss(self):
        env = self._make_env(h2_transport_loss=0.1)
        env.profiles["load_h"][1, 0] = 90.0
        initial_h2 = float(env.h2_level[1])
        first = self._info(env, self._actions(env))
        shipment = first["h2_transport_shipments"][0]
        self.assertAlmostEqual(
            shipment["gross_quantity"],
            shipment["net_quantity"] + shipment["loss_quantity"],
            delta=1e-6,
        )
        self.assertAlmostEqual(
            first["pending_h2_energy_total"], shipment["net_quantity"], delta=1e-3
        )
        self.assertAlmostEqual(
            first["h2_market_traded"], shipment["gross_quantity"], delta=1e-3
        )
        self.assertLess(
            abs(sum(first["h2_cda_paid"]) - sum(first["h2_cda_received"])), 1e-3
        )

        quiet = np.zeros((env.agent_num, env.action_dim), dtype=np.float32)
        quiet[:, 0] = -1.0
        quiet[:, 5] = -1.0
        final = first
        while env.t < shipment["deliver_at"]:
            final = self._info(env, quiet)
        self.assertAlmostEqual(
            final["delivered_h2_energy"][1], shipment["net_quantity"], delta=1e-3
        )
        stored = (float(env.h2_level[1]) - initial_h2) * env.cfg["LHV_H2"]
        self.assertAlmostEqual(stored, shipment["net_quantity"], delta=1e-3)

    def test_traffic_adds_no_reward_or_transport_cost_term(self):
        direct = self._make_env()
        detour = self._make_env()
        for env in (direct, detour):
            env.profiles["load_h"][1, 0] = 90.0
        direct_step = direct.step(self._actions(direct, buyer_route=-1.0))
        detour_step = detour.step(self._actions(detour, buyer_route=1.0))
        direct_info = direct_step[3][0]
        detour_info = detour_step[3][0]
        self.assertNotIn("C_transport", direct_info)
        self.assertAlmostEqual(direct_info["total_cost"], detour_info["total_cost"], delta=1e-6)
        self.assertAlmostEqual(direct_step[1][0][0], detour_step[1][0][0], delta=1e-6)

    def test_step_info_is_strict_json_serializable(self):
        env = self._make_env()
        env.profiles["load_h"][1, 0] = 90.0
        info = self._info(env, self._actions(env))
        encoded = json.dumps(info, allow_nan=False, sort_keys=True)
        self.assertIn('"h2_transport_shipments"', encoded)

    def test_no_lag_bypasses_transport_queue(self):
        env = self._make_env(h2_market_lag_enable=False, h2_delivery_lag=0)
        env.profiles["load_h"][1, 0] = 90.0
        info = self._info(env, self._actions(env, buyer_route=1.0))
        self.assertGreater(info["h2_market_traded"], 0.0)
        self.assertEqual(info["h2_transport_shipments"], [])
        self.assertEqual(info["h2_pending_count"], 0)
        self.assertGreaterEqual(info["e_h2_ext"][1], -1e-6)


if __name__ == "__main__":
    unittest.main()
