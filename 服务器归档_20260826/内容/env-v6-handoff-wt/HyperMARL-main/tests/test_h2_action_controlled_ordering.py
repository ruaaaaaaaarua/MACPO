import unittest

import numpy as np

from envs.microgrid.microgrid_env import MicrogridEnv
from scripts.run_stas_mechanism_ablation import LHV_H2, planned_experiments


class H2ActionControlledOrderingTest(unittest.TestCase):
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
            "h2_cap_aware_buy_enable": True,
            "h2_delivery_reservation_enable": True,
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
            env.profiles[key] = np.zeros(
                (env.agent_num, env.T), dtype=np.float32
            )
        return env

    @staticmethod
    def _actions(env, a5=-1.0):
        actions = np.zeros((env.agent_num, env.action_dim), dtype=np.float32)
        actions[:, 0] = -1.0
        actions[:, 5] = a5
        return actions

    @staticmethod
    def _info(env, actions):
        return env.step(actions)[3][0]

    def test_minus_one_submits_no_buy_despite_positive_net_demand(self):
        env = self._make_env()
        env.profiles["load_h"][:, 0] = env.load_h_peak * 0.25
        info = self._info(env, self._actions(env, a5=-1.0))
        self.assertTrue(np.all(np.asarray(info["net_h2_demand"]) > 0.0))
        np.testing.assert_allclose(
            info["h2_action_requested_buy_quantity"], 0.0, atol=1e-6
        )
        np.testing.assert_allclose(info["h2_order_quantity"], 0.0, atol=1e-6)
        self.assertEqual(info["h2_order_source"], ["none"] * env.agent_num)

    def test_zero_net_demand_positive_a5_submits_action_buy(self):
        env = self._make_env()
        info = self._info(env, self._actions(env, a5=1.0))
        np.testing.assert_allclose(info["net_h2_demand"], 0.0, atol=1e-6)
        self.assertTrue(
            np.all(np.asarray(info["h2_action_requested_buy_quantity"]) > 0.0)
        )
        effective = np.asarray(info["h2_action_effective_buy_quantity"])
        expected = np.minimum(
            np.asarray(info["h2_action_requested_buy_quantity"]),
            np.asarray(info["h2_buy_future_headroom"]),
        )
        np.testing.assert_allclose(effective, expected, rtol=0.0, atol=1e-4)
        self.assertEqual(
            info["h2_order_source"],
            ["action_buy"] * env.agent_num,
        )
        np.testing.assert_allclose(info["h2_order_quantity"], effective, atol=1e-4)

    def test_zero_and_one_request_half_and_full_static_peak_hour(self):
        for action_value, fraction in ((0.0, 0.5), (1.0, 1.0)):
            with self.subTest(action_value=action_value):
                env = self._make_env()
                env.profiles["load_h"][:, 0] = env.load_h_peak * 0.25
                expected_qmax = env.load_h_peak / env.cfg["boiler_eff"] * env.dt
                info = self._info(env, self._actions(env, a5=action_value))
                np.testing.assert_allclose(
                    info["h2_order_qmax"], expected_qmax, rtol=0.0, atol=1e-4
                )
                np.testing.assert_allclose(
                    info["h2_action_requested_buy_quantity"],
                    fraction * expected_qmax,
                    rtol=0.0,
                    atol=1e-4,
                )
                np.testing.assert_allclose(
                    info["h2_action_effective_buy_quantity"],
                    fraction * expected_qmax,
                    rtol=0.0,
                    atol=1e-4,
                )
                self.assertEqual(
                    info["h2_order_source"], ["action_buy"] * env.agent_num
                )

    def test_canonical_peak_hours_scales_qmax_and_group_metadata(self):
        max_peak_hours = 2.5
        env = self._make_env(
            h2_action_order_max_peak_hours=max_peak_hours,
            h2_learnable_rolling_order_max_fraction=99.0,
        )
        expected_qmax = (
            env.load_h_peak
            / env.cfg["boiler_eff"]
            * env.dt
            * max_peak_hours
        )
        np.testing.assert_allclose(
            env.h2_order_qmax, expected_qmax, rtol=0.0, atol=1e-4
        )

        spec = next(
            item for item in planned_experiments() if item.group == "group_abc"
        )
        self.assertEqual(
            spec.env_overrides["h2_action_order_max_peak_hours"], 1.0
        )

    def test_action_order_peak_hours_must_be_finite_and_positive(self):
        for invalid in (0.0, -1.0, float("nan"), float("inf")):
            with self.subTest(invalid=invalid):
                with self.assertRaisesRegex(
                    ValueError,
                    "h2_action_order_max_peak_hours.*finite.*positive",
                ):
                    self._make_env(h2_action_order_max_peak_hours=invalid)

    def test_seller_uses_exact_physical_surplus_and_ignores_a5(self):
        quantities = []
        for action_value in (-1.0, 1.0):
            env = self._make_env()
            actions = self._actions(env, a5=-1.0)
            actions[0, 0] = 1.0
            actions[0, 5] = action_value
            info = self._info(env, actions)
            self.assertLess(info["net_h2_demand"][0], 0.0)
            self.assertEqual(info["h2_order_source"][0], "physical_surplus")
            self.assertAlmostEqual(
                info["h2_order_quantity"][0],
                -info["net_h2_demand"][0],
                delta=1e-4,
            )
            self.assertEqual(info["h2_action_effective_buy_quantity"][0], 0.0)
            quantities.append(info["h2_order_quantity"][0])
        self.assertAlmostEqual(quantities[0], quantities[1], delta=1e-4)

    def test_action_ordering_conflicts_with_heuristic_reservation(self):
        with self.assertRaisesRegex(
            ValueError, "action-controlled.*heuristic|heuristic.*action-controlled"
        ):
            MicrogridEnv(
                {
                    "h2_learnable_rolling_order_enable": True,
                    "h2_buyer_reservation_demand_enable": True,
                }
            )

    def test_legacy_buy_diagnostics_are_not_reported_as_action_buys(self):
        automatic = self._make_env(
            h2_learnable_rolling_order_enable=False,
        )
        automatic.profiles["load_h"][:, 0] = automatic.load_h_peak * 0.25
        automatic_actions = np.zeros(
            (automatic.agent_num, automatic.action_dim), dtype=np.float32
        )
        automatic_actions[:, 0] = -1.0
        automatic_info = self._info(automatic, automatic_actions)
        np.testing.assert_allclose(
            automatic_info["h2_action_requested_buy_quantity"], 0.0, atol=1e-6
        )
        np.testing.assert_allclose(
            automatic_info["h2_action_effective_buy_quantity"], 0.0, atol=1e-6
        )
        self.assertEqual(
            automatic_info["h2_order_source"],
            ["automatic_deficit"] * automatic.agent_num,
        )

        heuristic = self._make_env(
            h2_learnable_rolling_order_enable=False,
            h2_buyer_reservation_demand_enable=True,
            h2_buyer_reservation_agent_indices=[1],
            h2_buyer_reservation_target_ratios=[0.0, 1.0, 0.0, 0.0],
            h2_buyer_reservation_max_order_fraction=0.25,
        )
        heuristic_actions = np.zeros(
            (heuristic.agent_num, heuristic.action_dim), dtype=np.float32
        )
        heuristic_actions[:, 0] = -1.0
        heuristic_info = self._info(heuristic, heuristic_actions)
        self.assertGreater(
            heuristic_info["h2_buyer_reservation_extra_order"][1], 0.0
        )
        np.testing.assert_allclose(
            heuristic_info["h2_action_requested_buy_quantity"], 0.0, atol=1e-6
        )
        np.testing.assert_allclose(
            heuristic_info["h2_action_effective_buy_quantity"], 0.0, atol=1e-6
        )
        self.assertEqual(
            heuristic_info["h2_order_source"][1], "heuristic_reservation"
        )

    def _seller_buyer_actions(self, env):
        actions = self._actions(env, a5=-1.0)
        actions[0, 0] = 1.0
        actions[0, 3] = -1.0
        actions[1, 3] = 1.0
        actions[1, 5] = 1.0
        return actions

    def test_t0_trade_arrives_in_state4_and_conserves_pending_and_cash(self):
        env = self._make_env()
        env.profiles["load_h"][1, 0] = 90.0
        initial_h2 = float(env.h2_level[1])
        first = self._info(env, self._seller_buyer_actions(env))
        traded = float(first["h2_market_traded"])
        self.assertGreater(traded, 0.0)
        self.assertAlmostEqual(
            first["pending_h2_energy_total"], traded, delta=1e-3
        )
        self.assertLess(
            abs(sum(first["h2_cda_paid"]) - sum(first["h2_cda_received"])),
            1e-3,
        )
        quiet = self._actions(env, a5=-1.0)
        delivered = None
        for _ in range(3):
            delivered = self._info(env, quiet)
        self.assertEqual(env.t, 4)
        self.assertAlmostEqual(delivered["delivered_h2_energy"][1], traded, delta=1e-3)
        self.assertAlmostEqual(delivered["pending_h2_energy_total"], 0.0, delta=1e-3)
        stored_energy = (float(env.h2_level[1]) - initial_h2) * env.cfg["LHV_H2"]
        self.assertAlmostEqual(stored_energy, traded, delta=1e-3)

    def test_t19_is_deliverable_but_t20_is_horizon_clipped(self):
        env19 = self._make_env()
        env19.t = 19
        env19.profiles["load_h"][1, 19] = 90.0
        info19 = self._info(env19, self._seller_buyer_actions(env19))
        self.assertGreater(info19["h2_action_effective_buy_quantity"][1], 0.0)
        self.assertAlmostEqual(info19["h2_buy_horizon_clip_amount"][1], 0.0)

        env20 = self._make_env()
        env20.t = 20
        env20.profiles["load_h"][1, 20] = 90.0
        info20 = self._info(env20, self._seller_buyer_actions(env20))
        self.assertEqual(info20["h2_order_source"][1], "action_buy")
        self.assertAlmostEqual(info20["h2_action_effective_buy_quantity"][1], 0.0)
        self.assertAlmostEqual(
            info20["h2_buy_horizon_clip_amount"][1],
            info20["h2_action_requested_buy_quantity"][1],
            delta=1e-4,
        )


    def test_horizon_clip_is_unconditional_when_cap_aware_disabled(self):
        env = self._make_env(h2_cap_aware_buy_enable=False)
        env.t = 20
        env.profiles["load_h"][1, 20] = 90.0
        info = self._info(env, self._seller_buyer_actions(env))

        self.assertGreater(info["h2_action_requested_buy_quantity"][1], 0.0)
        self.assertEqual(info["h2_order_source"][1], "action_buy")
        self.assertAlmostEqual(info["h2_action_effective_buy_quantity"][1], 0.0)
        self.assertAlmostEqual(info["h2_order_quantity"][1], 0.0)
        self.assertAlmostEqual(
            info["h2_buy_horizon_clip_amount"][1],
            info["h2_action_requested_buy_quantity"][1],
            delta=1e-4,
        )
        self.assertAlmostEqual(info["h2_market_traded"], 0.0)
        self.assertEqual(info["h2_pending_count"], 0)
        self.assertEqual(len(env.pending_h2_deliveries), 0)

    def test_group_abc_shape_prices_and_shaping_contract(self):
        spec = next(
            item for item in planned_experiments() if item.group == "group_abc"
        )
        env = MicrogridEnv(spec.env_overrides)
        env.seed(11)
        env.reset()
        for key in ("pv", "wt", "load_e", "load_h"):
            env.profiles[key] = np.zeros(
                (env.agent_num, env.T), dtype=np.float32
            )
        env.profiles["load_h"][:, 0] = 90.0
        actions = self._actions(env, a5=-1.0)
        actions[:, 3] = np.asarray([-1.0, 0.0, 1.0, -1.0], dtype=np.float32)
        info = self._info(env, actions)
        self.assertEqual((env.agent_num, env.T, env.obs_dim, env.action_dim), (4, 24, 19, 6))
        np.testing.assert_allclose(
            np.asarray(info["h2_bid_price"][:3]) * LHV_H2,
            [3.0, 16.5, 30.0],
            rtol=0.0,
            atol=1e-5,
        )
        self.assertAlmostEqual(env.lambda_h2_buy * LHV_H2, 45.0, delta=1e-5)
        for key in (
            "penalty_enable",
            "low_inventory_penalty_enable",
            "terminal_h2_floor_penalty_enable",
            "terminal_h2_shortfall_value_enable",
            "terminal_soc_floor_penalty_enable",
            "terminal_battery_salvage_enable",
            "stepwise_h2_floor_penalty_enable",
            "h2_internal_trade_bonus_enable",
            "action_reg_enable",
        ):
            self.assertIs(spec.env_overrides[key], False, key)

    def test_reward_is_pure_cost_even_when_all_diagnostics_are_enabled(self):
        shaping_on = {
            "penalty_enable": True,
            "soc_penalty_coef": 1000.0,
            "h2_penalty_coef": 1000.0,
            "low_inventory_penalty_enable": True,
            "low_inventory_penalty_coef": 1000.0,
            "terminal_h2_floor_penalty_enable": True,
            "terminal_h2_floor_penalty_coef": 1000.0,
            "terminal_h2_shortfall_value_enable": True,
            "terminal_soc_floor_penalty_enable": True,
            "terminal_soc_floor_penalty_coef": 1000.0,
            "terminal_battery_salvage_enable": True,
            "terminal_battery_salvage_value_coef": 1000.0,
            "stepwise_h2_floor_penalty_enable": True,
            "stepwise_h2_floor_penalty_coef": 1000.0,
            "action_reg_enable": True,
            "action_magnitude_penalty_coef": 1000.0,
            "external_h2_dependency_penalty_enable": True,
            "external_h2_dependency_penalty_coef": 2.0,
        }
        enabled = self._make_env(**shaping_on)
        disabled = self._make_env(
            external_h2_dependency_penalty_enable=True,
            external_h2_dependency_penalty_coef=2.0,
        )
        for env in (enabled, disabled):
            env.t = env.T - 1
            env.profiles["load_h"][:, env.t] = 90.0
        actions = self._actions(enabled, a5=-1.0)
        enabled_step = enabled.step(actions)
        disabled_step = disabled.step(actions)
        enabled_info = enabled_step[3][0]
        disabled_info = disabled_step[3][0]
        enabled_reward = enabled_step[1][0][0]
        disabled_reward = disabled_step[1][0][0]
        self.assertGreater(enabled_info["penalty_total"], 0.0)
        self.assertAlmostEqual(enabled_reward, disabled_reward, delta=1e-6)
        for reward, info, env in (
            (enabled_reward, enabled_info, enabled),
            (disabled_reward, disabled_info, disabled),
        ):
            expected = -(
                info["C_grid"]
                + info["C_h2"]
                + info["external_h2_dependency_penalty"]
            ) / env.reward_scale
            self.assertAlmostEqual(reward, expected, delta=1e-6)

    def test_matched_trade_bonus_is_diagnostic_only(self):
        coefficient = 7.0
        enabled = self._make_env(
            h2_internal_trade_bonus_enable=True,
            h2_internal_trade_bonus_coef=coefficient,
        )
        disabled = self._make_env(
            h2_internal_trade_bonus_enable=False,
            h2_internal_trade_bonus_coef=coefficient,
        )
        for env in (enabled, disabled):
            env.profiles["load_h"][1, 0] = 90.0
        enabled_step = enabled.step(self._seller_buyer_actions(enabled))
        disabled_step = disabled.step(self._seller_buyer_actions(disabled))
        enabled_info = enabled_step[3][0]
        disabled_info = disabled_step[3][0]
        enabled_reward = enabled_step[1][0][0]
        disabled_reward = disabled_step[1][0][0]
        self.assertGreater(enabled_info["h2_market_traded"], 0.0)
        self.assertAlmostEqual(
            enabled_info["h2_internal_trade_bonus"],
            coefficient * enabled_info["h2_market_traded"],
            delta=1e-6,
        )
        self.assertAlmostEqual(
            enabled_info["market_bonus"],
            enabled_info["h2_internal_trade_bonus"],
            delta=1e-6,
        )
        self.assertAlmostEqual(enabled_reward, disabled_reward, delta=1e-6)
        for reward, info, env in (
            (enabled_reward, enabled_info, enabled),
            (disabled_reward, disabled_info, disabled),
        ):
            expected = -(
                info["C_grid"]
                + info["C_h2"]
                + info["external_h2_dependency_penalty"]
            ) / env.reward_scale
            self.assertAlmostEqual(reward, expected, delta=1e-6)

    def test_no_lag_excess_matched_buy_enters_tank_without_external_resale(self):
        env = self._make_env(
            h2_market_lag_enable=False,
            h2_delivery_lag=0,
            h2_learnable_rolling_order_agent_indices=[0, 1, 2, 3],
        )
        env.profiles["load_h"][1, 0] = 90.0
        initial_h2 = float(env.h2_level[1])
        info = self._info(env, self._seller_buyer_actions(env))
        matched = sum(
            float(trade["quantity"])
            for trade in info["h2_market_trades"]
            if int(trade["buyer_id"]) == 1
        )
        demand = float(info["net_h2_demand"][1])
        demand_offset = min(max(demand, 0.0), matched)
        expected_excess = matched - demand_offset
        stored_energy = (float(env.h2_level[1]) - initial_h2) * env.cfg["LHV_H2"]
        self.assertGreater(expected_excess, 0.0)
        self.assertAlmostEqual(stored_energy, expected_excess, delta=1e-3)
        self.assertGreaterEqual(info["e_h2_ext"][1], -1e-6)
        self.assertAlmostEqual(
            info["e_h2_ext"][1], max(0.0, demand - matched), delta=1e-4
        )
        self.assertLess(abs(matched - demand_offset - stored_energy), 1e-3)
        self.assertLess(
            abs(sum(info["h2_cda_paid"]) - sum(info["h2_cda_received"])),
            1e-3,
        )

    def test_no_lag_cap_unaware_overflow_is_conserved_not_resold(self):
        env = self._make_env(
            h2_market_lag_enable=False,
            h2_delivery_lag=0,
            h2_cap_aware_buy_enable=False,
        )
        env.profiles["load_h"][1, 0] = 90.0
        env.h2_level[1] = env.h2_max[1]
        info = self._info(env, self._seller_buyer_actions(env))
        matched = sum(
            float(trade["quantity"])
            for trade in info["h2_market_trades"]
            if int(trade["buyer_id"]) == 1
        )
        demand = float(info["net_h2_demand"][1])
        demand_offset = float(info["h2_no_lag_demand_offset"][1])
        stored = float(info["h2_immediate_stored_energy"][1])
        overflow = float(info["h2_immediate_overflow_energy"][1])
        self.assertGreater(matched, 0.0)
        self.assertGreater(overflow, 0.0)
        self.assertLess(
            abs(matched - demand_offset - stored - overflow),
            1e-3,
        )
        self.assertLess(
            abs(info["h2_no_lag_conservation_residual"][1]), 1e-3
        )
        self.assertGreaterEqual(info["e_h2_ext"][1], -1e-6)
        self.assertAlmostEqual(
            info["e_h2_ext"][1],
            max(0.0, demand - demand_offset),
            delta=1e-4,
        )
        self.assertAlmostEqual(
            info["gas_external_injection_agent"][1],
            0.0,
            delta=1e-6,
        )


if __name__ == "__main__":
    unittest.main()
