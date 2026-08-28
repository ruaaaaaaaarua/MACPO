import json
import unittest
from unittest.mock import patch

import numpy as np

from envs.microgrid.microgrid_env import MicrogridEnv


class SafeMicrogridEnvironmentTest(unittest.TestCase):
    def test_power_flow_model_is_constructed_by_the_configured_factory(self):
        sentinel = object()
        with patch(
            "envs.microgrid.power_flow.build_power_flow", return_value=sentinel
        ) as factory:
            env = MicrogridEnv(
                {
                    "profile_source": "synthetic",
                    "italian_split_enable": False,
                    "power_flow_enable": True,
                    "power_flow_model": "swiss_mv",
                    "power_flow_case_dir": "/unused/by/factory/mock",
                    "power_flow_pcc_bus_ids": [10, 20, 30, 40],
                }
            )

        self.assertIs(env.power_flow, sentinel)
        factory.assert_called_once()
        self.assertEqual(factory.call_args.args[0]["power_flow_model"], "swiss_mv")

    def test_eta_counterfactual_can_keep_the_pending_observation_window_fixed(self):
        env = MicrogridEnv(
            {
                "profile_source": "synthetic",
                "italian_split_enable": False,
                "h2_traffic_enable": True,
                "h2_route_action_enable": False,
                "h2_pending_obs_enable": True,
                "h2_pending_obs_horizon": 2,
                "h2_pending_obs_auto_expand_to_eta": False,
                "power_flow_enable": False,
            }
        )

        self.assertEqual(env.h2_pending_obs_horizon, 2)

    def test_day_ahead_hydrogen_features_are_local_and_zero_beyond_day_end(self):
        base = self._make_env(terminal_settlement=False)
        forecast = MicrogridEnv(
            {
                "profile_source": "synthetic",
                "italian_split_enable": False,
                "episode_length": 4,
                "h2_day_ahead_forecast_enable": True,
                "h2_day_ahead_forecast_horizons": [1, 2],
                "power_flow_enable": False,
            }
        )
        base_obs = base._get_obs()[0]
        forecast.seed(17)
        forecast.reset()
        self.assertEqual(len(forecast._get_obs()[0]), len(base_obs) + 4)
        for _ in range(3):
            forecast.step(np.zeros((forecast.agent_num, forecast.action_dim), dtype=np.float32))
        np.testing.assert_allclose(forecast._get_obs()[0][-4:], 0.0)

    def test_supply_intent_facts_append_four_local_nonleaking_values(self):
        baseline = MicrogridEnv(
            {
                "profile_source": "synthetic", "italian_split_enable": False,
                "episode_length": 4, "power_flow_enable": False,
            }
        )
        env = MicrogridEnv(
            {
                "profile_source": "synthetic", "italian_split_enable": False,
                "episode_length": 4, "power_flow_enable": False,
                "h2_supply_intent_message_enable": True,
            }
        )
        baseline.seed(30); baseline.reset()
        env.seed(30); env.reset()
        self.assertEqual(len(env._get_obs()[0]), len(baseline._get_obs()[0]) + 4)
        for _ in range(3):
            env.step(np.zeros((env.agent_num, env.action_dim), dtype=np.float32))
        np.testing.assert_allclose(env._get_obs()[0][-4:-1], 0.0)

    def test_local_supply_facts_do_not_require_an_intent_message(self):
        baseline = MicrogridEnv(
            {
                "profile_source": "synthetic",
                "italian_split_enable": False,
                "episode_length": 4,
                "power_flow_enable": False,
            }
        )
        local = MicrogridEnv(
            {
                "profile_source": "synthetic",
                "italian_split_enable": False,
                "episode_length": 4,
                "power_flow_enable": False,
                "h2_supply_intent_message_enable": False,
                "h2_local_supply_facts_enable": True,
            }
        )

        self.assertEqual(local.obs_dim, baseline.obs_dim + 4)
        self.assertTrue(local.h2_local_supply_facts_enable)
        self.assertFalse(local.h2_supply_intent_message_enable)

    def test_external_min_eta_uses_only_current_background_traffic(self):
        env = MicrogridEnv(
            {
                "profile_source": "synthetic",
                "italian_split_enable": False,
                "episode_length": 24,
                "power_flow_enable": False,
                "h2_traffic_enable": True,
                "h2_traffic_external_node_enable": True,
                "h2_traffic_eta_min": 3,
                "h2_traffic_eta_max": 10,
                "h2_supply_intent_message_enable": True,
            }
        )
        env.seed(30)
        env.reset()
        network = env.h2_transport_network
        current = network.background_utilization(0)
        expected_eta = min(
            network._route_eta(path, current)[0]
            for path in network.route_options(network.external_node_id, 0)
        )
        expected = (expected_eta - network.min_eta) / (
            network.max_eta - network.min_eta
        )

        def current_only(hour):
            if int(hour) != 0:
                raise AssertionError("future traffic was queried")
            return current

        with patch.object(network, "background_utilization", side_effect=current_only):
            facts = env._supply_intent_facts(
                0,
                0,
                np.zeros((env.agent_num, env.h2_pending_obs_horizon), dtype=np.float32),
            )
        self.assertAlmostEqual(facts[-1], expected)

    @staticmethod
    def _make_env(terminal_settlement):
        env = MicrogridEnv(
            {
                "profile_source": "synthetic",
                "italian_split_enable": False,
                "h2_tank_init_ratio": 0.40,
                "episode_length": 4,
                "reward_emission_mode": "dense",
                "power_flow_enable": True,
                "terminal_economic_settlement_enable": terminal_settlement,
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
        env.seed(17)
        env.reset()
        return env

    def test_enabled_power_flow_emits_shared_global_safety_cost(self):
        env = self._make_env(terminal_settlement=False)
        actions = np.zeros((env.agent_num, env.action_dim), dtype=np.float32)

        _, rewards, dones, infos = env.step(actions)
        info = infos[0]

        self.assertFalse(dones[0])
        self.assertTrue(info["pf_converged"])
        self.assertIn("voltage_cost", info)
        self.assertEqual(len(info["voltages_pu"]), 33)
        self.assertAlmostEqual(
            float(np.mean(rewards)), -info["total_cost"] / env.reward_scale
        )

    def test_pcc_scale_only_changes_the_power_flow_interface(self):
        raw = self._make_env(terminal_settlement=False)
        scaled = MicrogridEnv({**raw.cfg, "power_flow_pcc_injection_scale": 0.5})
        raw.seed(30); scaled.seed(30)
        raw.reset(); scaled.reset()
        actions = np.zeros((raw.agent_num, raw.action_dim), dtype=np.float32)
        _, raw_rewards, _, raw_infos = raw.step(actions)
        _, scaled_rewards, _, scaled_infos = scaled.step(actions)
        raw_info, scaled_info = raw_infos[0], scaled_infos[0]
        np.testing.assert_allclose(raw_info["pcc_p_kw"], scaled_info["pcc_p_kw"])
        np.testing.assert_allclose(raw_info["pcc_q_kvar"], scaled_info["pcc_q_kvar"])
        np.testing.assert_allclose(
            scaled_info["power_flow_pcc_p_kw"], 0.5 * np.asarray(raw_info["pcc_p_kw"])
        )
        np.testing.assert_allclose(
            scaled_info["power_flow_pcc_q_kvar"], 0.5 * np.asarray(raw_info["pcc_q_kvar"])
        )
        self.assertEqual(scaled_info["power_flow_pcc_injection_scale"], 0.5)
        np.testing.assert_allclose(raw_rewards, scaled_rewards)

    def test_power_flow_failure_info_is_strict_json_serializable(self):
        env = self._make_env(terminal_settlement=False)
        actions = np.zeros((env.agent_num, env.action_dim), dtype=np.float32)

        with patch("envs.microgrid.power_flow.runpf", side_effect=ArithmeticError):
            _, _, _, infos = env.step(actions)

        info = infos[0]
        self.assertIsNone(info["voltage_min_pu"])
        self.assertIsNone(info["voltage_max_pu"])
        json.dumps(info, allow_nan=False, sort_keys=True)

    def test_terminal_settlement_only_changes_terminal_cost(self):
        without_settlement = self._make_env(terminal_settlement=False)
        with_settlement = self._make_env(terminal_settlement=True)
        actions = np.zeros(
            (without_settlement.agent_num, without_settlement.action_dim), dtype=np.float32
        )
        actions[:, 4] = -1.0

        for _ in range(without_settlement.T):
            _, _, _, without_infos = without_settlement.step(actions)
            _, _, _, with_infos = with_settlement.step(actions)

        without_info = without_infos[0]
        with_info = with_infos[0]
        self.assertNotEqual(with_info["terminal_settlement_cost"], 0.0)
        self.assertAlmostEqual(
            with_info["total_cost"] - without_info["total_cost"],
            with_info["terminal_settlement_cost"],
            places=5,
        )
        self.assertAlmostEqual(
            with_info["terminal_asset_value"],
            with_info["terminal_h2_asset_value"]
            + with_info["terminal_battery_asset_value"]
            + with_info["terminal_pending_h2_asset_value"],
            places=5,
        )


    def test_terminal_pending_hydrogen_is_paid_but_has_no_terminal_asset_value(self):
        without_settlement = self._make_env(terminal_settlement=False)
        with_settlement = self._make_env(terminal_settlement=True)
        pending = {
            "deliver_at": 99,
            "buyer_id": 0,
            "seller_id": -1,
            "quantity": 100.0,
            "price": with_settlement.lambda_h2_buy,
        }
        without_settlement.pending_h2_deliveries.append(dict(pending))
        with_settlement.pending_h2_deliveries.append(dict(pending))
        actions = np.zeros(
            (without_settlement.agent_num, without_settlement.action_dim), dtype=np.float32
        )

        for _ in range(without_settlement.T):
            _, _, _, without_infos = without_settlement.step(actions)
            _, _, _, with_infos = with_settlement.step(actions)

        without_info = without_infos[0]
        with_info = with_infos[0]
        self.assertEqual(with_info["terminal_pending_h2_asset_value"], 0.0)
        self.assertAlmostEqual(with_info["terminal_undelivered_h2_energy"], 100.0)
        self.assertAlmostEqual(
            with_info["terminal_settlement_cost"],
            with_info["initial_terminal_asset_value"]
            - with_info["terminal_battery_asset_value"]
            - with_info["terminal_h2_asset_value"],
        )
        self.assertAlmostEqual(
            with_info["total_cost"] - without_info["total_cost"],
            with_info["terminal_settlement_cost"],
        )

    def test_reset_uses_configured_initial_soc_without_changing_default(self):
        configured = MicrogridEnv(
            {
                "profile_source": "synthetic",
                "italian_split_enable": False,
                "power_flow_enable": False,
                "soc_init": 0.5,
            }
        )
        historical_default = MicrogridEnv(
            {
                "profile_source": "synthetic",
                "italian_split_enable": False,
                "power_flow_enable": False,
            }
        )

        configured.reset()
        historical_default.reset()

        np.testing.assert_allclose(configured.soc, 0.5)
        np.testing.assert_allclose(historical_default.soc, 0.1)

if __name__ == "__main__":
    unittest.main()
