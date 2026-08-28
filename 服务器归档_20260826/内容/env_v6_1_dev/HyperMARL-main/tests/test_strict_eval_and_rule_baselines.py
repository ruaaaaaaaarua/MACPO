import unittest

import numpy as np

from baselines.utils.fixed_scenario_eval import (
    EvaluationContext,
    TEST_DAYS,
    VALIDATION_DAYS,
    build_scenarios,
    evaluate_contextual_policy,
    evaluate_policy,
    load_manifest_splits,
)
from baselines.utils.rule_baselines import (
    current_deficit_rule,
    physical_idle,
    privileged_t4_rule,
)


ABC_MINIMAL = {
    "episode_length": 24,
    "italian_split_enable": True,
    "h2_pending_obs_enable": True,
    "h2_pending_obs_horizon": 4,
    "h2_pending_summary_obs_enable": True,
    "h2_learnable_rolling_order_enable": True,
}


class ManifestEvaluationTest(unittest.TestCase):
    def test_manifest_splits_are_disjoint_and_match_fixed_days(self):
        splits = load_manifest_splits()
        expected_train = {
            0, 2, 3, 4, 5, 6, 9, 10, 11, 12,
            13, 15, 16, 18, 19, 20, 22, 25, 26, 27,
        }
        self.assertEqual(set(splits["train"]), expected_train)
        self.assertEqual(tuple(splits["validation"]), VALIDATION_DAYS)
        self.assertEqual(tuple(splits["test"]), TEST_DAYS)
        self.assertFalse(set(splits["train"]) & set(splits["validation"]))
        self.assertFalse(set(splits["train"]) & set(splits["test"]))
        self.assertFalse(set(splits["validation"]) & set(splits["test"]))
        self.assertEqual(
            set(splits["train"]) | set(splits["validation"]) | set(splits["test"]),
            set(range(28)),
        )

    def test_validation_rejects_test_day_before_env_construction(self):
        with self.assertRaisesRegex(ValueError, "not in manifest split 'validation'"):
            evaluate_policy(
                lambda obs: np.zeros((obs.shape[0], 6), dtype=np.float32),
                ABC_MINIMAL,
                build_scenarios([TEST_DAYS[0]], [4200]),
                algorithm="invalid",
                split_name="validation",
            )

    def test_test_rejects_train_and_validation_days(self):
        for day in (0, VALIDATION_DAYS[0]):
            with self.subTest(day=day):
                with self.assertRaisesRegex(ValueError, "not in manifest split 'test'"):
                    evaluate_policy(
                        lambda obs: np.zeros((obs.shape[0], 6), dtype=np.float32),
                        ABC_MINIMAL,
                        build_scenarios([day], [5200]),
                        algorithm="invalid",
                        split_name="test",
                    )

    def test_strict_split_rejects_any_nonlocked_noise_seed(self):
        with self.assertRaisesRegex(ValueError, "requires scenario seed 4200"):
            evaluate_policy(
                lambda obs: np.zeros((obs.shape[0], 6), dtype=np.float32),
                ABC_MINIMAL,
                build_scenarios([VALIDATION_DAYS[0]], [4201]),
                algorithm="wrong-seed",
                split_name="validation",
            )

    def test_contextual_policy_receives_read_only_step_config_and_profiles(self):
        seen_steps = []
        first_profile = []

        def contextual(obs, context):
            seen_steps.append(context["episode_step"])
            self.assertEqual(context.episode_step, context["episode_step"])
            self.assertEqual(context["config"]["italian_split_strategy"], "manifest")
            self.assertFalse(context["profiles"]["load_h"].flags.writeable)
            first_profile.append(float(context["profiles"]["load_h"][0, 0]))
            return np.zeros((obs.shape[0], 6), dtype=np.float32)

        result = evaluate_contextual_policy(
            contextual,
            ABC_MINIMAL,
            build_scenarios([VALIDATION_DAYS[0]], [4200]),
            algorithm="contextual",
            split_name="validation",
        )
        self.assertEqual(seen_steps, list(range(24)))
        self.assertTrue(all(value == first_profile[0] for value in first_profile))
        self.assertEqual(result["episodes"][0]["steps"], 24)

    def test_traffic_evaluation_reports_route_and_eta_metrics(self):
        overrides = {
            **ABC_MINIMAL,
            "h2_traffic_enable": True,
            "h2_route_action_enable": True,
            "h2_traffic_max_eta": 6,
            "h2_delivery_reservation_horizon": 6,
        }
        result = evaluate_policy(
            lambda obs: np.tile(
                np.asarray([-1.0, 0.0, 0.0, 0.0, 0.0, -1.0, -1.0], dtype=np.float32),
                (obs.shape[0], 1),
            ),
            overrides,
            build_scenarios([VALIDATION_DAYS[0]], [4200]),
            algorithm="traffic-idle",
            split_name="validation",
        )
        for key in (
            "transport_shipment_count_mean",
            "transport_gross_mean",
            "transport_loss_mean",
            "transport_eta_mean",
            "transport_delayed_rate_mean",
            "route_entropy_mean",
            "horizon_clipped_buy_mean",
            "edge_utilization_max_mean",
        ):
            self.assertIn(key, result["summary"])
            self.assertTrue(np.isfinite(result["summary"][key]))


class RuleBaselinesTest(unittest.TestCase):
    def setUp(self):
        self.config = {
            "num_agents": 4,
            "episode_length": 8,
            "dt": 1.0,
            "pv_cap": [100.0] * 4,
            "wt_cap": [100.0] * 4,
            "load_e_peak": [100.0] * 4,
            "load_h_peak": [100.0] * 4,
            "el_cap": [100.0] * 4,
            "el_eff": [0.5] * 4,
            "boiler_eff": 1.0,
        }
        self.profiles = {
            name: np.zeros((4, 8), dtype=np.float32)
            for name in ("pv", "wt", "load_e", "load_h")
        }
        self.profiles["load_h"][:, 4] = 80.0
        self.obs = np.zeros((4, 13), dtype=np.float32)
        self.obs[:, 1] = 0.6
        self.obs[:, 2] = 0.2
        self.obs[:, 3] = 0.3
        self.context = EvaluationContext(
            episode_step=0,
            config=self.config,
            profiles=self.profiles,
        )

    def test_physical_idle_is_exact_and_deterministic(self):
        expected = np.tile(
            np.asarray([-1.0, 0.0, 0.0, 0.0, 0.0, -1.0], dtype=np.float32),
            (4, 1),
        )
        first = physical_idle(self.obs, self.context)
        second = physical_idle(self.obs.copy(), self.context)
        np.testing.assert_array_equal(first, expected)
        np.testing.assert_array_equal(first, second)

    def test_current_and_privileged_are_deterministic_and_only_a5_differs(self):
        current = current_deficit_rule(self.obs, self.context)
        current_again = current_deficit_rule(self.obs.copy(), self.context)
        privileged = privileged_t4_rule(self.obs, self.context)
        privileged_again = privileged_t4_rule(self.obs.copy(), self.context)
        np.testing.assert_array_equal(current, current_again)
        np.testing.assert_array_equal(privileged, privileged_again)
        np.testing.assert_array_equal(current[:, :5], privileged[:, :5])
        self.assertTrue(np.any(current[:, 5] != privileged[:, 5]))
        self.assertTrue(privileged_t4_rule.privileged_diagnostic)

    def test_current_uses_obs_now_and_privileged_uses_profile_t_plus_four(self):
        current = current_deficit_rule(self.obs, self.context)
        privileged = privileged_t4_rule(self.obs, self.context)
        # Current: renewable surplus 40 -> 20 H2, current load 30 -> deficit 10.
        np.testing.assert_allclose(current[:, 5], -0.8, atol=1e-6)
        # Future: no renewable, H2 load 80 -> deficit 80.
        np.testing.assert_allclose(privileged[:, 5], 0.6, atol=1e-6)

        changed_future = {key: value.copy() for key, value in self.profiles.items()}
        changed_future["load_h"][:, 4] = 10.0
        changed_context = EvaluationContext(0, self.config, changed_future)
        np.testing.assert_array_equal(
            current_deficit_rule(self.obs, changed_context),
            current,
        )
        self.assertTrue(
            np.any(privileged_t4_rule(self.obs, changed_context)[:, 5] != privileged[:, 5])
        )

    def test_privileged_tail_has_no_order(self):
        context = EvaluationContext(4, self.config, self.profiles)
        action = privileged_t4_rule(self.obs, context)
        np.testing.assert_array_equal(action[:, 5], -1.0)

    def test_h2_crossing_bid_uses_seller_min_for_projected_surplus(self):
        surplus_obs = self.obs.copy()
        surplus_obs[:, 1] = 1.0
        surplus_obs[:, 2] = 0.0
        surplus_obs[:, 3] = 0.1
        action = current_deficit_rule(surplus_obs, self.context)
        np.testing.assert_array_equal(action[:, 3], -1.0)
        np.testing.assert_array_equal(action[:, 5], -1.0)

    def test_traffic_rules_return_seven_actions_and_default_to_direct_route(self):
        config = dict(self.config)
        config["h2_route_action_enable"] = True
        context = EvaluationContext(0, config, self.profiles)
        traffic_obs = np.pad(self.obs, ((0, 0), (0, 11)))
        for policy in (physical_idle, current_deficit_rule, privileged_t4_rule):
            action = policy(traffic_obs, context)
            self.assertEqual(action.shape, (4, 7))
            np.testing.assert_array_equal(action[:, 6], -1.0)


if __name__ == "__main__":
    unittest.main()
