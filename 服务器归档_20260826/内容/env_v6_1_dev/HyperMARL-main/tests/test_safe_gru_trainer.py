import unittest
from unittest.mock import patch

import numpy as np
import jax.numpy as jnp
from flax.core import freeze, unfreeze

from baselines.MAPPO.safe_gru_trainer import SafeGRUMAPPOTrainer


class SafeGRUMAPPOTrainerTest(unittest.TestCase):
    @staticmethod
    def _config():
        return {
            "seed": 30,
            "num_envs": 1,
            "num_steps": 4,
            "hidden_size": 16,
            "lr": 3e-4,
            "gamma": 1.0,
            "gae_lambda": 0.95,
            "clip_eps": 0.2,
            "entropy_coef": 0.0,
            "lagrange_lr": 0.5,
            "cost_budget": 0.0,
            "env_overrides": {
                "profile_source": "synthetic",
                "italian_split_enable": False,
                "episode_length": 4,
                "reward_emission_mode": "dense",
                "power_flow_enable": True,
                "terminal_economic_settlement_enable": False,
                "penalty_enable": False,
                "low_inventory_penalty_enable": False,
                "terminal_h2_floor_penalty_enable": False,
                "terminal_h2_shortfall_value_enable": False,
                "terminal_h2_settlement_in_reward_enable": False,
                "terminal_soc_floor_penalty_enable": False,
                "terminal_battery_salvage_enable": False,
                "stepwise_h2_floor_penalty_enable": False,
                "action_reg_enable": False,
            },
        }

    def test_rollout_keeps_separate_reward_and_global_voltage_cost(self):
        trainer = SafeGRUMAPPOTrainer(self._config())
        rollout = trainer.collect_rollout()

        self.assertEqual(rollout.rewards.shape, (4, 1))
        self.assertEqual(rollout.costs.shape, (4, 1))
        self.assertEqual(rollout.actions.shape[:3], (4, 1, 4))
        self.assertTrue(np.all(np.asarray(rollout.costs) >= 0.0))
        self.assertGreater(np.max(np.abs(np.asarray(rollout.rewards))), 0.0)
        np.testing.assert_allclose(trainer.actor_hidden, 0.0)
        np.testing.assert_allclose(trainer.reward_critic_hidden, 0.0)
        np.testing.assert_allclose(trainer.cost_critic_hidden, 0.0)
        trainer.close()

    def test_voltage_cost_scale_keeps_raw_diagnostics_and_normalized_training_cost(self):
        config = self._config()
        config.update({"voltage_cost_scale": 0.02, "cost_budget": 1.0})
        trainer = SafeGRUMAPPOTrainer(config)
        try:
            rollout = trainer.collect_rollout()
            np.testing.assert_allclose(
                np.asarray(rollout.raw_costs),
                0.02 * np.asarray(rollout.costs),
                rtol=1e-6,
                atol=1e-8,
            )
            metrics = trainer.update(rollout, algorithm="mappo")
            self.assertAlmostEqual(
                metrics["daily_voltage_cost_raw"],
                0.02 * metrics["daily_voltage_cost_normalized"],
                places=5,
            )
            self.assertAlmostEqual(metrics["daily_voltage_cost"], metrics["daily_voltage_cost_raw"])
            self.assertAlmostEqual(metrics["cost_budget_raw"], 0.02)
            self.assertAlmostEqual(metrics["cost_budget_normalized"], 1.0)
        finally:
            trainer.close()

    def test_fixed_penalty_mappo_uses_explicit_dimensionless_coefficient(self):
        config = self._config()
        config.update(
            {
                "voltage_cost_scale": 0.02,
                "fixed_cost_penalty_coef": 1.0,
            }
        )
        trainer = SafeGRUMAPPOTrainer(config)
        try:
            metrics = trainer.update(
                trainer.collect_rollout(), algorithm="mappo_penalty"
            )
            self.assertEqual(metrics["algorithm_cost_mode"], "fixed_penalty")
            self.assertAlmostEqual(metrics["fixed_cost_penalty_coef"], 1.0)
        finally:
            trainer.close()

    def test_no_communication_actor_keeps_only_local_action_history_and_supply_facts(self):
        config = self._config()
        config.update(
            {
                "include_previous_action": True,
                "include_transaction_message": False,
                "two_stage_intent": False,
                "h2_supply_intent_message_enable": False,
                "env_overrides": {
                    **self._config()["env_overrides"],
                    "h2_local_supply_facts_enable": True,
                },
            }
        )
        trainer = SafeGRUMAPPOTrainer(config)
        try:
            self.assertEqual(trainer.transaction_message_dim, 0)
            self.assertEqual(trainer.obs_dim, trainer.base_obs_dim + trainer.action_dim)
            self.assertFalse(trainer.two_stage_intent)
            self.assertEqual(trainer.supply_message_dim, 0)
            rollout = trainer.collect_rollout()
            self.assertEqual(rollout.local_obs.shape[-1], trainer.obs_dim)
        finally:
            trainer.close()

    def test_fused_rollout_kernel_matches_legacy_forward_with_fixed_rng(self):
        config = self._config()
        config["fused_rollout_kernel"] = True
        trainer = SafeGRUMAPPOTrainer(config)
        try:
            parity = trainer.rollout_kernel_parity(update_index=1)
            self.assertEqual(parity["fields_compared"], 10)
            self.assertLessEqual(parity["max_abs_difference"], 1e-6)
        finally:
            trainer.close()

    def test_optional_action_history_and_transaction_message_are_broadcast_to_actors(self):
        config = self._config()
        config.update(
            {
                "include_previous_action": True,
                "include_transaction_message": True,
            }
        )
        trainer = SafeGRUMAPPOTrainer(config)
        rollout = trainer.collect_rollout()

        base_dim = trainer.base_obs_dim
        action_end = base_dim + trainer.action_dim
        self.assertEqual(trainer.obs_dim, action_end + 2 * trainer.num_agents)
        np.testing.assert_allclose(rollout.local_obs[0, 0, :, base_dim:], 0.0)
        np.testing.assert_allclose(
            rollout.local_obs[1, 0, :, base_dim:action_end], rollout.actions[0, 0]
        )
        np.testing.assert_allclose(
            rollout.local_obs[1, 0, :, action_end:],
            np.broadcast_to(
                rollout.local_obs[1, 0, 0, action_end:],
                rollout.local_obs[1, 0, :, action_end:].shape,
            ),
        )
        trainer.close()

    def test_self_only_masks_other_agents_historical_transactions(self):
        config = self._config()
        config.update(
            {
                "include_transaction_message": True,
                "communication_scope": "self_only",
            }
        )
        trainer = SafeGRUMAPPOTrainer(config)
        try:
            raw = np.arange(1, 2 * trainer.num_agents + 1, dtype=np.float32)[None]
            flat = np.zeros(
                (trainer.num_agents, trainer.base_obs_dim), dtype=np.float32
            )
            shaped = np.asarray(
                trainer._reshape_local_obs(flat, transaction_messages=raw)
            )
            history = shaped[0, :, trainer.base_obs_dim:]
            for agent_index in range(trainer.num_agents):
                expected = np.zeros(2 * trainer.num_agents, dtype=np.float32)
                expected[agent_index] = raw[0, agent_index]
                expected[trainer.num_agents + agent_index] = raw[
                    0, trainer.num_agents + agent_index
                ]
                np.testing.assert_allclose(history[agent_index], expected)
        finally:
            trainer.close()

    def test_deterministic_rollout_supports_history_and_transaction_features(self):
        config = self._config()
        config.update(
            {
                "include_previous_action": True,
                "include_transaction_message": True,
            }
        )
        trainer = SafeGRUMAPPOTrainer(config)
        try:
            report = trainer.deterministic_rollout(seed=30)
            self.assertEqual(report["summary"]["steps"], 4)
        finally:
            trainer.close()

    def test_deterministic_rollout_supports_history_and_eta_counterfactuals(self):
        config = self._config()
        config.update(
            {
                "include_previous_action": True,
                "include_transaction_message": True,
                "two_stage_intent": True,
                "intent_residual_limit": 0.25,
                "env_overrides": {
                    **self._config()["env_overrides"],
                    "h2_learnable_rolling_order_enable": True,
                },
            }
        )
        trainer = SafeGRUMAPPOTrainer(config)
        try:
            report = trainer.deterministic_rollout(
                seed=30, history_off=True, eta_delay_hours=2
            )
            self.assertEqual(report["summary"]["steps"], 4)
            self.assertEqual(report["summary"]["eta_delay_hours"], 2)
            self.assertTrue(report["summary"]["history_off"])
            self.assertIn("intent_action_mae", report["summary"])
            self.assertIn("h2_late_order_energy", report["steps"][0])
        finally:
            trainer.close()

    def test_deterministic_rollout_separates_hidden_and_previous_action_ablations(self):
        config = self._config()
        config.update({"include_previous_action": True})
        trainer = SafeGRUMAPPOTrainer(config)
        try:
            hidden_off = trainer.deterministic_rollout(
                seed=30, gru_hidden_off=True
            )
            action_off = trainer.deterministic_rollout(
                seed=30, previous_action_off=True
            )
        finally:
            trainer.close()

        self.assertTrue(hidden_off["summary"]["gru_hidden_off"])
        self.assertFalse(hidden_off["summary"]["previous_action_off"])
        self.assertFalse(action_off["summary"]["gru_hidden_off"])
        self.assertTrue(action_off["summary"]["previous_action_off"])

    def test_lagrangian_update_trains_both_centralized_critics(self):
        trainer = SafeGRUMAPPOTrainer(self._config())
        metrics = trainer.update(trainer.collect_rollout(), algorithm="lagrangian")

        self.assertIn("actor_loss", metrics)
        self.assertIn("reward_critic_loss", metrics)
        self.assertIn("cost_critic_loss", metrics)
        self.assertIn("lagrange_multiplier", metrics)
        trainer.close()


    def test_macpo_update_reports_sample_kl_and_global_cost_check(self):
        config = self._config()
        config["cost_budget"] = 100.0
        config["macpo_max_kl"] = 0.05
        trainer = SafeGRUMAPPOTrainer(config)
        metrics = trainer.update(trainer.collect_rollout(), algorithm="macpo")

        self.assertIn("accepted", metrics)
        self.assertIn("kl_after", metrics)
        self.assertIn("cost_after", metrics)
        self.assertLessEqual(float(metrics["kl_after"]), 0.05 + 1e-6)
        self.assertLessEqual(float(metrics["cost_after"]), 100.0 + 1e-6)
        trainer.close()

    def test_macpo_kl_compares_effective_clipped_log_stds_on_both_sides(self):
        config = self._config()
        config.update({"cost_budget": 10.0, "log_std_max": -1.0})
        trainer = SafeGRUMAPPOTrainer(config)
        try:
            rollout = trainer.collect_rollout()
            raw_params = unfreeze(trainer.actor_state.params)
            raw_params["params"]["log_std"] = jnp.full_like(
                raw_params["params"]["log_std"], 2.0
            )
            trainer.actor_state = trainer.actor_state.replace(params=freeze(raw_params))
            metrics = trainer.update(rollout, algorithm="macpo")

            self.assertLessEqual(abs(float(metrics["kl_before"])), 1e-6)
            self.assertLessEqual(float(metrics["kl_after"]), 0.01 + 1e-6)
        finally:
            trainer.close()

    def test_annealing_schedule_reaches_zero_at_the_requested_update_count(self):
        config = self._config()
        config.update({"anneal_lr": True, "total_updates": 4})
        trainer = SafeGRUMAPPOTrainer(config)

        self.assertAlmostEqual(trainer.learning_rate(0), config["lr"])
        self.assertAlmostEqual(trainer.learning_rate(4), 0.0)
        trainer.close()

    def test_curriculum_interpolates_budget_and_log_std_cap(self):
        config = self._config()
        config.update(
            {
                "cost_budget": 0.4,
                "curriculum_d_start": 3.5,
                "curriculum_d_target": 0.5,
                "curriculum_updates": 200,
                "curriculum_log_std_start": -1.0,
                "curriculum_log_std_end": -2.3,
            }
        )
        trainer = SafeGRUMAPPOTrainer(config)
        try:
            self.assertAlmostEqual(trainer.current_cost_budget(1), 3.5)
            self.assertAlmostEqual(trainer.current_cost_budget(200), 0.5)
            self.assertAlmostEqual(trainer.current_log_std_max(1), -1.0)
            self.assertAlmostEqual(trainer.current_log_std_max(200), -2.3)
        finally:
            trainer.close()

    def test_deterministic_rollout_reports_economic_and_safety_metrics(self):
        trainer = SafeGRUMAPPOTrainer(self._config())
        report = trainer.deterministic_rollout(seed=30)

        self.assertEqual(report["summary"]["steps"], 4)
        self.assertIn("economic_cost", report["summary"])
        self.assertIn("daily_voltage_cost", report["summary"])
        self.assertIn("pf_failure_rate", report["summary"])
        self.assertEqual(len(report["steps"]), 4)
        self.assertEqual(np.asarray(report["steps"][0]["actions"]).shape, (4, 5))
        trainer.close()

    def test_deterministic_rollout_survives_power_flow_failure(self):
        trainer = SafeGRUMAPPOTrainer(self._config())
        with patch("envs.microgrid.power_flow.runpf", side_effect=ArithmeticError):
            report = trainer.deterministic_rollout(seed=30)

        self.assertEqual(report["summary"]["pf_failure_rate"], 1.0)
        self.assertIsNone(report["summary"]["voltage_min_pu"])
        self.assertIsNone(report["summary"]["voltage_max_pu"])
        self.assertTrue(all(not record["pf_converged"] for record in report["steps"]))
        trainer.close()

if __name__ == "__main__":
    unittest.main()
