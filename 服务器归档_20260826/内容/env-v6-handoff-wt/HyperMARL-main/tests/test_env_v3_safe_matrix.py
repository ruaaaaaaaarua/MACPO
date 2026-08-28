import unittest

from scripts.run_env_v3_safe_matrix import EXPERIMENTS, build_gru_config


class EnvV3SafeMatrixTest(unittest.TestCase):
    def test_matrix_keeps_v3_baselines_and_adds_exploratory_v4_variants(self):
        self.assertTrue(
            {
                "dense_ff_mappo_anneal",
                "dense_gru_mappo_anneal",
                "dense_gru_mappo_lagrangian",
                "dense_gru_macpo",
                "dense_gru_macpo_history_communication",
            }.issubset(EXPERIMENTS)
        )
        self.assertTrue(
            {"v4_full_intent_curriculum", "v4_no_broadcast_curriculum"}.issubset(
                EXPERIMENTS
            )
        )
        self.assertTrue(all(spec["seed"] == 30 for spec in EXPERIMENTS.values()))
        self.assertTrue(all(spec["exploratory"] for spec in EXPERIMENTS.values()))

    def test_gru_variants_use_the_dense_safe_environment_and_global_cost_budget(self):
        config = build_gru_config("dense_gru_mappo_lagrangian", updates=2)

        self.assertEqual(config["seed"], 30)
        self.assertEqual(config["gamma"], 1.0)
        self.assertEqual(config["cost_budget"], 0.0)
        self.assertTrue(config["env_overrides"]["power_flow_enable"])
        self.assertEqual(config["env_overrides"]["reward_emission_mode"], "dense")

    def test_history_communication_variant_exposes_previous_actions_and_public_trades(self):
        config = build_gru_config("dense_gru_macpo_history_communication", updates=2)

        self.assertTrue(config["include_previous_action"])
        self.assertTrue(config["include_transaction_message"])

    def test_v52_matrix_has_the_three_locked_single_seed_variants(self):
        names = {
            "v52_full_gru_mappo",
            "v52_full_gru_macpo",
            "v52_self_only_gru_macpo",
        }
        self.assertTrue(names.issubset(EXPERIMENTS))
        full_mappo = build_gru_config("v52_full_gru_mappo", updates=1000)
        full_macpo = build_gru_config("v52_full_gru_macpo", updates=1000)
        self_only = build_gru_config("v52_self_only_gru_macpo", updates=1000)
        self.assertEqual(EXPERIMENTS["v52_full_gru_mappo"]["algorithm"], "mappo")
        self.assertEqual(EXPERIMENTS["v52_full_gru_macpo"]["algorithm"], "macpo")
        self.assertEqual(full_mappo["communication_scope"], "full")
        self.assertEqual(full_macpo["communication_scope"], "full")
        self.assertEqual(self_only["communication_scope"], "self_only")
        self.assertTrue(full_macpo["h2_supply_intent_message_enable"])
        self.assertEqual(full_macpo["curriculum_updates"], 300)
        self.assertEqual(full_macpo["curriculum_d_target"], 0.0)
        self.assertEqual(full_macpo["curriculum_log_std_start"], -1.0)
        self.assertEqual(full_macpo["curriculum_log_std_end"], -2.3)
        self.assertEqual(full_macpo["env_overrides"]["soc_init"], 0.5)


if __name__ == "__main__":
    unittest.main()
