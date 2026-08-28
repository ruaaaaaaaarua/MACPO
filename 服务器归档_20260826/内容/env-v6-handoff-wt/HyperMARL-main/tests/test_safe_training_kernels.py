import unittest

from baselines.MAPPO.safe_gru_trainer import SafeGRUMAPPOTrainer


class SafeTrainingKernelsTest(unittest.TestCase):
    def test_warmup_compiles_the_fixed_shape_gru_training_kernels(self):
        trainer = SafeGRUMAPPOTrainer(
            {
                "seed": 30,
                "num_envs": 1,
                "num_steps": 4,
                "hidden_size": 16,
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
        )
        try:
            self.assertEqual(
                trainer.warmup_training_kernels(),
                {"actor": True, "reward_critic": True, "cost_critic": True, "gae": True},
            )
        finally:
            trainer.close()


if __name__ == "__main__":
    unittest.main()
