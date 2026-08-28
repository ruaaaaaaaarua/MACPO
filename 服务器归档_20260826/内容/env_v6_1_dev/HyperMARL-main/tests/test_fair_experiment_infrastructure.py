import unittest

import numpy as np

from baselines.utils.microgrid_vec_env import MicrogridVecEnv
from envs.microgrid.config import MICROGRID_CONFIG


class InstanceConfigIsolationTest(unittest.TestCase):
    def test_per_environment_overrides_do_not_mutate_global_config(self):
        original_split = MICROGRID_CONFIG["italian_split_name"]
        original_days = MICROGRID_CONFIG.get("italian_day_indices")
        env = MicrogridVecEnv(
            num_envs=2,
            auto_reset=False,
            config_overrides_by_env=[
                {"italian_split_name": "validation", "italian_day_indices": [8]},
                {"italian_split_name": "validation", "italian_day_indices": [17]},
            ],
        )
        try:
            first_cfg = env.envs[0].env.cfg
            second_cfg = env.envs[1].env.cfg
            self.assertEqual(first_cfg["italian_day_indices"], [8])
            self.assertEqual(second_cfg["italian_day_indices"], [17])
            self.assertEqual(first_cfg["italian_split_name"], "validation")
            self.assertIsNot(first_cfg, second_cfg)
            self.assertEqual(MICROGRID_CONFIG["italian_split_name"], original_split)
            self.assertEqual(MICROGRID_CONFIG.get("italian_day_indices"), original_days)
        finally:
            env.close()

    def test_fixed_seed_reset_is_reproducible(self):
        env = MicrogridVecEnv(
            num_envs=1,
            auto_reset=False,
            config_overrides={
                "italian_split_enable": True,
                "italian_split_name": "validation",
                "italian_day_indices": [8],
            },
        )
        try:
            first, _ = env.reset(seed=4200)
            second, _ = env.reset(seed=4200)
            np.testing.assert_allclose(first, second, rtol=0.0, atol=0.0)
        finally:
            env.close()

    def test_rolling_order_override_expands_action_space(self):
        env = MicrogridVecEnv(
            num_envs=1,
            config_overrides={"h2_learnable_rolling_order_enable": True},
        )
        try:
            self.assertEqual(env.action_dim, 6)
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
