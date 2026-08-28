import unittest

import numpy as np

from baselines.utils.microgrid_vec_env import MicrogridVecEnv


class MicrogridProcessVectorEnvTest(unittest.TestCase):
    def test_process_backend_matches_serial_safe_step(self):
        overrides = {
            "profile_source": "synthetic",
            "italian_split_enable": False,
            "episode_length": 2,
            "reward_emission_mode": "dense",
            "power_flow_enable": True,
            "penalty_enable": False,
            "action_reg_enable": False,
        }
        serial = MicrogridVecEnv(2, auto_reset=False, config_overrides=overrides)
        process = MicrogridVecEnv(
            2,
            auto_reset=False,
            config_overrides=overrides,
            parallel_backend="process",
        )
        try:
            serial_obs, _ = serial.reset(seed=30)
            process_obs, _ = process.reset(seed=30)
            np.testing.assert_allclose(process_obs, serial_obs)
            actions = np.zeros((2, serial.num_agents, serial.action_dim), dtype=np.float32)
            serial_step = serial.step(actions)
            process_step = process.step(actions)
            for process_value, serial_value in zip(process_step[:4], serial_step[:4]):
                np.testing.assert_allclose(process_value, serial_value)
            self.assertEqual(
                process_step[4][0]["pf_converged"], serial_step[4][0]["pf_converged"]
            )
            self.assertAlmostEqual(
                process_step[4][0]["voltage_cost"], serial_step[4][0]["voltage_cost"]
            )
        finally:
            serial.close()
            process.close()


if __name__ == "__main__":
    unittest.main()
