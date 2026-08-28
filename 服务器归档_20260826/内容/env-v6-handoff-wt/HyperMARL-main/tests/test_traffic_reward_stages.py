import tempfile
import unittest
from pathlib import Path

import numpy as np

from baselines.utils.fixed_scenario_eval import _same_time_correlation
from envs.microgrid.microgrid_env import MicrogridEnv
from scripts.run_traffic_stas_reward_stages import (
    LHV_H2,
    STAGES,
    _price_metadata,
    build_specs,
    stage_overrides,
)


class TrafficRewardStagesTest(unittest.TestCase):
    def test_stage_prices_terminal_contract_and_mix_schedule(self):
        expected = {
            "stage60_no_terminal": 60.0,
            "stage45_no_terminal": 45.0,
            "stage45_terminal20": 45.0,
        }
        with tempfile.TemporaryDirectory() as tmp:
            specs = build_specs(Path(tmp), episodes=10000)
            for name in STAGES:
                overrides = stage_overrides(name)
                prices = _price_metadata(overrides)
                self.assertAlmostEqual(
                    prices["effective_external_cost_yuan_per_kg"],
                    expected[name],
                    places=6,
                )
                command = specs[name].command
                self.assertIn("+STAS.RAMP_EPISODES=4000", command)
                self.assertIn("+STAS.MAX_MIX_COEF=0.05", command)
                self.assertTrue(
                    any(item.startswith("+BEST_VALIDATION_CHECKPOINT_DIR=") for item in command)
                )
            terminal = stage_overrides("stage45_terminal20")
            self.assertEqual(
                terminal["terminal_h2_shortfall_value_targets"], [0.20] * 4
            )
            self.assertEqual(
                terminal["terminal_h2_shortfall_value_agent_indices"], [0, 1, 2, 3]
            )
            self.assertTrue(terminal["terminal_h2_settlement_in_reward_enable"])

    def test_terminal_settlement_is_exact_and_only_changes_terminal_total_cost(self):
        config = stage_overrides("stage45_terminal20")
        config.update(
            {
                "profile_source": "synthetic",
                "italian_split_enable": False,
            }
        )
        env = MicrogridEnv(config)
        env.seed(7)
        env.reset()
        for key in ("pv", "wt", "load_e", "load_h"):
            env.profiles[key] = np.zeros((env.agent_num, env.T), dtype=np.float32)
        env.h2_level = env.h2_tank_cap * 0.05
        env.t = env.T - 1
        actions = np.zeros((env.agent_num, env.action_dim), dtype=np.float32)
        actions[:, 0] = -1.0
        actions[:, 5] = -1.0
        actions[:, 6] = -1.0
        info = env.step(actions)[3][0]
        self.assertAlmostEqual(info["terminal_h2_shortfall_kg"], 240.0, places=3)
        self.assertAlmostEqual(info["terminal_h2_settlement_cost"], 10800.0, places=2)
        self.assertAlmostEqual(
            info["total_cost"] - info["base_cost"], 10800.0, places=2
        )
        self.assertAlmostEqual(
            info["effective_external_h2_cost_yuan_per_kg"], 45.0, places=6
        )

    def test_arrival_load_correlation_uses_same_time_agent_values(self):
        self.assertAlmostEqual(
            _same_time_correlation([0.0, 1.0, 2.0], [0.0, 2.0, 4.0]),
            1.0,
            places=12,
        )
        self.assertIsNone(_same_time_correlation([1.0, 1.0], [0.0, 1.0]))


if __name__ == "__main__":
    unittest.main()
