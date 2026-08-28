import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from envs.microgrid.microgrid_env import MicrogridEnv
from scripts.run_traffic_experiment import (
    build_traffic_specs,
    traffic_group_abc_overrides,
    write_traffic_manifest,
)


class TrafficExperimentPipelineTest(unittest.TestCase):
    def test_traffic_overrides_lock_environment_contract(self):
        overrides = traffic_group_abc_overrides()
        self.assertTrue(overrides["h2_traffic_enable"])
        self.assertTrue(overrides["h2_route_action_enable"])
        self.assertEqual(overrides["h2_traffic_min_eta"], 4)
        self.assertEqual(overrides["h2_traffic_max_eta"], 6)
        self.assertEqual(overrides["h2_pending_obs_horizon"], 6)
        self.assertEqual(overrides["h2_delivery_reservation_horizon"], 6)
        self.assertEqual(overrides["h2_traffic_morning_peak_amplitude"], 1.0)
        self.assertEqual(overrides["h2_traffic_evening_peak_amplitude"], 1.1)
        self.assertEqual(overrides["h2_traffic_directional_phase_hours"], 4.0)
        self.assertFalse(overrides["h2_buyer_reservation_demand_enable"])
        env = MicrogridEnv(overrides)
        self.assertEqual((env.agent_num, env.T, env.obs_dim, env.action_dim), (4, 24, 24, 7))

    def test_smoke_specs_have_four_locked_candidates_and_dynamic_dimensions(self):
        with tempfile.TemporaryDirectory() as tmp:
            specs = build_traffic_specs(Path(tmp), episodes=100)
        self.assertEqual(
            set(specs),
            {"mappo_256", "matd3_256", "stas_mix010", "stas_mix020"},
        )
        for name, spec in specs.items():
            command = " ".join(spec.command)
            self.assertIn("h2_traffic_enable:true", command.replace('"', "").replace(" ", ""))
            if name == "matd3_256":
                arg = spec.command[spec.command.index("--microgrid-overrides-json") + 1]
                overrides = json.loads(arg)
                self.assertTrue(overrides["h2_route_action_enable"])
                self.assertIn("--episodes 100", command)
            else:
                self.assertIn("TOTAL_TIMESTEPS=2400", command)
        self.assertIn("+STAS.MAX_MIX_COEF=0.1", " ".join(specs["stas_mix010"].command))
        self.assertIn("+STAS.MAX_MIX_COEF=0.2", " ".join(specs["stas_mix020"].command))

    def test_manifest_records_traffic_contract_and_runtime_dimensions(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            specs = build_traffic_specs(root, episodes=100)
            with patch(
                "scripts.run_traffic_experiment._git_metadata",
                return_value=("codex/traffic", "a" * 40),
            ):
                write_traffic_manifest(root, 100, specs)
            payload = json.loads((root / "manifest.json").read_text())
        self.assertEqual(payload["branch"], "codex/traffic")
        self.assertEqual(payload["commit"], "a" * 40)
        self.assertEqual(payload["dimensions"], {
            "agents": 4,
            "episode_steps": 24,
            "obs_dim": 24,
            "action_dim": 7,
        })
        self.assertEqual(payload["traffic"]["eta_range"], [4, 6])
        self.assertEqual(payload["stas_mix_candidates"], [0.1, 0.2])


if __name__ == "__main__":
    unittest.main()
