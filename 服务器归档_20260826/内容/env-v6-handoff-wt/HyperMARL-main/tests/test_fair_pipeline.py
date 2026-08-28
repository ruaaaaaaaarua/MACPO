import tempfile
import unittest
from pathlib import Path

from scripts.run_fair_stas_pipeline import build_specs


class FairPipelineSpecTest(unittest.TestCase):
    def test_smoke_contains_all_six_candidates_with_fixed_eval(self):
        with tempfile.TemporaryDirectory() as tmp:
            specs = build_specs(Path(tmp), episodes=100)
        self.assertEqual(
            set(specs),
            {
                "legacy_mappo_128",
                "stable_mappo_128",
                "stable_mappo_256",
                "matd3_256",
                "stas_causal",
                "stas_bidirectional",
            },
        )
        for name, spec in specs.items():
            command_text = " ".join(spec.command)
            if name == "matd3_256":
                self.assertIn("--episodes 100", command_text)
            else:
                self.assertIn("TOTAL_TIMESTEPS=2400", command_text)
            self.assertTrue(str(spec.eval_jsonl).endswith("validation_eval.jsonl"))
            self.assertTrue(spec.checkpoints)

    def test_stage_one_uses_stable_ppo_and_untouched_matd3_capacity(self):
        with tempfile.TemporaryDirectory() as tmp:
            specs = build_specs(Path(tmp), episodes=10000)
        stable = " ".join(specs["stable_mappo_256"].command)
        self.assertIn("+POLICY_MODE=squashed_gaussian", stable)
        self.assertIn("ACTOR_LAYERS=[256,256]", stable)
        self.assertIn("ACTIVATION=relu", stable)
        self.assertIn("ENT_COEF=0.0", stable)
        matd3 = " ".join(specs["matd3_256"].command)
        self.assertIn("--hidden-dim 256", matd3)
        self.assertIn("--episodes 10000", matd3)
        self.assertIn("--eval-interval-episodes 500", matd3)


if __name__ == "__main__":
    unittest.main()
