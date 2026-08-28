import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


try:
    from scripts.run_paper_stas_sparse import (
        build_specs,
        sparse_traffic_overrides,
        write_manifest,
    )
except ImportError:
    build_specs = None
    sparse_traffic_overrides = None
    write_manifest = None

try:
    from scripts.finalize_paper_stas_sparse import selected_best_checkpoints
except ImportError:
    selected_best_checkpoints = None


class PaperSparsePipelineTest(unittest.TestCase):
    def _require_pipeline(self):
        self.assertIsNotNone(build_specs, "paper sparse launcher is missing")

    def test_stage_a_cost_and_sparse_reward_contract(self):
        self._require_pipeline()
        overrides = sparse_traffic_overrides()
        self.assertEqual(overrides["reward_emission_mode"], "terminal_total")
        self.assertAlmostEqual(overrides["lambda_h2_buy"] * 33.33, 45.0, places=6)
        self.assertTrue(overrides["external_h2_dependency_penalty_enable"])
        self.assertEqual(overrides["external_h2_dependency_penalty_kg"], 15.0)
        self.assertFalse(overrides["terminal_h2_settlement_in_reward_enable"])
        self.assertTrue(overrides["h2_traffic_enable"])
        self.assertEqual(
            (overrides["h2_traffic_min_eta"], overrides["h2_traffic_max_eta"]),
            (4, 6),
        )
        self.assertEqual(overrides["h2_transport_loss"], 0.0)

    def test_smoke_has_three_algorithms_and_exercises_full_paper_path(self):
        self._require_pipeline()
        with tempfile.TemporaryDirectory() as tmp:
            specs = build_specs(Path(tmp), episodes=100, smoke=True)
        self.assertEqual(set(specs), {"stas", "mappo", "matd3"})
        for spec in specs.values():
            command = " ".join(spec.command)
            if spec.name == "matd3":
                self.assertIn("--gamma 1.0", command)
                raw = spec.command[spec.command.index("--microgrid-overrides-json") + 1]
                self.assertEqual(json.loads(raw)["reward_emission_mode"], "terminal_total")
            else:
                self.assertIn("GAMMA=1.0", command)
                self.assertIn("ENT_COEF=0.01", command)
                self.assertIn("POLICY_MODE=squashed_gaussian", command)
        stas = " ".join(specs["stas"].command)
        self.assertEqual(
            sum(
                item.startswith("+BEST_VALIDATION_CHECKPOINT_DIR=")
                for item in specs["stas"].command
            ),
            1,
        )
        self.assertIn("STAS.MODE=paper", stas)
        self.assertIn("STAS.POLICY_WARMUP_EPISODES=40", stas)
        self.assertIn("STAS.REWARD_MODEL_UPDATE_INTERVAL_EPISODES=8", stas)
        self.assertIn("STAS.BATCH_SIZE=16", stas)
        self.assertIn("STAS.REWARD_MODEL_UPDATES_PER_INTERVAL=5", stas)

    def test_formal_manifest_locks_validation_only_and_paper_hyperparameters(self):
        self._require_pipeline()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            specs = build_specs(root, episodes=10000, smoke=False)
            with patch(
                "scripts.run_paper_stas_sparse._git_metadata",
                return_value=("codex/stas-faithful-sparse-reward", "4" * 40),
            ):
                write_manifest(root, 10000, specs, smoke=False)
            manifest = json.loads((root / "manifest.json").read_text())
        stas = " ".join(specs["stas"].command)
        self.assertIn("STAS.EMB_DIM=128", stas)
        self.assertIn("STAS.N_HEADS=4", stas)
        self.assertIn("STAS.N_LAYERS=3", stas)
        self.assertIn("STAS.SAMPLE_NUM=5", stas)
        self.assertIn("STAS.BUFFER_SIZE=15000", stas)
        self.assertIn("STAS.BATCH_SIZE=256", stas)
        self.assertIn("STAS.LR=0.0005", stas)
        self.assertIn("STAS.WEIGHT_DECAY=1e-05", stas)
        self.assertIn("STAS.POLICY_WARMUP_EPISODES=4000", stas)
        self.assertIn("STAS.REWARD_MODEL_UPDATE_INTERVAL_EPISODES=800", stas)
        self.assertIn("STAS.REWARD_MODEL_UPDATES_PER_INTERVAL=50", stas)
        self.assertEqual(manifest["selection_split"], "validation")
        self.assertFalse(manifest["test_accessed"])
        self.assertEqual(manifest["validation_days"], [8, 17, 21, 23])
        self.assertEqual(manifest["validation_noise_seed"], 4200)
        self.assertEqual(manifest["effective_external_h2_cost_yuan_per_kg"], 60.0)

    def test_finalizer_selects_only_best_validation_checkpoints(self):
        self.assertIsNotNone(selected_best_checkpoints, "sparse finalizer is missing")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            expected = {}
            for name, key in (
                ("stas", "jax_checkpoint"),
                ("mappo", "jax_checkpoint"),
                ("matd3", "checkpoint"),
            ):
                best = root / name / "output" / "checkpoints" / "best_validation"
                best.mkdir(parents=True)
                checkpoint = best / f"{name}.ckpt"
                checkpoint.write_bytes(name.encode())
                (best / "best_validation.json").write_text(
                    json.dumps(
                        {
                            key: str(checkpoint),
                            "episode": 500,
                            "validation_return": -10.0,
                        }
                    )
                )
                expected[name.upper()] = checkpoint
            selected = selected_best_checkpoints(root, verify_hashes=False)
        self.assertEqual(selected, expected)


if __name__ == "__main__":
    unittest.main()
