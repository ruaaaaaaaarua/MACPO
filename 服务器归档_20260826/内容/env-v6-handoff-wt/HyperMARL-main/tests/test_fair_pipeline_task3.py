import json
import tempfile
import unittest
from pathlib import Path

import scripts.run_fair_stas_pipeline as pipeline


TRAIN_DAYS = [0, 2, 3, 4, 5, 6, 9, 10, 11, 12, 13, 15, 16, 18, 19, 20, 22, 25, 26, 27]
VALIDATION_DAYS = [8, 17, 21, 23]
TEST_DAYS = [1, 7, 14, 24]


class FairPipelineTask3Test(unittest.TestCase):
    def test_default_root_and_single_seed_constants_are_locked(self):
        self.assertEqual(
            pipeline.DEFAULT_OUTPUT.name,
            "fair-stas-h2-action-order-20260715",
        )
        self.assertEqual(getattr(pipeline, "TRAIN_SEED", None), 30)
        self.assertEqual(getattr(pipeline, "VALIDATION_NOISE_SEED", None), 4200)
        self.assertEqual(getattr(pipeline, "TEST_NOISE_SEED", None), 5200)

    def test_specs_lock_manifest_training_and_validation_evaluation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            specs = pipeline.build_specs(root, episodes=100)

        self.assertEqual(len(specs), 6)
        for spec in specs.values():
            command = " ".join(spec.command)
            if spec.kind == "torch":
                arg_index = spec.command.index("--microgrid-overrides-json") + 1
                overrides = json.loads(spec.command[arg_index])
                self.assertEqual(overrides["italian_split_strategy"], "manifest")
                self.assertEqual(overrides["italian_split_name"], "train")
                self.assertTrue(overrides["h2_learnable_rolling_order_enable"])
                self.assertEqual(
                    overrides["h2_learnable_rolling_order_agent_indices"],
                    [0, 1, 2, 3],
                )
                self.assertFalse(overrides["h2_buyer_reservation_demand_enable"])
                self.assertIn("--fixed-eval-split validation", command)
                self.assertIn("--fixed-eval-noise-seed 4200", command)
            else:
                self.assertIn("italian_split_strategy:manifest", command)
                self.assertIn("italian_split_name:train", command)
                self.assertIn("h2_learnable_rolling_order_enable:true", command)
                self.assertIn(
                    "h2_learnable_rolling_order_agent_indices:[0,1,2,3]",
                    command,
                )
                self.assertIn("h2_buyer_reservation_demand_enable:false", command)
                self.assertIn("+FIXED_EVAL_SPLIT=validation", command)
                self.assertIn("+FIXED_EVAL_NOISE_SEED=4200", command)

    def test_manifest_records_provenance_units_splits_and_four_item_grids(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            specs = pipeline.build_specs(root, episodes=100)
            pipeline._manifest(root, 100, specs)
            payload = json.loads((root / "manifest.json").read_text())

        self.assertTrue(payload["branch"])
        self.assertEqual(len(payload["commit"]), 40)
        self.assertEqual(payload["output_root"], str(root))
        self.assertEqual(payload["qmax_basis"], "peak_heat_hour")
        self.assertEqual(payload["train_days"], TRAIN_DAYS)
        self.assertEqual(payload["validation_days"], VALIDATION_DAYS)
        self.assertEqual(payload["test_days"], TEST_DAYS)
        self.assertEqual(
            payload["seeds"],
            {"training": 30, "validation": 4200, "test": 5200},
        )
        self.assertEqual(
            payload["evaluation_grids"]["validation"],
            [{"day": day, "seed": 4200} for day in VALIDATION_DAYS],
        )
        self.assertEqual(
            payload["evaluation_grids"]["test"],
            [{"day": day, "seed": 5200} for day in TEST_DAYS],
        )
        self.assertEqual(len(payload["evaluation_grids"]["validation"]), 4)
        self.assertEqual(len(payload["evaluation_grids"]["test"]), 4)

        overrides = payload["abc_overrides"]
        self.assertEqual(overrides["italian_split_strategy"], "manifest")
        self.assertEqual(overrides["italian_split_name"], "train")
        self.assertEqual(
            overrides["h2_learnable_rolling_order_agent_indices"],
            [0, 1, 2, 3],
        )
        self.assertFalse(overrides["h2_buyer_reservation_demand_enable"])

        prices = payload["h2_prices"]
        self.assertEqual(prices["internal_unit"], "yuan/kWh-H2")
        self.assertEqual(prices["display_unit"], "yuan/kg")
        lhv = float(prices["lhv_kwh_per_kg"])
        for key, value in prices["yuan_per_kwh_h2"].items():
            self.assertAlmostEqual(
                prices["yuan_per_kg"][key],
                float(value) * lhv,
                places=10,
            )
        self.assertAlmostEqual(
            prices["yuan_per_kg"]["lambda_h2_buy"], 45.0, places=8
        )
        self.assertAlmostEqual(
            prices["yuan_per_kg"]["h2_price_max"], 30.0, places=8
        )

    def test_fresh_root_rejects_existing_artifacts_without_writing(self):
        ensure_fresh = getattr(pipeline, "_ensure_fresh_output_root", None)
        self.assertIsNotNone(
            ensure_fresh,
            "fresh output-root guard has not been implemented",
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sentinel = root / "old-checkpoint.bin"
            sentinel.write_bytes(b"old")
            before = sentinel.read_bytes()
            with self.assertRaisesRegex(FileExistsError, "refusing to mix"):
                ensure_fresh(root)
            self.assertEqual(sentinel.read_bytes(), before)
            self.assertEqual(list(root.iterdir()), [sentinel])

    def test_fresh_specs_never_reference_old_result_roots(self):
        specs = pipeline.build_specs(pipeline.DEFAULT_OUTPUT / "smoke_100", 100)
        old_root = "fair-stas-results-20260710"
        self.assertNotIn(old_root, str(pipeline.DEFAULT_OUTPUT))
        for spec in specs.values():
            self.assertNotIn(old_root, " ".join(spec.command))
            for checkpoint in spec.checkpoints:
                self.assertTrue(checkpoint.is_relative_to(pipeline.DEFAULT_OUTPUT))


if __name__ == "__main__":
    unittest.main()
