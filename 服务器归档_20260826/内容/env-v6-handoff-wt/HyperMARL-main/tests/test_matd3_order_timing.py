import unittest

import numpy as np

from scripts import evaluate_matd3_order_timing as timing


class MATD3OrderTimingTransformationTest(unittest.TestCase):
    def test_variants_preserve_non_a5_actions_boundary_and_budget(self):
        actions = (
            np.arange(24 * 4 * 6, dtype=np.float32).reshape(24, 4, 6) / 1000
        )

        variants = timing.build_a5_variants(actions)

        invariants = timing.validate_variant_invariants(actions, variants)
        self.assertEqual(tuple(variants), timing.VARIANT_NAMES)
        for transformed in variants.values():
            np.testing.assert_array_equal(
                transformed[:, :, :5], actions[:, :, :5]
            )
            np.testing.assert_array_equal(
                transformed[20:, :, 5], actions[20:, :, 5]
            )
            np.testing.assert_allclose(
                transformed[:20, :, 5].sum(axis=0),
                actions[:20, :, 5].sum(axis=0),
                atol=1e-6,
            )
        self.assertEqual(tuple(invariants), timing.VARIANT_NAMES)
        self.assertTrue(
            all(item["budget_preserved"] for item in invariants.values())
        )

    def test_shift_shuffle_and_constant_mean_definitions(self):
        actions = np.zeros((24, 4, 6), dtype=np.float32)
        actions[:20, :, 5] = np.arange(20, dtype=np.float32)[:, None]

        variants = timing.build_a5_variants(actions)

        np.testing.assert_array_equal(
            variants["shift_earlier_4"][:20, 0, 5],
            np.roll(actions[:20, 0, 5], -4),
        )
        np.testing.assert_array_equal(
            variants["shift_later_4"][:20, 0, 5],
            np.roll(actions[:20, 0, 5], 4),
        )
        shuffle_indices = np.random.default_rng(20260715).permutation(20)
        np.testing.assert_array_equal(
            variants["shuffle_fixed"][:20, :, 5],
            actions[shuffle_indices, :, 5],
        )
        expected_mean = np.broadcast_to(
            actions[:20, :, 5].mean(axis=0), (20, 4)
        )
        np.testing.assert_allclose(
            variants["constant_mean"][:20, :, 5],
            expected_mean,
        )

    def test_constant_mean_preserves_budget_for_normalized_float32_actions(self):
        actions = np.random.default_rng(38).uniform(
            -1.0, 1.0, size=(24, 4, 6)
        ).astype(np.float32)

        transformed = timing.build_a5_variants(actions)["constant_mean"]

        np.testing.assert_allclose(
            transformed[:20, :, 5].sum(axis=0, dtype=np.float64),
            actions[:20, :, 5].sum(axis=0, dtype=np.float64),
            rtol=0.0,
            atol=1e-6,
        )

    def test_validate_variant_invariants_rejects_budget_changes(self):
        actions = np.zeros((24, 4, 6), dtype=np.float32)
        changed = actions.copy()
        changed[0, 0, 5] = 0.5

        with self.assertRaisesRegex(ValueError, "budget"):
            timing.validate_variant_invariants(actions, {"changed": changed})


if __name__ == "__main__":
    unittest.main()
