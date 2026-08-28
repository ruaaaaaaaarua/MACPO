import copy
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch


STAS_ROOT = Path(__file__).resolve().parents[1] / "baselines" / "STAS-MAPPO"
if str(STAS_ROOT) not in sys.path:
    sys.path.insert(0, str(STAS_ROOT))

import stas_mappo.checkpoint as checkpoint_module  # noqa: E402
from stas_mappo.conserved_credit import ConservedSTASCreditAssigner  # noqa: E402
from stas_mappo.credit import STASCreditConfig  # noqa: E402
from stas_mappo.diagnostics import build_rollout_record  # noqa: E402


SAVE_CHECKPOINT = getattr(checkpoint_module, "save_credit_assigner_checkpoint", None)
LOAD_CHECKPOINT = getattr(checkpoint_module, "load_credit_assigner_checkpoint", None)


def make_config(**overrides):
    values = {
        "obs_dim": 4,
        "action_dim": 1,
        "n_agents": 2,
        "seq_length": 3,
        "emb_dim": 8,
        "n_heads": 2,
        "n_layers": 1,
        "sample_num": 2,
        "dropout": 0.0,
        "buffer_size": 16,
        "batch_size": 2,
        "warmup_rollouts": 1,
        "updates_per_step": 1,
        "conserve_discounted": True,
        "warmup_episodes": 0,
        "ramp_episodes": 1,
        "negative_patience": 3,
        "eval_mask_seed": 3030,
        "eval_mask_count": 8,
        "device": "cpu",
    }
    values.update(overrides)
    return STASCreditConfig(**values)


def make_assigner(config=None):
    config = config or make_config()
    torch.manual_seed(17)
    assigner = ConservedSTASCreditAssigner(config)
    rng = np.random.default_rng(5)
    obs = rng.normal(size=(10, 2, 3, 4)).astype(np.float32)
    actions = rng.normal(size=(10, 2, 3, 1)).astype(np.float32)
    rewards = rng.normal(size=(10, 2, 3)).astype(np.float32)
    dones = np.zeros((10, 2, 3), dtype=np.float32)
    assigner.add_rollout(obs, actions, rewards, dones)
    assigner.train_if_ready()
    for _ in range(3):
        assigner.gate.mix_coef(assigner.episodes_seen, -0.1)
    assigner.last_explained_variance = -0.1
    assigner.last_conservation_error = 9e-6
    assigner.last_mix_coef = 0.0
    assigner.last_gate_phase = "ev_blocked"
    assigner.last_gate_active = False
    assigner.last_gate_reason = "explained_variance_below_threshold"
    stored = assigner.buffer.storage[0]
    assigner.buffer.storage[0] = (*stored[:4], float(stored[4]) + 0.25)
    build_rollout_record(assigner, update=1, episode=10, global_step=240)
    return assigner


class NestedStateAssertions(unittest.TestCase):
    def assertNestedEqual(self, actual, expected):
        if torch.is_tensor(expected):
            self.assertTrue(torch.is_tensor(actual))
            torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
        elif isinstance(expected, dict):
            self.assertEqual(set(actual), set(expected))
            for key in expected:
                self.assertNestedEqual(actual[key], expected[key])
        elif isinstance(expected, (list, tuple)):
            self.assertEqual(len(actual), len(expected))
            for actual_item, expected_item in zip(actual, expected):
                self.assertNestedEqual(actual_item, expected_item)
        elif isinstance(expected, np.ndarray):
            np.testing.assert_array_equal(actual, expected)
        elif isinstance(expected, float) and np.isnan(expected):
            self.assertTrue(np.isnan(actual))
        else:
            self.assertEqual(actual, expected)

    def assertStorageEqual(self, actual, expected):
        self.assertEqual(len(actual), len(expected))
        for actual_item, expected_item in zip(actual, expected):
            for index in range(4):
                np.testing.assert_array_equal(actual_item[index], expected_item[index])
            self.assertEqual(actual_item[4], expected_item[4])

    def assertStorageIndependent(self, actual, other):
        self.assertEqual(len(actual), len(other))
        for actual_item, other_item in zip(actual, other):
            for index in range(4):
                self.assertFalse(
                    np.shares_memory(actual_item[index], other_item[index])
                )


class CompleteCreditCheckpointTest(NestedStateAssertions):
    def test_file_round_trip_restores_complete_state_and_rng(self):
        self.assertIsNotNone(SAVE_CHECKPOINT, "atomic save API is missing")
        self.assertIsNotNone(LOAD_CHECKPOINT, "atomic load API is missing")
        original = make_assigner()
        restored = ConservedSTASCreditAssigner(original.config)
        expected_train = copy.deepcopy(original.buffer.storage)
        expected_holdout = copy.deepcopy(original.holdout_buffer.storage)
        expected_model = copy.deepcopy(original.model.state_dict())
        expected_optimizer = copy.deepcopy(original.optimizer.state_dict())
        expected_target_error = original.last_target_error
        expected_train_capacity = original.buffer.capacity
        expected_holdout_capacity = original.holdout_buffer.capacity
        restored.buffer.capacity = 1
        restored.holdout_buffer.capacity = 1

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "nested" / "stas.pt"
            np.random.seed(111)
            torch.manual_seed(222)
            result = SAVE_CHECKPOINT(
                path,
                original,
                update=7,
                episode=28,
                global_step=672,
            )
            expected_numpy = np.random.random(5)
            expected_torch = torch.rand(5)
            original.buffer.storage[0][0].fill(999.0)
            original.holdout_buffer.storage[0][2].fill(-999.0)
            np.random.seed(999)
            torch.manual_seed(999)

            metadata = LOAD_CHECKPOINT(path, restored)
            actual_numpy = np.random.random(5)
            actual_torch = torch.rand(5)

            self.assertEqual(result, path)
            self.assertEqual(
                metadata,
                {"update": 7, "episode": 28, "global_step": 672},
            )
            self.assertTrue(path.is_file())
            self.assertFalse(path.with_suffix(path.suffix + ".tmp").exists())
            np.testing.assert_array_equal(actual_numpy, expected_numpy)
            torch.testing.assert_close(actual_torch, expected_torch, rtol=0.0, atol=0.0)

        self.assertStorageEqual(restored.buffer.storage, expected_train)
        self.assertStorageEqual(restored.holdout_buffer.storage, expected_holdout)
        self.assertStorageIndependent(restored.buffer.storage, original.buffer.storage)
        self.assertStorageIndependent(
            restored.holdout_buffer.storage, original.holdout_buffer.storage
        )
        self.assertEqual(restored.buffer.capacity, expected_train_capacity)
        self.assertEqual(restored.holdout_buffer.capacity, expected_holdout_capacity)
        self.assertNestedEqual(restored.model.state_dict(), expected_model)
        self.assertNestedEqual(restored.optimizer.state_dict(), expected_optimizer)
        self.assertEqual(restored.rollouts_seen, original.rollouts_seen)
        self.assertEqual(restored.episodes_seen, original.episodes_seen)
        self.assertEqual(restored.normalizer.count, original.normalizer.count)
        self.assertEqual(restored.normalizer.mean, original.normalizer.mean)
        self.assertEqual(restored.normalizer.m2, original.normalizer.m2)
        self.assertEqual(restored.last_explained_variance, -0.1)
        self.assertEqual(restored.last_conservation_error, 9e-6)
        self.assertEqual(restored.last_mix_coef, 0.0)
        self.assertEqual(restored.last_target_error, expected_target_error)
        self.assertGreater(restored.last_target_error, 0.24)
        self.assertEqual(restored.last_gate_phase, "ev_blocked")
        self.assertFalse(restored.last_gate_active)
        self.assertEqual(
            restored.last_gate_reason,
            "explained_variance_below_threshold",
        )
        self.assertEqual(restored.gate.negative_streak, 3)
        self.assertFalse(restored.gate.disabled)

    def test_in_memory_snapshot_owns_buffer_arrays(self):
        original = make_assigner()
        state = checkpoint_module.credit_assigner_state_dict(original)
        expected_train = copy.deepcopy(state["buffer_storage"])
        expected_holdout = copy.deepcopy(state["holdout_buffer_storage"])
        original.buffer.storage[0][0].fill(1234.0)
        original.holdout_buffer.storage[0][2].fill(-1234.0)
        self.assertStorageEqual(state["buffer_storage"], expected_train)
        self.assertStorageEqual(state["holdout_buffer_storage"], expected_holdout)
        self.assertStorageIndependent(
            state["buffer_storage"], original.buffer.storage
        )
        self.assertStorageIndependent(
            state["holdout_buffer_storage"], original.holdout_buffer.storage
        )

    def test_restored_next_training_step_matches_uninterrupted_step(self):
        self.assertIsNotNone(SAVE_CHECKPOINT, "atomic save API is missing")
        self.assertIsNotNone(LOAD_CHECKPOINT, "atomic load API is missing")
        original = make_assigner()
        restored = ConservedSTASCreditAssigner(original.config)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "stas.pt"
            np.random.seed(333)
            torch.manual_seed(444)
            SAVE_CHECKPOINT(path, original, update=2, episode=10, global_step=240)
            expected_loss = original.train_if_ready()
            expected_model = copy.deepcopy(original.model.state_dict())
            expected_optimizer = copy.deepcopy(original.optimizer.state_dict())
            np.random.seed(999)
            torch.manual_seed(999)
            LOAD_CHECKPOINT(path, restored)
            actual_loss = restored.train_if_ready()

        self.assertAlmostEqual(actual_loss, expected_loss, places=7)
        self.assertNestedEqual(restored.model.state_dict(), expected_model)
        self.assertNestedEqual(restored.optimizer.state_dict(), expected_optimizer)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
    def test_cuda_rng_and_next_training_step_match_after_restore(self):
        original = make_assigner(make_config(device="cuda"))
        restored = ConservedSTASCreditAssigner(original.config)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "stas-cuda.pt"
            np.random.seed(333)
            torch.manual_seed(444)
            torch.cuda.manual_seed_all(555)
            SAVE_CHECKPOINT(path, original, update=2, episode=10, global_step=240)
            expected_cuda = torch.rand(5, device="cuda")
            expected_loss = original.train_if_ready()
            expected_model = copy.deepcopy(original.model.state_dict())
            expected_optimizer = copy.deepcopy(original.optimizer.state_dict())

            np.random.seed(999)
            torch.manual_seed(999)
            torch.cuda.manual_seed_all(999)
            LOAD_CHECKPOINT(path, restored)
            actual_cuda = torch.rand(5, device="cuda")
            actual_loss = restored.train_if_ready()

        torch.testing.assert_close(
            actual_cuda, expected_cuda, rtol=0.0, atol=0.0
        )
        self.assertAlmostEqual(actual_loss, expected_loss, places=7)
        self.assertNestedEqual(restored.model.state_dict(), expected_model)
        self.assertNestedEqual(restored.optimizer.state_dict(), expected_optimizer)

    def test_config_mismatch_and_future_schema_fail_before_mutation(self):
        original = make_assigner()
        state = checkpoint_module.credit_assigner_state_dict(original)
        for label, mutate in (
            ("config", lambda payload: payload["config"].__setitem__("obs_dim", 999)),
            ("schema", lambda payload: payload.__setitem__("version", 999)),
            (
                "non_integral_schema",
                lambda payload: payload.__setitem__("version", 2.5),
            ),
        ):
            with self.subTest(label=label):
                payload = copy.deepcopy(state)
                mutate(payload)
                destination = ConservedSTASCreditAssigner(original.config)
                before = copy.deepcopy(destination.model.state_dict())
                with self.assertRaises(ValueError):
                    checkpoint_module.load_credit_assigner_state(destination, payload)
                self.assertNestedEqual(destination.model.state_dict(), before)
                self.assertEqual(destination.rollouts_seen, 0)

    def test_version_one_state_still_loads(self):
        original = make_assigner()
        legacy = checkpoint_module.credit_assigner_state_dict(original)
        legacy["version"] = 1
        legacy["config"].pop("eval_mask_seed")
        legacy["config"].pop("eval_mask_count")
        for key in (
            "last_target_error",
            "last_gate_phase",
            "last_gate_active",
            "last_gate_reason",
        ):
            legacy.pop(key, None)
        restored = ConservedSTASCreditAssigner(original.config)
        checkpoint_module.load_credit_assigner_state(restored, legacy)
        self.assertNestedEqual(restored.model.state_dict(), original.model.state_dict())
        self.assertEqual(restored.rollouts_seen, original.rollouts_seen)

    def test_versionless_model_only_state_still_loads(self):
        original = make_assigner()
        legacy = checkpoint_module.credit_assigner_state_dict(original)
        legacy.pop("version")
        for field in ("causal", "eval_mask_seed", "eval_mask_count"):
            legacy["config"].pop(field)
        for key in (
            "buffer_capacity",
            "buffer_storage",
            "holdout_buffer_capacity",
            "holdout_buffer_storage",
            "numpy_random_state",
            "torch_rng_state",
            "torch_cuda_rng_state_all",
            "normalizer",
            "gate",
        ):
            legacy.pop(key, None)

        restored = ConservedSTASCreditAssigner(original.config)
        checkpoint_module.load_credit_assigner_state(restored, legacy)

        self.assertNestedEqual(restored.model.state_dict(), original.model.state_dict())
        self.assertNestedEqual(
            restored.optimizer.state_dict(), original.optimizer.state_dict()
        )
        self.assertEqual(restored.rollouts_seen, original.rollouts_seen)
        self.assertEqual(restored.last_loss, original.last_loss)

    def test_training_entry_uses_atomic_full_stas_file_apis(self):
        source = (
            STAS_ROOT / "mappo_stas.py"
        ).read_text(encoding="utf-8")
        self.assertIn("load_credit_assigner_checkpoint", source)
        self.assertIn("save_credit_assigner_checkpoint", source)
        self.assertNotIn("torch.save(", source)


if __name__ == "__main__":
    unittest.main()
