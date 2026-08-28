import inspect
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from torch import nn


STAS_ROOT = Path(__file__).resolve().parents[1] / "baselines" / "STAS-MAPPO"
if str(STAS_ROOT) not in sys.path:
    sys.path.insert(0, str(STAS_ROOT))

from mappo_stas import _make_credit_assigner  # noqa: E402
from stas_mappo.checkpoint import (  # noqa: E402
    credit_assigner_state_dict,
    load_credit_assigner_state,
)
from stas_mappo.conserved_credit import ConservedSTASCreditAssigner  # noqa: E402
from stas_mappo.credit import EpisodeCreditBuffer, STASCreditConfig  # noqa: E402
from stas_mappo.credit_conservation import explained_variance  # noqa: E402
from stas_mappo.reward_model import ShapleyAttention, STASRewardModel  # noqa: E402


def _credit_config(**overrides):
    values = {
        "obs_dim": 2,
        "action_dim": 1,
        "n_agents": 3,
        "seq_length": 2,
        "gamma": 1.0,
        "emb_dim": 8,
        "n_heads": 2,
        "n_layers": 1,
        "sample_num": 2,
        "dropout": 0.0,
        "buffer_size": 32,
        "batch_size": 2,
        "conserve_discounted": True,
    }
    values.update(overrides)
    return STASCreditConfig(**values)


class _EchoObservationCredit(nn.Module):
    """Return observation channel zero as agent-time credit."""

    def __init__(self):
        super().__init__()
        self.batch_sizes = []

    def forward(self, obs, actions, rewards, dones):
        self.batch_sizes.append(int(obs.shape[0]))
        return obs[..., 0]


class DeterministicEvaluationMaskTest(unittest.TestCase):
    def test_pre_task_2_model_state_loads_strictly(self):
        model = STASRewardModel(
            obs_dim=2,
            action_dim=1,
            n_agents=3,
            seq_length=2,
            emb_dim=8,
            n_heads=2,
            n_layers=1,
            dropout=0.0,
            eval_mask_seed=71,
            eval_mask_count=7,
        )
        legacy_state = {
            name: value
            for name, value in model.state_dict().items()
            if name != "shapley._eval_mask_bank"
        }
        restored = STASRewardModel(
            obs_dim=2,
            action_dim=1,
            n_agents=3,
            seq_length=2,
            emb_dim=8,
            n_heads=2,
            n_layers=1,
            dropout=0.0,
            eval_mask_seed=71,
            eval_mask_count=7,
        )

        try:
            incompatible = restored.load_state_dict(legacy_state, strict=True)
        except RuntimeError as exc:
            self.fail(f"pre-Task-2 model state must load strictly: {exc}")

        self.assertEqual(incompatible.missing_keys, [])
        self.assertEqual(incompatible.unexpected_keys, [])

    def test_credit_config_and_training_entry_wire_evaluation_mask_settings(self):
        defaults = _credit_config()
        self.assertEqual(getattr(defaults, "eval_mask_seed", None), 3030)
        self.assertEqual(getattr(defaults, "eval_mask_count", None), 8)

        explicit = _make_credit_assigner(
            {
                "NUM_STEPS": 2,
                "GAMMA": 0.99,
                "STAS": {
                    "EVAL_MASK_SEED": 91,
                    "EVAL_MASK_COUNT": 5,
                },
            },
            raw_obs_dim=2,
            action_dim=1,
            num_agents=3,
        )
        self.assertEqual(explicit.config.eval_mask_seed, 91)
        self.assertEqual(explicit.config.eval_mask_count, 5)

        entry_defaults = _make_credit_assigner(
            {"NUM_STEPS": 2, "GAMMA": 0.99, "STAS": {}},
            raw_obs_dim=2,
            action_dim=1,
            num_agents=3,
        )
        self.assertEqual(entry_defaults.config.eval_mask_seed, 3030)
        self.assertEqual(entry_defaults.config.eval_mask_count, 8)

    def test_eval_forward_is_independent_of_global_rng_state(self):
        parameters = inspect.signature(STASRewardModel).parameters
        self.assertIn("eval_mask_seed", parameters)
        self.assertIn("eval_mask_count", parameters)

        torch.manual_seed(13)
        model = STASRewardModel(
            obs_dim=2,
            action_dim=1,
            n_agents=4,
            seq_length=3,
            emb_dim=8,
            n_heads=2,
            n_layers=1,
            sample_num=2,
            dropout=0.4,
            eval_mask_seed=71,
            eval_mask_count=7,
        )
        generator = torch.Generator(device="cpu")
        generator.manual_seed(8)
        obs = torch.randn(2, 4, 3, 2, generator=generator)
        actions = torch.randn(2, 4, 3, 1, generator=generator)
        rewards = torch.randn(2, 4, 3, generator=generator)
        dones = torch.zeros(2, 4, 3)

        model.eval()
        first = model(obs, actions, rewards, dones)
        np.random.seed(129)
        np.random.random(4096)
        torch.manual_seed(993)
        torch.rand(4096)
        second = model(obs, actions, rewards, dones)

        torch.testing.assert_close(first, second, rtol=0.0, atol=1e-6)
        masks = model.shapley._attention_masks(torch.device("cpu"))
        self.assertEqual(tuple(masks.shape), (7, 4, 4))
        self.assertFalse(bool(torch.diagonal(masks, dim1=-2, dim2=-1).any()))

    def test_training_mode_uses_random_sample_num_masks(self):
        parameters = inspect.signature(ShapleyAttention).parameters
        self.assertIn("eval_mask_seed", parameters)
        self.assertIn("eval_mask_count", parameters)

        attention = ShapleyAttention(
            emb_dim=8,
            n_heads=2,
            n_agents=5,
            sample_num=3,
            dropout=0.0,
            eval_mask_seed=17,
            eval_mask_count=9,
        )
        attention.train()
        torch.manual_seed(1)
        first = attention._attention_masks(torch.device("cpu"))
        torch.manual_seed(2)
        second = attention._attention_masks(torch.device("cpu"))

        self.assertEqual(tuple(first.shape), (3, 5, 5))
        self.assertEqual(tuple(second.shape), (3, 5, 5))
        self.assertFalse(torch.equal(first, second))
        self.assertFalse(bool(torch.diagonal(first, dim1=-2, dim2=-1).any()))

        attention.eval()
        eval_masks = attention._attention_masks(torch.device("cpu"))
        self.assertEqual(tuple(eval_masks.shape), (9, 5, 5))


class FullHoldoutExplainedVarianceTest(unittest.TestCase):
    def test_full_buffer_read_preserves_stable_insertion_order(self):
        buffer = EpisodeCreditBuffer(capacity=3)
        for value in range(4):
            item = np.full((1, 1, 1), value, dtype=np.float32)
            buffer.add(item, item, item[..., 0], item[..., 0], float(value))

        self.assertTrue(hasattr(buffer, "get_all"))
        obs, actions, rewards, dones, returns = buffer.get_all()

        np.testing.assert_array_equal(returns, np.array([1.0, 2.0, 3.0]))
        np.testing.assert_array_equal(obs[:, 0, 0, 0], returns)
        np.testing.assert_array_equal(actions[:, 0, 0, 0], returns)
        np.testing.assert_array_equal(rewards[:, 0, 0], returns)
        np.testing.assert_array_equal(dones[:, 0, 0], returns)

    def test_holdout_larger_than_batch_uses_every_item_for_known_ev(self):
        assigner = ConservedSTASCreditAssigner(_credit_config(batch_size=2))
        model = _EchoObservationCredit()
        assigner.model = model
        targets = np.array([0.0, 1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        predictions = np.array([0.0, 1.0, 2.0, 3.0, 0.0], dtype=np.float32)
        for target, prediction in zip(targets, predictions):
            obs = np.zeros((3, 2, 2), dtype=np.float32)
            obs[0, 0, 0] = prediction
            actions = np.zeros((3, 2, 1), dtype=np.float32)
            rewards = np.zeros((3, 2), dtype=np.float32)
            dones = np.zeros((3, 2), dtype=np.float32)
            assigner.holdout_buffer.add(
                obs, actions, rewards, dones, float(target)
            )

        actual = assigner.holdout_explained_variance()
        expected = explained_variance(targets, predictions)

        self.assertAlmostEqual(actual, expected, delta=1e-12)
        self.assertEqual(model.batch_sizes, [len(targets)])
        repeated = assigner.holdout_explained_variance()
        self.assertAlmostEqual(actual, repeated, delta=1e-6)
        self.assertEqual(model.batch_sizes, [len(targets), len(targets)])

    def test_repeated_holdout_ev_is_deterministic_for_reward_model(self):
        parameters = inspect.signature(STASCreditConfig).parameters
        self.assertIn("eval_mask_seed", parameters)
        self.assertIn("eval_mask_count", parameters)
        torch.manual_seed(44)
        assigner = ConservedSTASCreditAssigner(
            _credit_config(
                batch_size=2,
                eval_mask_seed=3030,
                eval_mask_count=8,
            )
        )
        rng = np.random.default_rng(204)
        for index in range(7):
            obs = rng.normal(size=(3, 2, 2)).astype(np.float32)
            actions = rng.normal(size=(3, 2, 1)).astype(np.float32)
            rewards = rng.normal(size=(3, 2)).astype(np.float32)
            dones = np.zeros((3, 2), dtype=np.float32)
            assigner.holdout_buffer.add(
                obs, actions, rewards, dones, float(index - 3)
            )

        values = [assigner.holdout_explained_variance() for _ in range(3)]

        self.assertLessEqual(max(values) - min(values), 1e-6)

    def test_serialized_restore_reconstructs_full_holdout_ev_deterministically(self):
        config = _credit_config(
            batch_size=2,
            eval_mask_seed=3030,
            eval_mask_count=8,
        )
        torch.manual_seed(44)
        original = ConservedSTASCreditAssigner(config)
        rng = np.random.default_rng(912)
        for index in range(7):
            obs = rng.normal(size=(3, 2, 2)).astype(np.float32)
            actions = rng.normal(size=(3, 2, 1)).astype(np.float32)
            rewards = rng.normal(size=(3, 2)).astype(np.float32)
            dones = np.zeros((3, 2), dtype=np.float32)
            original.holdout_buffer.add(
                obs, actions, rewards, dones, float(index - 3)
            )
        before_save = original.holdout_explained_variance()

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "stas-credit.pt"
            torch.save(credit_assigner_state_dict(original), path)
            np.random.seed(1201)
            np.random.random(4096)
            torch.manual_seed(1202)
            torch.rand(4096)

            restored = ConservedSTASCreditAssigner(config)
            serialized_state = torch.load(path, map_location="cpu")
            load_credit_assigner_state(restored, serialized_state)

        batch_sizes = []

        def record_batch_size(_module, inputs):
            batch_sizes.append(int(inputs[0].shape[0]))

        hook = restored.model.register_forward_pre_hook(record_batch_size)
        try:
            after_restore = restored.holdout_explained_variance()
            np.random.seed(2201)
            np.random.random(4096)
            torch.manual_seed(2202)
            torch.rand(4096)
            repeated = restored.holdout_explained_variance()
        finally:
            hook.remove()

        self.assertEqual(len(restored.holdout_buffer), 7)
        self.assertEqual(batch_sizes, [7, 7])
        self.assertLessEqual(abs(before_save - after_restore), 1e-6)
        self.assertLessEqual(abs(after_restore - repeated), 1e-6)
        self.assertNotIn(
            "shapley._eval_mask_bank", serialized_state["model_state_dict"]
        )


if __name__ == "__main__":
    unittest.main()
