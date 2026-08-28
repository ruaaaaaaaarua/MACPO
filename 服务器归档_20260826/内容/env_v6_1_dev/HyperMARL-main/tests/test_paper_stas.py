import copy
import inspect
import random
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch


STAS_ROOT = Path(__file__).resolve().parents[1] / "baselines" / "STAS-MAPPO"
if str(STAS_ROOT) not in sys.path:
    sys.path.insert(0, str(STAS_ROOT))

try:
    from stas_mappo.paper_credit import (  # noqa: E402
        PaperSTASCreditAssigner,
        PaperSTASRewardModel,
        PaperShapleyAttention,
        shared_team_return,
    )
except ImportError:
    PaperSTASCreditAssigner = None
    PaperSTASRewardModel = None
    PaperShapleyAttention = None
    shared_team_return = None

from mappo_stas import (  # noqa: E402
    _make_credit_assigner,
    _paper_policy_should_update,
)
from stas_mappo.checkpoint import (  # noqa: E402
    load_credit_assigner_checkpoint,
    save_credit_assigner_checkpoint,
)


def paper_config(**overrides):
    values = {
        "NUM_STEPS": 3,
        "GAMMA": 1.0,
        "STAS": {
            "MODE": "paper",
            "DEVICE": "cpu",
            "LR": 5e-4,
            "WEIGHT_DECAY": 1e-5,
            "EMB_DIM": 16,
            "N_HEADS": 4,
            "N_LAYERS": 1,
            "SAMPLE_NUM": 5,
            "DROPOUT": 0.0,
            "EVAL_MASK_SEED": 3030,
            "EVAL_MASK_COUNT": 5,
            "BUFFER_SIZE": 64,
            "BATCH_SIZE": 4,
            "REWARD_MODEL_UPDATE_INTERVAL_EPISODES": 8,
            "REWARD_MODEL_UPDATES_PER_INTERVAL": 2,
            "POLICY_WARMUP_EPISODES": 8,
        },
    }
    values["STAS"].update(overrides)
    return values


def rollout(num_envs=4):
    rng = np.random.default_rng(91)
    obs = rng.normal(size=(num_envs, 2, 3, 4)).astype(np.float32)
    actions = rng.normal(size=(num_envs, 2, 3, 2)).astype(np.float32)
    rewards = np.zeros((num_envs, 2, 3), dtype=np.float32)
    terminal = rng.normal(size=num_envs).astype(np.float32)
    rewards[:, :, -1] = terminal[:, None]
    dones = np.zeros_like(rewards)
    dones[:, :, -1] = 1.0
    return obs, actions, rewards, dones


class PaperSTASTest(unittest.TestCase):
    def _require_paper(self):
        self.assertIsNotNone(PaperSTASCreditAssigner, "paper STAS module is missing")

    def test_shared_reward_is_counted_once_not_once_per_agent(self):
        self._require_paper()
        stream = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        rewards = np.tile(stream, (2, 4, 1))
        np.testing.assert_allclose(
            shared_team_return(rewards, gamma=1.0), np.array([6.0, 6.0])
        )
        broken = rewards.copy()
        broken[0, 3, 1] += 0.25
        with self.assertRaisesRegex(ValueError, "shared reward"):
            shared_team_return(broken, gamma=1.0)

    def test_reward_model_public_interface_cannot_receive_reward_or_done(self):
        self._require_paper()
        parameters = list(inspect.signature(PaperSTASRewardModel.forward).parameters)
        self.assertEqual(parameters, ["self", "obs", "actions", "valid_mask"])
        model = PaperSTASRewardModel(
            obs_dim=4,
            action_dim=2,
            n_agents=2,
            seq_length=3,
            emb_dim=16,
            n_heads=4,
            n_layers=1,
            sample_num=5,
            dropout=0.0,
        )
        self.assertFalse(hasattr(model, "reward_emb"))
        self.assertFalse(hasattr(model, "done_emb"))
        output = model(
            torch.zeros(2, 2, 3, 4),
            torch.zeros(2, 2, 3, 2),
            torch.ones(2, 2, 3, dtype=torch.bool),
        )
        self.assertEqual(tuple(output.shape), (2, 2, 3))

    def test_coalitions_keep_target_agent_and_eval_bank_is_deterministic(self):
        self._require_paper()
        attention = PaperShapleyAttention(
            emb_dim=16,
            n_heads=4,
            n_agents=4,
            sample_num=5,
            dropout=0.0,
            eval_mask_seed=17,
            eval_mask_count=5,
        )
        for keep in attention.eval_coalition_keep_bank:
            self.assertTrue(torch.all(torch.diagonal(keep)))
        attention.eval()
        x = torch.randn(2, 4, 3, 16)
        valid = torch.ones(2, 4, 3, dtype=torch.bool)
        first = attention(x, valid)
        second = attention(x, valid)
        torch.testing.assert_close(first, second, rtol=0.0, atol=0.0)

    def test_internal_agent_permutation_restores_credit_alignment(self):
        self._require_paper()
        attention = PaperShapleyAttention(
            emb_dim=8,
            n_heads=2,
            n_agents=4,
            sample_num=1,
            dropout=0.0,
            eval_mask_seed=7,
            eval_mask_count=1,
        )
        attention.eval()
        x = torch.randn(2, 4, 3, 8)
        valid = torch.ones(2, 4, 3, dtype=torch.bool)
        identity = torch.arange(4)
        shuffled = torch.tensor([2, 0, 3, 1])
        first = attention._forward_sample(
            x, valid, identity, attention.eval_coalition_keep_bank[0]
        )
        second = attention._forward_sample(
            x, valid, shuffled, attention.eval_coalition_keep_bank[0]
        )
        torch.testing.assert_close(first, second, rtol=1e-5, atol=1e-6)

    def test_paper_mode_uses_fixed_defaults_and_never_projects_or_mixes(self):
        self._require_paper()
        assigner = _make_credit_assigner(
            paper_config(), raw_obs_dim=4, action_dim=2, num_agents=2
        )
        self.assertIsInstance(assigner, PaperSTASCreditAssigner)
        self.assertEqual(assigner.config.mode, "paper")
        self.assertEqual(assigner.optimizer.param_groups[0]["weight_decay"], 1e-5)
        obs, actions, rewards, dones = rollout()
        assigner.add_rollout(obs, actions, rewards, dones)
        assigner.add_rollout(obs, actions, rewards, dones)
        expected = np.full_like(rewards, 7.0)
        with patch.object(assigner, "credit_rewards", return_value=expected), patch(
            "stas_mappo.credit_conservation.project_discounted_credits",
            side_effect=AssertionError("paper mode must not project"),
        ):
            actual, _ = assigner.process_rollout(obs, actions, rewards, dones)
        np.testing.assert_array_equal(actual, expected)
        self.assertEqual(assigner.last_mix_coef, 1.0)

    def test_reward_model_updates_on_episode_schedule_and_policy_waits(self):
        self._require_paper()
        assigner = _make_credit_assigner(
            paper_config(REWARD_MODEL_UPDATES_PER_INTERVAL=5),
            raw_obs_dim=4,
            action_dim=2,
            num_agents=2,
        )
        obs, actions, rewards, dones = rollout()
        before = copy.deepcopy(assigner.model.state_dict())
        assigner.process_rollout(obs, actions, rewards, dones)
        self.assertEqual(assigner.reward_model_updates, 0)
        for name, value in before.items():
            torch.testing.assert_close(assigner.model.state_dict()[name], value)
        self.assertFalse(_paper_policy_should_update("paper", 8, 8))
        assigner.process_rollout(obs, actions, rewards, dones)
        self.assertEqual(assigner.reward_model_updates, 5)
        self.assertTrue(_paper_policy_should_update("paper", 9, 8))
        self.assertTrue(_paper_policy_should_update("conserved", 0, 4000))

    def test_checkpoint_restores_paper_buffer_schedule_and_random_sequences(self):
        self._require_paper()
        config = paper_config(REWARD_MODEL_UPDATES_PER_INTERVAL=1)
        original = _make_credit_assigner(
            config, raw_obs_dim=4, action_dim=2, num_agents=2
        )
        restored = _make_credit_assigner(
            config, raw_obs_dim=4, action_dim=2, num_agents=2
        )
        obs, actions, rewards, dones = rollout()
        original.process_rollout(obs, actions, rewards, dones)
        original.process_rollout(obs, actions, rewards, dones)

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "paper.pt"
            random.seed(111)
            np.random.seed(222)
            torch.manual_seed(333)
            save_credit_assigner_checkpoint(
                path, original, update=2, episode=8, global_step=192
            )
            expected_python = [random.random() for _ in range(3)]
            expected_numpy = np.random.random(3)
            expected_torch = torch.rand(3)
            random.seed(999)
            np.random.seed(999)
            torch.manual_seed(999)
            metadata = load_credit_assigner_checkpoint(path, restored)

        self.assertEqual(metadata["episode"], 8)
        self.assertEqual(len(restored.buffer), len(original.buffer))
        self.assertEqual(restored.reward_model_updates, original.reward_model_updates)
        self.assertEqual(
            restored.next_reward_model_update_episode,
            original.next_reward_model_update_episode,
        )
        self.assertEqual([random.random() for _ in range(3)], expected_python)
        np.testing.assert_array_equal(np.random.random(3), expected_numpy)
        torch.testing.assert_close(torch.rand(3), expected_torch, rtol=0.0, atol=0.0)


if __name__ == "__main__":
    unittest.main()
