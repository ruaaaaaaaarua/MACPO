"""Smoke tests for the MATD3 microgrid baseline."""

from __future__ import annotations

import importlib

import numpy as np


def test_matd3_modules_importable():
    train_mod = importlib.import_module("baselines.MATD3.train_matd3_microgrid")
    alg_mod = importlib.import_module("baselines.MATD3.matd3")

    assert hasattr(train_mod, "train")
    assert hasattr(alg_mod, "MATD3")
    assert hasattr(alg_mod, "ReplayBuffer")


def test_replay_buffer_samples_joint_transition_shapes():
    alg_mod = importlib.import_module("baselines.MATD3.matd3")
    buffer = alg_mod.ReplayBuffer(
        obs_dim=3,
        state_dim=12,
        action_dim=2,
        num_agents=4,
        capacity=8,
        seed=0,
    )

    obs = np.zeros((4, 3), dtype=np.float32)
    state = np.zeros(12, dtype=np.float32)
    action = np.zeros((4, 2), dtype=np.float32)
    next_obs = np.ones((4, 3), dtype=np.float32)
    next_state = np.ones(12, dtype=np.float32)
    buffer.add(obs, state, action, -1.0, next_obs, next_state, False)

    batch = buffer.sample(batch_size=1, device="cpu")

    assert batch["obs"].shape == (1, 4, 3)
    assert batch["actions"].shape == (1, 4, 2)
    assert batch["states"].shape == (1, 12)
    assert batch["rewards"].shape == (1, 1)
