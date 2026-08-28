"""Minimal verification for STAS-MAPPO components."""

from __future__ import annotations

import os
import sys

import numpy as np

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from envs.microgrid.microgrid_continuous_env import MicrogridContinuousEnv
from stas_mappo.credit import STASCreditAssigner, STASCreditConfig


class _NoOpRun:
    def log(self, *args, **kwargs):
        return None


def check_reward_model():
    env = MicrogridContinuousEnv()
    env.seed(123)
    obs = env.reset()
    n_agents, obs_dim = obs.shape
    action_dim = env.signal_action_dim
    obs_rollout = []
    action_rollout = []
    reward_rollout = []
    done_rollout = []
    for _ in range(2):
        actions = np.zeros((n_agents, action_dim), dtype=np.float32)
        next_obs, reward, done, _ = env.step(actions)
        obs_rollout.append(obs)
        action_rollout.append(actions)
        reward_rollout.append(reward.reshape(n_agents))
        done_rollout.append(done.reshape(n_agents))
        obs = next_obs

    obs_eat = np.asarray(obs_rollout, dtype=np.float32).transpose(1, 0, 2)[None, ...]
    actions_eat = np.asarray(action_rollout, dtype=np.float32).transpose(1, 0, 2)[None, ...]
    rewards_eat = np.asarray(reward_rollout, dtype=np.float32).transpose(1, 0)[None, ...]
    dones_eat = np.asarray(done_rollout, dtype=np.float32).transpose(1, 0)[None, ...]
    assigner = STASCreditAssigner(
        STASCreditConfig(
            obs_dim=obs_dim,
            action_dim=action_dim,
            n_agents=n_agents,
            seq_length=2,
            mix_coef=1.0,
            buffer_size=4,
            batch_size=1,
            warmup_rollouts=0,
            emb_dim=32,
            n_heads=4,
            n_layers=1,
            sample_num=1,
        )
    )
    credit = assigner.credit_rewards(obs_eat, actions_eat, rewards_eat, dones_eat)
    assert credit.shape == (1, n_agents, 2), credit.shape
    env.close()
    return credit.shape


def check_one_update():
    import jax
    from mappo_stas import make_train

    config = {
        "ENV_NAME": "microgrid",
        "NUM_ENVS": 1,
        "NUM_STEPS": 2,
        "TOTAL_TIMESTEPS": 2,
        "NUM_MINIBATCHES": 1,
        "UPDATE_EPOCHS": 1,
        "LR": 1e-4,
        "GAMMA": 0.99,
        "GAE_LAMBDA": 0.95,
        "CLIP_EPS": 0.2,
        "ENT_COEF": 0.0,
        "VF_COEF": 0.5,
        "MAX_GRAD_NORM": 1.0,
        "ACTIVATION": "tanh",
        "ANNEAL_LR": False,
        "LOG_STD_INIT": -1.0,
        "ACTOR_LAYERS": [16],
        "CRITIC_LAYERS": [16],
        "ACTION_SPACE_TYPE": "continuous",
        "TRAIN_ENV_KWARGS": {"auto_reset": True},
        "TEST_ENV_KWARGS": {"auto_reset": True},
        "EVAL_INTERVAL": 10**9,
        "EVAL_EPISODES": 1,
        "EVAL_PARALLEL": False,
        "CHECKPOINT": False,
        "CHECKPOINT_INTERVAL": None,
        "SEED": 0,
        "EXP_NAME": "smoke",
        "CHP_DIR": ".",
        "STAS": {
            "MIX_COEF": 1.0,
            "LR": 1e-3,
            "EMB_DIM": 32,
            "N_HEADS": 4,
            "N_LAYERS": 1,
            "SAMPLE_NUM": 1,
            "DROPOUT": 0.0,
            "BUFFER_SIZE": 4,
            "BATCH_SIZE": 1,
            "UPDATE_FREQ": 1,
            "UPDATES_PER_STEP": 1,
            "WARMUP_ROLLOUTS": 0,
            "GLOBAL_REWARD_AGG": "sum",
            "DEVICE": "cpu",
        },
    }
    train = make_train(config)
    out, env = train(jax.random.PRNGKey(0), _NoOpRun())
    env.close()
    assert "metrics" in out
    return sorted(out["metrics"].keys())


if __name__ == "__main__":
    shape = check_reward_model()
    print(f"STAS credit shape: {shape}")
    try:
        keys = check_one_update()
    except ModuleNotFoundError as exc:
        if exc.name in {"jax", "flax", "optax", "distrax"}:
            print(f"One MAPPO update skipped: missing dependency {exc.name}")
        else:
            raise
    else:
        print(f"One MAPPO update metric keys: {keys}")
