"""Shape conversions between flat MAPPO rollouts and STAS tensors."""

from __future__ import annotations

import numpy as np


def flat_time_agent_to_env_agent_time(
    x: np.ndarray,
    num_envs: int,
    num_agents: int,
) -> np.ndarray:
    """Convert ``[time, env * agent, ...]`` to ``[env, agent, time, ...]``."""
    time_len = x.shape[0]
    tail = x.shape[2:]
    return x.reshape(time_len, num_envs, num_agents, *tail).transpose(1, 2, 0, *range(3, 3 + len(tail)))


def env_agent_time_to_flat_time_agent(
    x: np.ndarray,
    num_envs: int,
    num_agents: int,
) -> np.ndarray:
    """Convert ``[env, agent, time]`` to ``[time, env * agent]``."""
    return x.transpose(2, 0, 1).reshape(x.shape[2], num_envs * num_agents)
