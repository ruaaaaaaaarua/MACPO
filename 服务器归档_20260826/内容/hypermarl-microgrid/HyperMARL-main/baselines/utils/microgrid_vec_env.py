"""Vectorized adapter for the in-repo microgrid environment."""

from pathlib import Path
import sys
from typing import Any, Optional

import numpy as np

try:
    from gymnasium import spaces
except ImportError:
    from gym import spaces


def _ensure_hypermarl_on_path() -> None:
    hypermarl_root = Path(__file__).resolve().parents[2]
    hypermarl_root_str = str(hypermarl_root)
    if hypermarl_root_str not in sys.path:
        sys.path.insert(0, hypermarl_root_str)


_ensure_hypermarl_on_path()
from envs.microgrid.microgrid_continuous_env import MicrogridContinuousEnv  # noqa: E402


class MicrogridVecEnv:
    """Small NumPy vector env wrapper around MicrogridContinuousEnv."""

    metadata = {"render_modes": ["rgb_array"], "name": "microgrid"}

    def __init__(self, num_envs: int = 1, auto_reset: bool = True):
        if num_envs < 1:
            raise ValueError(f"num_envs must be >= 1, got {num_envs}")

        self.num_envs = int(num_envs)
        self.auto_reset = bool(auto_reset)
        self.envs = [MicrogridContinuousEnv() for _ in range(self.num_envs)]

        template = self.envs[0]
        self.num_agents = int(template.num_agent)
        self.obs_dim = int(template.signal_obs_dim)
        self.action_dim = int(template.signal_action_dim)
        self.agents = [f"agent_{i}" for i in range(self.num_agents)]
        self.possible_agents = list(self.agents)

        self.observation_space = [
            spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(self.obs_dim,),
                dtype=np.float32,
            )
            for _ in self.agents
        ]
        self.action_space = [
            spaces.Box(
                low=-1.0,
                high=1.0,
                shape=(self.action_dim,),
                dtype=np.float32,
            )
            for _ in self.agents
        ]
        self.single_observation_space = self.observation_space[0]
        self.single_action_space = self.action_space[0]
        self.is_vector_env = True

    def reset(self, seed: Optional[int] = None, **_: Any):
        observations = []
        for env_index, env in enumerate(self.envs):
            if seed is not None:
                env.seed(int(seed) + env_index)
            observations.append(env.reset())

        return self._flatten_obs(observations), {}

    def step(self, actions):
        actions = self._reshape_actions(actions)

        observations = []
        rewards = []
        terminations = []
        truncations = []
        infos = []

        for env_index, env in enumerate(self.envs):
            obs, rew, done, info = env.step(actions[env_index])
            obs = np.asarray(obs, dtype=np.float32)
            rew = np.asarray(rew, dtype=np.float32).reshape(self.num_agents)
            done = np.asarray(done, dtype=bool).reshape(self.num_agents)
            trunc = np.zeros(self.num_agents, dtype=bool)
            if not np.any(done):
                info0 = info[0] if isinstance(info, (list, tuple)) and info else info
                if isinstance(info0, dict):
                    raw_env = getattr(env, "env", None)
                    trunc[:] = bool(
                        getattr(raw_env, "daily_truncation_enable", False)
                        and info0.get("day_boundary", False)
                    )

            if self.auto_reset and np.any(done):
                terminal_obs = obs.copy()
                obs = np.asarray(env.reset(), dtype=np.float32)
                info = self._with_terminal_obs(info, terminal_obs)

            observations.append(obs)
            rewards.append(rew)
            terminations.append(done)
            truncations.append(trunc)
            infos.extend(self._flatten_infos(info, env_index))

        return (
            self._flatten_obs(observations),
            np.concatenate(rewards).astype(np.float32),
            np.concatenate(terminations).astype(bool),
            np.concatenate(truncations).astype(bool),
            infos,
        )

    def close(self):
        for env in self.envs:
            env.close()

    def render(self, mode: str = "rgb_array", **kwargs: Any):
        if self.num_envs == 1:
            return self.envs[0].render(mode=mode)
        return [env.render(mode=mode) for env in self.envs]

    def _reshape_actions(self, actions):
        actions = np.asarray(actions, dtype=np.float32)
        expected_size = self.num_envs * self.num_agents * self.action_dim
        if actions.size != expected_size:
            raise ValueError(
                f"Expected {expected_size} action values for shape "
                f"({self.num_envs}, {self.num_agents}, {self.action_dim}), "
                f"got array with shape {actions.shape}"
            )
        actions = actions.reshape(self.num_envs, self.num_agents, self.action_dim)
        return np.clip(actions, -1.0, 1.0)

    def _flatten_obs(self, observations):
        return np.asarray(observations, dtype=np.float32).reshape(
            self.num_envs * self.num_agents, self.obs_dim
        )

    def _flatten_infos(self, info, env_index: int):
        if isinstance(info, (list, tuple)):
            raw_infos = info
        else:
            raw_infos = [info for _ in range(self.num_agents)]

        flat_infos = []
        for agent_index in range(self.num_agents):
            agent_info = raw_infos[agent_index] if agent_index < len(raw_infos) else {}
            if isinstance(agent_info, dict):
                agent_info = dict(agent_info)
            else:
                agent_info = {"info": agent_info}
            agent_info["env_index"] = env_index
            flat_infos.append(agent_info)
        return flat_infos

    def _with_terminal_obs(self, info, terminal_obs):
        if isinstance(info, (list, tuple)):
            updated = []
            for agent_index, agent_info in enumerate(info):
                if isinstance(agent_info, dict):
                    agent_info = dict(agent_info)
                else:
                    agent_info = {"info": agent_info}
                agent_info["terminal_observation"] = terminal_obs[agent_index]
                updated.append(agent_info)
            return updated

        updated = dict(info) if isinstance(info, dict) else {"info": info}
        updated["terminal_observation"] = terminal_obs
        return updated
