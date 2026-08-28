"""Continuous-action wrapper for the microgrid environment."""

import numpy as np

try:
    from gymnasium import spaces
except ImportError:
    try:
        from gym import spaces
    except ImportError:
        class _Box:
            def __init__(self, low, high, shape, dtype):
                self.low = low
                self.high = high
                self.shape = shape
                self.dtype = dtype

        class _Spaces:
            Box = _Box

        spaces = _Spaces()

from envs.microgrid.microgrid_env import MicrogridEnv


class MicrogridContinuousEnv(object):
    """Expose the environment using stacked NumPy tensors and Box spaces."""

    def __init__(self, config_overrides=None):
        self.env = MicrogridEnv(config_overrides=config_overrides)
        self.num_agent = self.env.agent_num
        self.signal_obs_dim = self.env.obs_dim
        self.signal_action_dim = self.env.action_dim
        self.discrete_action_input = False
        self.movable = True

        self.action_space = []
        self.observation_space = []
        self.share_observation_space = []

        share_obs_dim = 0
        for _ in range(self.num_agent):
            self.action_space.append(
                spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(self.signal_action_dim,),
                    dtype=np.float32,
                )
            )

            share_obs_dim += self.signal_obs_dim
            self.observation_space.append(
                spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(self.signal_obs_dim,),
                    dtype=np.float32,
                )
            )

        self.share_observation_space = [
            spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(share_obs_dim,),
                dtype=np.float32,
            )
            for _ in range(self.num_agent)
        ]

    def step(self, actions):
        obs, rews, dones, infos = self.env.step(actions)
        return np.stack(obs), np.stack(rews), np.stack(dones), infos

    def reset(self):
        return np.stack(self.env.reset())

    def close(self):
        self.env.close()

    def render(self, mode="rgb_array"):
        return self.env.render(mode)

    def seed(self, seed):
        self.env.seed(seed)
