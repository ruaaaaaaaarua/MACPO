"""Vectorized adapter for the in-repo microgrid environment."""

from pathlib import Path
import multiprocessing as mp
import sys
from typing import Any, Mapping, Optional, Sequence

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


def _step_one_env(env, action, *, auto_reset: bool, num_agents: int):
    """Run one environment step; shared by serial and worker execution."""
    obs, rew, done, info = env.step(action)
    obs = np.asarray(obs, dtype=np.float32)
    rew = np.asarray(rew, dtype=np.float32).reshape(num_agents)
    done = np.asarray(done, dtype=bool).reshape(num_agents)
    trunc = np.zeros(num_agents, dtype=bool)
    if not np.any(done):
        info0 = info[0] if isinstance(info, (list, tuple)) and info else info
        if isinstance(info0, dict):
            raw_env = getattr(env, "env", None)
            trunc[:] = bool(
                getattr(raw_env, "daily_truncation_enable", False)
                and info0.get("day_boundary", False)
            )
    if auto_reset and np.any(done):
        terminal_obs = obs.copy()
        obs = np.asarray(env.reset(), dtype=np.float32)
        info = MicrogridVecEnv._with_terminal_obs_static(info, terminal_obs)
    return obs, rew, done, trunc, info


def _process_env_worker(connection, config, auto_reset: bool):
    """Own exactly one PyPower environment in a persistent CPU-only process."""
    env = MicrogridContinuousEnv(config)
    try:
        while True:
            command, payload = connection.recv()
            if command == "reset":
                if payload is not None:
                    env.seed(int(payload))
                connection.send(("ok", np.asarray(env.reset(), dtype=np.float32)))
            elif command == "step":
                connection.send(
                    ("ok", _step_one_env(env, payload, auto_reset=auto_reset, num_agents=env.num_agent))
                )
            elif command == "close":
                env.close()
                connection.send(("ok", None))
                break
            else:
                raise ValueError(f"unknown vector-worker command: {command}")
    except BaseException as error:
        connection.send(("error", repr(error)))
    finally:
        connection.close()


class MicrogridVecEnv:
    """Small NumPy vector env wrapper around MicrogridContinuousEnv."""

    metadata = {"render_modes": ["rgb_array"], "name": "microgrid"}

    def __init__(
        self,
        num_envs: int = 1,
        auto_reset: bool = True,
        config_overrides: Optional[Mapping[str, Any]] = None,
        config_overrides_by_env: Optional[Sequence[Mapping[str, Any]]] = None,
        parallel_backend: str = "serial",
    ):
        if num_envs < 1:
            raise ValueError(f"num_envs must be >= 1, got {num_envs}")
        if config_overrides is not None and config_overrides_by_env is not None:
            raise ValueError("Use config_overrides or config_overrides_by_env, not both")
        if config_overrides_by_env is not None and len(config_overrides_by_env) != num_envs:
            raise ValueError(
                "config_overrides_by_env length must equal num_envs: "
                f"{len(config_overrides_by_env)} != {num_envs}"
            )
        if parallel_backend not in {"serial", "process"}:
            raise ValueError("parallel_backend must be 'serial' or 'process'")

        self.num_envs = int(num_envs)
        self.auto_reset = bool(auto_reset)
        self.parallel_backend = str(parallel_backend)
        env_configs = (
            list(config_overrides_by_env)
            if config_overrides_by_env is not None
            else [config_overrides] * self.num_envs
        )
        self.envs = [MicrogridContinuousEnv(cfg) for cfg in env_configs]
        self._worker_connections = []
        self._worker_processes = []
        if self.parallel_backend == "process":
            # JAX may already have initialized worker threads in the parent.
            # ``fork`` after that point can deadlock; a clean spawned Python
            # process is slower to create but remains safe for persistent runs.
            context = mp.get_context("spawn")
            for cfg in env_configs:
                parent, child = context.Pipe()
                worker = context.Process(
                    target=_process_env_worker,
                    args=(child, cfg, self.auto_reset),
                    daemon=True,
                )
                worker.start()
                child.close()
                self._worker_connections.append(parent)
                self._worker_processes.append(worker)

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
        if self.parallel_backend == "process":
            for env_index, connection in enumerate(self._worker_connections):
                connection.send(("reset", None if seed is None else int(seed) + env_index))
            observations = [self._receive_worker(connection) for connection in self._worker_connections]
            return self._flatten_obs(observations), {}
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

        if self.parallel_backend == "process":
            for env_index, connection in enumerate(self._worker_connections):
                connection.send(("step", actions[env_index]))
            transitions = [self._receive_worker(connection) for connection in self._worker_connections]
        else:
            transitions = [
                _step_one_env(env, actions[env_index], auto_reset=self.auto_reset, num_agents=self.num_agents)
                for env_index, env in enumerate(self.envs)
            ]

        for env_index, (obs, rew, done, trunc, info) in enumerate(transitions):

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
        for connection in self._worker_connections:
            try:
                connection.send(("close", None))
            except (BrokenPipeError, EOFError):
                pass
        for connection in self._worker_connections:
            try:
                self._receive_worker(connection)
            except (BrokenPipeError, EOFError, RuntimeError):
                pass
            connection.close()
        for worker in self._worker_processes:
            worker.join(timeout=5)
            if worker.is_alive():
                worker.terminate()
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

    @staticmethod
    def _with_terminal_obs_static(info, terminal_obs):
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

    def _with_terminal_obs(self, info, terminal_obs):
        return self._with_terminal_obs_static(info, terminal_obs)

    @staticmethod
    def _receive_worker(connection):
        status, payload = connection.recv()
        if status != "ok":
            raise RuntimeError(f"microgrid process worker failed: {payload}")
        return payload
