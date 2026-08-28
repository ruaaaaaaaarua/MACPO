"""Multi-Agent TD3 components for continuous microgrid control."""

from __future__ import annotations

from dataclasses import dataclass
import copy
from typing import Dict, List

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


def _mlp(input_dim: int, hidden_dim: int, output_dim: int, output_tanh: bool = False):
    layers: List[nn.Module] = [
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, output_dim),
    ]
    if output_tanh:
        layers.append(nn.Tanh())
    return nn.Sequential(*layers)


class Actor(nn.Module):
    """Per-agent deterministic actor."""

    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = _mlp(obs_dim, hidden_dim, action_dim, output_tanh=True)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


class TwinCritic(nn.Module):
    """Centralized twin Q critic over global state and joint action."""

    def __init__(self, state_dim: int, joint_action_dim: int, hidden_dim: int = 256):
        super().__init__()
        input_dim = state_dim + joint_action_dim
        self.q1 = _mlp(input_dim, hidden_dim, 1)
        self.q2 = _mlp(input_dim, hidden_dim, 1)

    def forward(self, state: torch.Tensor, joint_action: torch.Tensor):
        x = torch.cat([state, joint_action], dim=-1)
        return self.q1(x), self.q2(x)

    def q1_value(self, state: torch.Tensor, joint_action: torch.Tensor):
        return self.q1(torch.cat([state, joint_action], dim=-1))


class ReplayBuffer:
    """Replay buffer storing joint multi-agent transitions."""

    def __init__(
        self,
        obs_dim: int,
        state_dim: int,
        action_dim: int,
        num_agents: int,
        capacity: int,
        seed: int = 0,
    ) -> None:
        self.capacity = int(capacity)
        self.num_agents = int(num_agents)
        self.obs = np.zeros((capacity, num_agents, obs_dim), dtype=np.float32)
        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, num_agents, action_dim), dtype=np.float32)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.next_obs = np.zeros((capacity, num_agents, obs_dim), dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.dones = np.zeros((capacity, 1), dtype=np.float32)
        self.ptr = 0
        self.size = 0
        self.rng = np.random.default_rng(seed)

    def add(self, obs, state, action, reward, next_obs, next_state, done) -> None:
        idx = self.ptr
        self.obs[idx] = np.asarray(obs, dtype=np.float32)
        self.states[idx] = np.asarray(state, dtype=np.float32)
        self.actions[idx] = np.asarray(action, dtype=np.float32)
        self.rewards[idx, 0] = float(reward)
        self.next_obs[idx] = np.asarray(next_obs, dtype=np.float32)
        self.next_states[idx] = np.asarray(next_state, dtype=np.float32)
        self.dones[idx, 0] = float(done)
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, device) -> Dict[str, torch.Tensor]:
        if self.size == 0:
            raise ValueError("Cannot sample from an empty ReplayBuffer")
        idx = self.rng.integers(0, self.size, size=int(batch_size))
        return {
            "obs": torch.as_tensor(self.obs[idx], dtype=torch.float32, device=device),
            "states": torch.as_tensor(self.states[idx], dtype=torch.float32, device=device),
            "actions": torch.as_tensor(self.actions[idx], dtype=torch.float32, device=device),
            "rewards": torch.as_tensor(self.rewards[idx], dtype=torch.float32, device=device),
            "next_obs": torch.as_tensor(self.next_obs[idx], dtype=torch.float32, device=device),
            "next_states": torch.as_tensor(
                self.next_states[idx], dtype=torch.float32, device=device
            ),
            "dones": torch.as_tensor(self.dones[idx], dtype=torch.float32, device=device),
        }

    def state_dict(self) -> Dict[str, object]:
        return {
            "version": 1,
            "capacity": self.capacity,
            "num_agents": self.num_agents,
            "ptr": self.ptr,
            "size": self.size,
            "obs": self.obs.copy(),
            "states": self.states.copy(),
            "actions": self.actions.copy(),
            "rewards": self.rewards.copy(),
            "next_obs": self.next_obs.copy(),
            "next_states": self.next_states.copy(),
            "dones": self.dones.copy(),
            "rng_state": copy.deepcopy(self.rng.bit_generator.state),
        }

    def load_state_dict(self, state: Dict[str, object]) -> None:
        stored_capacity = int(state["capacity"])
        if stored_capacity != self.capacity:
            raise ValueError(
                f"ReplayBuffer capacity mismatch: {stored_capacity} != {self.capacity}"
            )
        if int(state["num_agents"]) != self.num_agents:
            raise ValueError("ReplayBuffer num_agents mismatch")
        for name in (
            "obs",
            "states",
            "actions",
            "rewards",
            "next_obs",
            "next_states",
            "dones",
        ):
            destination = getattr(self, name)
            source = np.asarray(state[name], dtype=destination.dtype)
            if source.shape != destination.shape:
                raise ValueError(f"ReplayBuffer {name} shape mismatch")
            destination[...] = source
        self.ptr = int(state["ptr"])
        self.size = int(state["size"])
        self.rng.bit_generator.state = copy.deepcopy(state["rng_state"])

    def __len__(self) -> int:
        return self.size


@dataclass
class MATD3Config:
    obs_dim: int
    state_dim: int
    action_dim: int
    num_agents: int
    hidden_dim: int = 256
    actor_lr: float = 1e-3
    critic_lr: float = 1e-3
    gamma: float = 0.99
    tau: float = 0.005
    policy_noise: float = 0.2
    noise_clip: float = 0.5
    policy_delay: int = 2
    max_grad_norm: float = 10.0


class MATD3:
    """MATD3 with decentralized actors and centralized twin critics."""

    def __init__(self, config: MATD3Config, device: torch.device) -> None:
        self.config = config
        self.device = device
        self.total_it = 0
        self.actors = nn.ModuleList(
            [
                Actor(config.obs_dim, config.action_dim, config.hidden_dim)
                for _ in range(config.num_agents)
            ]
        ).to(device)
        self.actor_targets = nn.ModuleList(
            [
                Actor(config.obs_dim, config.action_dim, config.hidden_dim)
                for _ in range(config.num_agents)
            ]
        ).to(device)
        joint_action_dim = config.num_agents * config.action_dim
        self.critic = TwinCritic(config.state_dim, joint_action_dim, config.hidden_dim).to(device)
        self.critic_target = TwinCritic(
            config.state_dim, joint_action_dim, config.hidden_dim
        ).to(device)
        self.actor_targets.load_state_dict(self.actors.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.actor_optimizers = [
            torch.optim.Adam(actor.parameters(), lr=config.actor_lr)
            for actor in self.actors
        ]
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=config.critic_lr)

    def select_action(self, obs: np.ndarray, noise_std: float = 0.0) -> np.ndarray:
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        actions = []
        with torch.no_grad():
            for agent_idx, actor in enumerate(self.actors):
                action = actor(obs_t[agent_idx : agent_idx + 1]).squeeze(0)
                actions.append(action.cpu().numpy())
        action_arr = np.asarray(actions, dtype=np.float32)
        if noise_std > 0:
            action_arr += np.random.normal(0.0, noise_std, size=action_arr.shape).astype(
                np.float32
            )
        return np.clip(action_arr, -1.0, 1.0)

    def update(self, replay: ReplayBuffer, batch_size: int) -> Dict[str, float]:
        self.total_it += 1
        cfg = self.config
        batch = replay.sample(batch_size, self.device)
        obs = batch["obs"]
        states = batch["states"]
        actions = batch["actions"]
        rewards = batch["rewards"]
        next_obs = batch["next_obs"]
        next_states = batch["next_states"]
        dones = batch["dones"]

        with torch.no_grad():
            next_actions = []
            for agent_idx, actor_targ in enumerate(self.actor_targets):
                next_action = actor_targ(next_obs[:, agent_idx, :])
                noise = torch.randn_like(next_action) * cfg.policy_noise
                noise = noise.clamp(-cfg.noise_clip, cfg.noise_clip)
                next_actions.append((next_action + noise).clamp(-1.0, 1.0))
            next_joint_action = torch.cat(next_actions, dim=-1)
            target_q1, target_q2 = self.critic_target(next_states, next_joint_action)
            target_q = rewards + cfg.gamma * (1.0 - dones) * torch.minimum(target_q1, target_q2)

        joint_action = actions.reshape(actions.shape[0], -1)
        current_q1, current_q2 = self.critic(states, joint_action)
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)
        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), cfg.max_grad_norm)
        self.critic_optimizer.step()

        actor_loss_value = 0.0
        if self.total_it % max(1, cfg.policy_delay) == 0:
            policy_actions = []
            for agent_idx, actor in enumerate(self.actors):
                policy_actions.append(actor(obs[:, agent_idx, :]))
            policy_joint_action = torch.cat(policy_actions, dim=-1)
            actor_loss = -self.critic.q1_value(states, policy_joint_action).mean()
            for opt in self.actor_optimizers:
                opt.zero_grad(set_to_none=True)
            actor_loss.backward()
            for actor in self.actors:
                nn.utils.clip_grad_norm_(actor.parameters(), cfg.max_grad_norm)
            for opt in self.actor_optimizers:
                opt.step()
            actor_loss_value = float(actor_loss.detach().cpu())
            self._soft_update(self.actor_targets, self.actors, cfg.tau)
            self._soft_update(self.critic_target, self.critic, cfg.tau)

        return {
            "critic_loss": float(critic_loss.detach().cpu()),
            "actor_loss": actor_loss_value,
        }

    @staticmethod
    def _soft_update(target: nn.Module, source: nn.Module, tau: float) -> None:
        with torch.no_grad():
            for target_param, param in zip(target.parameters(), source.parameters()):
                target_param.data.mul_(1.0 - tau).add_(param.data, alpha=tau)

    def checkpoint_state(self) -> Dict[str, object]:
        return {
            "config": self.config.__dict__,
            "total_it": self.total_it,
            "actors": self.actors.state_dict(),
            "actor_targets": self.actor_targets.state_dict(),
            "critic": self.critic.state_dict(),
            "critic_target": self.critic_target.state_dict(),
            "actor_optimizers": [opt.state_dict() for opt in self.actor_optimizers],
            "critic_optimizer": self.critic_optimizer.state_dict(),
        }

    def load_checkpoint_state(self, state: Dict[str, object]) -> None:
        self.total_it = int(state.get("total_it", 0))
        self.actors.load_state_dict(state["actors"])  # type: ignore[arg-type]
        self.actor_targets.load_state_dict(state["actor_targets"])  # type: ignore[arg-type]
        self.critic.load_state_dict(state["critic"])  # type: ignore[arg-type]
        self.critic_target.load_state_dict(state["critic_target"])  # type: ignore[arg-type]
        for optimizer, optimizer_state in zip(
            self.actor_optimizers, state.get("actor_optimizers", [])
        ):
            optimizer.load_state_dict(optimizer_state)
        if "critic_optimizer" in state:
            self.critic_optimizer.load_state_dict(state["critic_optimizer"])  # type: ignore[arg-type]
