"""Credit assignment utilities used between rollout collection and MAPPO update."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import torch
from torch import nn

from .reward_model import STASRewardModel

from .credit_conservation import (
    CreditQualityGate,
    discounted_team_return,
    explained_variance,
    project_discounted_credits,
)

@dataclass
class STASCreditConfig:
    obs_dim: int
    action_dim: int
    n_agents: int
    seq_length: int
    gamma: float = 0.99
    mix_coef: float = 1.0
    lr: float = 1e-3
    emb_dim: int = 128
    n_heads: int = 4
    n_layers: int = 2
    sample_num: int = 4
    dropout: float = 0.1
    eval_mask_seed: int = 3030
    eval_mask_count: int = 8
    buffer_size: int = 128
    batch_size: int = 16
    update_freq: int = 1
    updates_per_step: int = 1
    warmup_rollouts: int = 1
    global_reward_agg: str = "sum"
    device: str = "cpu"
    causal: bool = True
    conserve_discounted: bool = False
    quality_gate_enable: bool = False
    warmup_episodes: int = 2000
    ramp_episodes: int = 8000
    max_mix_coef: float = 0.1
    explained_variance_threshold: float = 0.2
    negative_patience: int = 3
    mode: str = "legacy"
    weight_decay: float = 0.0
    reward_model_update_interval_episodes: int = 800
    reward_model_updates_per_interval: int = 50
    policy_warmup_episodes: int = 4000


class EpisodeCreditBuffer:
    """Small episode buffer for reward model supervised training."""

    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.storage: List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]] = []

    def add(
        self,
        obs: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        dones: np.ndarray,
        global_return: float,
    ) -> None:
        if len(self.storage) >= self.capacity:
            self.storage.pop(0)
        self.storage.append(
            (
                obs.copy(),
                actions.copy(),
                rewards.copy(),
                dones.copy(),
                float(global_return),
            )
        )

    def sample(self, batch_size: int):
        size = min(batch_size, len(self.storage))
        indices = np.random.choice(len(self.storage), size=size, replace=False)
        batch = [self.storage[i] for i in indices]
        obs, actions, rewards, dones, returns = zip(*batch)
        return (
            np.asarray(obs, dtype=np.float32),
            np.asarray(actions, dtype=np.float32),
            np.asarray(rewards, dtype=np.float32),
            np.asarray(dones, dtype=np.float32),
            np.asarray(returns, dtype=np.float32),
        )

    def get_all(self):
        """Return every stored episode in stable insertion order."""
        obs, actions, rewards, dones, returns = zip(*self.storage)
        return (
            np.asarray(obs, dtype=np.float32),
            np.asarray(actions, dtype=np.float32),
            np.asarray(rewards, dtype=np.float32),
            np.asarray(dones, dtype=np.float32),
            np.asarray(returns, dtype=np.float32),
        )

    def __len__(self) -> int:
        return len(self.storage)


class STASCreditAssigner:
    """Train STAS reward model and produce MAPPO-compatible credit rewards."""

    def __init__(self, config: STASCreditConfig) -> None:
        self.config = config
        self.device = torch.device(config.device)
        self.model = STASRewardModel(
            obs_dim=config.obs_dim,
            action_dim=config.action_dim,
            n_agents=config.n_agents,
            seq_length=config.seq_length,
            emb_dim=config.emb_dim,
            n_heads=config.n_heads,
            n_layers=config.n_layers,
            sample_num=config.sample_num,
            dropout=config.dropout,
            eval_mask_seed=config.eval_mask_seed,
            eval_mask_count=config.eval_mask_count,
        ).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.lr)
        self.loss_fn = nn.MSELoss()
        self.buffer = EpisodeCreditBuffer(config.buffer_size)
        self.rollouts_seen = 0
        self.last_loss = np.nan

    def _discounted_global_return(self, rewards: np.ndarray) -> float:
        if self.config.global_reward_agg == "mean":
            global_step_reward = rewards.mean(axis=0)
        else:
            global_step_reward = rewards.sum(axis=0)
        discounts = self.config.gamma ** np.arange(len(global_step_reward))
        return float(np.sum(discounts * global_step_reward))

    def add_rollout(
        self,
        obs_eat: np.ndarray,
        actions_eat: np.ndarray,
        rewards_eat: np.ndarray,
        dones_eat: np.ndarray,
    ) -> None:
        """Add rollout chunks shaped ``[env, agent, time, ...]`` to the buffer."""
        num_envs = obs_eat.shape[0]
        for env_idx in range(num_envs):
            global_return = self._discounted_global_return(rewards_eat[env_idx])
            self.buffer.add(
                obs_eat[env_idx],
                actions_eat[env_idx],
                rewards_eat[env_idx],
                dones_eat[env_idx],
                global_return,
            )
        self.rollouts_seen += 1

    def train_if_ready(self) -> float:
        cfg = self.config
        if self.rollouts_seen < cfg.warmup_rollouts:
            return self.last_loss
        if len(self.buffer) < max(1, min(cfg.batch_size, cfg.buffer_size)):
            return self.last_loss
        if self.rollouts_seen % max(1, cfg.update_freq) != 0:
            return self.last_loss

        self.model.train()
        losses = []
        for _ in range(max(1, cfg.updates_per_step)):
            obs, actions, rewards, dones, returns = self.buffer.sample(cfg.batch_size)
            obs_t = torch.as_tensor(obs, device=self.device)
            actions_t = torch.as_tensor(actions, device=self.device)
            rewards_t = torch.as_tensor(rewards, device=self.device)
            dones_t = torch.as_tensor(dones, device=self.device)
            returns_t = torch.as_tensor(returns, device=self.device)

            pred = self.model(obs_t, actions_t, rewards_t, dones_t)
            pred_return = pred.sum(dim=(1, 2))
            loss = self.loss_fn(pred_return, returns_t)

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()
            losses.append(float(loss.detach().cpu()))

        self.last_loss = float(np.mean(losses))
        return self.last_loss

    @torch.no_grad()
    def credit_rewards(
        self,
        obs_eat: np.ndarray,
        actions_eat: np.ndarray,
        rewards_eat: np.ndarray,
        dones_eat: np.ndarray,
    ) -> np.ndarray:
        """Return credit rewards shaped ``[env, agent, time]``."""
        self.model.eval()
        obs_t = torch.as_tensor(obs_eat, dtype=torch.float32, device=self.device)
        actions_t = torch.as_tensor(actions_eat, dtype=torch.float32, device=self.device)
        rewards_t = torch.as_tensor(rewards_eat, dtype=torch.float32, device=self.device)
        dones_t = torch.as_tensor(dones_eat, dtype=torch.float32, device=self.device)
        credit = self.model(obs_t, actions_t, rewards_t, dones_t)
        return credit.detach().cpu().numpy().astype(np.float32)

    def process_rollout(
        self,
        obs_eat: np.ndarray,
        actions_eat: np.ndarray,
        rewards_eat: np.ndarray,
        dones_eat: np.ndarray,
    ) -> Tuple[np.ndarray, float]:
        """Store rollout, train the reward model, and return blended rewards."""
        self.add_rollout(obs_eat, actions_eat, rewards_eat, dones_eat)
        loss = self.train_if_ready()
        min_batch = max(1, min(self.config.batch_size, self.config.buffer_size))
        if self.rollouts_seen < self.config.warmup_rollouts or len(self.buffer) < min_batch:
            return rewards_eat.astype(np.float32), loss
        credit = self.credit_rewards(obs_eat, actions_eat, rewards_eat, dones_eat)
        blended = (1.0 - self.config.mix_coef) * rewards_eat + self.config.mix_coef * credit
        return blended.astype(np.float32), loss
