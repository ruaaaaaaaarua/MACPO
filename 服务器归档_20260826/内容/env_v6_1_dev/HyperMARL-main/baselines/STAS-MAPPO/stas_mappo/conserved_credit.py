"""Return-conserving STAS credit assigner with held-out quality gating."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from .credit import EpisodeCreditBuffer, STASCreditConfig
from .credit_conservation import (
    CreditQualityGate,
    discounted_team_return,
    explained_variance,
    project_discounted_credits,
)
from .diagnostics import compute_conservation_error, compute_target_error
from .reward_model import STASRewardModel


class RunningReturnNormalizer:
    def __init__(self) -> None:
        self.count = 0
        self.mean = 0.0
        self.m2 = 0.0

    def update(self, values) -> None:
        for value in np.asarray(values, dtype=np.float64).reshape(-1):
            self.count += 1
            delta = float(value) - self.mean
            self.mean += delta / self.count
            self.m2 += delta * (float(value) - self.mean)

    @property
    def std(self) -> float:
        if self.count < 2:
            return 1.0
        return max(float(np.sqrt(self.m2 / (self.count - 1))), 1e-6)

    def normalize(self, values):
        return (np.asarray(values, dtype=np.float32) - self.mean) / self.std


class ConservedSTASCreditAssigner:
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
            causal=config.causal,
            eval_mask_seed=config.eval_mask_seed,
            eval_mask_count=config.eval_mask_count,
        ).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.lr)
        self.loss_fn = nn.MSELoss()
        self.buffer = EpisodeCreditBuffer(config.buffer_size)
        self.holdout_buffer = EpisodeCreditBuffer(max(8, config.buffer_size // 4))
        self.normalizer = RunningReturnNormalizer()
        self.gate = CreditQualityGate(
            config.warmup_episodes,
            config.ramp_episodes,
            config.max_mix_coef,
            config.explained_variance_threshold,
            config.negative_patience,
        )
        self.rollouts_seen = 0
        self.episodes_seen = 0
        self.last_loss = np.nan
        self.last_explained_variance = 0.0
        self.last_conservation_error = 0.0
        self.last_mix_coef = 0.0
        self.last_target_error = 0.0

    def _weights_torch(self, seq_len: int):
        return torch.pow(
            torch.as_tensor(self.config.gamma, device=self.device),
            torch.arange(seq_len, device=self.device),
        )

    def add_rollout(self, obs, actions, rewards, dones) -> None:
        targets = discounted_team_return(rewards, self.config.gamma)
        self.normalizer.update(targets)
        for env_idx, target in enumerate(targets):
            item = (obs[env_idx], actions[env_idx], rewards[env_idx], dones[env_idx], target)
            self.episodes_seen += 1
            destination = self.holdout_buffer if self.episodes_seen % 5 == 0 else self.buffer
            destination.add(*item)
        self.rollouts_seen += 1
        self.last_target_error = compute_target_error(
            (self.buffer, self.holdout_buffer), self.config.gamma
        )

    def _predict_normalized_returns(self, obs, actions, rewards, dones):
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        actions_t = torch.as_tensor(actions, dtype=torch.float32, device=self.device)
        rewards_t = torch.as_tensor(rewards, dtype=torch.float32, device=self.device)
        dones_t = torch.as_tensor(dones, dtype=torch.float32, device=self.device)
        credit = self.model(obs_t, actions_t, rewards_t, dones_t)
        weights = self._weights_torch(credit.shape[2])
        return credit, torch.sum(credit * weights[None, None, :], dim=(1, 2))

    def train_if_ready(self) -> float:
        cfg = self.config
        minimum = max(1, min(cfg.batch_size, cfg.buffer_size))
        if self.rollouts_seen < cfg.warmup_rollouts or len(self.buffer) < minimum:
            return self.last_loss
        if self.rollouts_seen % max(1, cfg.update_freq) != 0:
            return self.last_loss
        self.model.train()
        losses = []
        for _ in range(max(1, cfg.updates_per_step)):
            obs, actions, rewards, dones, returns = self.buffer.sample(cfg.batch_size)
            normalized = torch.as_tensor(
                self.normalizer.normalize(returns), dtype=torch.float32, device=self.device
            )
            _, predicted = self._predict_normalized_returns(obs, actions, rewards, dones)
            loss = self.loss_fn(predicted, normalized)
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()
            losses.append(float(loss.detach().cpu()))
        self.last_loss = float(np.mean(losses))
        return self.last_loss

    @torch.no_grad()
    def holdout_explained_variance(self) -> float:
        if len(self.holdout_buffer) < 2:
            return 0.0
        self.model.eval()
        obs, actions, rewards, dones, returns = self.holdout_buffer.get_all()
        _, predicted_normalized = self._predict_normalized_returns(
            obs, actions, rewards, dones
        )
        predicted = predicted_normalized.detach().cpu().numpy() * self.normalizer.std
        predicted += self.normalizer.mean
        return explained_variance(returns, predicted)

    @torch.no_grad()
    def credit_rewards(self, obs, actions, rewards, dones) -> np.ndarray:
        self.model.eval()
        normalized_credit, _ = self._predict_normalized_returns(obs, actions, rewards, dones)
        normalized_credit = normalized_credit.detach().cpu().numpy().astype(np.float64)
        seq_len = normalized_credit.shape[2]
        weights = np.power(self.config.gamma, np.arange(seq_len, dtype=np.float64))
        mean_allocation = self.normalizer.mean * weights / (
            self.config.n_agents * np.square(weights).sum()
        )
        raw_credit = normalized_credit * self.normalizer.std
        raw_credit += mean_allocation[None, None, :]
        targets = discounted_team_return(rewards, self.config.gamma)
        projected, errors = project_discounted_credits(
            raw_credit, targets, self.config.gamma
        )
        self.last_conservation_error = float(np.max(np.abs(errors)))
        return projected

    def _finalize_rollout_rewards(self, training_rewards, original_rewards, loss):
        """Record conservation for the exact float32 rewards MAPPO receives."""
        actual_rewards = np.asarray(training_rewards, dtype=np.float32)
        self.last_conservation_error = compute_conservation_error(
            actual_rewards,
            original_rewards,
            self.config.gamma,
        )
        return actual_rewards, loss

    def process_rollout(self, obs, actions, rewards, dones):
        self.add_rollout(obs, actions, rewards, dones)
        loss = self.train_if_ready()
        minimum = max(1, min(self.config.batch_size, self.config.buffer_size))
        if len(self.buffer) < minimum:
            self.last_mix_coef = 0.0
            return self._finalize_rollout_rewards(rewards, rewards, loss)
        self.last_explained_variance = self.holdout_explained_variance()
        mix = self.gate.mix_coef(self.episodes_seen, self.last_explained_variance)
        self.last_mix_coef = mix
        if mix <= 0.0:
            return self._finalize_rollout_rewards(rewards, rewards, loss)
        credit = self.credit_rewards(obs, actions, rewards, dones)
        blended = (1.0 - mix) * rewards + mix * credit
        return self._finalize_rollout_rewards(blended, rewards, loss)


class UniformCreditAssigner(ConservedSTASCreditAssigner):
    """IRCR 风格照妖镜基线: credit = 按折扣权重均匀摊回的团队回报。

    复用守恒/门控/调度/checkpoint 全套管线, 只把注意力模型的逐步逐体
    分配换成常数分配。STAS 若赢不了它, 说明收益只来自方差整形而非
    真正的信用识别; 赢了则是注意力结构价值的最硬证据。
    """

    def train_if_ready(self) -> float:
        # 无模型可训; 保留缓冲计数使调度与父类完全同步。
        return 0.0

    def holdout_explained_variance(self) -> float:
        # 均匀分配对"episode 总回报"的解释是精确的 (守恒), 逐步归因
        # 无信息量 —— 这正是该基线的定义, EV 恒 1 使门控行为退化为纯调度。
        return 1.0

    @torch.no_grad()
    def credit_rewards(self, obs, actions, rewards, dones) -> np.ndarray:
        rewards = np.asarray(rewards, dtype=np.float64)
        n_envs, n_agents, seq_len = rewards.shape
        weights = np.power(self.config.gamma, np.arange(seq_len, dtype=np.float64))
        targets = discounted_team_return(rewards, self.config.gamma)
        flat = targets.reshape(-1, 1, 1) / (n_agents * max(float(weights.sum()), 1e-9))
        credit = np.broadcast_to(flat, rewards.shape).astype(np.float64).copy()
        projected, errors = project_discounted_credits(
            credit, targets, self.config.gamma
        )
        self.last_conservation_error = float(np.max(np.abs(errors)))
        return projected
