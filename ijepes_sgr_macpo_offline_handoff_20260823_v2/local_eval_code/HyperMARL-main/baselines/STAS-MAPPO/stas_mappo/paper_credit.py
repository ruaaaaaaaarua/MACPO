"""Paper-style spatial-temporal return decomposition for terminal team returns."""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
import torch
from torch import nn

from .credit import STASCreditConfig
from .credit_conservation import explained_variance


def _valid_mask_from_dones(dones: np.ndarray) -> np.ndarray:
    dones = np.asarray(dones, dtype=bool)
    valid = np.ones_like(dones, dtype=bool)
    if dones.shape[-1] > 1:
        valid[..., 1:] = np.cumsum(dones[..., :-1], axis=-1) == 0
    return valid


def shared_team_return(
    rewards: np.ndarray,
    gamma: float,
    valid_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Return one team stream per episode from repeated shared rewards."""

    rewards = np.asarray(rewards, dtype=np.float64)
    if rewards.ndim != 3:
        raise ValueError("shared rewards must have shape [episode, agent, time]")
    if valid_mask is None:
        valid = np.ones_like(rewards, dtype=bool)
    else:
        valid = np.asarray(valid_mask, dtype=bool)
        if valid.shape != rewards.shape:
            raise ValueError("valid_mask must match shared reward shape")
    reference = rewards[:, :1, :]
    comparable = valid & valid[:, :1, :]
    if np.any(np.abs(rewards - reference) > 1e-6, where=comparable):
        raise ValueError("agents did not receive an identical shared reward")
    if not np.all(valid == valid[:, :1, :]):
        raise ValueError("agents did not receive an identical shared reward mask")
    stream = rewards.mean(axis=1)
    stream_valid = valid[:, 0, :]
    discounts = np.power(float(gamma), np.arange(rewards.shape[-1], dtype=np.float64))
    return np.sum(stream * stream_valid * discounts[None, :], axis=-1)


class PaperTemporalEncoderLayer(nn.Module):
    def __init__(self, emb_dim: int, n_heads: int, dropout: float) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(
            emb_dim, n_heads, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(emb_dim)
        self.ffn = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(emb_dim, emb_dim),
        )
        self.norm2 = nn.LayerNorm(emb_dim)

    def forward(
        self,
        x: torch.Tensor,
        causal_mask: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        attended, _ = self.attn(
            x,
            x,
            x,
            attn_mask=causal_mask,
            key_padding_mask=~valid_mask,
            need_weights=False,
        )
        x = self.norm1(x + attended)
        x = self.norm2(x + self.ffn(x))
        return x * valid_mask.unsqueeze(-1)


class PaperShapleyAttention(nn.Module):
    """Agent-specific spatial attention with MC coalitions and permutations."""

    def __init__(
        self,
        emb_dim: int,
        n_heads: int,
        n_agents: int,
        sample_num: int,
        dropout: float,
        eval_mask_seed: int = 3030,
        eval_mask_count: int = 8,
    ) -> None:
        super().__init__()
        if emb_dim % n_heads:
            raise ValueError("emb_dim must be divisible by n_heads")
        self.emb_dim = int(emb_dim)
        self.n_heads = int(n_heads)
        self.head_dim = self.emb_dim // self.n_heads
        self.n_agents = int(n_agents)
        self.sample_num = max(1, int(sample_num))
        self.query_projections = nn.ModuleList(
            [nn.Linear(emb_dim, emb_dim) for _ in range(n_agents)]
        )
        self.key_projection = nn.Linear(emb_dim, emb_dim)
        self.value_projection = nn.Linear(emb_dim, emb_dim)
        self.output_projection = nn.Linear(emb_dim, emb_dim)
        self.agent_embedding = nn.Embedding(n_agents, emb_dim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(emb_dim)

        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(eval_mask_seed))
        count = max(1, int(eval_mask_count))
        keep = torch.rand((count, n_agents, n_agents), generator=generator) < 0.5
        diagonal = torch.arange(n_agents)
        keep[:, diagonal, diagonal] = True
        permutations = torch.stack(
            [torch.randperm(n_agents, generator=generator) for _ in range(count)]
        )
        self.register_buffer("eval_coalition_keep_bank", keep, persistent=False)
        self.register_buffer("eval_permutation_bank", permutations, persistent=False)

    def _sample_keep(self, device: torch.device) -> torch.Tensor:
        keep = torch.rand(
            (self.n_agents, self.n_agents), device=device
        ) < 0.5
        keep.fill_diagonal_(True)
        return keep

    def _forward_sample(
        self,
        x: torch.Tensor,
        valid_mask: torch.Tensor,
        permutation: torch.Tensor,
        coalition_keep_by_identity: torch.Tensor,
    ) -> torch.Tensor:
        batch, agents, seq_len, emb_dim = x.shape
        if agents != self.n_agents:
            raise ValueError(f"expected {self.n_agents} agents, got {agents}")
        permutation = permutation.to(device=x.device, dtype=torch.long)
        inverse = torch.argsort(permutation)
        identities = torch.arange(agents, device=x.device)[permutation]

        original = x.permute(0, 2, 1, 3).reshape(batch * seq_len, agents, emb_dim)
        valid = valid_mask.permute(0, 2, 1).reshape(batch * seq_len, agents)
        ordered = original[:, permutation, :]
        ordered_valid = valid[:, permutation]
        ordered = ordered + self.agent_embedding(identities)[None, :, :]

        query = torch.stack(
            [
                self.query_projections[int(identity)](ordered[:, position, :])
                for position, identity in enumerate(identities.tolist())
            ],
            dim=1,
        )
        key = self.key_projection(ordered)
        value = self.value_projection(ordered)

        def split_heads(tensor):
            return tensor.reshape(
                batch * seq_len, agents, self.n_heads, self.head_dim
            ).permute(0, 2, 1, 3)

        query = split_heads(query)
        key = split_heads(key)
        value = split_heads(value)
        scores = torch.matmul(query, key.transpose(-1, -2)) / np.sqrt(self.head_dim)
        keep = coalition_keep_by_identity.to(device=x.device, dtype=torch.bool)
        keep = keep[identities[:, None], identities[None, :]]
        allowed = keep[None, None, :, :] & ordered_valid[:, None, None, :]
        scores = scores.masked_fill(~allowed, -1e9)
        probabilities = self.dropout(torch.softmax(scores, dim=-1))
        attended = torch.matmul(probabilities, value)
        attended = attended.permute(0, 2, 1, 3).reshape(
            batch * seq_len, agents, emb_dim
        )
        attended = self.output_projection(attended) * ordered_valid.unsqueeze(-1)
        attended = attended[:, inverse, :]
        output = self.norm(original + attended) * valid.unsqueeze(-1)
        return output.reshape(batch, seq_len, agents, emb_dim).permute(0, 2, 1, 3)

    def forward(self, x: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        outputs = []
        if self.training:
            for _ in range(self.sample_num):
                outputs.append(
                    self._forward_sample(
                        x,
                        valid_mask,
                        torch.randperm(self.n_agents, device=x.device),
                        self._sample_keep(x.device),
                    )
                )
        else:
            for permutation, keep in zip(
                self.eval_permutation_bank, self.eval_coalition_keep_bank
            ):
                outputs.append(
                    self._forward_sample(x, valid_mask, permutation, keep)
                )
        return torch.stack(outputs, dim=0).mean(dim=0)


class PaperSTASRewardModel(nn.Module):
    """Causal STAS model whose inputs contain no reward or termination values."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        n_agents: int,
        seq_length: int,
        emb_dim: int = 128,
        n_heads: int = 4,
        n_layers: int = 3,
        sample_num: int = 5,
        dropout: float = 0.1,
        eval_mask_seed: int = 3030,
        eval_mask_count: int = 8,
    ) -> None:
        super().__init__()
        self.n_agents = int(n_agents)
        self.seq_length = int(seq_length)
        self.obs_emb = nn.Linear(obs_dim, emb_dim)
        self.action_emb = nn.Linear(action_dim, emb_dim)
        self.pos_embedding = nn.Embedding(seq_length, emb_dim)
        self.temporal_layers = nn.ModuleList(
            [PaperTemporalEncoderLayer(emb_dim, n_heads, dropout) for _ in range(n_layers)]
        )
        self.shapley = PaperShapleyAttention(
            emb_dim,
            n_heads,
            n_agents,
            sample_num,
            dropout,
            eval_mask_seed,
            eval_mask_count,
        )
        self.out = nn.Linear(emb_dim, 1)

    def forward(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch, agents, seq_len, _ = obs.shape
        if agents != self.n_agents:
            raise ValueError(f"expected {self.n_agents} agents, got {agents}")
        if seq_len > self.seq_length:
            raise ValueError("sequence exceeds configured STAS length")
        valid_mask = valid_mask.to(dtype=torch.bool, device=obs.device)
        positions = self.pos_embedding(torch.arange(seq_len, device=obs.device))
        x = self.obs_emb(obs) + self.action_emb(actions) + positions[None, None, :, :]
        x = x.reshape(batch * agents, seq_len, -1)
        valid_time = valid_mask.reshape(batch * agents, seq_len)
        causal = torch.triu(
            torch.ones(seq_len, seq_len, device=obs.device, dtype=torch.bool),
            diagonal=1,
        )
        for layer in self.temporal_layers:
            x = layer(x, causal, valid_time)
        x = x.reshape(batch, agents, seq_len, -1)
        credit = self.out(self.shapley(x, valid_mask)).squeeze(-1)
        return credit * valid_mask


class PaperEpisodeBuffer:
    def __init__(self, capacity: int) -> None:
        self.capacity = int(capacity)
        self.storage: List[
            Tuple[np.ndarray, np.ndarray, np.ndarray, float]
        ] = []

    def add(self, obs, actions, valid_mask, team_return) -> None:
        if len(self.storage) >= self.capacity:
            self.storage.pop(0)
        self.storage.append(
            (
                np.asarray(obs, dtype=np.float32).copy(),
                np.asarray(actions, dtype=np.float32).copy(),
                np.asarray(valid_mask, dtype=bool).copy(),
                float(team_return),
            )
        )

    def sample(self, batch_size: int):
        size = min(int(batch_size), len(self.storage))
        indices = np.random.choice(len(self.storage), size=size, replace=False)
        batch = [self.storage[index] for index in indices]
        obs, actions, valid, returns = zip(*batch)
        return (
            np.asarray(obs, dtype=np.float32),
            np.asarray(actions, dtype=np.float32),
            np.asarray(valid, dtype=bool),
            np.asarray(returns, dtype=np.float32),
        )

    def get_all(self):
        obs, actions, valid, returns = zip(*self.storage)
        return (
            np.asarray(obs, dtype=np.float32),
            np.asarray(actions, dtype=np.float32),
            np.asarray(valid, dtype=bool),
            np.asarray(returns, dtype=np.float32),
        )

    def __len__(self) -> int:
        return len(self.storage)


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


class PaperSTASCreditAssigner:
    def __init__(self, config: STASCreditConfig) -> None:
        self.config = config
        self.device = torch.device(config.device)
        self.model = PaperSTASRewardModel(
            config.obs_dim,
            config.action_dim,
            config.n_agents,
            config.seq_length,
            config.emb_dim,
            config.n_heads,
            config.n_layers,
            config.sample_num,
            config.dropout,
            config.eval_mask_seed,
            config.eval_mask_count,
        ).to(self.device)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=config.lr,
            weight_decay=config.weight_decay,
        )
        self.loss_fn = nn.MSELoss()
        self.buffer = PaperEpisodeBuffer(config.buffer_size)
        self.holdout_buffer = PaperEpisodeBuffer(max(8, config.buffer_size // 5))
        self.normalizer = RunningReturnNormalizer()
        self.rollouts_seen = 0
        self.episodes_seen = 0
        self.reward_model_updates = 0
        self.next_reward_model_update_episode = max(
            1, config.reward_model_update_interval_episodes
        )
        self.last_loss = float("nan")
        self.last_explained_variance = 0.0
        self.last_reconstruction_rmse = 0.0
        self.last_agent_credit_variance = 0.0
        self.last_time_credit_variance = 0.0
        self.last_conservation_error = float("nan")
        self.last_mix_coef = 0.0

    def add_rollout(self, obs, actions, rewards, dones) -> None:
        valid = _valid_mask_from_dones(dones)
        targets = shared_team_return(rewards, self.config.gamma, valid)
        self.normalizer.update(targets)
        for env_index, target in enumerate(targets):
            self.episodes_seen += 1
            destination = (
                self.holdout_buffer if self.episodes_seen % 5 == 0 else self.buffer
            )
            destination.add(obs[env_index], actions[env_index], valid[env_index], target)
        self.rollouts_seen += 1

    def _predict_normalized(self, obs, actions, valid):
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        actions_t = torch.as_tensor(actions, dtype=torch.float32, device=self.device)
        valid_t = torch.as_tensor(valid, dtype=torch.bool, device=self.device)
        credit = self.model(obs_t, actions_t, valid_t)
        discounts = torch.pow(
            torch.as_tensor(self.config.gamma, device=self.device),
            torch.arange(credit.shape[-1], device=self.device),
        )
        predicted = torch.sum(
            credit * valid_t * discounts[None, None, :], dim=(1, 2)
        )
        return credit, predicted, valid_t, discounts

    def _run_reward_model_updates(self, count: int) -> None:
        minimum = max(1, min(self.config.batch_size, self.config.buffer_size))
        if len(self.buffer) < minimum:
            return
        self.model.train()
        losses = []
        for _ in range(int(count)):
            obs, actions, valid, returns = self.buffer.sample(self.config.batch_size)
            target = torch.as_tensor(
                self.normalizer.normalize(returns),
                dtype=torch.float32,
                device=self.device,
            )
            _, predicted, _, _ = self._predict_normalized(obs, actions, valid)
            loss = self.loss_fn(predicted, target)
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()
            losses.append(float(loss.detach().cpu()))
            self.reward_model_updates += 1
        if losses:
            self.last_loss = float(np.mean(losses))

    def train_if_due(self) -> float:
        interval = max(1, self.config.reward_model_update_interval_episodes)
        while self.episodes_seen >= self.next_reward_model_update_episode:
            self._run_reward_model_updates(
                self.config.reward_model_updates_per_interval
            )
            self.next_reward_model_update_episode += interval
        return self.last_loss

    @torch.no_grad()
    def _update_holdout_diagnostics(self) -> None:
        if len(self.holdout_buffer) < 2:
            return
        self.model.eval()
        obs, actions, valid, returns = self.holdout_buffer.get_all()
        _, normalized, _, _ = self._predict_normalized(obs, actions, valid)
        predicted = normalized.cpu().numpy() * self.normalizer.std + self.normalizer.mean
        self.last_explained_variance = explained_variance(returns, predicted)
        self.last_reconstruction_rmse = float(
            np.sqrt(np.mean(np.square(predicted - returns)))
        )

    @torch.no_grad()
    def credit_rewards(self, obs, actions, rewards, dones) -> np.ndarray:
        del rewards
        self.model.eval()
        valid = _valid_mask_from_dones(dones)
        normalized, _, valid_t, discounts = self._predict_normalized(
            obs, actions, valid
        )
        weighted_count = torch.sum(
            valid_t * discounts[None, None, :], dim=(1, 2), keepdim=True
        ).clamp_min(1.0)
        raw = normalized * self.normalizer.std
        raw = raw + self.normalizer.mean / weighted_count
        raw = raw * valid_t
        result = raw.cpu().numpy().astype(np.float32)
        self.last_agent_credit_variance = float(
            np.mean(np.var(result, axis=1))
        )
        self.last_time_credit_variance = float(
            np.mean(np.var(result, axis=2))
        )
        return result

    def process_rollout(self, obs, actions, rewards, dones):
        self.add_rollout(obs, actions, rewards, dones)
        loss = self.train_if_due()
        self._update_holdout_diagnostics()
        if self.episodes_seen <= self.config.policy_warmup_episodes:
            self.last_mix_coef = 0.0
            return np.asarray(rewards, dtype=np.float32), loss
        self.last_mix_coef = 1.0
        return self.credit_rewards(obs, actions, rewards, dones), loss
