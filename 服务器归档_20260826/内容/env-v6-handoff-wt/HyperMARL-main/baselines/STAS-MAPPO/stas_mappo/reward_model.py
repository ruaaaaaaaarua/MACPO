"""PyTorch spatial-temporal reward decomposition model for continuous actions."""

from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F


class TemporalEncoderLayer(nn.Module):
    """Causal self-attention over time for each agent trajectory."""

    def __init__(self, emb_dim: int, n_heads: int, dropout: float) -> None:
        super().__init__()
        if emb_dim % n_heads != 0:
            raise ValueError("emb_dim must be divisible by n_heads")
        self.attn = nn.MultiheadAttention(
            embed_dim=emb_dim,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(emb_dim)
        self.ffn = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(emb_dim, emb_dim),
        )
        self.norm2 = nn.LayerNorm(emb_dim)

    def forward(self, x: torch.Tensor, causal_mask: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.attn(x, x, x, attn_mask=causal_mask)
        x = self.norm1(x + attn_out)
        return self.norm2(x + self.ffn(x))


class ShapleyAttention(nn.Module):
    """Monte Carlo coalition attention across agents at each time step."""

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
        if emb_dim % n_heads != 0:
            raise ValueError("emb_dim must be divisible by n_heads")
        self.n_agents = n_agents
        self.sample_num = max(1, sample_num)
        self.eval_mask_count = int(eval_mask_count)
        if self.eval_mask_count < 1:
            raise ValueError("eval_mask_count must be positive")
        self.attn = nn.MultiheadAttention(
            embed_dim=emb_dim,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.agent_embedding = nn.Embedding(n_agents, emb_dim)
        self.norm = nn.LayerNorm(emb_dim)

        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(eval_mask_seed))
        keep = torch.rand(
            (self.eval_mask_count, n_agents, n_agents), generator=generator
        ) < 0.5
        diagonal = torch.arange(n_agents)
        keep[:, diagonal, diagonal] = True
        self.register_buffer("_eval_mask_bank", ~keep, persistent=False)

    def _coalition_mask(self, device: torch.device) -> torch.Tensor:
        keep = torch.bernoulli(
            torch.full((self.n_agents, self.n_agents), 0.5, device=device)
        ).bool()
        keep.fill_diagonal_(True)
        return ~keep

    def _attention_masks(self, device: torch.device) -> torch.Tensor:
        if self.training:
            return torch.stack(
                [self._coalition_mask(device) for _ in range(self.sample_num)]
            )
        return self._eval_mask_bank.to(device=device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, n_agents, seq_len, emb_dim = x.shape
        x = x.permute(0, 2, 1, 3).reshape(batch * seq_len, n_agents, emb_dim)
        agent_ids = torch.arange(n_agents, device=x.device)
        x = x + self.agent_embedding(agent_ids)[None, :, :]

        outputs = []
        for mask in self._attention_masks(x.device):
            out, _ = self.attn(x, x, x, attn_mask=mask)
            outputs.append(out)
        out = torch.stack(outputs, dim=0).mean(dim=0)
        out = self.norm(x + out)
        return out.reshape(batch, seq_len, n_agents, emb_dim).permute(0, 2, 1, 3)


class STASRewardModel(nn.Module):
    """Return decomposition model that outputs agent-time credit rewards.

    Args:
        obs_dim: Local observation size per agent.
        action_dim: Continuous action size per agent.
        n_agents: Number of agents.
        seq_length: Rollout or episode length.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        n_agents: int,
        seq_length: int,
        emb_dim: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        sample_num: int = 4,
        dropout: float = 0.1,
        causal: bool = True,
        eval_mask_seed: int = 3030,
        eval_mask_count: int = 8,
    ) -> None:
        super().__init__()
        self.seq_length = seq_length
        self.n_agents = n_agents
        self.causal = bool(causal)
        self.obs_emb = nn.Linear(obs_dim, emb_dim)
        self.action_emb = nn.Linear(action_dim, emb_dim)
        self.reward_emb = nn.Linear(1, emb_dim)
        self.done_emb = nn.Linear(1, emb_dim)
        self.pos_embedding = nn.Embedding(seq_length, emb_dim)
        self.temporal_layers = nn.ModuleList(
            [TemporalEncoderLayer(emb_dim, n_heads, dropout) for _ in range(n_layers)]
        )
        self.shapley = ShapleyAttention(
            emb_dim=emb_dim,
            n_heads=n_heads,
            n_agents=n_agents,
            sample_num=sample_num,
            dropout=dropout,
            eval_mask_seed=eval_mask_seed,
            eval_mask_count=eval_mask_count,
        )
        self.out = nn.Linear(emb_dim, 1)

    def _causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        return torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()

    def forward(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        dones: torch.Tensor,
    ) -> torch.Tensor:
        """Return credit rewards with shape ``[batch, n_agents, seq_len]``."""

        batch, n_agents, seq_len, _ = obs.shape
        if n_agents != self.n_agents:
            raise ValueError(f"expected {self.n_agents} agents, got {n_agents}")
        if seq_len > self.seq_length:
            raise ValueError(f"seq_len {seq_len} exceeds configured {self.seq_length}")

        positions = self.pos_embedding(torch.arange(seq_len, device=obs.device))
        positions = positions[None, None, :, :]
        x = (
            self.obs_emb(obs)
            + self.action_emb(actions)
            + self.reward_emb(rewards.unsqueeze(-1))
            + self.done_emb(dones.unsqueeze(-1).float())
            + positions
        )
        x = x.reshape(batch * n_agents, seq_len, -1)
        causal_mask = self._causal_mask(seq_len, obs.device) if self.causal else None
        for layer in self.temporal_layers:
            x = layer(x, causal_mask)
        x = x.reshape(batch, n_agents, seq_len, -1)
        x = self.shapley(x)
        return self.out(x).squeeze(-1)
