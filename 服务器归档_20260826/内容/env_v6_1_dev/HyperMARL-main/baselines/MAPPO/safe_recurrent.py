"""Shared recurrent rollout utilities for the env-v3-safe MAPPO variants."""

from __future__ import annotations

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from flax.linen.initializers import constant, orthogonal


def mask_cross_agent_channels(
    message_tensor: jnp.ndarray,
    observer_index: int,
    scope: str,
) -> jnp.ndarray:
    """Apply the same full/self-only scope to one observer's agent messages."""
    if scope == "full":
        return message_tensor
    if scope in {"self_only", "other_zero"}:
        masked = jnp.zeros_like(message_tensor)
        return masked.at[:, int(observer_index), :].set(
            message_tensor[:, int(observer_index), :]
        )
    if scope == "all_zero":
        return jnp.zeros_like(message_tensor)
    raise ValueError("communication scope must be 'full', 'self_only', or 'all_zero'")


class IndependentGRUActors(nn.Module):
    """Four parameter-independent local GRU policies."""

    num_agents: int
    action_dim: int
    hidden_size: int
    log_std_init: float = -1.0
    two_stage_intent: bool = False
    intent_dim: int = 3
    intent_broadcast_mode: str = "full"
    intent_residual_limit: float = 0.25
    supply_message_dim: int = 0

    @nn.compact
    def __call__(
        self,
        local_obs: jnp.ndarray,
        hidden: jnp.ndarray,
        *,
        return_intents: bool = False,
        return_messages: bool = False,
        intent_broadcast_mode: str | None = None,
    ):
        """Produce local actions, optionally via a same-hour public intent phase.

        The intent is deterministic and is computed from the current local GRU
        state.  All agents then receive the complete fixed-order message before
        the final stochastic action distribution is parameterised.
        """
        means = []
        next_hidden = []
        actor_features = []
        intents = []
        for agent_index in range(self.num_agents):
            agent_hidden, _ = nn.GRUCell(
                features=self.hidden_size,
                name=f"actor_{agent_index}_gru",
            )(hidden[:, agent_index], local_obs[:, agent_index])
            features = nn.tanh(
                nn.Dense(
                    self.hidden_size,
                    kernel_init=orthogonal(np.sqrt(2.0)),
                    bias_init=constant(0.0),
                    name=f"actor_{agent_index}_hidden",
                )(agent_hidden)
            )
            actor_features.append(features)
            if self.two_stage_intent:
                intents.append(
                    nn.tanh(
                        nn.Dense(
                            self.intent_dim,
                            kernel_init=orthogonal(0.01),
                            bias_init=constant(0.0),
                            name=f"actor_{agent_index}_intent",
                        )(agent_hidden)
                    )
                )
            next_hidden.append(agent_hidden)

        broadcast_mode = (
            self.intent_broadcast_mode
            if intent_broadcast_mode is None
            else intent_broadcast_mode
        )
        if self.two_stage_intent:
            intent_tensor = jnp.stack(intents, axis=1)
            if self.supply_message_dim:
                supply_facts = local_obs[..., -self.supply_message_dim:]
                message_tensor = jnp.concatenate((intent_tensor, supply_facts), axis=-1)
            else:
                message_tensor = intent_tensor
            full_message = message_tensor.reshape(message_tensor.shape[0], -1)
            for agent_index, features in enumerate(actor_features):
                message = mask_cross_agent_channels(
                    message_tensor,
                    observer_index=agent_index,
                    scope=broadcast_mode,
                ).reshape(message_tensor.shape[0], -1)
                stage2_features = nn.tanh(
                    nn.Dense(
                        self.hidden_size,
                        kernel_init=orthogonal(np.sqrt(2.0)),
                        bias_init=constant(0.0),
                        name=f"actor_{agent_index}_stage2_hidden",
                    )(jnp.concatenate((features, message), axis=-1))
                )
                means.append(
                    nn.Dense(
                        self.action_dim,
                        kernel_init=orthogonal(0.01),
                        bias_init=constant(0.0),
                        name=f"actor_{agent_index}_mean",
                    )(stage2_features)
                )
                raw_action = means[-1]
                residual = jnp.tanh(raw_action[:, (0, 1, 5)])
                committed = jnp.clip(
                    intent_tensor[:, agent_index, :]
                    + float(self.intent_residual_limit) * residual,
                    -0.999,
                    0.999,
                )
                means[-1] = raw_action.at[:, (0, 1, 5)].set(jnp.arctanh(committed))
        else:
            for agent_index, features in enumerate(actor_features):
                means.append(
                    nn.Dense(
                    self.action_dim,
                    kernel_init=orthogonal(0.01),
                    bias_init=constant(0.0),
                    name=f"actor_{agent_index}_mean",
                    )(features)
                )
        means = jnp.stack(means, axis=1)
        next_hidden = jnp.stack(next_hidden, axis=1)
        log_std = self.param(
            "log_std", constant(self.log_std_init), (self.num_agents, self.action_dim)
        )
        log_stds = jnp.broadcast_to(log_std[None, :, :], means.shape)
        if return_intents:
            if return_messages:
                return means, log_stds, next_hidden, intent_tensor, message_tensor
            if self.two_stage_intent:
                return means, log_stds, next_hidden, intent_tensor
            return means, log_stds, next_hidden, jnp.zeros(
                (*means.shape[:2], self.intent_dim), dtype=means.dtype
            )
        return means, log_stds, next_hidden


class CentralGRUCritic(nn.Module):
    """One centralized GRU critic, used independently for reward or cost."""

    hidden_size: int

    @nn.compact
    def __call__(self, global_obs: jnp.ndarray, hidden: jnp.ndarray):
        next_hidden, _ = nn.GRUCell(
            features=self.hidden_size, name="central_critic_gru"
        )(hidden, global_obs)
        features = nn.tanh(
            nn.Dense(
                self.hidden_size,
                kernel_init=orthogonal(np.sqrt(2.0)),
                bias_init=constant(0.0),
                name="central_critic_hidden",
            )(next_hidden)
        )
        values = nn.Dense(
            1,
            kernel_init=orthogonal(1.0),
            bias_init=constant(0.0),
            name="central_critic_value",
        )(features)
        return jnp.squeeze(values, axis=-1), next_hidden


def reset_actor_hidden(hidden: jnp.ndarray, done: jnp.ndarray) -> jnp.ndarray:
    """Zero all four actor states for vector environments that have terminated."""
    done = jnp.asarray(done, dtype=bool)
    return jnp.where(done[:, None, None], jnp.zeros_like(hidden), hidden)


def reset_global_hidden(hidden: jnp.ndarray, done: jnp.ndarray) -> jnp.ndarray:
    """Zero one centralized critic state per terminated vector environment."""
    done = jnp.asarray(done, dtype=bool)
    return jnp.where(done[:, None], jnp.zeros_like(hidden), hidden)


def compute_gae(
    rewards: jnp.ndarray,
    values: jnp.ndarray,
    last_values: jnp.ndarray,
    dones: jnp.ndarray,
    *,
    gamma: float,
    gae_lambda: float,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Compute batched time-major GAE without bootstrapping across episode ends."""
    def scan_step(carry, transition):
        next_advantage, next_value = carry
        reward, value, done = transition
        nonterminal = 1.0 - done.astype(values.dtype)
        delta = reward + gamma * next_value * nonterminal - value
        advantage = delta + gamma * gae_lambda * nonterminal * next_advantage
        return (advantage, value), advantage

    _, advantages = jax.lax.scan(
        scan_step,
        (jnp.zeros_like(last_values), last_values),
        (rewards, values, dones),
        reverse=True,
    )
    return advantages, advantages + values


def update_lagrange_multiplier(
    multiplier: float, *, cost_mean: float, budget: float, lr: float
) -> float:
    """Projected dual ascent for the one shared system voltage constraint."""
    return max(0.0, float(multiplier) + float(lr) * (float(cost_mean) - float(budget)))
