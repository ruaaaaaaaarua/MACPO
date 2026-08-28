"""Runnable GRU-MAPPO, GRU-MAPPO-Lagrangian, and shared-system MACPO trainer."""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import serialization
from flax.training.train_state import TrainState

from baselines.MAPPO.continuous_policy import sample_squashed_gaussian, squashed_log_prob
from baselines.MAPPO.safe_recurrent import (
    CentralGRUCritic,
    IndependentGRUActors,
    compute_gae,
    reset_actor_hidden,
    reset_global_hidden,
    update_lagrange_multiplier,
)
from baselines.MAPPO.shared_system_macpo import SharedSystemMACPOUpdater
from baselines.utils.microgrid_vec_env import MicrogridVecEnv


@dataclass
class SafeRollout:
    local_obs: jnp.ndarray
    global_obs: jnp.ndarray
    dones_before: jnp.ndarray
    dones: jnp.ndarray
    actions: jnp.ndarray
    log_probs: jnp.ndarray
    rewards: jnp.ndarray
    costs: jnp.ndarray
    raw_costs: jnp.ndarray
    reward_values: jnp.ndarray
    cost_values: jnp.ndarray
    intents: jnp.ndarray
    last_reward_value: jnp.ndarray
    last_cost_value: jnp.ndarray
    initial_actor_hidden: jnp.ndarray
    initial_reward_critic_hidden: jnp.ndarray
    initial_cost_critic_hidden: jnp.ndarray


class SafeGRUMAPPOTrainer:
    """Small explicit trainer that keeps economic reward and voltage cost apart."""

    def __init__(self, config: Mapping[str, Any]):
        self.config = {
            "seed": 30,
            "num_envs": 1,
            "num_steps": 24,
            "hidden_size": 64,
            "lr": 3e-4,
            "total_updates": 1,
            "anneal_lr": False,
            "gamma": 1.0,
            "gae_lambda": 0.95,
            "clip_eps": 0.2,
            "entropy_coef": 0.01,
            "log_std_min": -2.5,
            "log_std_max": -0.5,
            "lagrange_lr": 0.05,
            "cost_budget": 0.0,
            "voltage_cost_scale": 1.0,
            "fixed_cost_penalty_coef": 1.0,
            "fused_rollout_kernel": False,
            "macpo_max_kl": 0.01,
            "macpo_cg_iterations": 10,
            "macpo_damping": 1e-2,
            "include_previous_action": False,
            "include_transaction_message": False,
            "two_stage_intent": False,
            "intent_dim": 3,
            "intent_broadcast_mode": "full",
            "communication_scope": None,
            "intent_residual_limit": 0.25,
            "intent_residual_coef": 0.0,
            "h2_supply_intent_message_enable": False,
            "curriculum_d_start": None,
            "curriculum_d_target": None,
            "curriculum_updates": 0,
            "curriculum_log_std_start": None,
            "curriculum_log_std_end": None,
            "env_parallel_backend": "serial",
            "env_overrides": {},
        }
        self.config.update(dict(config))
        self.voltage_cost_scale = float(self.config["voltage_cost_scale"])
        if not np.isfinite(self.voltage_cost_scale) or self.voltage_cost_scale <= 0.0:
            raise ValueError("voltage_cost_scale must be finite and positive")
        self.env = MicrogridVecEnv(
            num_envs=int(self.config["num_envs"]),
            auto_reset=True,
            config_overrides=self.config["env_overrides"],
            parallel_backend=str(self.config["env_parallel_backend"]),
        )
        self.num_envs = self.env.num_envs
        self.num_agents = self.env.num_agents
        self.base_obs_dim = self.env.obs_dim
        self.action_dim = self.env.action_dim
        self.include_previous_action = bool(self.config["include_previous_action"])
        self.include_transaction_message = bool(self.config["include_transaction_message"])
        self.two_stage_intent = bool(self.config["two_stage_intent"])
        self.intent_dim = int(self.config["intent_dim"])
        configured_scope = self.config.get("communication_scope")
        if configured_scope is None:
            configured_scope = self.config["intent_broadcast_mode"]
        self.communication_scope = str(configured_scope)
        if self.communication_scope == "other_zero":
            self.communication_scope = "self_only"
        if self.communication_scope not in {"full", "self_only", "all_zero"}:
            raise ValueError(
                "communication_scope must be 'full', 'self_only', or 'all_zero'"
            )
        self.supply_message_dim = 4 if bool(
            self.config["h2_supply_intent_message_enable"]
        ) else 0
        self.transaction_message_dim = 2 * self.num_agents if self.include_transaction_message else 0
        self.obs_dim = (
            self.base_obs_dim
            + (self.action_dim if self.include_previous_action else 0)
            + self.transaction_message_dim
        )
        self.hidden_size = int(self.config["hidden_size"])
        self._rng = jax.random.PRNGKey(int(self.config["seed"]))
        self._previous_actions = np.zeros(
            (self.num_envs, self.num_agents, self.action_dim), dtype=np.float32
        )
        self._transaction_messages = np.zeros(
            (self.num_envs, self.transaction_message_dim), dtype=np.float32
        )
        self._obs = self._reshape_local_obs(self.env.reset(seed=int(self.config["seed"]))[0])
        self._done = np.ones(self.num_envs, dtype=bool)
        self.actor_hidden = jnp.zeros(
            (self.num_envs, self.num_agents, self.hidden_size), dtype=jnp.float32
        )
        self.reward_critic_hidden = jnp.zeros(
            (self.num_envs, self.hidden_size), dtype=jnp.float32
        )
        self.cost_critic_hidden = jnp.zeros_like(self.reward_critic_hidden)

        self.actor = IndependentGRUActors(
            num_agents=self.num_agents,
            action_dim=self.action_dim,
            hidden_size=self.hidden_size,
            two_stage_intent=self.two_stage_intent,
            intent_dim=self.intent_dim,
            intent_broadcast_mode=self.communication_scope,
            intent_residual_limit=float(self.config["intent_residual_limit"]),
            supply_message_dim=self.supply_message_dim,
        )
        self.reward_critic = CentralGRUCritic(hidden_size=self.hidden_size)
        self.cost_critic = CentralGRUCritic(hidden_size=self.hidden_size)
        self._rng, actor_key, reward_key, cost_key = jax.random.split(self._rng, 4)
        local_zeros = jnp.zeros(
            (self.num_envs, self.num_agents, self.obs_dim), dtype=jnp.float32
        )
        global_zeros = self._global_obs(local_zeros)
        actor_params = self.actor.init(actor_key, local_zeros, self.actor_hidden)
        reward_params = self.reward_critic.init(
            reward_key, global_zeros, self.reward_critic_hidden
        )
        cost_params = self.cost_critic.init(cost_key, global_zeros, self.cost_critic_hidden)
        if self.config["anneal_lr"]:
            self._learning_rate_schedule = optax.linear_schedule(
                float(self.config["lr"]),
                0.0,
                transition_steps=max(1, int(self.config["total_updates"])),
            )
        else:
            self._learning_rate_schedule = lambda _: jnp.asarray(float(self.config["lr"]))
        optimizer = optax.chain(
            optax.clip_by_global_norm(5.0), optax.adam(self._learning_rate_schedule, eps=1e-5)
        )
        # Gradient steps the critics take per update against the same (fixed) GAE
        # targets.  This was hard-wired to 1, far below the 4-15 epochs standard
        # MAPPO implementations use, and the 2026-07-30 diagnosis showed what that
        # costs: on the successful runs the cost critic out-of-sample RMSE equals
        # the cost value itself (retry1 4.46 vs 4.02; N1 1.63 vs 1.65), so the cost
        # advantages are noise-dominated and the CPO cost surrogate carries no
        # usable magnitude (rho = actual/predicted cost drop ~ 0 at every step
        # size, in every run including the ones that worked).
        #
        # optax schedules advance once per apply_gradients call, so looping the
        # critic step would decay the critic LR critic_epochs times too fast; the
        # critics get their own schedule stretched by the same factor so the
        # LR-vs-update-index curve is unchanged.  Default 1 reproduces every
        # historical run bit-for-bit.
        self.critic_epochs = int(self.config.get("critic_epochs", 1))
        if self.critic_epochs < 1:
            raise ValueError("critic_epochs must be >= 1")
        if self.config["anneal_lr"]:
            critic_schedule = optax.linear_schedule(
                float(self.config["lr"]),
                0.0,
                transition_steps=max(
                    1, int(self.config["total_updates"]) * self.critic_epochs
                ),
            )
        else:
            critic_schedule = self._learning_rate_schedule
        critic_optimizer = optax.chain(
            optax.clip_by_global_norm(5.0), optax.adam(critic_schedule, eps=1e-5)
        )
        self.actor_state = TrainState.create(
            apply_fn=self.actor.apply, params=actor_params, tx=optimizer
        )
        self.reward_critic_state = TrainState.create(
            apply_fn=self.reward_critic.apply, params=reward_params, tx=critic_optimizer
        )
        self.cost_critic_state = TrainState.create(
            apply_fn=self.cost_critic.apply, params=cost_params, tx=critic_optimizer
        )
        self.lagrange_multiplier = 0.0
        self.macpo = SharedSystemMACPOUpdater(
            max_kl=float(self.config["macpo_max_kl"]),
            cg_iterations=int(self.config["macpo_cg_iterations"]),
            damping=float(self.config["macpo_damping"]),
        )
        self._build_training_kernels()

    def _reshape_local_obs(
        self,
        flat_obs: np.ndarray,
        *,
        previous_actions: np.ndarray | None = None,
        transaction_messages: np.ndarray | None = None,
        communication_scope: str | None = None,
    ) -> jnp.ndarray:
        """Attach optional local action history and public confirmed-trade message."""
        env_count = int(np.asarray(flat_obs).shape[0] // self.num_agents)
        base_obs = jnp.asarray(flat_obs, dtype=jnp.float32).reshape(
            env_count, self.num_agents, self.base_obs_dim
        )
        features = [base_obs]
        if self.include_previous_action:
            actions = self._previous_actions if previous_actions is None else previous_actions
            features.append(jnp.asarray(actions, dtype=jnp.float32))
        if self.include_transaction_message:
            scope = self.communication_scope if communication_scope is None else str(
                communication_scope
            )
            if scope == "other_zero":
                scope = "self_only"
            messages = (
                self._transaction_messages
                if transaction_messages is None
                else transaction_messages
            )
            broadcast = np.broadcast_to(
                np.asarray(messages, dtype=np.float32)[:, None, :],
                (env_count, self.num_agents, self.transaction_message_dim),
            ).copy()
            if scope == "self_only":
                for observer in range(self.num_agents):
                    keep = np.zeros(self.transaction_message_dim, dtype=bool)
                    keep[observer] = True
                    keep[self.num_agents + observer] = True
                    broadcast[:, observer, ~keep] = 0.0
            elif scope == "all_zero":
                broadcast.fill(0.0)
            elif scope != "full":
                raise ValueError(
                    "communication_scope must be 'full', 'self_only', or 'all_zero'"
                )
            features.append(jnp.asarray(broadcast, dtype=jnp.float32))
        return jnp.concatenate(features, axis=-1)

    def _transaction_message_from_infos(self, infos: list[dict[str, Any]]) -> np.ndarray:
        """Broadcast confirmed planned orders and in-transit H2 for all agents.

        This is a public market message available only after step clearing; it
        deliberately excludes private future profiles and critic-only signals.
        """
        if not self.include_transaction_message:
            return np.zeros((self.num_envs, 0), dtype=np.float32)
        raw_env = self.env.envs[0].env
        scale = np.maximum(np.asarray(raw_env.pending_scale, dtype=np.float32), 1e-6)
        messages = []
        env_count = len(infos) // self.num_agents
        for env_index in range(env_count):
            info = infos[env_index * self.num_agents]
            planned = np.asarray(
                info.get("h2_planned_external_order_energy", np.zeros(self.num_agents)),
                dtype=np.float32,
            )
            pending = np.asarray(
                info.get("pending_h2_energy_agent", np.zeros(self.num_agents)),
                dtype=np.float32,
            )
            messages.append(np.concatenate((planned / scale, pending / scale)))
        return np.clip(np.asarray(messages, dtype=np.float32), -1.0, 1.0)

    @staticmethod
    def _global_obs(local_obs: jnp.ndarray) -> jnp.ndarray:
        return local_obs.reshape(local_obs.shape[0], -1)

    def learning_rate(self, update: int) -> float:
        """Expose the shared actor/critic annealing schedule for diagnostics."""
        return float(self._learning_rate_schedule(int(update)))

    def _curriculum_fraction(self, update: int) -> float:
        """Return the inclusive 1..N interpolation fraction for v4 courses."""
        updates = int(self.config["curriculum_updates"])
        if updates <= 1:
            return 1.0
        return float(np.clip((int(update) - 1) / (updates - 1), 0.0, 1.0))

    def current_cost_budget(self, update: int) -> float:
        """Return the system-cost budget active for this optimisation update."""
        start = self.config["curriculum_d_start"]
        target = self.config["curriculum_d_target"]
        if start is None or target is None or int(self.config["curriculum_updates"]) <= 0:
            return float(self.config["cost_budget"])
        fraction = self._curriculum_fraction(update)
        return float(start) + fraction * (float(target) - float(start))

    def current_log_std_max(self, update: int) -> float:
        """Return the rollout/replay log-standard-deviation cap for this update."""
        start = self.config["curriculum_log_std_start"]
        end = self.config["curriculum_log_std_end"]
        if start is None or end is None or int(self.config["curriculum_updates"]) <= 0:
            return float(self.config["log_std_max"])
        fraction = self._curriculum_fraction(update)
        return float(start) + fraction * (float(end) - float(start))

    def _build_training_kernels(self) -> None:
        """Create fixed-shape JAX kernels; environment interaction stays in Python/PyPower."""
        def rollout_policy_step(
            actor_params,
            reward_critic_params,
            cost_critic_params,
            local_obs,
            actor_hidden,
            reward_hidden,
            cost_hidden,
            done_before,
            rng,
            log_std_max,
        ):
            actor_hidden = reset_actor_hidden(actor_hidden, done_before)
            reward_hidden = reset_global_hidden(reward_hidden, done_before)
            cost_hidden = reset_global_hidden(cost_hidden, done_before)
            global_obs = local_obs.reshape(local_obs.shape[0], -1)
            means, log_stds, next_actor_hidden, intents = self.actor.apply(
                actor_params,
                local_obs,
                actor_hidden,
                return_intents=True,
            )
            reward_value, next_reward_hidden = self.reward_critic.apply(
                reward_critic_params, global_obs, reward_hidden
            )
            cost_value, next_cost_hidden = self.cost_critic.apply(
                cost_critic_params, global_obs, cost_hidden
            )
            next_rng, sample_key = jax.random.split(rng)
            action, log_prob = sample_squashed_gaussian(
                means,
                log_stds,
                sample_key,
                log_std_min=float(self.config["log_std_min"]),
                log_std_max=log_std_max,
            )
            return (
                next_rng,
                global_obs,
                action,
                log_prob,
                reward_value,
                cost_value,
                next_actor_hidden,
                next_reward_hidden,
                next_cost_hidden,
                intents,
            )

        self._fused_rollout_policy_step = jax.jit(rollout_policy_step)

        def actor_unroll(params, local_obs, dones_before, initial_hidden):
            def step(hidden, inputs):
                obs, done = inputs
                hidden = reset_actor_hidden(hidden, done)
                mean, log_std, next_hidden = self.actor.apply(params, obs, hidden)
                return next_hidden, (mean, log_std)

            _, (means, log_stds) = jax.lax.scan(
                step, initial_hidden, (local_obs, dones_before)
            )
            return means, log_stds

        self._actor_unroll = jax.jit(actor_unroll)

        def ppo_actor_loss_and_grad(
            params, local_obs, dones_before, initial_hidden, actions, old_log_probs, advantages,
            intents, log_std_max,
        ):
            def loss_fn(candidate_params):
                means, log_stds = actor_unroll(
                    candidate_params, local_obs, dones_before, initial_hidden
                )
                new_log_probs = squashed_log_prob(
                    means,
                    log_stds,
                    actions,
                    log_std_min=float(self.config["log_std_min"]),
                    log_std_max=log_std_max,
                )
                ratio = jnp.exp(new_log_probs - old_log_probs)
                advantage = advantages[:, :, None]
                surrogate = jnp.minimum(
                    ratio * advantage,
                    jnp.clip(
                        ratio,
                        1.0 - float(self.config["clip_eps"]),
                        1.0 + float(self.config["clip_eps"]),
                    )
                    * advantage,
                )
                entropy = jnp.mean(-new_log_probs)
                residual_penalty = 0.0
                if self.two_stage_intent:
                    residual = (
                        jnp.tanh(means[..., (0, 1, 5)]) - intents
                    ) / float(self.config["intent_residual_limit"])
                    residual_penalty = float(self.config["intent_residual_coef"]) * jnp.mean(
                        jnp.square(residual)
                    )
                return (
                    -jnp.mean(surrogate)
                    - float(self.config["entropy_coef"]) * entropy
                    + residual_penalty
                )

            return jax.value_and_grad(loss_fn)(params)

        self._ppo_actor_loss_and_grad = jax.jit(ppo_actor_loss_and_grad)

        def make_critic_loss_and_grad(critic: CentralGRUCritic):
            def loss_and_grad(params, global_obs, dones_before, initial_hidden, targets):
                def loss_fn(candidate_params):
                    def step(hidden, inputs):
                        obs, done = inputs
                        hidden = reset_global_hidden(hidden, done)
                        value, next_hidden = critic.apply(candidate_params, obs, hidden)
                        return next_hidden, value

                    _, predicted = jax.lax.scan(
                        step, initial_hidden, (global_obs, dones_before)
                    )
                    return jnp.mean(jnp.square(predicted - targets))

                return jax.value_and_grad(loss_fn)(params)

            return jax.jit(loss_and_grad)

        self._reward_critic_loss_and_grad = make_critic_loss_and_grad(self.reward_critic)
        self._cost_critic_loss_and_grad = make_critic_loss_and_grad(self.cost_critic)
        self._gae = jax.jit(
            lambda rewards, values, last_values, dones: compute_gae(
                rewards,
                values,
                last_values,
                dones,
                gamma=float(self.config["gamma"]),
                gae_lambda=float(self.config["gae_lambda"]),
            )
        )

    def _legacy_rollout_policy_step(
        self, done_before: jnp.ndarray, rng: jnp.ndarray, log_std_max: float
    ):
        actor_hidden = reset_actor_hidden(self.actor_hidden, done_before)
        reward_hidden = reset_global_hidden(self.reward_critic_hidden, done_before)
        cost_hidden = reset_global_hidden(self.cost_critic_hidden, done_before)
        global_obs = self._global_obs(self._obs)
        means, log_stds, next_actor_hidden, intents = self.actor.apply(
            self.actor_state.params,
            self._obs,
            actor_hidden,
            return_intents=True,
        )
        reward_value, next_reward_hidden = self.reward_critic.apply(
            self.reward_critic_state.params, global_obs, reward_hidden
        )
        cost_value, next_cost_hidden = self.cost_critic.apply(
            self.cost_critic_state.params, global_obs, cost_hidden
        )
        next_rng, sample_key = jax.random.split(rng)
        action, log_prob = sample_squashed_gaussian(
            means,
            log_stds,
            sample_key,
            log_std_min=float(self.config["log_std_min"]),
            log_std_max=log_std_max,
        )
        return (
            next_rng,
            global_obs,
            action,
            log_prob,
            reward_value,
            cost_value,
            next_actor_hidden,
            next_reward_hidden,
            next_cost_hidden,
            intents,
        )

    def _fused_rollout_step(
        self, done_before: jnp.ndarray, rng: jnp.ndarray, log_std_max: float
    ):
        return self._fused_rollout_policy_step(
            self.actor_state.params,
            self.reward_critic_state.params,
            self.cost_critic_state.params,
            self._obs,
            self.actor_hidden,
            self.reward_critic_hidden,
            self.cost_critic_hidden,
            done_before,
            rng,
            jnp.asarray(log_std_max, dtype=jnp.float32),
        )

    def rollout_kernel_parity(self, *, update_index: int = 1) -> dict[str, float | int]:
        """Compare fused and legacy rollout inference without mutating trainer state."""
        done_before = jnp.asarray(self._done)
        log_std_max = self.current_log_std_max(update_index)
        legacy = self._legacy_rollout_policy_step(done_before, self._rng, log_std_max)
        fused = self._fused_rollout_step(done_before, self._rng, log_std_max)
        field_names = (
            "rng", "global_obs", "actions", "log_probs", "reward_values",
            "cost_values", "actor_hidden", "reward_hidden", "cost_hidden",
            "intents",
        )
        differences = {
            name: (
                float(np.max(np.abs(np.asarray(first) - np.asarray(second))))
                if np.asarray(first).size
                else 0.0
            )
            for name, first, second in zip(field_names, legacy, fused)
        }
        return {
            "fields_compared": len(differences),
            "field_abs_differences": differences,
            "max_abs_difference": max(differences.values(), default=0.0),
        }

    def warmup_training_kernels(self) -> dict[str, bool]:
        """Compile fixed rollout-shape training kernels without stepping the environment."""
        steps = int(self.config["num_steps"])
        local_obs = jnp.zeros((steps, self.num_envs, self.num_agents, self.obs_dim))
        global_obs = jnp.zeros((steps, self.num_envs, self.num_agents * self.obs_dim))
        dones = jnp.zeros((steps, self.num_envs), dtype=bool)
        actions = jnp.zeros((steps, self.num_envs, self.num_agents, self.action_dim))
        log_probs = jnp.zeros((steps, self.num_envs, self.num_agents))
        advantages = jnp.zeros((steps, self.num_envs))
        _, actor_grads = self._ppo_actor_loss_and_grad(
            self.actor_state.params,
            local_obs,
            dones,
            self.actor_hidden,
            actions,
            log_probs,
            advantages,
            jnp.zeros((steps, self.num_envs, self.num_agents, self.intent_dim)),
            float(self.config["log_std_max"]),
        )
        _, reward_grads = self._reward_critic_loss_and_grad(
            self.reward_critic_state.params,
            global_obs,
            dones,
            self.reward_critic_hidden,
            advantages,
        )
        _, cost_grads = self._cost_critic_loss_and_grad(
            self.cost_critic_state.params,
            global_obs,
            dones,
            self.cost_critic_hidden,
            advantages,
        )
        gae, _ = self._gae(advantages, advantages, jnp.zeros(self.num_envs), dones)
        for tree in (actor_grads, reward_grads, cost_grads):
            for leaf in jax.tree_util.tree_leaves(tree):
                leaf.block_until_ready()
        gae.block_until_ready()
        return {"actor": True, "reward_critic": True, "cost_critic": True, "gae": True}

    def collect_rollout(self, *, update_index: int = 1) -> SafeRollout:
        """Collect one rollout with independent local actor state and global critics."""
        local_obses, global_obses, dones_before, dones = [], [], [], []
        actions, log_probs, rewards, costs, raw_costs = [], [], [], [], []
        reward_values, cost_values, intents = [], [], []
        initial_actor_hidden = self.actor_hidden
        initial_reward_hidden = self.reward_critic_hidden
        initial_cost_hidden = self.cost_critic_hidden

        log_std_max = self.current_log_std_max(update_index)
        for _ in range(int(self.config["num_steps"])):
            done_before = jnp.asarray(self._done)
            policy_step = (
                self._fused_rollout_step
                if bool(self.config["fused_rollout_kernel"])
                else self._legacy_rollout_policy_step
            )
            (
                self._rng,
                global_obs,
                action,
                log_prob,
                reward_value,
                cost_value,
                next_actor_hidden,
                next_reward_hidden,
                next_cost_hidden,
                current_intents,
            ) = policy_step(done_before, self._rng, log_std_max)
            next_obs, reward, termination, truncation, info = self.env.step(
                np.asarray(action, dtype=np.float32).reshape(-1, self.action_dim)
            )
            done = np.logical_or(termination, truncation).reshape(
                self.num_envs, self.num_agents
            ).any(axis=1)
            shared_reward = np.asarray(reward, dtype=np.float32).reshape(
                self.num_envs, self.num_agents
            ).mean(axis=1)
            shared_raw_cost = np.asarray(
                [
                    float(info[env_index * self.num_agents].get("voltage_cost", 0.0))
                    for env_index in range(self.num_envs)
                ],
                dtype=np.float32,
            )
            shared_cost = shared_raw_cost / self.voltage_cost_scale
            local_obses.append(self._obs)
            global_obses.append(global_obs)
            dones_before.append(done_before)
            dones.append(jnp.asarray(done))
            actions.append(action)
            log_probs.append(log_prob)
            rewards.append(jnp.asarray(shared_reward))
            costs.append(jnp.asarray(shared_cost))
            raw_costs.append(jnp.asarray(shared_raw_cost))
            reward_values.append(reward_value)
            cost_values.append(cost_value)
            intents.append(current_intents)
            self._done = done
            self._previous_actions = np.where(
                done[:, None, None],
                np.zeros_like(np.asarray(action, dtype=np.float32)),
                np.asarray(action, dtype=np.float32),
            )
            self._transaction_messages = np.where(
                done[:, None],
                np.zeros_like(self._transaction_message_from_infos(info)),
                self._transaction_message_from_infos(info),
            )
            self._obs = self._reshape_local_obs(next_obs)
            # Clearing immediately makes end-of-episode state observable and
            # prevents accidental memory carryover before the next action.
            self.actor_hidden = reset_actor_hidden(next_actor_hidden, jnp.asarray(done))
            self.reward_critic_hidden = reset_global_hidden(
                next_reward_hidden, jnp.asarray(done)
            )
            self.cost_critic_hidden = reset_global_hidden(next_cost_hidden, jnp.asarray(done))

        last_global_obs = self._global_obs(self._obs)
        reward_bootstrap_hidden = reset_global_hidden(
            self.reward_critic_hidden, jnp.asarray(self._done)
        )
        cost_bootstrap_hidden = reset_global_hidden(
            self.cost_critic_hidden, jnp.asarray(self._done)
        )
        last_reward_value, _ = self.reward_critic.apply(
            self.reward_critic_state.params, last_global_obs, reward_bootstrap_hidden
        )
        last_cost_value, _ = self.cost_critic.apply(
            self.cost_critic_state.params, last_global_obs, cost_bootstrap_hidden
        )
        return SafeRollout(
            local_obs=jnp.stack(local_obses),
            global_obs=jnp.stack(global_obses),
            dones_before=jnp.stack(dones_before),
            dones=jnp.stack(dones),
            actions=jnp.stack(actions),
            log_probs=jnp.stack(log_probs),
            rewards=jnp.stack(rewards),
            costs=jnp.stack(costs),
            raw_costs=jnp.stack(raw_costs),
            reward_values=jnp.stack(reward_values),
            cost_values=jnp.stack(cost_values),
            intents=jnp.stack(intents),
            last_reward_value=last_reward_value,
            last_cost_value=last_cost_value,
            initial_actor_hidden=initial_actor_hidden,
            initial_reward_critic_hidden=initial_reward_hidden,
            initial_cost_critic_hidden=initial_cost_hidden,
        )

    def _unroll_actor(self, params: Any, rollout: SafeRollout):
        return self._actor_unroll(
            params,
            rollout.local_obs,
            rollout.dones_before,
            rollout.initial_actor_hidden,
        )

    def _unroll_critic(self, critic: CentralGRUCritic, params: Any, initial_hidden, rollout):
        def step(hidden, inputs):
            obs, done = inputs
            hidden = reset_global_hidden(hidden, done)
            value, next_hidden = critic.apply(params, obs, hidden)
            return next_hidden, value

        _, values = jax.lax.scan(
            step, initial_hidden, (rollout.global_obs, rollout.dones_before)
        )
        return values

    def update(
        self,
        rollout: SafeRollout,
        *,
        algorithm: str = "mappo",
        update_index: int = 1,
    ) -> dict[str, float | str | bool]:
        """Update PPO critics plus MAPPO, Lagrangian, or shared-system MACPO actor."""
        if algorithm not in {"mappo", "mappo_penalty", "lagrangian", "macpo"}:
            raise ValueError(
                "algorithm must be 'mappo', 'mappo_penalty', 'lagrangian', or 'macpo'"
            )
        reward_advantages, reward_returns = self._gae(
            rollout.rewards,
            rollout.reward_values,
            rollout.last_reward_value,
            rollout.dones,
        )
        cost_advantages, cost_returns = self._gae(
            rollout.costs,
            rollout.cost_values,
            rollout.last_cost_value,
            rollout.dones,
        )
        daily_cost = jnp.mean(jnp.sum(rollout.costs, axis=0))
        daily_raw_cost = jnp.mean(jnp.sum(rollout.raw_costs, axis=0))
        daily_reward_normalized = jnp.mean(jnp.sum(rollout.rewards, axis=0))
        reward_scale_yuan = float(
            self.config.get("env_overrides", {}).get("reward_scale", 1.0)
        )
        daily_reward_raw_yuan = daily_reward_normalized * reward_scale_yuan
        cost_budget = self.current_cost_budget(update_index)
        log_std_max = self.current_log_std_max(update_index)

        def actor_terms(params):
            means, log_stds = self._unroll_actor(params, rollout)
            log_stds = jnp.clip(
                log_stds,
                float(self.config["log_std_min"]),
                log_std_max,
            )
            new_log_probs = squashed_log_prob(
                means,
                log_stds,
                rollout.actions,
                log_std_min=float(self.config["log_std_min"]),
                log_std_max=log_std_max,
            )
            return new_log_probs, means, log_stds

        if algorithm == "macpo":
            old_means, old_log_stds = self._unroll_actor(self.actor_state.params, rollout)
            old_log_stds = jnp.clip(
                old_log_stds,
                float(self.config["log_std_min"]),
                log_std_max,
            )

            def episode_surrogate(ratio, advantages):
                # First-order estimate of the change in the *episode-sum* return:
                # sum over the 24 steps and the 4 agents (the joint ratio is the
                # product of per-agent ratios, so to first order the per-agent
                # (ratio - 1) terms add), mean over parallel envs.  A plain mean
                # over (T, E, N) under-states the effect of a policy change on
                # the daily cost by T*N (~96x), which starves the CPO updater:
                # the feasibility test then always fails and the line search
                # cannot see the cost moving, so training degenerates into
                # always-accepted pure-recovery steps.
                return jnp.mean(
                    jnp.sum((ratio - 1.0) * advantages[:, :, None], axis=(0, 2))
                )

            def reward_objective(params):
                new_log_probs, means, _ = actor_terms(params)
                ratio = jnp.exp(new_log_probs - rollout.log_probs)
                objective = episode_surrogate(ratio, reward_advantages)
                if self.two_stage_intent:
                    residual = (
                        jnp.tanh(means[..., (0, 1, 5)]) - rollout.intents
                    ) / float(self.config["intent_residual_limit"])
                    objective -= float(self.config["intent_residual_coef"]) * jnp.mean(
                        jnp.square(residual)
                    )
                return objective

            def cost_objective(params):
                new_log_probs, _, _ = actor_terms(params)
                ratio = jnp.exp(new_log_probs - rollout.log_probs)
                # Same episode-sum units as ``daily_cost`` and ``budget``.
                return daily_cost + episode_surrogate(ratio, cost_advantages)

            def kl_divergence(params):
                _, means, log_stds = actor_terms(params)
                old_variance = jnp.exp(2.0 * old_log_stds)
                new_variance = jnp.exp(2.0 * log_stds)
                normal_kl = 0.5 * (
                    2.0 * (log_stds - old_log_stds)
                    + (old_variance + jnp.square(old_means - means)) / new_variance
                    - 1.0
                )
                return jnp.mean(jnp.sum(normal_kl, axis=-1))

            actor_params, macpo_metrics = self.macpo.update(
                self.actor_state.params,
                reward_objective=reward_objective,
                cost_objective=cost_objective,
                kl_divergence=kl_divergence,
                budget=cost_budget,
            )
            self.actor_state = self.actor_state.replace(params=actor_params)
            actor_loss = -float(reward_objective(actor_params))
        else:
            if algorithm == "lagrangian":
                combined_advantages = reward_advantages - self.lagrange_multiplier * cost_advantages
            elif algorithm == "mappo_penalty":
                combined_advantages = (
                    reward_advantages
                    - float(self.config["fixed_cost_penalty_coef"]) * cost_advantages
                )
            else:
                combined_advantages = reward_advantages
            combined_advantages = (combined_advantages - combined_advantages.mean()) / (
                combined_advantages.std() + 1e-8
            )

            actor_loss, actor_grads = self._ppo_actor_loss_and_grad(
                self.actor_state.params,
                rollout.local_obs,
                rollout.dones_before,
                rollout.initial_actor_hidden,
                rollout.actions,
                rollout.log_probs,
                combined_advantages,
                rollout.intents,
                log_std_max,
            )
            self.actor_state = self.actor_state.apply_gradients(grads=actor_grads)
            actor_loss = float(actor_loss)
            macpo_metrics = {}

        # ``*_critic_loss`` is deliberately the FIRST pass loss: it is measured on
        # a freshly sampled rollout before any of this update's critic steps, so it
        # is an out-of-sample error and stays comparable with every run recorded
        # before ``critic_epochs`` existed.  ``*_critic_loss_last`` is the in-sample
        # fit after the extra passes; the gap between them is the generalization
        # gap of the critic.
        reward_loss = reward_loss_last = 0.0
        for epoch in range(self.critic_epochs):
            loss, grads = self._reward_critic_loss_and_grad(
                self.reward_critic_state.params,
                rollout.global_obs,
                rollout.dones_before,
                rollout.initial_reward_critic_hidden,
                reward_returns,
            )
            self.reward_critic_state = self.reward_critic_state.apply_gradients(grads=grads)
            reward_loss_last = float(loss)
            if epoch == 0:
                reward_loss = reward_loss_last
        cost_loss = cost_loss_last = 0.0
        for epoch in range(self.critic_epochs):
            loss, grads = self._cost_critic_loss_and_grad(
                self.cost_critic_state.params,
                rollout.global_obs,
                rollout.dones_before,
                rollout.initial_cost_critic_hidden,
                cost_returns,
            )
            self.cost_critic_state = self.cost_critic_state.apply_gradients(grads=grads)
            cost_loss_last = float(loss)
            if epoch == 0:
                cost_loss = cost_loss_last
        if algorithm == "lagrangian":
            self.lagrange_multiplier = update_lagrange_multiplier(
                self.lagrange_multiplier,
                cost_mean=float(daily_cost),
                budget=cost_budget,
                lr=float(self.config["lagrange_lr"]),
            )
        metrics: dict[str, float | str | bool] = {
            "actor_loss": float(actor_loss),
            "reward_critic_loss": float(reward_loss),
            "cost_critic_loss": float(cost_loss),
            "reward_critic_loss_last": float(reward_loss_last),
            "cost_critic_loss_last": float(cost_loss_last),
            "critic_epochs": int(self.critic_epochs),
            "daily_voltage_cost": float(daily_raw_cost),
            "daily_voltage_cost_raw": float(daily_raw_cost),
            "daily_voltage_cost_normalized": float(daily_cost),
            "daily_economic_return_raw_yuan": float(daily_reward_raw_yuan),
            "daily_economic_return_normalized": float(daily_reward_normalized),
            "daily_economic_cost_raw_yuan": float(-daily_reward_raw_yuan),
            "economic_reward_scale_yuan": float(reward_scale_yuan),
            "lagrange_multiplier": float(self.lagrange_multiplier),
            "cost_budget": float(cost_budget),
            "cost_budget_raw": float(cost_budget) * self.voltage_cost_scale,
            "cost_budget_normalized": float(cost_budget),
            "algorithm_cost_mode": (
                "fixed_penalty" if algorithm == "mappo_penalty" else "separate_cost"
            ),
            "fixed_cost_penalty_coef": (
                float(self.config["fixed_cost_penalty_coef"])
                if algorithm == "mappo_penalty"
                else 0.0
            ),
            "log_std_max": float(log_std_max),
        }
        metrics.update(macpo_metrics)
        return metrics

    def deterministic_rollout(
        self,
        *,
        seed: int = 30,
        intent_broadcast_mode: str | None = None,
        history_off: bool = False,
        gru_hidden_off: bool = False,
        previous_action_off: bool = False,
        cross_agent_off: bool = False,
        eta_delay_hours: int = 0,
    ) -> dict[str, Any]:
        """Evaluate the current GRU policies for one fixed day without sampling."""
        if eta_delay_hours < 0:
            raise ValueError("eta_delay_hours must be non-negative")
        evaluation_overrides = dict(self.config["env_overrides"])
        if eta_delay_hours:
            base_env = self.env.envs[0].env
            # Preserve the checkpoint's fixed observation shape.  Deliveries
            # beyond this existing window remain physically present but are
            # intentionally not visible to the actor in the ETA-shift test.
            evaluation_overrides["h2_pending_obs_horizon"] = int(
                base_env.h2_pending_obs_horizon
            )
            evaluation_overrides["h2_pending_obs_auto_expand_to_eta"] = False
            evaluation_overrides["h2_traffic_eta_min"] = (
                int(base_env.h2_traffic_min_eta) + int(eta_delay_hours)
            )
            evaluation_overrides["h2_traffic_eta_max"] = (
                int(base_env.h2_traffic_max_eta) + int(eta_delay_hours)
            )
        evaluation_env = MicrogridVecEnv(
            num_envs=1,
            auto_reset=False,
            config_overrides=evaluation_overrides,
        )
        try:
            evaluation_scope = "self_only" if cross_agent_off else self.communication_scope
            actor_broadcast_mode = (
                "self_only"
                if cross_agent_off
                else intent_broadcast_mode
            )
            flat_obs, _ = evaluation_env.reset(seed=int(seed))
            previous_actions = np.zeros((1, self.num_agents, self.action_dim), dtype=np.float32)
            transaction_messages = np.zeros((1, self.transaction_message_dim), dtype=np.float32)
            local_obs = self._reshape_local_obs(
                flat_obs,
                previous_actions=previous_actions,
                transaction_messages=transaction_messages,
                communication_scope=evaluation_scope,
            )
            actor_hidden = jnp.zeros(
                (1, self.num_agents, self.hidden_size), dtype=jnp.float32
            )
            done = jnp.ones(1, dtype=bool)
            records = []
            episode_length = int(evaluation_env.envs[0].env.T)
            for step in range(episode_length):
                actor_hidden = reset_actor_hidden(actor_hidden, done)
                if history_off or gru_hidden_off:
                    actor_hidden = jnp.zeros_like(actor_hidden)
                means, _, next_actor_hidden, current_intents = self.actor.apply(
                    self.actor_state.params,
                    local_obs,
                    actor_hidden,
                    return_intents=True,
                    intent_broadcast_mode=actor_broadcast_mode,
                )
                action = np.asarray(jnp.tanh(means[0]), dtype=np.float32)
                next_obs, _, termination, truncation, infos = evaluation_env.step(
                    action.reshape(-1, self.action_dim)
                )
                info = infos[0]
                records.append(
                    {
                        "step": step,
                        "actions": action.tolist(),
                        "intents": np.asarray(current_intents[0]).tolist(),
                        "economic_cost": float(info["economic_cost"]),
                        "step_total_cost": float(info["total_cost"]),
                        "terminal_settlement_cost": float(info["terminal_settlement_cost"]),
                        "terminal_battery_asset_value": float(
                            info["terminal_battery_asset_value"]
                        ),
                        "terminal_h2_asset_value": float(info["terminal_h2_asset_value"]),
                        "terminal_undelivered_h2_energy": float(
                            info["terminal_undelivered_h2_energy"]
                        ),
                        "voltage_cost": float(info["voltage_cost"]),
                        "voltage_violation_area": float(info["voltage_violation_area"]),
                        "voltage_min_pu": info["voltage_min_pu"],
                        "voltage_max_pu": info["voltage_max_pu"],
                        "pf_converged": bool(info["pf_converged"]),
                        "pcc_p_kw": list(info["pcc_p_kw"]),
                        "pcc_q_kvar": list(info["pcc_q_kvar"]),
                        "p_el": list(info["p_el"]),
                        "p_bat": list(info["p_bat"]),
                        "soc": list(info["soc"]),
                        "p_ht": list(info["p_ht"]),
                        "h2_emergency_buy_energy": list(info["h2_emergency_buy_energy"]),
                        "h2_planned_external_order_energy": list(
                            info["h2_planned_external_order_energy"]
                        ),
                        "h2_late_order_energy": list(info["h2_late_order_energy"]),
                        "h2_level": list(info["h2_level"]),
                        "e_h2_load": list(info["e_h2_load"]),
                        "pending_h2_energy_total": float(info["pending_h2_energy_total"]),
                        "pending_h2_energy_agent": list(info["pending_h2_energy_agent"]),
                        "external_min_eta": list(
                            info["h2_external_min_eta_normalized"]
                        ),
                    }
                )
                done_np = np.logical_or(termination, truncation).reshape(
                    1, self.num_agents
                ).any(axis=1)
                done = jnp.asarray(done_np)
                actor_hidden = reset_actor_hidden(next_actor_hidden, done)
                if history_off or previous_action_off:
                    previous_actions = np.zeros_like(previous_actions)
                else:
                    previous_actions = np.where(
                        done_np[:, None, None],
                        np.zeros_like(action[None, :, :]),
                        action[None, :, :],
                    )
                if history_off:
                    transaction_messages = np.zeros_like(transaction_messages)
                else:
                    transaction_messages = np.where(
                        done_np[:, None],
                        np.zeros_like(self._transaction_message_from_infos(infos)),
                        self._transaction_message_from_infos(infos),
                    )
                local_obs = self._reshape_local_obs(
                    next_obs,
                    previous_actions=previous_actions,
                    transaction_messages=transaction_messages,
                    communication_scope=evaluation_scope,
                )
                if bool(done_np[0]):
                    break
        finally:
            evaluation_env.close()

        finite_mins = [
            record["voltage_min_pu"]
            for record in records
            if record["voltage_min_pu"] is not None and np.isfinite(record["voltage_min_pu"])
        ]
        finite_maxs = [
            record["voltage_max_pu"]
            for record in records
            if record["voltage_max_pu"] is not None and np.isfinite(record["voltage_max_pu"])
        ]
        intent_stats: dict[str, float | None] = {
            "intent_action_mae": None,
            "intent_action_correlation": None,
            "intent_residual_max": None,
            "intent_residual_utilization_mean": None,
        }
        if self.two_stage_intent and records:
            intent_values = np.asarray([record["intents"] for record in records])
            action_values = np.asarray([record["actions"] for record in records])[..., (0, 1, 5)]
            residual = action_values - intent_values
            limit = float(self.config["intent_residual_limit"])
            flat_intent = intent_values.reshape(-1)
            flat_action = action_values.reshape(-1)
            correlation = None
            if np.std(flat_intent) > 1e-8 and np.std(flat_action) > 1e-8:
                correlation = float(np.corrcoef(flat_intent, flat_action)[0, 1])
            intent_stats = {
                "intent_action_mae": float(np.mean(np.abs(residual))),
                "intent_action_correlation": correlation,
                "intent_residual_max": float(np.max(np.abs(residual))),
                "intent_residual_utilization_mean": float(
                    np.mean(np.abs(residual)) / max(limit, 1e-8)
                ),
            }
        return {
            "summary": {
                "seed": int(seed),
                "steps": len(records),
                "economic_cost": float(sum(record["economic_cost"] for record in records)),
                "total_cost": float(sum(record["step_total_cost"] for record in records)),
                "terminal_settlement_cost": (
                    records[-1]["terminal_settlement_cost"] if records else 0.0
                ),
                "terminal_battery_asset_value": (
                    records[-1]["terminal_battery_asset_value"] if records else 0.0
                ),
                "terminal_h2_asset_value": (
                    records[-1]["terminal_h2_asset_value"] if records else 0.0
                ),
                "terminal_undelivered_h2_energy": (
                    records[-1]["terminal_undelivered_h2_energy"] if records else 0.0
                ),
                "daily_voltage_cost": float(sum(record["voltage_cost"] for record in records)),
                "voltage_violation_area": float(sum(record["voltage_violation_area"] for record in records)),
                "voltage_min_pu": float(min(finite_mins)) if finite_mins else None,
                "voltage_max_pu": float(max(finite_maxs)) if finite_maxs else None,
                "pf_failure_rate": float(np.mean([not record["pf_converged"] for record in records])),
                "safe_step_rate": float(np.mean([record["voltage_cost"] <= 0.0 for record in records])),
                "emergency_h2_buy": float(sum(sum(record["h2_emergency_buy_energy"]) for record in records)),
                "planned_h2_order": float(sum(sum(record["h2_planned_external_order_energy"]) for record in records)),
                "late_h2_order": float(sum(sum(record["h2_late_order_energy"]) for record in records)),
                "pending_h2_energy": records[-1]["pending_h2_energy_total"] if records else 0.0,
                "history_off": bool(history_off),
                "gru_hidden_off": bool(gru_hidden_off or history_off),
                "previous_action_off": bool(previous_action_off or history_off),
                "cross_agent_off": bool(cross_agent_off),
                "eta_delay_hours": int(eta_delay_hours),
                **intent_stats,
            },
            "steps": records,
        }

    def _config_fingerprint(self) -> str:
        """Return a stable identity for settings that affect checkpoint compatibility."""
        encoded = json.dumps(self.config, sort_keys=True, separators=(",", ":"), default=str)
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()

    @staticmethod
    def _metadata_path(checkpoint: Path) -> Path:
        return checkpoint.with_suffix(".json")

    @staticmethod
    def _atomic_write_bytes(path: Path, data: bytes) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.tmp")
        with temporary.open("wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)

    @staticmethod
    def _atomic_write_text(path: Path, text: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)

    def save_checkpoint(self, checkpoint: str | Path, *, update: int, algorithm: str) -> Path:
        """Atomically persist trainable state after a complete vectorized episode."""
        if algorithm not in {"mappo", "mappo_penalty", "lagrangian", "macpo"}:
            raise ValueError(
                "algorithm must be 'mappo', 'mappo_penalty', 'lagrangian', or 'macpo'"
            )
        if not bool(np.all(self._done)):
            raise RuntimeError("checkpoints are only supported at episode boundaries")
        checkpoint = Path(checkpoint)
        payload = {
            "actor_state": self.actor_state,
            "reward_critic_state": self.reward_critic_state,
            "cost_critic_state": self.cost_critic_state,
            "rng": self._rng,
            "lagrange_multiplier": float(self.lagrange_multiplier),
        }
        metadata = {
            "format_version": 1,
            "update": int(update),
            "algorithm": algorithm,
            "config_fingerprint": self._config_fingerprint(),
        }
        self._atomic_write_bytes(checkpoint, serialization.to_bytes(payload))
        self._atomic_write_text(
            self._metadata_path(checkpoint),
            json.dumps(metadata, sort_keys=True) + "\n",
        )
        return checkpoint

    def load_checkpoint(self, checkpoint: str | Path, *, algorithm: str) -> int:
        """Restore trainable state and re-enter at a deterministic episode boundary."""
        checkpoint = Path(checkpoint)
        metadata_path = self._metadata_path(checkpoint)
        if not checkpoint.is_file() or not metadata_path.is_file():
            raise FileNotFoundError(f"checkpoint and metadata must both exist: {checkpoint}")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("format_version") != 1:
            raise ValueError("unsupported checkpoint format")
        if metadata.get("algorithm") != algorithm:
            raise ValueError("checkpoint algorithm does not match requested algorithm")
        if metadata.get("config_fingerprint") != self._config_fingerprint():
            raise ValueError("checkpoint configuration fingerprint does not match this trainer")
        target = {
            "actor_state": self.actor_state,
            "reward_critic_state": self.reward_critic_state,
            "cost_critic_state": self.cost_critic_state,
            "rng": self._rng,
            "lagrange_multiplier": float(self.lagrange_multiplier),
        }
        restored = serialization.from_bytes(target, checkpoint.read_bytes())
        self.actor_state = restored["actor_state"]
        self.reward_critic_state = restored["reward_critic_state"]
        self.cost_critic_state = restored["cost_critic_state"]
        self._rng = restored["rng"]
        self.lagrange_multiplier = float(restored["lagrange_multiplier"])
        self._previous_actions = np.zeros_like(self._previous_actions)
        self._transaction_messages = np.zeros_like(self._transaction_messages)
        self._obs = self._reshape_local_obs(self.env.reset(seed=int(self.config["seed"]))[0])
        self._done = np.ones(self.num_envs, dtype=bool)
        self.actor_hidden = jnp.zeros_like(self.actor_hidden)
        self.reward_critic_hidden = jnp.zeros_like(self.reward_critic_hidden)
        self.cost_critic_hidden = jnp.zeros_like(self.cost_critic_hidden)
        return int(metadata["update"])

    def train(
        self,
        updates: int,
        *,
        algorithm: str = "mappo",
        start_update: int = 0,
        checkpoint_dir: str | Path | None = None,
        checkpoint_interval: int = 25,
        metrics_path: str | Path | None = None,
        validation_interval: int = 0,
        validation_callback: Callable[[int], None] | None = None,
    ) -> list[dict[str, float | str | bool | int]]:
        """Train sequential updates, emitting durable metrics and episode-boundary checkpoints."""
        if updates < 1:
            raise ValueError("updates must be positive")
        if start_update < 0:
            raise ValueError("start_update must be non-negative")
        if checkpoint_dir is not None and checkpoint_interval < 1:
            raise ValueError("checkpoint_interval must be positive")
        if validation_interval < 0:
            raise ValueError("validation_interval must be non-negative")
        if validation_interval and validation_callback is None:
            raise ValueError("validation_callback is required when validation_interval is set")
        checkpoint_root = Path(checkpoint_dir) if checkpoint_dir is not None else None
        metrics_file = Path(metrics_path) if metrics_path is not None else None
        if metrics_file is not None:
            metrics_file.parent.mkdir(parents=True, exist_ok=True)
        rows: list[dict[str, float | str | bool | int]] = []
        for offset in range(updates):
            update_index = start_update + offset + 1
            total_start = time.perf_counter()
            rollout_start = time.perf_counter()
            rollout = self.collect_rollout(update_index=update_index)
            rollout_seconds = time.perf_counter() - rollout_start
            update_start = time.perf_counter()
            row: dict[str, float | str | bool | int] = {
                "update": update_index,
                "algorithm": algorithm,
            }
            row.update(self.update(rollout, algorithm=algorithm, update_index=update_index))
            row["rollout_wall_seconds"] = float(rollout_seconds)
            row["update_wall_seconds"] = float(time.perf_counter() - update_start)
            row["total_wall_seconds"] = float(time.perf_counter() - total_start)
            rows.append(row)
            if metrics_file is not None:
                with metrics_file.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(row, sort_keys=True) + "\n")
                    handle.flush()
                    os.fsync(handle.fileno())
            if checkpoint_root is not None and (
                update_index % checkpoint_interval == 0 or offset == updates - 1
            ):
                self.save_checkpoint(
                    checkpoint_root / f"update_{update_index:06d}.msgpack",
                    update=update_index,
                    algorithm=algorithm,
                )
            if validation_callback is not None and validation_interval and (
                update_index % validation_interval == 0 or offset == updates - 1
            ):
                validation_callback(update_index)
        return rows

    def close(self) -> None:
        self.env.close()
