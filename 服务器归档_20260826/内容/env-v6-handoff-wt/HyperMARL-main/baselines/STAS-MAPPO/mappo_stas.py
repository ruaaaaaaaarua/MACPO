"""STAS-MAPPO training entry for the microgrid environment."""

from __future__ import annotations

import hashlib
import json
import multiprocessing as mp
import os
import random
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import distrax
import hydra
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training import orbax_utils
from flax.training.train_state import TrainState
from omegaconf import OmegaConf
from orbax.checkpoint import checkpointer
from orbax.checkpoint.pytree_checkpoint_handler import PyTreeCheckpointHandler

# Older wandb releases still refer to np.float_, which NumPy 2 removed.
if not hasattr(np, "float_"):
    np.float_ = np.float64  # type: ignore[attr-defined]
if not hasattr(np, "complex_"):
    np.complex_ = np.complex128  # type: ignore[attr-defined]

import wandb
from baselines.MAPPO.mappo import get_update_fn
from baselines.MAPPO.continuous_policy import (
    deterministic_action,
    sample_squashed_gaussian,
)
from baselines.MAPPO.mappo_ff_shared_weights import (
    ActorCritic,
    EpisodeStatistics,
    Transition,
    TransitionInfo,
    apply_microgrid_config_overrides,
    initialize_metrics_storage,
    update_metrics,
)
from baselines.utils.eval import run_eval_jax
from baselines.utils.environment_stream import reset_environment_stream
from baselines.utils.fixed_scenario_eval import (
    DEFAULT_NOISE_SEEDS,
    VALIDATION_DAYS,
    append_evaluation_record,
    build_scenarios,
    evaluate_policy,
)
from baselines.utils.training_checkpoint import (
    load_jax_training_checkpoint,
    save_jax_training_checkpoint,
)
from baselines.utils.utils import log_eval_metrics, log_train_metrics, log_update_progress
from baselines.utils.wrappers import make_env
from stas_mappo.credit import STASCreditAssigner, STASCreditConfig
# Initialize XLA before importing PyTorch-backed STAS modules. Loading PyTorch's
# CUDA libraries first can prevent JAX from resolving cuSOLVER in this image.
_JAX_DEVICES = jax.devices()

import torch

from stas_mappo.conserved_credit import (
    ConservedSTASCreditAssigner,
    UniformCreditAssigner,
)
from stas_mappo.paper_credit import PaperSTASCreditAssigner
from stas_mappo.checkpoint import (
    load_credit_assigner_checkpoint,
    load_credit_assigner_state,
    save_credit_assigner_checkpoint,
)
from stas_mappo.diagnostics import write_rollout_diagnostic
from stas_mappo.shape_utils import (
    env_agent_time_to_flat_time_agent,
    flat_time_agent_to_env_agent_time,
)


forkserver_available = "forkserver" in mp.get_all_start_methods()
mp.set_start_method("forkserver" if forkserver_available else "spawn", force=True)


def seed_stas_runtime(seed: int) -> None:
    """Seed every RNG used by the STAS reward-model path."""
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _save_best_validation_checkpoint(
    root,
    *,
    train_state,
    rng,
    credit_assigner,
    update: int,
    episode: int,
    global_step: int,
    validation_return: float,
) -> dict:
    """Save a complete generation, then atomically publish its metadata."""
    root = Path(root)
    generation = root / f"episode_{int(episode):05d}"
    jax_path = generation / "training_state.msgpack"
    stas_path = generation / "stas_credit.pt"
    save_jax_training_checkpoint(
        jax_path,
        train_state=train_state,
        rng=rng,
        update=update,
        episode=episode,
        global_step=global_step,
    )
    save_credit_assigner_checkpoint(
        stas_path,
        credit_assigner,
        update=update,
        episode=episode,
        global_step=global_step,
    )
    metadata = {
        "episode": int(episode),
        "update": int(update),
        "global_step": int(global_step),
        "validation_return": float(validation_return),
        "jax_checkpoint": str(jax_path),
        "stas_checkpoint": str(stas_path),
        "sha256": {
            "jax_checkpoint": _sha256(jax_path),
            "stas_checkpoint": _sha256(stas_path),
        },
    }
    root.mkdir(parents=True, exist_ok=True)
    destination = root / "best_validation.json"
    temporary = destination.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(destination)
    return metadata


def _existing_best_validation_return(root) -> float:
    if not root:
        return float("-inf")
    metadata = Path(root) / "best_validation.json"
    if not metadata.is_file():
        return float("-inf")
    return float(json.loads(metadata.read_text(encoding="utf-8"))["validation_return"])


def _make_credit_assigner(config, raw_obs_dim: int, action_dim: int, num_agents: int):
    stas_cfg = config.get("STAS", {})
    mode = str(stas_cfg.get("MODE", "legacy")).lower()
    device = stas_cfg.get("DEVICE", "cuda" if stas_cfg.get("CUDA", False) else "cpu")
    credit_config = STASCreditConfig(
        obs_dim=raw_obs_dim,
        action_dim=action_dim,
        n_agents=num_agents,
        seq_length=config["NUM_STEPS"],
        gamma=config["GAMMA"],
        mix_coef=stas_cfg.get("MIX_COEF", 1.0),
        lr=stas_cfg.get("LR", 1e-3),
        emb_dim=stas_cfg.get("EMB_DIM", 128),
        n_heads=stas_cfg.get("N_HEADS", 4),
        n_layers=stas_cfg.get("N_LAYERS", 2),
        sample_num=stas_cfg.get("SAMPLE_NUM", 4),
        dropout=stas_cfg.get("DROPOUT", 0.1),
        eval_mask_seed=stas_cfg.get("EVAL_MASK_SEED", 3030),
        eval_mask_count=stas_cfg.get("EVAL_MASK_COUNT", 8),
        buffer_size=stas_cfg.get("BUFFER_SIZE", 128),
        batch_size=stas_cfg.get("BATCH_SIZE", 16),
        update_freq=stas_cfg.get("UPDATE_FREQ", 1),
        updates_per_step=stas_cfg.get("UPDATES_PER_STEP", 1),
        warmup_rollouts=stas_cfg.get("WARMUP_ROLLOUTS", 1),
        global_reward_agg=stas_cfg.get("GLOBAL_REWARD_AGG", "sum"),
        device=device,
        causal=not stas_cfg.get("BIDIRECTIONAL", False),
        conserve_discounted=stas_cfg.get("CONSERVE_DISCOUNTED", False),
        quality_gate_enable=stas_cfg.get("QUALITY_GATE_ENABLE", False),
        warmup_episodes=stas_cfg.get("WARMUP_EPISODES", 2000),
        ramp_episodes=stas_cfg.get("RAMP_EPISODES", 8000),
        max_mix_coef=stas_cfg.get("MAX_MIX_COEF", 0.1),
        explained_variance_threshold=stas_cfg.get(
            "EXPLAINED_VARIANCE_THRESHOLD", 0.2
        ),
        negative_patience=stas_cfg.get("NEGATIVE_PATIENCE", 3),
        mode=mode,
        weight_decay=stas_cfg.get("WEIGHT_DECAY", 0.0),
        reward_model_update_interval_episodes=stas_cfg.get(
            "REWARD_MODEL_UPDATE_INTERVAL_EPISODES", 800
        ),
        reward_model_updates_per_interval=stas_cfg.get(
            "REWARD_MODEL_UPDATES_PER_INTERVAL", 50
        ),
        policy_warmup_episodes=stas_cfg.get("POLICY_WARMUP_EPISODES", 4000),
    )
    if mode == "paper":
        return PaperSTASCreditAssigner(credit_config)
    if mode == "uniform":
        # IRCR 风格照妖镜: 与 conserved 完全同调度, credit 换均匀摊。
        return UniformCreditAssigner(credit_config)
    if mode not in {"legacy", "conserved", "conserved-hybrid"}:
        raise ValueError(f"unknown STAS.MODE {mode!r}")
    assigner_cls = (
        ConservedSTASCreditAssigner
        if credit_config.conserve_discounted
        else STASCreditAssigner
    )
    return assigner_cls(credit_config)

def _credit_diagnostics(credit_assigner):
    diagnostics = {
        "stas_mix_coef": float(
            getattr(credit_assigner, "last_mix_coef", credit_assigner.config.mix_coef)
        ),
        "stas_explained_variance": float(getattr(credit_assigner, "last_explained_variance", 0.0)),
        "stas_conservation_error": float(getattr(credit_assigner, "last_conservation_error", 0.0)),
        "stas_episodes_seen": int(getattr(credit_assigner, "episodes_seen", 0)),
        "stas_gate_disabled": bool(
            getattr(getattr(credit_assigner, "gate", None), "disabled", False)
        ),
    }
    for name in (
        "last_reconstruction_rmse",
        "last_agent_credit_variance",
        "last_time_credit_variance",
        "reward_model_updates",
    ):
        if hasattr(credit_assigner, name):
            diagnostics[f"stas_{name.removeprefix('last_')}"] = float(
                getattr(credit_assigner, name)
            )
    return diagnostics


def _paper_policy_should_update(
    mode: str, episodes_seen: int, policy_warmup_episodes: int
) -> bool:
    """Freeze only paper-mode policy through the complete pretraining window."""
    return str(mode).lower() != "paper" or int(episodes_seen) > int(
        policy_warmup_episodes
    )


def _stas_mix_coef_for_update(config, update_idx: int) -> float:
    stas_cfg = config.get("STAS", {})
    schedule = stas_cfg.get("MIX_COEF_SCHEDULE", None)
    if not schedule:
        return float(stas_cfg.get("MIX_COEF", 1.0))
    episode = float(update_idx * config.get("NUM_ENVS", 1))
    points = sorted(
        (float(item["episode"]), float(item["coef"]))
        for item in schedule
        if "episode" in item and "coef" in item
    )
    if not points:
        return float(stas_cfg.get("MIX_COEF", 1.0))
    if episode <= points[0][0]:
        return points[0][1]
    for (left_ep, left_coef), (right_ep, right_coef) in zip(points, points[1:]):
        if episode <= right_ep:
            span = max(right_ep - left_ep, 1.0)
            frac = (episode - left_ep) / span
            return left_coef + frac * (right_coef - left_coef)
    return points[-1][1]


def make_train(config):
    env, _, action_dim, num_actions, observation_size = make_env(
        config["ENV_NAME"], num_envs=config["NUM_ENVS"], **config["TRAIN_ENV_KWARGS"]
    )
    config["NUM_ACTORS"] = env.num_agents * config["NUM_ENVS"]
    config["NUM_UPDATES"] = int(
        config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    )
    config["MINIBATCH_SIZE"] = int(
        config["NUM_ACTORS"] * config["NUM_STEPS"] // config["NUM_MINIBATCHES"]
    )

    def linear_schedule(count):
        frac = 1.0 - (
            count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"])
        ) / config["NUM_UPDATES"]
        return config["LR"] * frac

    def train(rng, wb_run=None):
        num_agents = len(env.agents)
        raw_observation_size = observation_size
        actor_observation_size = raw_observation_size + num_agents
        critic_obs_size = raw_observation_size * num_agents
        is_continuous = config.get("ACTION_SPACE_TYPE", "discrete") == "continuous"
        if not is_continuous:
            raise ValueError("STAS-MAPPO microgrid integration expects continuous actions")

        credit_assigner = _make_credit_assigner(
            config, raw_observation_size, num_actions, num_agents
        )
        stas_cfg = config.get("STAS", {})
        stas_checkpoint_load_path = stas_cfg.get("CHECKPOINT_LOAD_PATH")
        stas_load_path = stas_cfg.get("REWARD_MODEL_LOAD_PATH")
        if stas_checkpoint_load_path:
            stas_metadata = load_credit_assigner_checkpoint(
                stas_checkpoint_load_path, credit_assigner
            )
            print(
                f"Loaded full STAS state from {stas_checkpoint_load_path} "
                f"(global_step={stas_metadata['global_step']}, "
                f"rollouts_seen={credit_assigner.rollouts_seen})"
            )
        elif stas_load_path:
            import torch
            checkpoint = torch.load(stas_load_path, map_location=credit_assigner.config.device)
            load_credit_assigner_state(credit_assigner, checkpoint)

        @jax.jit
        def append_agent_ids(obs):
            num_envs = obs.shape[0] // num_agents
            agent_ids = jnp.tile(jnp.eye(num_agents, dtype=obs.dtype), (num_envs, 1))
            return jnp.concatenate([obs, agent_ids], axis=-1)

        network = ActorCritic(
            action_dim,
            activation=config["ACTIVATION"],
            actor_layers=config.get("ACTOR_LAYERS"),
            critic_layers=config.get("CRITIC_LAYERS"),
            num_agents=env.num_agents,
            observation_dim=raw_observation_size,
            is_continuous=True,
            log_std_init=config.get("LOG_STD_INIT", 0.0),
            log_std_min=config.get("LOG_STD_MIN", -20.0),
            log_std_max=config.get("LOG_STD_MAX", 2.0),
        )

        rng, _rng = jax.random.split(rng)
        init_x = jnp.zeros((1, actor_observation_size))
        init_x_critic = jnp.zeros((1, critic_obs_size))
        network_params = network.init(_rng, init_x, init_x_critic)
        tx = optax.chain(
            optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
            optax.adam(linear_schedule if config["ANNEAL_LR"] else config["LR"], eps=1e-5),
        )
        train_state = TrainState.create(
            apply_fn=network.apply,
            params=network_params,
            tx=tx,
        )
        network.apply = jax.jit(network.apply)

        zero_shot_eval = False
        start_update = 0
        restored_global_step = 0
        restored_rng = None
        training_checkpoint_load_path = config.get("TRAINING_CHECKPOINT_LOAD_PATH")
        if training_checkpoint_load_path:
            restored = load_jax_training_checkpoint(
                training_checkpoint_load_path, train_state
            )
            train_state = restored.train_state
            restored_rng = restored.rng
            start_update = restored.update
            restored_global_step = restored.global_step
            zero_shot_eval = True
            print(
                f"Loaded full training checkpoint from {training_checkpoint_load_path} "
                f"(episode={restored.episode}, update={restored.update})"
            )
        elif config.get("CHECKPOINT_LOAD_DIR") is not None:
            load_checkpointer = checkpointer.Checkpointer(
                PyTreeCheckpointHandler(aggregate_filename="checkpoints")
            )
            loaded = load_checkpointer.restore(
                config.get("CHECKPOINT_LOAD_DIR"), item=train_state.params
            )
            train_state = TrainState.create(apply_fn=train_state.apply_fn, params=loaded, tx=tx)
            zero_shot_eval = True

        episode_stats = EpisodeStatistics(
            episode_returns=jnp.zeros((config["NUM_ENVS"] * num_agents), dtype=jnp.float32),
            episode_lengths=jnp.zeros((config["NUM_ENVS"] * num_agents), dtype=jnp.int32),
            returned_episode_returns=jnp.zeros(
                (config["NUM_ENVS"] * num_agents), dtype=jnp.float32
            ),
            returned_episode_lengths=jnp.zeros(
                (config["NUM_ENVS"] * num_agents), dtype=jnp.int32
            ),
        )

        transitions = Transition(
            done=np.zeros((config["NUM_STEPS"], config["NUM_ENVS"] * num_agents), dtype=bool),
            action=np.zeros(
                (config["NUM_STEPS"], config["NUM_ENVS"] * num_agents, num_actions),
                dtype=jnp.float32,
            ),
            value=np.zeros((config["NUM_STEPS"], config["NUM_ENVS"] * num_agents), dtype=np.float32),
            reward=np.zeros((config["NUM_STEPS"], config["NUM_ENVS"] * num_agents), dtype=np.float32),
            log_prob=np.zeros((config["NUM_STEPS"], config["NUM_ENVS"] * num_agents), dtype=np.float32),
            obs=np.zeros(
                (config["NUM_STEPS"], config["NUM_ENVS"] * num_agents, actor_observation_size),
                dtype=jnp.float32,
            ),
            global_obs=np.zeros(
                (config["NUM_STEPS"], config["NUM_ENVS"] * num_agents, critic_obs_size),
                dtype=jnp.float32,
            ),
            info=TransitionInfo(
                returned_episode_returns=np.zeros(
                    (config["NUM_STEPS"], config["NUM_ENVS"] * num_agents),
                    dtype=np.float32,
                ),
                returned_episode_lengths=np.zeros(
                    (config["NUM_STEPS"], config["NUM_ENVS"] * num_agents),
                    dtype=np.int32,
                ),
            ),
        )

        @jax.jit
        def concat_local_obs_into_global_obs(obs):
            reshaped_obs = obs.reshape((config["NUM_ENVS"], num_agents, -1))
            obs_dim = reshaped_obs.shape[-1]
            global_obs = jnp.zeros((config["NUM_ENVS"] * num_agents, num_agents * obs_dim))
            for env_idx in range(config["NUM_ENVS"]):
                flat_global_obs_per_env = reshaped_obs[env_idx].flatten()
                env_begin = env_idx * num_agents
                env_end = env_begin + num_agents
                global_obs = global_obs.at[env_begin:env_end].set(
                    jnp.tile(flat_global_obs_per_env, (num_agents,)).reshape(
                        num_agents, num_agents * obs_dim
                    )
                )
            return global_obs

        rng, reset_rng = jax.random.split(rng)
        int_seed = jax.random.randint(
            reset_rng, shape=(1,), minval=1, maxval=1000000
        ).item()
        raw_obsv, _ = reset_environment_stream(
            env,
            seed=int_seed,
            completed_resets=start_update,
        )
        obsv = append_agent_ids(raw_obsv)
        global_obsv = concat_local_obs_into_global_obs(raw_obsv)
        env_state = {}
        env_step = env.step

        def step_env_wrapped(action: Any) -> Any:
            next_raw_obs, reward, termination, truncs, info = env_step(action.flatten())
            next_actor_obs = append_agent_ids(next_raw_obs)
            global_obs = concat_local_obs_into_global_obs(next_raw_obs)
            return next_raw_obs, next_actor_obs, reward, termination, truncs, info, global_obs

        @jax.jit
        def update_episode_stats(episode_stats, reward, done):
            new_episode_return = episode_stats.episode_returns + reward
            new_episode_length = episode_stats.episode_lengths + 1
            returned_episode_returns = jnp.where(
                done, new_episode_return, episode_stats.returned_episode_returns
            )
            returned_episode_lengths = jnp.where(
                done, new_episode_length, episode_stats.returned_episode_lengths
            )
            return episode_stats.replace(
                episode_returns=new_episode_return * (1 - done),
                episode_lengths=new_episode_length * (1 - done),
                returned_episode_returns=returned_episode_returns,
                returned_episode_lengths=returned_episode_lengths,
            )

        @jax.jit
        def _select_action(params, obs, obs_critic, _rng):
            actor_output, value = network.apply(params, obs, obs_critic)
            actor_mean, actor_log_std = actor_output
            if config.get("POLICY_MODE") == "squashed_gaussian":
                action, log_prob = sample_squashed_gaussian(
                    actor_mean,
                    actor_log_std,
                    _rng,
                    log_std_min=config.get("LOG_STD_MIN", -2.5),
                    log_std_max=config.get("LOG_STD_MAX", -0.5),
                )
                return action, log_prob, value
            pi = distrax.MultivariateNormalDiag(actor_mean, jnp.exp(actor_log_std))
            action = pi.sample(seed=_rng)
            log_prob = pi.log_prob(action)
            return action, log_prob, value

        @jax.jit
        def _select_action_eval(params, obs, _rng):
            dummy_obs_critic = jnp.zeros((obs.shape[0], critic_obs_size))
            actor_obs = append_agent_ids(obs)
            actor_output, _ = network.apply(params, actor_obs, dummy_obs_critic)
            actor_mean, actor_log_std = actor_output
            if config.get("eval_stochastic"):
                if config.get("POLICY_MODE") == "squashed_gaussian":
                    action, _ = sample_squashed_gaussian(
                        actor_mean,
                        actor_log_std,
                        _rng,
                        log_std_min=config.get("LOG_STD_MIN", -2.5),
                        log_std_max=config.get("LOG_STD_MAX", -0.5),
                    )
                else:
                    pi = distrax.MultivariateNormalDiag(actor_mean, jnp.exp(actor_log_std))
                    action = deterministic_action(pi.sample(seed=_rng))
            else:
                action = deterministic_action(actor_mean)
            return (action,)

        calculate_advantage_and_update_ppo = get_update_fn(config, network)

        def _update_step(runner_state, unused):
            train_state, env_state, last_raw_obs, last_obs, last_global_obs, rng, episode_stats = runner_state
            stas_obs = np.zeros(
                (config["NUM_STEPS"], config["NUM_ENVS"] * num_agents, raw_observation_size),
                dtype=np.float32,
            )
            stas_actions = np.zeros(
                (config["NUM_STEPS"], config["NUM_ENVS"] * num_agents, num_actions),
                dtype=np.float32,
            )

            for t in range(config["NUM_STEPS"]):
                rng, _rng = jax.random.split(rng)
                action, log_prob, value = _select_action(
                    train_state.params, last_obs, last_global_obs, _rng
                )
                if config.get("POLICY_MODE") == "squashed_gaussian":
                    env_action = action
                else:
                    env_action = jnp.tanh(action)
                np_action = np.asarray(env_action, dtype=np.float32)
                (
                    next_raw_obs,
                    obsv,
                    reward,
                    termination,
                    truncs,
                    info,
                    global_obs,
                ) = step_env_wrapped(np_action)
                done = np.logical_or(termination, truncs)
                episode_stats = update_episode_stats(episode_stats, reward, done)

                stas_obs[t] = np.asarray(last_raw_obs, dtype=np.float32)
                stas_actions[t] = np_action
                transitions.done[t] = done
                transitions.action[t] = action
                transitions.value[t] = value
                transitions.reward[t] = reward
                transitions.log_prob[t] = log_prob
                transitions.obs[t] = last_obs
                transitions.global_obs[t] = last_global_obs
                transitions.info.returned_episode_returns[t] = (
                    episode_stats.returned_episode_returns
                )
                transitions.info.returned_episode_lengths[t] = (
                    episode_stats.returned_episode_lengths
                )

                last_raw_obs = next_raw_obs
                last_obs = obsv
                last_global_obs = global_obs

            rewards_eat = flat_time_agent_to_env_agent_time(
                np.asarray(transitions.reward), config["NUM_ENVS"], num_agents
            )
            dones_eat = flat_time_agent_to_env_agent_time(
                np.asarray(transitions.done, dtype=np.float32),
                config["NUM_ENVS"],
                num_agents,
            )
            obs_eat = flat_time_agent_to_env_agent_time(
                stas_obs, config["NUM_ENVS"], num_agents
            )
            actions_eat = flat_time_agent_to_env_agent_time(
                stas_actions, config["NUM_ENVS"], num_agents
            )
            if (
                credit_assigner.config.mode != "paper"
                and not credit_assigner.config.conserve_discounted
            ):
                credit_assigner.config.mix_coef = _stas_mix_coef_for_update(
                    config, update
                )
            training_rewards_eat, stas_loss = credit_assigner.process_rollout(
                obs_eat, actions_eat, rewards_eat, dones_eat
            )
            training_rewards = env_agent_time_to_flat_time_agent(
                training_rewards_eat, config["NUM_ENVS"], num_agents
            )
            transitions_for_update = transitions.replace(reward=training_rewards)

            policy_update_enabled = _paper_policy_should_update(
                credit_assigner.config.mode,
                credit_assigner.episodes_seen,
                credit_assigner.config.policy_warmup_episodes,
            )
            if policy_update_enabled:
                update_state, loss_info = calculate_advantage_and_update_ppo(
                    transitions_for_update, last_obs, last_global_obs, train_state, rng
                )
                train_state = update_state[0]
                rng = update_state[-1]
            else:
                loss_info = {
                    name: jnp.asarray(0.0, dtype=jnp.float32)
                    for name in (
                        "total_loss",
                        "actor_loss",
                        "critic_loss",
                        "entropy",
                        "ratio",
                        "approx_kl",
                        "clip_fraction",
                        "total_grad_mean",
                        "total_grad_var",
                        "total_grad_norm",
                    )
                }

            metric = {
                "returned_episode_returns": episode_stats.returned_episode_returns,
                "returned_episode_lengths": episode_stats.returned_episode_lengths,
                "stas_reward_model_loss": jnp.asarray(
                    0.0 if np.isnan(stas_loss) else stas_loss, dtype=jnp.float32
                ),
                "stas_policy_frozen": jnp.asarray(
                    not policy_update_enabled, dtype=jnp.float32
                ),
                **{
                    name: jnp.asarray(value, dtype=jnp.float32)
                    for name, value in _credit_diagnostics(credit_assigner).items()
                },
            }
            loss_info = jax.tree_util.tree_map(lambda x: x.mean(), loss_info)
            metric = jax.tree_util.tree_map(lambda x: x.mean(), metric)
            metric = {**metric, **loss_info}

            runner_state = (
                train_state,
                env_state,
                last_raw_obs,
                last_obs,
                last_global_obs,
                rng,
                episode_stats,
            )
            return runner_state, metric

        rng, _rng = jax.random.split(rng)
        runner_state = (
            train_state,
            env_state,
            raw_obsv,
            obsv,
            global_obsv,
            restored_rng if restored_rng is not None else _rng,
            episode_stats,
        )

        if zero_shot_eval:
            eval_data = run_eval_jax(
                cfg=config,
                agent_state=runner_state[0].params,
                writer=wb_run,
                acting_fns=_select_action_eval,
                eval_seed=42,
                global_step=restored_global_step,
                capture_video=False,
                recurrent=False,
                shared_weights=True,
                parallel=config.get("EVAL_PARALLEL", True),
            )

        training_metrics = None
        eval_metrics = []
        start_time = time.time()
        global_step = restored_global_step
        eval_interval = config["EVAL_INTERVAL"]
        next_eval_step = (global_step // eval_interval + 1) * eval_interval
        checkpoint_interval = config.get("CHECKPOINT_INTERVAL", None)
        next_checkpoint_step = (
            (global_step // checkpoint_interval + 1) * checkpoint_interval
            if checkpoint_interval else None
        )
        best_validation_return = _existing_best_validation_return(
            config.get("BEST_VALIDATION_CHECKPOINT_DIR")
        )

        if config.get("EVAL_ONLY", False):
            return {"metrics": {}, "eval_metrics": [(0, eval_data)]}, env

        for update in range(start_update, config["NUM_UPDATES"]):
            final_update = update == config["NUM_UPDATES"] - 1
            update_time_start = time.time()
            runner_state, ret_metric = _update_step(runner_state, None)
            global_step += config["NUM_STEPS"] * config["NUM_ENVS"]
            write_rollout_diagnostic(
                stas_cfg.get("DIAGNOSTICS_PATH"),
                credit_assigner,
                update=update + 1,
                episode=global_step // config["NUM_STEPS"],
                global_step=global_step,
            )

            if update % config.get("LOG_INTERVAL", 100) == 0:
                print(f"Update: {update}/{config['NUM_UPDATES']}")
                sps = int(global_step / max(time.time() - start_time, 1e-6))
                sps_update = int(
                    config["NUM_ENVS"]
                    * config["NUM_STEPS"]
                    / max(time.time() - update_time_start, 1e-6)
                )
                print("SPS:", sps, sps_update)
                if wb_run is not None:
                    wb_run.log({"charts/SPS": sps, "charts/SPS_update": sps_update}, global_step)

            if training_metrics is None:
                training_metrics = initialize_metrics_storage(config, ret_metric.keys())
            training_metrics = update_metrics(training_metrics, ret_metric, update)
            log_update_progress(
                config,
                update,
                float(np.asarray(ret_metric["returned_episode_returns"]).mean()),
            )

            if global_step >= next_eval_step:
                if config.get("FIXED_EVAL_OUTPUT"):
                    params = runner_state[0].params

                    def fixed_action(obs):
                        return np.asarray(
                            _select_action_eval(
                                params, jnp.asarray(obs), jax.random.PRNGKey(0)
                            )[0]
                        )

                    fixed_eval_split = str(
                        config.get("FIXED_EVAL_SPLIT", "validation")
                    )
                    if fixed_eval_split == "validation":
                        fixed_eval_days = VALIDATION_DAYS
                        default_eval_seed = DEFAULT_NOISE_SEEDS[0]
                    elif fixed_eval_split == "test":
                        from baselines.utils.fixed_scenario_eval import (
                            TEST_DAYS,
                            TEST_NOISE_SEEDS,
                        )

                        fixed_eval_days = TEST_DAYS
                        default_eval_seed = TEST_NOISE_SEEDS[0]
                    else:
                        raise ValueError(
                            "FIXED_EVAL_SPLIT must be 'validation' or 'test'"
                        )
                    fixed_eval_seed = int(
                        config.get("FIXED_EVAL_NOISE_SEED", default_eval_seed)
                    )
                    eval_data = evaluate_policy(
                        fixed_action,
                        config.get("MICROGRID_CONFIG_OVERRIDES") or {},
                        build_scenarios(fixed_eval_days, (fixed_eval_seed,)),
                        algorithm=config["ALG"],
                        split_name=fixed_eval_split,
                    )
                    eval_data["stas_diagnostics"] = _credit_diagnostics(
                        credit_assigner
                    )
                    current_episode = global_step // config["NUM_STEPS"]
                    append_evaluation_record(
                        config["FIXED_EVAL_OUTPUT"], eval_data,
                        training_episode=current_episode,
                    )
                    best_root = config.get("BEST_VALIDATION_CHECKPOINT_DIR")
                    validation_return = float(
                        eval_data["summary"]["return_mean"]
                    )
                    if (
                        fixed_eval_split == "validation"
                        and best_root
                        and validation_return > best_validation_return
                    ):
                        _save_best_validation_checkpoint(
                            best_root,
                            train_state=runner_state[0],
                            rng=runner_state[-2],
                            credit_assigner=credit_assigner,
                            update=update + 1,
                            episode=current_episode,
                            global_step=global_step,
                            validation_return=validation_return,
                        )
                        best_validation_return = validation_return
                        print(
                            "Best validation checkpoint saved at "
                            f"episode {current_episode}: {validation_return:.6f}"
                        )
                else:
                    eval_data = run_eval_jax(
                        cfg=config, agent_state=runner_state[0].params, writer=wb_run,
                        acting_fns=_select_action_eval, eval_seed=42,
                        global_step=global_step, capture_video=False,
                        recurrent=False, shared_weights=True,
                        parallel=config.get("EVAL_PARALLEL", True),
                    )
                eval_metrics.append((global_step, eval_data))
                next_eval_step += config["EVAL_INTERVAL"]

            if config.get("CHECKPOINT", True) and (
                final_update or (next_checkpoint_step and global_step >= next_checkpoint_step)
            ):
                model_path = config.get("TRAINING_CHECKPOINT_PATH") or (
                    f"{config['CHP_DIR']}/{config['EXP_NAME']}_latest."
                    f"agent_{config['SEED']}_seed.msgpack"
                )
                stas_path = Path(
                    stas_cfg.get("CHECKPOINT_PATH")
                    or Path(config["CHP_DIR"]) / "stas_credit_latest.pt"
                )
                stas_path.parent.mkdir(parents=True, exist_ok=True)
                save_jax_training_checkpoint(
                    model_path,
                    train_state=runner_state[0],
                    rng=runner_state[-2],
                    update=update + 1,
                    episode=global_step // config["NUM_STEPS"],
                    global_step=global_step,
                )
                save_credit_assigner_checkpoint(
                    stas_path,
                    credit_assigner,
                    update=update + 1,
                    episode=global_step // config["NUM_STEPS"],
                    global_step=global_step,
                )
                print(f"Full training state saved to {model_path} at step {global_step}")
                print(f"Full STAS state saved to {stas_path} at step {global_step}")
                if next_checkpoint_step:
                    next_checkpoint_step += config["CHECKPOINT_INTERVAL"]

        return {
            "runner_state": runner_state,
            "metrics": training_metrics,
            "eval_metrics": eval_metrics,
            "stas_credit_assigner": credit_assigner,
        }, env

    return train


@hydra.main(version_base=None, config_path="config", config_name="stas_mappo_microgrid")
def main(config):
    print("Starting STAS-MAPPO training")
    config = OmegaConf.to_container(config, resolve=True)
    apply_microgrid_config_overrides(config)
    run = wandb.init(
        entity=config["ENTITY"],
        project=config["PROJECT"],
        tags=config["EXP_TAGS"],
        config=config,
        mode=config["WANDB_MODE"],
        name=config["RUN_NAME"],
        save_code=True,
        reinit=True,
        group=config["GROUP"],
    )
    config["CHP_DIR"] = f"{wandb.run.dir}/models/{config['RUN_NAME']}"  # type: ignore
    config["TEST_ENV_KWARGS"].update(config["env"]["ENV_KWARGS"])
    config["TRAIN_ENV_KWARGS"].update(config["env"]["ENV_KWARGS"])
    seed_stas_runtime(config["SEED"])
    rng = jax.random.PRNGKey(config["SEED"])
    train = make_train(config)
    out, env = train(rng, run)
    log_train_metrics(config, out["metrics"], run)
    log_eval_metrics(config, out["eval_metrics"], run)
    env.close()
    wandb.finish()


if __name__ == "__main__":
    main()
