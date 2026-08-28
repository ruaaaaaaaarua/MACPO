"""
Standard MAPPO (no hypernetwork) with shared actor weights on Microgrid.

The actor is a fixed MLP on local observations; the centralized critic uses a
shared MLP on concatenated global observations (CTDE).
"""

import multiprocessing as mp

forkserver_available = "forkserver" in mp.get_all_start_methods()
start_method = "forkserver" if forkserver_available else "spawn"
mp.set_start_method(start_method, force=True)

import os
from collections import defaultdict
from enum import Enum
from pathlib import Path
import sys
import tempfile
import time
from typing import Any, Dict, List, NamedTuple, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from baselines.utils import jax_runtime_bootstrap  # noqa: F401
LOCAL_TMP = PROJECT_ROOT / ".tmp"
WANDB_LOCAL_DIR = PROJECT_ROOT / "wandb"
for _path in (LOCAL_TMP, WANDB_LOCAL_DIR, WANDB_LOCAL_DIR / "cache", WANDB_LOCAL_DIR / "data"):
    _path.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("TMP", str(LOCAL_TMP))
os.environ.setdefault("TEMP", str(LOCAL_TMP))
os.environ.setdefault("TMPDIR", str(LOCAL_TMP))
os.environ.setdefault("WANDB_DIR", str(WANDB_LOCAL_DIR))
os.environ.setdefault("WANDB_CACHE_DIR", str(WANDB_LOCAL_DIR / "cache"))
os.environ.setdefault("WANDB_DATA_DIR", str(WANDB_LOCAL_DIR / "data"))
if os.name == "nt":
    _TemporaryDirectory = tempfile.TemporaryDirectory

    class _IgnoreCleanupTemporaryDirectory(_TemporaryDirectory):
        def __init__(self, *args, **kwargs):
            kwargs["ignore_cleanup_errors"] = True
            super().__init__(*args, **kwargs)

        def cleanup(self):
            try:
                super().cleanup()
            except Exception:
                pass

    tempfile.TemporaryDirectory = _IgnoreCleanupTemporaryDirectory

import distrax
import flax.linen as nn
import hydra
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
from flax.linen.initializers import constant, orthogonal
from flax.struct import dataclass
from flax.training import orbax_utils
from flax.training.train_state import TrainState
from omegaconf import OmegaConf
from orbax.checkpoint import checkpointer
from orbax.checkpoint.pytree_checkpoint_handler import PyTreeCheckpointHandler

import wandb
from baselines.MAPPO.mappo import get_update_fn
from baselines.utils.utils import (
    calculate_team_diversity,
    extract_and_save_agent_embeddings,
    log_eval_metrics,
    log_train_metrics,
    log_videos,
)
from baselines.utils.eval import run_eval_jax
from baselines.utils.wrappers import make_env


def apply_microgrid_config_overrides(config):
    overrides = config.get("MICROGRID_CONFIG_OVERRIDES")
    if not overrides:
        return
    from envs.microgrid.config import MICROGRID_CONFIG

    MICROGRID_CONFIG.update(dict(overrides))
    print(f"Applied MICROGRID_CONFIG_OVERRIDES: {dict(overrides)}")


@dataclass
class EpisodeStatistics:
    """Tracks statistics about episodes during training."""
    episode_returns: jnp.array
    episode_lengths: jnp.array
    returned_episode_returns: jnp.array
    returned_episode_lengths: jnp.array


class ActionType(Enum):
    """Supported action space types."""
    DISCRETE = "discrete"
    CONTINUOUS = "continuous"


class ActorCritic(nn.Module):
    """Shared-weight actor MLP + centralized critic MLP for CTDE MAPPO."""

    action_dim: int
    num_agents: int
    actor_layers: List[int]
    critic_layers: List[int]
    activation: str = "tanh"
    is_continuous: bool = False
    log_std_init: float = 0.0

    @nn.compact
    def __call__(self, x, x_critic):
        activation_fn = nn.relu if self.activation == "relu" else nn.tanh

        def build_network(inp, layer_sizes: List[int]):
            h = inp
            for size in layer_sizes:
                h = nn.Dense(
                    size,
                    kernel_init=orthogonal(np.sqrt(2)),
                    bias_init=constant(0.0),
                )(h)
                h = activation_fn(h)
            return h

        actor_hidden = build_network(x, list(self.actor_layers))
        if self.is_continuous:
            actor_mean = nn.Dense(
                self.action_dim,
                kernel_init=orthogonal(0.01),
                bias_init=constant(0.0),
            )(actor_hidden)
            log_std = self.param(
                "log_std",
                nn.initializers.constant(self.log_std_init),
                (self.action_dim,),
            )
            actor_output = (actor_mean, log_std)
        else:
            actor_output = nn.Dense(
                self.action_dim,
                kernel_init=orthogonal(0.01),
                bias_init=constant(0.0),
            )(actor_hidden)

        critic_hidden = build_network(x_critic, list(self.critic_layers))
        critic = nn.Dense(
            1, kernel_init=orthogonal(1.0), bias_init=constant(0.0)
        )(critic_hidden)

        return actor_output, jnp.squeeze(critic, axis=-1)


class TransitionInfo(NamedTuple):
    returned_episode_returns: jnp.array
    returned_episode_lengths: jnp.array


@dataclass
class Transition:
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    global_obs: jnp.ndarray
    info: TransitionInfo


def initialize_metrics_storage(config, metric_keys):
    return {k: np.zeros(config["NUM_UPDATES"]) for k in metric_keys}


@jax.jit
def update_metrics(metrics: dict, new_values: dict, update_idx: int) -> dict:
    def _update(arr, val):
        if isinstance(arr, jnp.ndarray):
            return arr.at[update_idx].set(val)
        elif isinstance(val, dict):
            return jax.tree_map(lambda a, v: _update(a, v), arr, val)
        return val

    return jax.tree_map(lambda x, y: _update(x, y), metrics, new_values)


def make_train(config):
    assert config["env"]["ENV_KWARGS"].get("one_hot_encode_agent_id"), "one_hot_ids must be True"

    env, possible_agents, action_dim, num_actions, observation_size = make_env(
        config["ENV_NAME"], num_envs=config["NUM_ENVS"], **config["TRAIN_ENV_KWARGS"]
    )

    config["NUM_ACTORS"] = env.num_agents * config["NUM_ENVS"]
    config["NUM_UPDATES"] = int(
        config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    )
    config["MINIBATCH_SIZE"] = int(
        config["NUM_ACTORS"] * config["NUM_STEPS"] // config["NUM_MINIBATCHES"]
    )

    action_space_type = config["ACTION_SPACE_TYPE"].lower()
    if action_space_type == ActionType.DISCRETE.value:
        action_type = ActionType.DISCRETE
    elif action_space_type == ActionType.CONTINUOUS.value:
        action_type = ActionType.CONTINUOUS
    else:
        raise ValueError(f"Unknown action space type: {config['ACTION_SPACE_TYPE']}")

    def linear_schedule(count):
        frac = (
            1.0
            - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"]))
            / config["NUM_UPDATES"]
        )
        return config["LR"] * frac

    def train(rng, wb_run=None):
        critic_obs_size = (observation_size - env.num_agents) * env.num_agents

        network = ActorCritic(
            action_dim,
            activation=config["ACTIVATION"],
            actor_layers=config.get("ACTOR_LAYERS"),
            critic_layers=config.get("CRITIC_LAYERS"),
            num_agents=env.num_agents,
            is_continuous=action_type is ActionType.CONTINUOUS,
            log_std_init=float(config.get("LOG_STD_INIT", 0.0)),
        )

        rng, _rng = jax.random.split(rng)
        init_x = jnp.zeros((1, observation_size))
        num_agents = len(env.agents)
        init_x_critic = jnp.zeros((1, critic_obs_size))
        network_params = network.init(_rng, init_x, init_x_critic)

        if config["ANNEAL_LR"]:
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(learning_rate=linear_schedule, eps=1e-5),
            )
        else:
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(config["LR"], eps=1e-5),
            )

        train_state = TrainState.create(
            apply_fn=network.apply,
            params=network_params,
            tx=tx,
        )

        network.apply = jax.jit(network.apply)

        zero_shot_eval = False
        if config.get("CHECKPOINT_LOAD_DIR") is not None:
            print(f"Loading from {config.get('CHECKPOINT_LOAD_DIR')}")
            load_checkpointer = checkpointer.Checkpointer(
                PyTreeCheckpointHandler(aggregate_filename="checkpoints")
            )
            loaded_checkpoints = load_checkpointer.restore(
                config.get("CHECKPOINT_LOAD_DIR"), item=train_state.params
            )
            train_state = TrainState.create(
                apply_fn=train_state.apply_fn, params=loaded_checkpoints, tx=tx
            )
            zero_shot_eval = True

        episode_stats = EpisodeStatistics(
            episode_returns=jnp.zeros(
                (config["NUM_ENVS"] * num_agents), dtype=jnp.float32
            ),
            episode_lengths=jnp.zeros(
                (config["NUM_ENVS"] * num_agents), dtype=jnp.int32
            ),
            returned_episode_returns=jnp.zeros(
                (config["NUM_ENVS"] * num_agents), dtype=jnp.float32
            ),
            returned_episode_lengths=jnp.zeros(
                (config["NUM_ENVS"] * num_agents), dtype=jnp.int32
            ),
        )

        transitions = Transition(
            done=np.zeros(
                (config["NUM_STEPS"], config["NUM_ENVS"] * num_agents), dtype=bool
            ),
            action=np.zeros(
                (config["NUM_STEPS"], config["NUM_ENVS"] * num_agents)
                + ((num_actions,) if action_type is ActionType.CONTINUOUS else ()),
                dtype=jnp.float32,
            ),
            value=np.zeros(
                (config["NUM_STEPS"], config["NUM_ENVS"] * num_agents), dtype=np.float32
            ),
            reward=np.zeros(
                (config["NUM_STEPS"], config["NUM_ENVS"] * num_agents), dtype=np.float32
            ),
            log_prob=np.zeros(
                (config["NUM_STEPS"], config["NUM_ENVS"] * num_agents), dtype=np.float32
            ),
            obs=np.zeros(
                (
                    config["NUM_STEPS"],
                    config["NUM_ENVS"] * num_agents,
                    observation_size,
                ),
                dtype=jnp.float32,
            ),
            global_obs=np.zeros(
                (
                    config["NUM_STEPS"],
                    config["NUM_ENVS"] * num_agents,
                    critic_obs_size,
                ),
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
            if config["env"]["ENV_KWARGS"].get("one_hot_encode_agent_id"):
                reshaped_obs = obs.reshape((config["NUM_ENVS"], num_agents, -1))[:, :, :-num_agents]
            else:
                reshaped_obs = obs.reshape((config["NUM_ENVS"], num_agents, -1))

            obs_dim = reshaped_obs.shape[-1]
            global_obs = jnp.zeros(
                (config["NUM_ENVS"] * num_agents, num_agents * obs_dim)
            )

            for i in range(config["NUM_ENVS"]):
                flat_global_obs_per_env = reshaped_obs[i].flatten()
                env_begin = i * num_agents
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
        obsv, infos = env.reset(seed=int_seed)
        global_obsv = concat_local_obs_into_global_obs(obsv)
        env_state = {}
        env_step = env.step

        def step_env_wrapped(action: Any) -> Any:
            next_obs, reward, termination, truncs, info = env_step(action.flatten())
            global_obs = concat_local_obs_into_global_obs(next_obs)
            return next_obs, reward, termination, truncs, info, global_obs

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
            if action_type is ActionType.DISCRETE:
                pi = distrax.Categorical(logits=actor_output)
            else:
                actor_mean, actor_logtstd = actor_output
                pi = distrax.MultivariateNormalDiag(
                    loc=actor_mean,
                    scale_diag=jnp.exp(actor_logtstd),
                )
            action = pi.sample(seed=_rng)
            log_prob = pi.log_prob(action)
            return action, log_prob, value

        @jax.jit
        def _select_action_eval(params, obs, _rng):
            dummy_obs_critic = jnp.zeros((obs.shape[0], critic_obs_size))
            actor_output, _ = network.apply(params, obs, dummy_obs_critic)
            if action_type is ActionType.DISCRETE:
                pi = distrax.Categorical(logits=actor_output)
            else:
                actor_mean, actor_logtstd = actor_output
                pi = distrax.MultivariateNormalDiag(
                    loc=actor_mean,
                    scale_diag=jnp.exp(actor_logtstd),
                )

            if config.get("EVAL_DETERMINISTIC") is False:
                action = pi.sample(seed=_rng)
            else:
                action = pi.mode()
            if action_type is ActionType.CONTINUOUS:
                action = jnp.tanh(action)

            return (action,)

        calculate_advantage_and_update_ppo = get_update_fn(config, network)

        def _update_step(runner_state, unused):
            train_state, env_state, last_obs, last_global_obs, rng, episode_stats = runner_state

            for t in range(config["NUM_STEPS"]):
                rng, _rng = jax.random.split(rng)
                action, log_prob, value = _select_action(
                    train_state.params, last_obs, last_global_obs, _rng
                )

                env_action = jnp.tanh(action) if action_type is ActionType.CONTINUOUS else action
                np_action = np.array(env_action)
                obsv, reward, termination, truncs, info, global_obs = step_env_wrapped(np_action)

                done = np.logical_or(termination, truncs)
                episode_stats = update_episode_stats(episode_stats, reward, done)

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

                last_obs = obsv
                last_global_obs = global_obs

            update_state, loss_info = calculate_advantage_and_update_ppo(
                transitions, last_obs, last_global_obs, train_state, rng
            )
            train_state = update_state[0]

            metric = {
                "returned_episode_returns": episode_stats.returned_episode_returns,
                "returned_episode_lengths": episode_stats.returned_episode_lengths,
            }
            rng = update_state[-1]

            loss_info = jax.tree_util.tree_map(lambda x: x.mean(), loss_info)
            metric = jax.tree_util.tree_map(lambda x: x.mean(), metric)
            metric = {**metric, **loss_info}

            runner_state = (train_state, env_state, last_obs, last_global_obs, rng, episode_stats)
            return runner_state, metric

        rng, _rng = jax.random.split(rng)
        runner_state = (train_state, env_state, obsv, global_obsv, _rng, episode_stats)

        checkpointers = checkpointer.Checkpointer(
            PyTreeCheckpointHandler(aggregate_filename=f"checkpoints")
        )

        if zero_shot_eval:
            print("Running zero shot eval")
            capture_video = config.get("EVAL_EPISODES") <= 40
            eval_data = run_eval_jax(
                cfg=config,
                agent_state=runner_state[0].params,
                writer=wb_run,
                acting_fns=_select_action_eval,
                eval_seed=42,
                global_step=0,
                capture_video=capture_video,
                recurrent=False,
                shared_weights=True,
                parallel=config.get("EVAL_PARALLEL", True),
            )

        training_metrics = None

        calculate_team_diversity_metrics = config.get(
            "CALCULATE_TEAM_DIVERSITY_METRICS", False
        )
        if calculate_team_diversity_metrics:
            _keys = jax.random.PRNGKey(config["SEED"])
            calculate_team_diversity(
                network, runner_state[0], _keys, num_agents,
                centralised_critic=True, param_sharing=True
            )

        eval_metrics = []
        start_time = time.time()
        global_step = 0
        next_eval_step = config["EVAL_INTERVAL"]
        next_capture_video_step = config.get("CAPTURE_VIDEO_INTERVAL", None)
        next_checkpoint_step = config.get("CHECKPOINT_INTERVAL", None)

        eval_only = config.get("EVAL_ONLY", False)
        if eval_only:
            return {"metrics": {}, "eval_metrics": [(0, eval_data)]}, env

        for update in range(config["NUM_UPDATES"]):
            final_update = update == config["NUM_UPDATES"] - 1
            update_time_start = time.time()

            runner_state, ret_metric = _update_step(runner_state, None)
            global_step += config["NUM_STEPS"] * config["NUM_ENVS"]

            if update % 100 == 0:
                print(f"Update: {update}/{config['NUM_UPDATES']}")
                sps = int(global_step / (time.time() - start_time))
                sps_update = int(
                    config["NUM_ENVS"]
                    * config["NUM_STEPS"]
                    / (time.time() - update_time_start)
                )
                print("SPS:", sps, sps_update)
                wb_run.log({"charts/SPS": sps}, global_step)
                wb_run.log({"charts/SPS_update": sps_update}, global_step)

            if training_metrics is None:
                training_metrics = initialize_metrics_storage(config, ret_metric.keys())
            training_metrics = update_metrics(training_metrics, ret_metric, update)

            record_final_episode = config.get("CAPTURE_VIDEO_INTERVAL") and final_update
            if (global_step >= next_eval_step) or record_final_episode:
                if (next_capture_video_step and global_step >= next_capture_video_step) or record_final_episode:
                    next_capture_video_step += config["CAPTURE_VIDEO_INTERVAL"]
                    capture_video = True
                else:
                    capture_video = False

                eval_data = run_eval_jax(
                    cfg=config,
                    agent_state=runner_state[0].params,
                    writer=wb_run,
                    acting_fns=_select_action_eval,
                    eval_seed=42,
                    global_step=global_step,
                    capture_video=capture_video,
                    recurrent=False,
                    shared_weights=True,
                    parallel=config.get("EVAL_PARALLEL", True),
                )
                eval_metrics.append((global_step, eval_data))
                next_eval_step += config["EVAL_INTERVAL"]

            if update % config.get("SAVE_EMBEDDINGS_INTERVAL", 5000) == 0 or final_update:
                extract_and_save_agent_embeddings(runner_state[0].params, global_step, config, wb_run)

            if (final_update) or (next_checkpoint_step and global_step >= next_checkpoint_step):
                agent_identity = f"agent_{config['SEED']}_seed"
                model_path = f"{config['CHP_DIR']}/{config['EXP_NAME']}_{global_step}_steps_{update}_updates.{agent_identity}"
                save_args = orbax_utils.save_args_from_target(runner_state[0].params)
                checkpointers.save(
                    model_path, runner_state[0].params, save_args=save_args
                )
                print(f"Model saved to {model_path} at step {global_step}")
                next_checkpoint_step += config["CHECKPOINT_INTERVAL"]

        return {
            "runner_state": runner_state,
            "metrics": training_metrics,
            "eval_metrics": eval_metrics,
        }, env

    return train


@hydra.main(
    version_base=None,
    config_path="config",
    config_name="mappo_ff_shared_weights_microgrid",
)
def main(config):
    print("Starting MAPPO baseline training")
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

    if config.get("CHECKPOINT_NAME") is not None:
        artifact = run.use_artifact(config.get("CHECKPOINT_NAME"), type="checkpoint")
        artifact_dir = artifact.download()
        checkpoint_load_dir = f"{artifact_dir}/{config.get('CHECKPOINT_FOLDER')}"
        config["CHECKPOINT_LOAD_DIR"] = checkpoint_load_dir
        print(f"checkpoint_load_dir: {checkpoint_load_dir}")

    rng = jax.random.PRNGKey(config["SEED"])
    rngs = jax.random.split(rng, config["NUM_SEEDS"])

    config["TEST_ENV_KWARGS"].update(config["env"]["ENV_KWARGS"])
    config["TRAIN_ENV_KWARGS"].update(config["env"]["ENV_KWARGS"])

    train = make_train(config)
    out, env = train(rngs[0], run)

    log_train_metrics(config, out["metrics"], run)
    log_eval_metrics(config, out["eval_metrics"], run)

    env.close()
    wandb.finish()


if __name__ == "__main__":
    main()
