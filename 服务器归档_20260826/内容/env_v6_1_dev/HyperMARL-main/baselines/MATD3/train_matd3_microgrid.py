#!/usr/bin/env python3
"""MATD3 training entry for the in-repo microgrid environment."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from baselines.MATD3.matd3 import MATD3, MATD3Config, ReplayBuffer  # noqa: E402
from baselines.utils.microgrid_vec_env import MicrogridVecEnv  # noqa: E402
from baselines.utils.fixed_scenario_eval import (  # noqa: E402
    DEFAULT_NOISE_SEEDS,
    VALIDATION_DAYS,
    append_evaluation_record,
    build_scenarios,
    evaluate_policy,
)
from envs.microgrid.config import MICROGRID_CONFIG  # noqa: E402
from scripts.microgrid_experiment_overrides import MICROGRID_EXPERIMENT_OVERRIDES  # noqa: E402


def apply_microgrid_config_overrides(overrides: Dict[str, Any] | None) -> None:
    if overrides is None:
        overrides = MICROGRID_EXPERIMENT_OVERRIDES
    if overrides:
        MICROGRID_CONFIG.update(dict(overrides))
        print(f"Applied MICROGRID_CONFIG_OVERRIDES: {dict(overrides)}")


def _output_root() -> Path:
    root = os.environ.get("HYPERMARL_OUTPUT_DIR")
    if root:
        base = Path(root).expanduser()
        if not base.is_absolute():
            base = PROJECT_ROOT.parent / base
    else:
        base = PROJECT_ROOT.parent / "result" / "generated"
    base.mkdir(parents=True, exist_ok=True)
    return base


def _safe_alg_tag(alg: str) -> str:
    return alg.replace("/", "_").replace(" ", "_")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _publish_best_validation(
    root: Path, checkpoint: Path, episode: int, validation_return: float
) -> None:
    root.mkdir(parents=True, exist_ok=True)
    payload = {
        "episode": int(episode),
        "validation_return": float(validation_return),
        "checkpoint": str(checkpoint),
        "sha256": {"checkpoint": _sha256(checkpoint)},
    }
    destination = root / "best_validation.json"
    temporary = destination.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(destination)


def _state_from_obs(obs: np.ndarray) -> np.ndarray:
    return np.asarray(obs, dtype=np.float32).reshape(-1)


def _save_checkpoint(
    matd3: MATD3,
    checkpoint_dir: Path,
    episode: int,
    obs_dim: int,
    action_dim: int,
    state_dim: int,
    args: argparse.Namespace,
    replay: ReplayBuffer,
    global_step: int,
    episode_returns: List[float],
    loss_history: List[Dict[str, float]],
) -> Path:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    path = checkpoint_dir / f"matd3_episode_{episode}.pt"
    state = matd3.checkpoint_state()
    state.update(
        {
            "episode": int(episode),
            "obs_dim": int(obs_dim),
            "action_dim": int(action_dim),
            "state_dim": int(state_dim),
            "algorithm": args.alg,
            "seed": int(args.seed),
            "args": vars(args),
            "global_step": int(global_step),
            "replay_buffer": replay.state_dict(),
            "episode_returns": list(episode_returns),
            "loss_history": list(loss_history),
            "numpy_random_state": np.random.get_state(),
            "torch_rng_state": torch.get_rng_state(),
        }
    )
    if torch.cuda.is_available():
        state["torch_cuda_rng_state_all"] = torch.cuda.get_rng_state_all()
    torch.save(state, path)
    print(f"[checkpoint save] wrote {path}")
    return path


def train(args: argparse.Namespace) -> Dict[str, Any]:
    apply_microgrid_config_overrides(args.microgrid_overrides)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    env = MicrogridVecEnv(num_envs=1, auto_reset=True)
    obs_flat, _ = env.reset(seed=args.seed)
    num_agents = env.num_agents
    obs_dim = env.obs_dim
    action_dim = env.action_dim
    state_dim = obs_dim * num_agents

    config = MATD3Config(
        obs_dim=obs_dim,
        state_dim=state_dim,
        action_dim=action_dim,
        num_agents=num_agents,
        hidden_dim=args.hidden_dim,
        actor_lr=args.actor_lr,
        critic_lr=args.critic_lr,
        gamma=args.gamma,
        tau=args.tau,
        policy_noise=args.policy_noise,
        noise_clip=args.noise_clip,
        policy_delay=args.policy_delay,
        max_grad_norm=args.max_grad_norm,
    )
    matd3 = MATD3(config, device)
    replay = ReplayBuffer(
        obs_dim=obs_dim,
        state_dim=state_dim,
        action_dim=action_dim,
        num_agents=num_agents,
        capacity=args.replay_size,
        seed=args.seed,
    )

    out_root = _output_root()
    returns_dir = out_root / "returns"
    logs_dir = out_root / "logs"
    checkpoint_dir = out_root / "checkpoints" / _safe_alg_tag(args.alg)
    start_episode = 0
    global_step = 0
    episode_returns: List[float] = []
    loss_history: List[Dict[str, float]] = []
    if args.resume_checkpoint:
        checkpoint = torch.load(
            args.resume_checkpoint, map_location=device, weights_only=False
        )
        matd3.load_checkpoint_state(checkpoint)
        replay.load_state_dict(checkpoint["replay_buffer"])
        start_episode = int(checkpoint["episode"])
        global_step = int(checkpoint["global_step"])
        episode_returns = [float(value) for value in checkpoint.get("episode_returns", [])]
        loss_history = list(checkpoint.get("loss_history", []))
        if "numpy_random_state" in checkpoint:
            np.random.set_state(checkpoint["numpy_random_state"])
        if "torch_rng_state" in checkpoint:
            torch.set_rng_state(checkpoint["torch_rng_state"].cpu())
        if torch.cuda.is_available() and "torch_cuda_rng_state_all" in checkpoint:
            torch.cuda.set_rng_state_all(
                [state.cpu() for state in checkpoint["torch_cuda_rng_state_all"]]
            )
        print(
            f"[checkpoint load] loaded {args.resume_checkpoint} "
            f"(episode={start_episode}, global_step={global_step}, "
            f"total_it={checkpoint.get('total_it')}, replay={len(replay)})"
        )
    returns_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    alg_tag = _safe_alg_tag(args.alg)

    print(
        "MATD3 training: "
        f"episodes={args.episodes}, episode_length={args.episode_length}, "
        f"seed={args.seed}, device={device}, obs_dim={obs_dim}, action_dim={action_dim}"
    )

    checkpoint_paths: List[str] = []
    best_validation_root = (
        Path(args.best_validation_checkpoint_dir).expanduser()
        if args.best_validation_checkpoint_dir
        else None
    )
    best_validation_return = float("-inf")
    if best_validation_root is not None:
        metadata = best_validation_root / "best_validation.json"
        if metadata.is_file():
            best_validation_return = float(
                json.loads(metadata.read_text(encoding="utf-8"))["validation_return"]
            )

    obs = obs_flat.reshape(num_agents, obs_dim)
    for episode in range(start_episode + 1, args.episodes + 1):
        if episode > 1:
            obs_flat, _ = env.reset(seed=args.seed + episode - 1)
            obs = obs_flat.reshape(num_agents, obs_dim)
        episode_return = 0.0

        for _ in range(args.episode_length):
            state = _state_from_obs(obs)
            if global_step < args.start_steps:
                action = np.random.uniform(-1.0, 1.0, size=(num_agents, action_dim)).astype(
                    np.float32
                )
            else:
                action = matd3.select_action(obs, noise_std=args.exploration_noise)

            next_obs_flat, reward_flat, term, trunc, _ = env.step(action)
            next_obs = next_obs_flat.reshape(num_agents, obs_dim)
            reward = float(np.mean(reward_flat))
            done = bool(np.any(term) or np.any(trunc))
            replay.add(obs, state, action, reward, next_obs, _state_from_obs(next_obs), done)

            if should_update(
                global_step,
                replay_size=len(replay),
                batch_size=args.batch_size,
                update_after=args.update_after,
                update_every=args.update_every,
            ):
                for _update_idx in range(args.updates_per_step):
                    losses = matd3.update(replay, args.batch_size)
                    loss_history.append({"step": float(global_step), **losses})

            obs = next_obs
            episode_return += reward
            global_step += 1
            if done:
                break

        episode_returns.append(episode_return)
        if episode % args.log_interval == 0 or episode == 1 or episode == args.episodes:
            recent = np.asarray(episode_returns[-min(len(episode_returns), 100) :])
            print(
                f"Episode {episode}/{args.episodes} return={episode_return:.3f} "
                f"recent100={recent.mean():.3f} buffer={len(replay)}"
            )

        if args.fixed_eval_output and episode % args.eval_interval_episodes == 0:
            if args.fixed_eval_split == "validation":
                fixed_eval_days = VALIDATION_DAYS
                default_eval_seed = DEFAULT_NOISE_SEEDS[0]
            elif args.fixed_eval_split == "test":
                from baselines.utils.fixed_scenario_eval import (
                    TEST_DAYS,
                    TEST_NOISE_SEEDS,
                )

                fixed_eval_days = TEST_DAYS
                default_eval_seed = TEST_NOISE_SEEDS[0]
            else:
                raise ValueError("fixed eval split must be 'validation' or 'test'")
            fixed_eval_seed = (
                default_eval_seed
                if args.fixed_eval_noise_seed is None
                else int(args.fixed_eval_noise_seed)
            )
            eval_data = evaluate_policy(
                lambda eval_obs: matd3.select_action(eval_obs, noise_std=0.0),
                args.microgrid_overrides or {},
                build_scenarios(fixed_eval_days, (fixed_eval_seed,)),
                algorithm=args.alg,
                split_name=args.fixed_eval_split,
            )
            append_evaluation_record(
                args.fixed_eval_output,
                eval_data,
                training_episode=episode,
            )
            validation_return = float(eval_data["summary"]["return_mean"])
            if (
                args.fixed_eval_split == "validation"
                and best_validation_root is not None
                and validation_return > best_validation_return
            ):
                best_path = _save_checkpoint(
                    matd3,
                    best_validation_root / f"episode_{episode:05d}",
                    episode,
                    obs_dim,
                    action_dim,
                    state_dim,
                    args,
                    replay,
                    global_step,
                    episode_returns,
                    loss_history,
                )
                _publish_best_validation(
                    best_validation_root, best_path, episode, validation_return
                )
                best_validation_return = validation_return

        if (
            args.checkpoint_interval > 0
            and episode % args.checkpoint_interval == 0
        ) or episode == args.episodes:
            path = _save_checkpoint(
                matd3,
                checkpoint_dir,
                episode,
                obs_dim,
                action_dim,
                state_dim,
                args,
                replay,
                global_step,
                episode_returns,
                loss_history,
            )
            checkpoint_paths.append(str(path))

    returns = np.asarray(episode_returns, dtype=np.float64)
    returns_path = returns_dir / f"returns_microgrid_{alg_tag}.npy"
    np.save(returns_path, returns)
    losses_path = logs_dir / f"losses_microgrid_{alg_tag}.jsonl"
    with losses_path.open("w", encoding="utf-8") as f:
        for row in loss_history:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {
        "algorithm": args.alg,
        "seed": args.seed,
        "episodes": args.episodes,
        "episode_length": args.episode_length,
        "device": str(device),
        "overall_mean": float(np.mean(returns)),
        "final_100_mean": float(np.mean(returns[-min(len(returns), 100) :])),
        "final_500_mean": float(np.mean(returns[-min(len(returns), 500) :])),
        "returns_npy": str(returns_path),
        "losses_jsonl": str(losses_path),
        "checkpoints": checkpoint_paths,
    }
    summary_path = logs_dir / f"summary_microgrid_{alg_tag}.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"[metrics save] wrote {returns_path}, {losses_path}, {summary_path}")
    env.close()
    return summary


def parse_overrides(raw: str | None) -> Dict[str, Any] | None:
    if not raw:
        return None
    return json.loads(raw.replace("'", '"'))


def should_update(
    global_step: int,
    *,
    replay_size: int,
    batch_size: int,
    update_after: int,
    update_every: int,
) -> bool:
    if update_every < 1:
        raise ValueError("update_every must be >= 1")
    return bool(
        replay_size >= batch_size
        and global_step >= update_after
        and global_step % update_every == 0
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=30)
    parser.add_argument("--episodes", type=int, default=10000)
    parser.add_argument("--episode-length", type=int, default=24)
    parser.add_argument("--alg", type=str, default="MATD3-10kEp")
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--actor-lr", type=float, default=1e-3)
    parser.add_argument("--critic-lr", type=float, default=1e-3)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--policy-noise", type=float, default=0.2)
    parser.add_argument("--noise-clip", type=float, default=0.5)
    parser.add_argument("--policy-delay", type=int, default=2)
    parser.add_argument("--exploration-noise", type=float, default=0.1)
    parser.add_argument("--replay-size", type=int, default=200000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--fixed-eval-output", type=str, default="")
    parser.add_argument(
        "--fixed-eval-split",
        choices=("validation", "test"),
        default="validation",
    )
    parser.add_argument("--fixed-eval-noise-seed", type=int, default=None)
    parser.add_argument("--best-validation-checkpoint-dir", type=str, default="")
    parser.add_argument("--eval-interval-episodes", type=int, default=500)
    parser.add_argument("--start-steps", type=int, default=1000)
    parser.add_argument("--update-after", type=int, default=1000)
    parser.add_argument("--updates-per-step", type=int, default=1)
    parser.add_argument("--update-every", type=int, default=1)
    parser.add_argument("--checkpoint-interval", type=int, default=1000)
    parser.add_argument("--resume-checkpoint", type=str, default="")
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument("--max-grad-norm", type=float, default=10.0)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--microgrid-overrides-json", type=str, default="")
    args = parser.parse_args()
    args.microgrid_overrides = MICROGRID_EXPERIMENT_OVERRIDES
    if args.microgrid_overrides_json:
        args.microgrid_overrides = parse_overrides(args.microgrid_overrides_json)
    train(args)


if __name__ == "__main__":
    main()
