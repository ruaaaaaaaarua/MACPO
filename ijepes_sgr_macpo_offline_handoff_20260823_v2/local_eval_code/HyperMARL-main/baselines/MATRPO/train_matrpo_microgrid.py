#!/usr/bin/env python3
"""MATRPO training entry for the in-repo microgrid environment."""

from __future__ import annotations

import argparse
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

from baselines.MATRPO.matrpo import MATRPO  # noqa: E402
from baselines.MATRPO.networks import Actor, Critic  # noqa: E402
from baselines.utils.experiment_progress import EpisodeProgressLogger  # noqa: E402
from baselines.utils.microgrid_vec_env import MicrogridVecEnv  # noqa: E402
from envs.microgrid.config import MICROGRID_CONFIG  # noqa: E402
from scripts.microgrid_experiment_overrides import MICROGRID_EXPERIMENT_OVERRIDES  # noqa: E402


def apply_microgrid_config_overrides(overrides: Dict[str, Any] | None) -> None:
    if overrides is None:
        overrides = MICROGRID_EXPERIMENT_OVERRIDES
    if overrides:
        MICROGRID_CONFIG.update(dict(overrides))
        print(f"Applied MICROGRID_CONFIG_OVERRIDES: {dict(overrides)}")


def _output_dir() -> Path:
    root = os.environ.get("HYPERMARL_OUTPUT_DIR")
    if root:
        base = Path(root).expanduser()
        if not base.is_absolute():
            base = PROJECT_ROOT.parent / base
    else:
        base = PROJECT_ROOT.parent / "result" / "generated"
    out = base / "returns"
    out.mkdir(parents=True, exist_ok=True)
    return out


def _collect_rollout(
    env: MicrogridVecEnv,
    actors: List[Actor],
    device: torch.device,
    num_steps: int,
    obs: np.ndarray | None = None,
) -> tuple[Dict[str, Any], float, np.ndarray]:
    num_agents = env.num_agents
    obs_dim = env.obs_dim

    if obs is None:
        obs_flat, _ = env.reset()
        obs = obs_flat.reshape(1, num_agents, obs_dim)

    obs_buf = [[] for _ in range(num_agents)]
    act_buf = [[] for _ in range(num_agents)]
    logp_buf = [[] for _ in range(num_agents)]
    states: List[np.ndarray] = []
    rewards: List[float] = []
    dones: List[float] = []
    next_states: List[np.ndarray] = []

    for _ in range(num_steps):
        state = np.concatenate(obs[0], axis=0).astype(np.float32)
        states.append(state)

        actions = []
        for agent_idx in range(num_agents):
            obs_t = torch.as_tensor(
                obs[0, agent_idx], dtype=torch.float32, device=device
            ).unsqueeze(0)
            dist = actors[agent_idx](obs_t)
            action = dist.sample()
            log_prob = dist.log_prob(action).sum(dim=-1)
            obs_buf[agent_idx].append(obs[0, agent_idx].copy())
            act_buf[agent_idx].append(action.squeeze(0).detach().cpu().numpy())
            logp_buf[agent_idx].append(float(log_prob.item()))
            actions.append(action.squeeze(0).detach().cpu().numpy())

        next_obs_flat, rew_flat, term, trunc, _ = env.step(np.asarray(actions))
        next_obs = next_obs_flat.reshape(1, num_agents, obs_dim)
        done = bool(np.any(term) or np.any(trunc))
        team_reward = float(np.mean(rew_flat))

        rewards.append(team_reward)
        dones.append(float(done))
        next_states.append(np.concatenate(next_obs[0], axis=0).astype(np.float32))
        obs = next_obs

    rollout = {
        "obs": [
            torch.as_tensor(np.asarray(obs_buf[i]), dtype=torch.float32, device=device)
            for i in range(num_agents)
        ],
        "actions": [
            torch.as_tensor(np.asarray(act_buf[i]), dtype=torch.float32, device=device)
            for i in range(num_agents)
        ],
        "log_probs": [
            torch.as_tensor(np.asarray(logp_buf[i]), dtype=torch.float32, device=device)
            for i in range(num_agents)
        ],
        "states": torch.as_tensor(np.asarray(states), dtype=torch.float32, device=device),
        "rewards": torch.as_tensor(np.asarray(rewards), dtype=torch.float32, device=device),
        "dones": torch.as_tensor(np.asarray(dones), dtype=torch.float32, device=device),
        "next_states": torch.as_tensor(
            np.asarray(next_states), dtype=torch.float32, device=device
        ),
    }
    episode_return = float(np.sum(rollout["rewards"].detach().cpu().numpy()))
    return rollout, episode_return, obs


def train(args: argparse.Namespace) -> None:
    apply_microgrid_config_overrides(args.microgrid_overrides)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    env = MicrogridVecEnv(num_envs=1, auto_reset=True)
    num_agents = env.num_agents
    obs_dim = env.obs_dim
    action_dim = env.action_dim
    state_dim = obs_dim * num_agents

    actors = [Actor(obs_dim, action_dim, hidden_dim=128).to(device) for _ in range(num_agents)]
    critic = Critic(state_dim, hidden_dim=128).to(device)
    matrpo = MATRPO(
        actors=actors,
        critic=critic,
        critic_lr=1e-3,
        max_kl=0.01,
        gamma=0.99,
        gae_lambda=0.95,
    )

    num_episodes = args.total_timesteps // args.episode_length
    episode_returns: List[float] = []
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    progress = EpisodeProgressLogger(
        log_path=Path(args.log_dir) / "progress_matrpo.jsonl",
        report_every=500,
        algorithm="MATRPO",
    )

    print(
        f"MATRPO training: episodes={num_episodes}, episode_length={args.episode_length}, "
        f"total_timesteps={args.total_timesteps}, seed={args.seed}, device={device}"
    )

    obs_state = None
    for episode_idx in range(num_episodes):
        rollout, episode_return, obs_state = _collect_rollout(
            env, actors, device, num_steps=args.episode_length, obs=obs_state
        )
        matrpo.update(rollout)
        episode_returns.append(episode_return)
        progress.record_episode(episode_idx + 1, episode_return)

        if episode_idx % 100 == 0 or episode_idx + 1 == num_episodes:
            print(
                f"Episode {episode_idx + 1}/{num_episodes} "
                f"return={episode_return:.2f}"
            )

    progress.finalize()
    out_dir = _output_dir()
    alg_tag = args.alg.replace("/", "_").replace(" ", "_")
    npy_path = out_dir / f"returns_microgrid_{alg_tag}.npy"
    np.save(npy_path, np.asarray(episode_returns, dtype=np.float64))
    print(f"[metrics save] wrote {npy_path}")

    actors_dir = Path(args.save_actors_dir) if args.save_actors_dir else out_dir.parent / "actors"
    actors_dir.mkdir(parents=True, exist_ok=True)
    for idx, actor in enumerate(actors):
        actor_path = actors_dir / f"matrpo_actor_agent{idx}.pt"
        torch.save(
            {
                "state_dict": actor.state_dict(),
                "obs_dim": obs_dim,
                "action_dim": action_dim,
                "hidden_dim": 128,
                "seed": args.seed,
                "algorithm": args.alg,
                "episodes": num_episodes,
            },
            actor_path,
        )
        print(f"[actor save] wrote {actor_path}")

    summary = {
        "algorithm": args.alg,
        "episodes": len(episode_returns),
        "overall_mean": float(np.mean(episode_returns)),
        "final_500_mean": float(np.mean(episode_returns[-500:])),
        "returns_npy": str(npy_path),
    }
    with (log_dir / "matrpo_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    env.close()


def parse_overrides(raw: str | None) -> Dict[str, Any] | None:
    if not raw:
        return None
    return json.loads(raw.replace("'", '"'))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=30)
    parser.add_argument("--total-timesteps", type=int, default=120000)
    parser.add_argument("--episode-length", type=int, default=24)
    parser.add_argument("--alg", type=str, default="MATRPO-24h-FullCDA-ReserveDemand-5kEp")
    parser.add_argument("--log-dir", type=str, default="")
    parser.add_argument("--save-actors-dir", type=str, default="")
    parser.add_argument("--microgrid-overrides-json", type=str, default="")
    args = parser.parse_args()

    if not args.log_dir:
        root = os.environ.get(
            "HYPERMARL_OUTPUT_DIR", str(PROJECT_ROOT.parent / "result/generated")
        )
        args.log_dir = str(Path(root) / "logs")

    args.microgrid_overrides = MICROGRID_EXPERIMENT_OVERRIDES
    if args.microgrid_overrides_json:
        args.microgrid_overrides = parse_overrides(args.microgrid_overrides_json)

    train(args)


if __name__ == "__main__":
    main()
