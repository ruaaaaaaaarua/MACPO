#!/usr/bin/env python3
"""One-shot locked-test evaluation after sparse model selection is complete."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Callable

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def selected_best_checkpoints(
    train_root: Path, *, verify_hashes: bool = True
) -> dict[str, Path]:
    train_root = Path(train_root)
    selected = {}
    for directory, label, key in (
        ("stas", "STAS", "jax_checkpoint"),
        ("mappo", "MAPPO", "jax_checkpoint"),
        ("matd3", "MATD3", "checkpoint"),
    ):
        metadata_path = (
            train_root
            / directory
            / "output"
            / "checkpoints"
            / "best_validation"
            / "best_validation.json"
        )
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        checkpoint = Path(metadata[key])
        if not checkpoint.is_file():
            raise FileNotFoundError(f"selected {label} checkpoint is missing: {checkpoint}")
        if verify_hashes:
            expected = metadata.get("sha256", {}).get(key)
            if not expected or _sha256(checkpoint) != expected:
                raise ValueError(f"selected {label} checkpoint hash mismatch")
        selected[label] = checkpoint
    return selected


def _jax_action_fn(checkpoint: Path, overrides: dict) -> Callable:
    import jax
    import jax.numpy as jnp
    import optax
    from flax.training.train_state import TrainState

    from baselines.MAPPO.continuous_policy import deterministic_action
    from baselines.MAPPO.mappo_ff_shared_weights import ActorCritic
    from baselines.utils.microgrid_vec_env import MicrogridVecEnv
    from baselines.utils.training_checkpoint import load_jax_training_checkpoint

    env = MicrogridVecEnv(num_envs=1, auto_reset=True, config_overrides=overrides)
    num_agents = env.num_agents
    obs_dim = env.obs_dim
    action_dim = env.action_dim
    env.close()
    network = ActorCritic(
        action_dim,
        activation="relu",
        actor_layers=[256, 256],
        critic_layers=[256, 256],
        num_agents=num_agents,
        observation_dim=obs_dim,
        is_continuous=True,
        log_std_init=-1.0,
        log_std_min=-2.5,
        log_std_max=-0.5,
    )
    actor_template = jnp.zeros((num_agents, obs_dim + num_agents), dtype=jnp.float32)
    critic_template = jnp.zeros((num_agents, obs_dim * num_agents), dtype=jnp.float32)
    params = network.init(jax.random.PRNGKey(0), actor_template, critic_template)
    tx = optax.chain(
        optax.clip_by_global_norm(5.0), optax.adam(3e-4, eps=1e-5)
    )
    template = TrainState.create(apply_fn=network.apply, params=params, tx=tx)
    trained = load_jax_training_checkpoint(checkpoint, template).train_state.params
    identities = jnp.eye(num_agents, dtype=jnp.float32)
    dummy_critic = jnp.zeros((num_agents, obs_dim * num_agents), dtype=jnp.float32)

    @jax.jit
    def act(obs):
        actor_obs = jnp.concatenate([obs, identities], axis=-1)
        actor_output, _ = network.apply(trained, actor_obs, dummy_critic)
        mean, _ = actor_output
        return deterministic_action(mean)

    return lambda obs: np.asarray(act(jnp.asarray(obs, dtype=jnp.float32)))


def _matd3_action_fn(checkpoint: Path) -> Callable:
    import torch

    from baselines.MATD3.matd3 import MATD3, MATD3Config

    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = MATD3(MATD3Config(**state["config"]), torch.device("cpu"))
    model.load_checkpoint_state(state)
    return lambda obs: model.select_action(obs, noise_std=0.0)


def _records_by_name(path: Path) -> dict[str, dict]:
    records = [json.loads(line) for line in path.read_text().splitlines() if line]
    return {record["result_name"]: record for record in records}


def _mechanism_gate(records: dict[str, dict]) -> dict:
    normal = records["STAS"]
    no_order = records["STAS__forced_no_order"]
    direct = records["STAS__forced_direct_route"]
    permuted = records["STAS__permuted_route"]

    def returns(record):
        return {int(row["day"]): float(row["return"]) for row in record["episodes"]}

    normal_days = returns(normal)

    def comparison(other):
        other_days = returns(other)
        deltas = {
            day: normal_days[day] - other_days[day] for day in sorted(normal_days)
        }
        return {
            "mean_return_delta": float(
                normal["summary"]["return_mean"] - other["summary"]["return_mean"]
            ),
            "days_normal_better": int(sum(value > 0.0 for value in deltas.values())),
            "per_day_return_delta": deltas,
        }

    versus_no_order = comparison(no_order)
    versus_direct = comparison(direct)
    versus_permuted = comparison(permuted)
    mechanism_success = bool(
        versus_no_order["mean_return_delta"] > 0.0
        and versus_permuted["mean_return_delta"] > 0.0
        and versus_no_order["days_normal_better"] >= 3
        and versus_permuted["days_normal_better"] >= 3
    )
    route_interpretation = "no buyer-route mechanism evidence"
    if versus_permuted["mean_return_delta"] > 0.0:
        route_interpretation = (
            "learned buyer-related routing and beat direct heuristic"
            if versus_direct["mean_return_delta"] > 0.0
            else "learned buyer-related routing but did not beat direct heuristic"
        )
    return {
        "mechanism_success": mechanism_success,
        "normal_minus_no_order": versus_no_order,
        "normal_minus_forced_direct": versus_direct,
        "normal_minus_permuted_route": versus_permuted,
        "route_interpretation": route_interpretation,
    }


def run(train_root: Path) -> dict:
    train_root = Path(train_root).resolve()
    output = train_root / "final_test"
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"locked test outputs already exist: {output}")
    selected = selected_best_checkpoints(train_root)
    manifest_path = train_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("test_accessed"):
        raise RuntimeError("manifest says locked test split was already accessed")
    overrides = dict(manifest["environment_overrides"])

    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    policies = {
        "MAPPO": _jax_action_fn(selected["MAPPO"], overrides),
        "STAS": _jax_action_fn(selected["STAS"], overrides),
        "MATD3": _matd3_action_fn(selected["MATD3"]),
    }
    from baselines.utils.final_comparison import run_final_comparison

    summary = run_final_comparison(
        policies,
        overrides,
        output,
        algorithm_names={
            "MAPPO": "Sparse-Terminal-MAPPO",
            "STAS": "Sparse-Terminal-Paper-STAS",
            "MATD3": "Sparse-Terminal-MATD3",
        },
        training_episode=max(
            int(
                json.loads(
                    (
                        train_root
                        / name
                        / "output/checkpoints/best_validation/best_validation.json"
                    ).read_text(encoding="utf-8")
                )["episode"]
            )
            for name in ("stas", "mappo", "matd3")
        ),
    )
    mechanism = _mechanism_gate(
        _records_by_name(output / "final_comparison.jsonl")
    )
    (output / "stas_mechanism_gate.json").write_text(
        json.dumps(mechanism, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    selection = {
        name: str(path) for name, path in selected.items()
    }
    (output / "selected_best_checkpoints.json").write_text(
        json.dumps(selection, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    manifest["test_accessed"] = True
    manifest["test_completed_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    manifest["selected_best_checkpoints"] = selection
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output / "FINAL_TEST_COMPLETE").write_text("complete\n", encoding="utf-8")
    from scripts.run_paper_stas_sparse import _write_hashes

    _write_hashes(train_root)
    return {"summary": summary, "mechanism": mechanism}


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False


def wait_and_run(train_root: Path, controller_pid: int | None, poll_seconds: int) -> dict:
    completion = Path(train_root) / "sha256.json"
    while not completion.is_file():
        if controller_pid is not None and not _pid_alive(controller_pid):
            raise RuntimeError("training controller exited without completion manifest")
        time.sleep(max(1, int(poll_seconds)))
    return run(train_root)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-root", type=Path, required=True)
    parser.add_argument("--wait", action="store_true")
    parser.add_argument("--controller-pid", type=int)
    parser.add_argument("--poll-seconds", type=int, default=30)
    args = parser.parse_args()
    if args.wait:
        wait_and_run(args.train_root, args.controller_pid, args.poll_seconds)
    else:
        run(args.train_root)


if __name__ == "__main__":
    main()
