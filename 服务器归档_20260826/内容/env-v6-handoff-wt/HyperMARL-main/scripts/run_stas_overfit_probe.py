#!/usr/bin/env python3
"""Run the audited 32-episode STAS reward-model capacity gate."""

from __future__ import annotations

import argparse
import copy
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import random
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Sequence

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
STAS_ROOT = ROOT / "baselines" / "STAS-MAPPO"
for candidate in (ROOT, STAS_ROOT):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from baselines.utils.microgrid_vec_env import MicrogridVecEnv  # noqa: E402
from scripts.run_stas_mechanism_ablation import planned_experiments  # noqa: E402
from stas_mappo.conserved_credit import ConservedSTASCreditAssigner  # noqa: E402
from stas_mappo.credit import EpisodeCreditBuffer, STASCreditConfig  # noqa: E402
from stas_mappo.diagnostics import compute_target_error  # noqa: E402


SCHEMA_VERSION = 1
LOCKED_SEED = 30
LOCKED_SAMPLES = 32
LOCKED_SEQUENCE_LENGTH = 24
LOCKED_MAX_UPDATES = 5000
LOCKED_MSE_THRESHOLD = 1e-3
LOCKED_EV_THRESHOLD = 0.95
EV_REPEATS = 3


@dataclass(frozen=True)
class ProbeDataset:
    obs: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    dones: np.ndarray

    @property
    def sample_count(self) -> int:
        return int(self.obs.shape[0])


def _group_ab_overrides() -> dict[str, Any]:
    for spec in planned_experiments():
        if spec.group == "group_ab":
            overrides = copy.deepcopy(spec.env_overrides)
            break
    else:
        raise RuntimeError("frozen A+B environment specification is unavailable")
    if overrides.get("italian_split_name") != "train":
        raise ValueError("capacity probe must use the training split")
    if "italian_day_indices" in overrides:
        raise ValueError("capacity probe must not select fixed validation/test days")
    return overrides


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def collect_probe_dataset(
    *, seed: int, samples: int, sequence_length: int = LOCKED_SEQUENCE_LENGTH
) -> tuple[ProbeDataset, dict[str, int], dict[str, Any]]:
    """Collect deterministic full episodes from the real frozen A+B environment."""
    overrides = _group_ab_overrides()
    env = MicrogridVecEnv(
        num_envs=1,
        auto_reset=False,
        config_overrides=overrides,
    )
    rng = np.random.default_rng(seed)
    observations: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    rewards: list[np.ndarray] = []
    dones: list[np.ndarray] = []
    try:
        dimensions = {
            "n_agents": int(env.num_agents),
            "obs_dim": int(env.obs_dim),
            "action_dim": int(env.action_dim),
        }
        for sample_index in range(samples):
            obs_flat, _ = env.reset(seed=seed + sample_index)
            sample_obs: list[np.ndarray] = []
            sample_actions: list[np.ndarray] = []
            sample_rewards: list[np.ndarray] = []
            sample_dones: list[np.ndarray] = []
            for step in range(sequence_length):
                obs = np.asarray(obs_flat, dtype=np.float32).reshape(
                    dimensions["n_agents"], dimensions["obs_dim"]
                )
                action = rng.uniform(
                    -1.0,
                    1.0,
                    size=(dimensions["n_agents"], dimensions["action_dim"]),
                ).astype(np.float32)
                next_obs, reward, terms, truncs, _ = env.step(action)
                done = np.logical_or(terms, truncs).reshape(dimensions["n_agents"])
                if step < sequence_length - 1 and bool(np.any(done)):
                    raise RuntimeError(
                        f"A+B episode {sample_index} ended before {sequence_length} steps"
                    )
                sample_obs.append(obs)
                sample_actions.append(action)
                sample_rewards.append(
                    np.asarray(reward, dtype=np.float32).reshape(dimensions["n_agents"])
                )
                sample_dones.append(np.asarray(done, dtype=np.float32))
                obs_flat = next_obs
            if not bool(np.any(sample_dones[-1])):
                raise RuntimeError(
                    f"A+B episode {sample_index} did not terminate at step {sequence_length}"
                )
            observations.append(np.stack(sample_obs, axis=1))
            actions.append(np.stack(sample_actions, axis=1))
            rewards.append(np.stack(sample_rewards, axis=1))
            dones.append(np.stack(sample_dones, axis=1))
    finally:
        env.close()
    dataset = ProbeDataset(
        obs=np.stack(observations).astype(np.float32),
        actions=np.stack(actions).astype(np.float32),
        rewards=np.stack(rewards).astype(np.float32),
        dones=np.stack(dones).astype(np.float32),
    )
    return dataset, dimensions, overrides


def _validate_probe_dataset(
    dataset: ProbeDataset, dimensions: dict[str, int]
) -> None:
    prefix = (
        LOCKED_SAMPLES,
        int(dimensions["n_agents"]),
        LOCKED_SEQUENCE_LENGTH,
    )
    expected = {
        "obs": prefix + (int(dimensions["obs_dim"]),),
        "actions": prefix + (int(dimensions["action_dim"]),),
        "rewards": prefix,
        "dones": prefix,
    }
    actual = {name: tuple(getattr(dataset, name).shape) for name in expected}
    if actual != expected:
        raise ValueError(
            f"probe dataset must contain exactly 32 samples of length 24; got {actual}"
        )


def build_credit_config(
    *,
    obs_dim: int,
    action_dim: int,
    n_agents: int,
    seq_length: int,
    device: str,
) -> STASCreditConfig:
    return STASCreditConfig(
        obs_dim=obs_dim,
        action_dim=action_dim,
        n_agents=n_agents,
        seq_length=seq_length,
        gamma=0.99,
        lr=1e-3,
        emb_dim=128,
        n_heads=4,
        n_layers=2,
        sample_num=4,
        dropout=0.0,
        eval_mask_seed=3030,
        eval_mask_count=8,
        buffer_size=LOCKED_SAMPLES,
        batch_size=LOCKED_SAMPLES,
        update_freq=1,
        updates_per_step=1,
        warmup_rollouts=1,
        global_reward_agg="sum",
        device=device,
        causal=True,
        conserve_discounted=True,
        quality_gate_enable=True,
        warmup_episodes=200,
        ramp_episodes=800,
        max_mix_coef=0.1,
        explained_variance_threshold=0.2,
    )


def route_memorization_dataset(
    assigner: ConservedSTASCreditAssigner,
    dataset: ProbeDataset,
) -> float:
    """Create production targets, then make independent train/eval copies."""
    assigner.add_rollout(
        dataset.obs,
        dataset.actions,
        dataset.rewards,
        dataset.dones,
    )
    if assigner.episodes_seen != dataset.sample_count:
        raise RuntimeError("production target path did not consume every probe sample")

    production_items = [
        *assigner.buffer.storage,
        *assigner.holdout_buffer.storage,
    ]
    if len(production_items) != dataset.sample_count:
        raise RuntimeError("production buffers did not retain every probe target")
    targets_by_reward = {
        np.asarray(item[2], dtype=np.float32).tobytes(order="C"): float(item[4])
        for item in production_items
    }
    source = EpisodeCreditBuffer(dataset.sample_count)
    for index in range(dataset.sample_count):
        reward_key = dataset.rewards[index].tobytes(order="C")
        if reward_key not in targets_by_reward:
            raise RuntimeError("a production target could not be matched to its sample")
        source.add(
            dataset.obs[index],
            dataset.actions[index],
            dataset.rewards[index],
            dataset.dones[index],
            targets_by_reward[reward_key],
        )

    assigner.buffer = copy.deepcopy(source)
    assigner.holdout_buffer = copy.deepcopy(source)
    return compute_target_error(
        (assigner.buffer, assigner.holdout_buffer), assigner.config.gamma
    )


def credit_buffer_sha256(buffer: EpisodeCreditBuffer) -> str:
    digest = hashlib.sha256()
    digest.update(f"stas-probe-buffer-v1:{len(buffer)}".encode("utf-8"))
    for item in buffer.storage:
        for array in item[:4]:
            value = np.asarray(array)
            digest.update(str(value.dtype).encode("ascii"))
            digest.update(json.dumps(value.shape).encode("ascii"))
            digest.update(np.ascontiguousarray(value).tobytes(order="C"))
        digest.update(np.asarray(float(item[4]), dtype="<f8").tobytes())
    return digest.hexdigest()


@torch.no_grad()
def measure_probe(assigner: ConservedSTASCreditAssigner) -> dict[str, Any]:
    assigner.model.eval()
    obs, actions, rewards, dones, targets = assigner.holdout_buffer.get_all()
    _, predicted_normalized = assigner._predict_normalized_returns(
        obs, actions, rewards, dones
    )
    predicted_normalized_np = predicted_normalized.detach().cpu().numpy()
    normalized_targets = assigner.normalizer.normalize(targets)
    normalized_mse = float(
        np.mean(np.square(predicted_normalized_np - normalized_targets))
    )
    predicted_raw = predicted_normalized_np * assigner.normalizer.std
    predicted_raw += assigner.normalizer.mean
    raw_mse = float(np.mean(np.square(predicted_raw - targets)))
    ev_values = [
        float(assigner.holdout_explained_variance()) for _ in range(EV_REPEATS)
    ]
    return {
        "normalized_mse": normalized_mse,
        "raw_mse": raw_mse,
        "ev_values": ev_values,
    }


def evaluate_probe_gate(
    *,
    sample_count: int,
    normalized_mse: float,
    raw_mse: float,
    ev_values: Sequence[float],
    target_error: float,
    mse_threshold: float,
    ev_threshold: float,
) -> dict[str, Any]:
    values = [normalized_mse, raw_mse, target_error, *ev_values]
    finite = all(math.isfinite(float(value)) for value in values)
    ev_range = (
        float(max(ev_values) - min(ev_values))
        if ev_values and all(math.isfinite(float(value)) for value in ev_values)
        else None
    )
    assertions = {
        "sample_count_exactly_32": sample_count == LOCKED_SAMPLES,
        "numeric_diagnostics_finite": finite,
        "normalized_mse_within_threshold": bool(
            finite and normalized_mse <= mse_threshold
        ),
        "all_final_ev_within_threshold": bool(
            len(ev_values) == EV_REPEATS
            and all(math.isfinite(float(value)) and value >= ev_threshold for value in ev_values)
        ),
        "deterministic_ev_range": bool(ev_range is not None and ev_range <= 1e-6),
        "target_consistency": bool(
            math.isfinite(float(target_error)) and target_error <= 1e-6
        ),
    }
    return {
        "assertions": assertions,
        "ev_range": ev_range,
        "passed": all(assertions.values()),
    }


def train_probe(
    assigner: ConservedSTASCreditAssigner,
    *,
    max_updates: int,
    mse_threshold: float,
    ev_threshold: float,
    target_error: float,
) -> tuple[int, float, dict[str, Any], dict[str, Any]]:
    last_loss = float("nan")
    metrics: dict[str, Any] | None = None
    gate: dict[str, Any] | None = None
    for update in range(1, max_updates + 1):
        last_loss = float(assigner.train_if_ready())
        if not math.isfinite(last_loss):
            raise RuntimeError(f"non-finite reward-model loss at update {update}")
        if update % 25 != 0 and update != max_updates:
            continue
        metrics = measure_probe(assigner)
        gate = evaluate_probe_gate(
            sample_count=len(assigner.holdout_buffer),
            normalized_mse=metrics["normalized_mse"],
            raw_mse=metrics["raw_mse"],
            ev_values=metrics["ev_values"],
            target_error=target_error,
            mse_threshold=mse_threshold,
            ev_threshold=ev_threshold,
        )
        if gate["passed"]:
            return update, last_loss, metrics, gate
    if metrics is None or gate is None:
        metrics = measure_probe(assigner)
        gate = evaluate_probe_gate(
            sample_count=len(assigner.holdout_buffer),
            normalized_mse=metrics["normalized_mse"],
            raw_mse=metrics["raw_mse"],
            ev_values=metrics["ev_values"],
            target_error=target_error,
            mse_threshold=mse_threshold,
            ev_threshold=ev_threshold,
        )
    return max_updates, last_loss, metrics, gate


def _source_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _validate_args(args: argparse.Namespace) -> None:
    locked = {
        "seed": (args.seed, LOCKED_SEED),
        "samples": (args.samples, LOCKED_SAMPLES),
        "max_updates": (args.max_updates, LOCKED_MAX_UPDATES),
        "mse_threshold": (args.mse_threshold, LOCKED_MSE_THRESHOLD),
        "ev_threshold": (args.ev_threshold, LOCKED_EV_THRESHOLD),
    }
    mismatches = [
        f"{name}={actual!r} (required {expected!r})"
        for name, (actual, expected) in locked.items()
        if actual != expected
    ]
    if mismatches:
        raise ValueError("locked probe arguments changed: " + ", ".join(mismatches))


def run(args: argparse.Namespace) -> dict[str, Any]:
    _validate_args(args)
    started = time.monotonic()
    seed_everything(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataset, dimensions, env_overrides = collect_probe_dataset(
        seed=args.seed,
        samples=args.samples,
        sequence_length=LOCKED_SEQUENCE_LENGTH,
    )
    _validate_probe_dataset(dataset, dimensions)
    config = build_credit_config(
        **dimensions,
        seq_length=LOCKED_SEQUENCE_LENGTH,
        device=device,
    )
    assigner = ConservedSTASCreditAssigner(config)
    target_error = route_memorization_dataset(assigner, dataset)
    dataset_digest = credit_buffer_sha256(assigner.buffer)
    update_count, last_loss, metrics, gate = train_probe(
        assigner,
        max_updates=args.max_updates,
        mse_threshold=args.mse_threshold,
        ev_threshold=args.ev_threshold,
        target_error=target_error,
    )
    elapsed = float(time.monotonic() - started)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "seed": args.seed,
        "sample_count": dataset.sample_count,
        "dataset_sha256": dataset_digest,
        "source_commit": _source_commit(),
        "resolved_config": {
            "credit": asdict(config),
            "environment": env_overrides,
            "dimensions": dimensions,
            "sequence_length": LOCKED_SEQUENCE_LENGTH,
            "max_updates": args.max_updates,
            "mse_threshold": args.mse_threshold,
            "ev_threshold": args.ev_threshold,
            "evaluation_repeats": EV_REPEATS,
        },
        "command": [sys.executable, *sys.argv],
        "update_count": update_count,
        "last_loss": last_loss,
        "normalized_mse": metrics["normalized_mse"],
        "raw_mse": metrics["raw_mse"],
        "ev_values": metrics["ev_values"],
        "ev_range": gate["ev_range"],
        "target_consistency_max_error": target_error,
        "elapsed_seconds": elapsed,
        "assertions": gate["assertions"],
        "passed": gate["passed"],
    }
    _write_json(args.output, payload)
    return payload


def _failure_payload(args: argparse.Namespace, error: BaseException) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "seed": args.seed,
        "sample_count": args.samples,
        "dataset_sha256": None,
        "source_commit": _source_commit(),
        "resolved_config": vars(args) | {"output": str(args.output)},
        "command": [sys.executable, *sys.argv],
        "update_count": 0,
        "normalized_mse": None,
        "raw_mse": None,
        "ev_values": [],
        "ev_range": None,
        "target_consistency_max_error": None,
        "elapsed_seconds": 0.0,
        "assertions": {"execution_completed": False},
        "error": f"{type(error).__name__}: {error}",
        "passed": False,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=LOCKED_SEED)
    parser.add_argument("--samples", type=int, default=LOCKED_SAMPLES)
    parser.add_argument("--max-updates", type=int, default=LOCKED_MAX_UPDATES)
    parser.add_argument("--mse-threshold", type=float, default=LOCKED_MSE_THRESHOLD)
    parser.add_argument("--ev-threshold", type=float, default=LOCKED_EV_THRESHOLD)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        payload = run(args)
    except BaseException as error:
        payload = _failure_payload(args, error)
        _write_json(args.output, payload)
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
