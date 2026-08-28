"""Auditable per-rollout diagnostics for STAS-MAPPO training."""

from __future__ import annotations

from collections.abc import Iterable
import json
from pathlib import Path

import numpy as np

from .credit_conservation import discounted_team_return


def compute_target_error(buffers: Iterable[object], gamma: float) -> float:
    """Return the worst saved-target mismatch across the supplied buffers."""
    maximum = 0.0
    for buffer in buffers:
        for item in getattr(buffer, "storage", ()):
            saved_rewards = np.asarray(item[2])
            saved_target = float(item[4])
            recomputed_target = float(
                discounted_team_return(saved_rewards[None, ...], gamma)[0]
            )
            mismatch = abs(saved_target - recomputed_target)
            if not np.isfinite(mismatch):
                return float("nan")
            maximum = max(maximum, mismatch)
    return float(maximum)


def compute_conservation_error(
    training_rewards: np.ndarray,
    original_rewards: np.ndarray,
    gamma: float,
) -> float:
    """Return the worst actual blended-vs-original discounted-return error."""
    actual = discounted_team_return(training_rewards, gamma)
    expected = discounted_team_return(original_rewards, gamma)
    if actual.shape != expected.shape:
        raise ValueError("training and original reward batches must match")
    if actual.size == 0:
        return 0.0
    return float(np.max(np.abs(actual - expected)))


def classify_gate_state(
    *,
    training_buffer_size: int,
    minimum_training_buffer: int,
    episodes_seen: int,
    warmup_episodes: int,
    explained_variance: float | None,
    explained_variance_threshold: float,
    mix_coef: float | None,
    negative_streak: int = 0,
    disabled: bool = False,
) -> dict[str, object]:
    """Describe the current gate decision without changing gate state."""
    ev_is_finite = explained_variance is not None and np.isfinite(
        explained_variance
    )
    mix_is_finite = mix_coef is not None and np.isfinite(mix_coef)
    mix_is_active = bool(mix_is_finite and mix_coef > 0.0)

    if training_buffer_size < minimum_training_buffer:
        phase = "insufficient_training_buffer"
        reason = "train_buffer_below_minimum"
    elif episodes_seen < warmup_episodes:
        phase = "warmup"
        reason = "episode_warmup"
    elif not ev_is_finite and mix_is_active:
        phase = "active"
        reason = "credit_mix_active_with_invalid_ev"
    elif not ev_is_finite:
        phase = "invalid_ev"
        reason = "nonfinite_explained_variance"
    elif explained_variance < explained_variance_threshold:
        phase = "ev_blocked"
        reason = "explained_variance_below_threshold"
    elif mix_is_active:
        phase = "active"
        reason = "credit_mix_active"
    elif not mix_is_finite:
        phase = "invalid_mix"
        reason = "nonfinite_mix_coef"
    else:
        phase = "ramp"
        reason = "ramp_mix_zero"

    return {
        "phase": phase,
        "active": phase == "active",
        "reason": reason,
        "negative_streak": int(negative_streak),
        "disabled": bool(disabled),
    }


def _finite_float(value: object) -> float | None:
    try:
        converted = float(value)
    except (TypeError, ValueError):
        return None
    return converted if np.isfinite(converted) else None


def _assigner_buffers(assigner: object) -> tuple[object, ...]:
    return tuple(
        buffer
        for name in ("buffer", "holdout_buffer")
        if (buffer := getattr(assigner, name, None)) is not None
    )


def build_rollout_record(
    assigner: object,
    *,
    update: int,
    episode: int,
    global_step: int,
) -> dict[str, object]:
    """Build one JSON-safe record from the assigner's current state."""
    config = assigner.config
    train_buffer = getattr(assigner, "buffer", ())
    minimum = max(
        1,
        min(
            int(getattr(config, "batch_size", 1)),
            int(getattr(config, "buffer_size", 1)),
        ),
    )
    explained_variance = _finite_float(
        getattr(assigner, "last_explained_variance", 0.0)
    )
    mix_coef = _finite_float(
        getattr(assigner, "last_mix_coef", getattr(config, "mix_coef", 0.0))
    )
    gate = getattr(assigner, "gate", None)
    gate_state = classify_gate_state(
        training_buffer_size=len(train_buffer),
        minimum_training_buffer=minimum,
        episodes_seen=int(getattr(assigner, "episodes_seen", len(train_buffer))),
        warmup_episodes=int(getattr(config, "warmup_episodes", 0)),
        explained_variance=explained_variance,
        explained_variance_threshold=float(
            getattr(config, "explained_variance_threshold", 0.0)
        ),
        mix_coef=mix_coef,
        negative_streak=int(getattr(gate, "negative_streak", 0)),
        disabled=bool(getattr(gate, "disabled", False)),
    )
    reward_model_loss = _finite_float(getattr(assigner, "last_loss", np.nan))
    current_target_error = compute_target_error(
        _assigner_buffers(assigner), float(config.gamma)
    )
    assigner.last_target_error = current_target_error
    target_error = _finite_float(current_target_error)
    return {
        "schema_version": 1,
        "update": int(update),
        "episode": int(episode),
        "global_step": int(global_step),
        "reward_model_loss": reward_model_loss,
        "explained_variance": explained_variance,
        "mix_coef": mix_coef,
        "target_error": target_error,
        "conservation_error": _finite_float(
            getattr(assigner, "last_conservation_error", 0.0)
        ),
        "gate": gate_state,
    }


def append_rollout_record(path: str | Path | None, record: dict[str, object]) -> bool:
    """Append one UTF-8 JSONL record, or do nothing when no path is configured."""
    if path is None or not str(path).strip():
        return False
    serialized = json.dumps(record, ensure_ascii=False, allow_nan=False)
    destination = Path(path).expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(serialized + "\n")
    return True


def write_rollout_diagnostic(
    path: str | Path | None,
    assigner: object,
    *,
    update: int,
    episode: int,
    global_step: int,
) -> bool:
    """Build and append one record only when diagnostics are configured."""
    if path is None or not str(path).strip():
        return False
    return append_rollout_record(
        path,
        build_rollout_record(
            assigner,
            update=update,
            episode=episode,
            global_step=global_step,
        ),
    )
