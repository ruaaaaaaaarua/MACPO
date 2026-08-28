"""Resumable training checkpoints for JAX MAPPO-family trainers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import numpy as np
from flax import serialization


@dataclass(frozen=True)
class RestoredJaxTrainingCheckpoint:
    train_state: Any
    rng: Any
    update: int
    episode: int
    global_step: int


def save_jax_training_checkpoint(
    path: str | Path,
    *,
    train_state: Any,
    rng: Any,
    update: int,
    episode: int,
    global_step: int,
    enabled: bool = True,
) -> Path | None:
    """Atomically save TrainState (including optimizer), RNG, and progress."""
    if not enabled:
        return None
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "train_state": train_state,
        "rng": jnp.asarray(rng, dtype=jnp.uint32),
        "update": np.asarray(update, dtype=np.int64),
        "episode": np.asarray(episode, dtype=np.int64),
        "global_step": np.asarray(global_step, dtype=np.int64),
    }
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_bytes(serialization.to_bytes(payload))
    temporary.replace(destination)
    return destination


def load_jax_training_checkpoint(
    path: str | Path,
    train_state_template: Any,
) -> RestoredJaxTrainingCheckpoint:
    """Restore into a freshly initialized TrainState with matching structure."""
    source = Path(path)
    template = {
        "train_state": train_state_template,
        "rng": jnp.zeros((2,), dtype=jnp.uint32),
        "update": np.asarray(0, dtype=np.int64),
        "episode": np.asarray(0, dtype=np.int64),
        "global_step": np.asarray(0, dtype=np.int64),
    }
    payload = serialization.from_bytes(template, source.read_bytes())
    return RestoredJaxTrainingCheckpoint(
        train_state=payload["train_state"],
        rng=payload["rng"],
        update=int(payload["update"]),
        episode=int(payload["episode"]),
        global_step=int(payload["global_step"]),
    )
