"""Deterministic environment-stream restoration for resumed training."""

from __future__ import annotations

from typing import Any


def reset_environment_stream(
    env: Any,
    *,
    seed: int,
    completed_resets: int,
):
    """Seed first, then replay completed reset draws and return the next state."""

    completed_resets = int(completed_resets)
    if completed_resets < 0:
        raise ValueError("completed_resets must be non-negative")

    observation, info = env.reset(seed=int(seed))
    for _ in range(completed_resets):
        observation, info = env.reset()
    return observation, info
