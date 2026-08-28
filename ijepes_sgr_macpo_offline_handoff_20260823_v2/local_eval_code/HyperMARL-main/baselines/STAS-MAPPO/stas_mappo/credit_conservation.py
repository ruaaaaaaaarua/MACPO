"""Discounted-return conservation and quality gating for STAS credits."""

from dataclasses import dataclass

import numpy as np


def discounted_team_return(rewards: np.ndarray, gamma: float) -> np.ndarray:
    array = np.asarray(rewards, dtype=np.float64)
    if array.ndim != 3:
        raise ValueError(f"expected [batch, agent, time], got {array.shape}")
    weights = np.power(float(gamma), np.arange(array.shape[2], dtype=np.float64))
    return np.sum(array * weights[None, None, :], axis=(1, 2))


def project_discounted_credits(
    credits: np.ndarray,
    target_returns: np.ndarray,
    gamma: float,
) -> tuple[np.ndarray, np.ndarray]:
    array = np.asarray(credits, dtype=np.float64)
    targets = np.asarray(target_returns, dtype=np.float64).reshape(-1)
    if array.ndim != 3 or array.shape[0] != targets.shape[0]:
        raise ValueError("credits and target return batch dimensions must match")
    weights = np.power(float(gamma), np.arange(array.shape[2], dtype=np.float64))
    current = np.sum(array * weights[None, None, :], axis=(1, 2))
    denominator = float(array.shape[1]) * float(np.square(weights).sum())
    correction = (targets - current)[:, None, None] * weights[None, None, :] / denominator
    projected = array + correction
    errors = discounted_team_return(projected, gamma) - targets
    return projected.astype(np.float32), errors.astype(np.float64)


def explained_variance(targets: np.ndarray, predictions: np.ndarray) -> float:
    targets = np.asarray(targets, dtype=np.float64)
    predictions = np.asarray(predictions, dtype=np.float64)
    variance = float(np.var(targets))
    if variance <= 1e-12:
        return 0.0
    return float(1.0 - np.var(targets - predictions) / variance)


@dataclass
class CreditQualityGate:
    warmup_episodes: int
    ramp_episodes: int
    max_mix_coef: float
    explained_variance_threshold: float
    negative_patience: int
    negative_streak: int = 0
    disabled: bool = False

    def mix_coef(self, episodes_seen: int, explained_variance: float) -> float:
        quality = float(explained_variance)
        # Quality gating is recoverable: poor predictions suppress credit for
        # the current rollout, but a later healthy holdout score can re-enable it.
        self.disabled = False
        gate_mature_episode = self.warmup_episodes + self.ramp_episodes
        if episodes_seen >= gate_mature_episode:
            if quality < 0.0:
                self.negative_streak += 1
            else:
                self.negative_streak = 0
        else:
            self.negative_streak = 0
        if (
            episodes_seen < self.warmup_episodes
            or quality < self.explained_variance_threshold
        ):
            return 0.0
        progress = min(
            1.0,
            max(0.0, (episodes_seen - self.warmup_episodes) / max(self.ramp_episodes, 1)),
        )
        return float(self.max_mix_coef * progress)
