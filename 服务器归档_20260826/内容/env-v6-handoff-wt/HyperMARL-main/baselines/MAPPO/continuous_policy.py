"""Numerically stable tanh-squashed diagonal Gaussian helpers."""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp


_LOG_TWO_PI = math.log(2.0 * math.pi)
_LOG_TWO = math.log(2.0)


def clamp_log_std(log_std, minimum: float, maximum: float):
    # Curriculum caps are runtime JAX scalars under a jitted PPO loss.  Keep
    # the helpful validation for ordinary Python configuration values, but do
    # not perform a Python boolean conversion on a tracer.
    if not isinstance(maximum, jax.core.Tracer) and minimum > maximum:
        raise ValueError(f"log std minimum {minimum} exceeds maximum {maximum}")
    return jnp.clip(jnp.asarray(log_std), minimum, maximum)


def _atanh(action, eps: float = 1e-6):
    bounded = jnp.clip(jnp.asarray(action), -1.0 + eps, 1.0 - eps)
    return 0.5 * (jnp.log1p(bounded) - jnp.log1p(-bounded))


def _normal_log_prob(mean, log_std, raw_action):
    inv_std = jnp.exp(-log_std)
    per_dim = -0.5 * (
        jnp.square((raw_action - mean) * inv_std) + 2.0 * log_std + _LOG_TWO_PI
    )
    return jnp.sum(per_dim, axis=-1)


def _tanh_log_abs_det_jacobian(raw_action):
    per_dim = 2.0 * (
        _LOG_TWO - raw_action - jax.nn.softplus(-2.0 * raw_action)
    )
    return jnp.sum(per_dim, axis=-1)


def squashed_log_prob(
    mean,
    log_std,
    action,
    *,
    log_std_min: float,
    log_std_max: float,
):
    """Log density of a bounded action under tanh(N(mean, std))."""
    clipped_log_std = clamp_log_std(log_std, log_std_min, log_std_max)
    raw_action = _atanh(action)
    return _normal_log_prob(mean, clipped_log_std, raw_action) - _tanh_log_abs_det_jacobian(
        raw_action
    )


def sample_squashed_gaussian(
    mean,
    log_std,
    rng,
    *,
    log_std_min: float,
    log_std_max: float,
):
    """Sample a bounded action and return its corrected log probability."""
    clipped_log_std = clamp_log_std(log_std, log_std_min, log_std_max)
    noise = jax.random.normal(rng, shape=jnp.shape(mean), dtype=jnp.asarray(mean).dtype)
    raw_action = jnp.asarray(mean) + jnp.exp(clipped_log_std) * noise
    action = jnp.tanh(raw_action)
    log_prob = _normal_log_prob(mean, clipped_log_std, raw_action) - _tanh_log_abs_det_jacobian(
        raw_action
    )
    return action, log_prob


def deterministic_action(mean):
    return jnp.tanh(jnp.asarray(mean))


def sampled_entropy(log_prob):
    """Monte Carlo entropy estimate for the transformed distribution."""
    return -jnp.asarray(log_prob)
