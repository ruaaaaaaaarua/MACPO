"""Pure action transformations for the MATD3 a5 timing diagnostic."""

from collections.abc import Mapping

import numpy as np


VARIANT_NAMES = (
    "normal_replay",
    "shift_earlier_4",
    "shift_later_4",
    "shuffle_fixed",
    "constant_mean",
)


def validate_variant_invariants(
    normal: np.ndarray,
    variants: Mapping[str, np.ndarray],
    *,
    usable_steps: int = 20,
    atol: float = 1e-6,
) -> dict[str, object]:
    """Validate that timing variants change only usable-window a5 timing.

    Returns JSON-serializable invariant metadata keyed by variant name and
    raises ``ValueError`` before rollout when any invariant is violated.
    """
    reference = np.asarray(normal)
    if reference.shape != (24, 4, 6) or not np.isfinite(reference).all():
        raise ValueError("normal must be finite with shape (24, 4, 6)")
    if (
        not isinstance(usable_steps, (int, np.integer))
        or not 1 <= usable_steps <= 24
    ):
        raise ValueError("usable_steps must be an integer from 1 through 24")
    if not isinstance(variants, Mapping):
        raise ValueError("variants must be a mapping")

    try:
        tolerance = float(atol)
    except (TypeError, ValueError) as exc:
        raise ValueError("atol must be a finite non-negative number") from exc
    if not np.isfinite(tolerance) or tolerance < 0:
        raise ValueError("atol must be a finite non-negative number")

    reference_budget = reference[:usable_steps, :, 5].sum(
        axis=0, dtype=np.float64
    )
    results: dict[str, object] = {}
    for name, transformed in variants.items():
        candidate = np.asarray(transformed)
        if candidate.shape != reference.shape or not np.isfinite(candidate).all():
            raise ValueError(
                f"variant {name!r} must be finite with shape (24, 4, 6)"
            )

        non_a5_preserved = bool(
            np.array_equal(candidate[:, :, :5], reference[:, :, :5])
        )
        boundary_preserved = bool(
            np.array_equal(
                candidate[usable_steps:, :, 5],
                reference[usable_steps:, :, 5],
            )
        )
        candidate_budget = candidate[:usable_steps, :, 5].sum(
            axis=0, dtype=np.float64
        )
        budget_error = np.abs(candidate_budget - reference_budget)
        budget_preserved = bool(np.all(budget_error <= tolerance))
        invariant = {
            "non_a5_preserved": non_a5_preserved,
            "boundary_preserved": boundary_preserved,
            "budget_preserved": budget_preserved,
            "max_abs_budget_error": float(budget_error.max()),
        }

        failed = [
            key
            for key, value in invariant.items()
            if key != "max_abs_budget_error" and not value
        ]
        if failed:
            raise ValueError(
                f"variant {name!r} violates invariants: {', '.join(failed)}"
            )
        results[name] = invariant

    return results


def build_a5_variants(
    actions: np.ndarray,
    *,
    usable_steps: int = 20,
    lag: int = 4,
    shuffle_seed: int = 20260715,
) -> dict[str, np.ndarray]:
    """Build deterministic, budget-preserving timing variants of action a5."""
    normal = np.asarray(actions, dtype=np.float32)
    if normal.shape != (24, 4, 6) or not np.isfinite(normal).all():
        raise ValueError("actions must be finite with shape (24, 4, 6)")
    if (
        not isinstance(usable_steps, (int, np.integer))
        or not 1 <= usable_steps <= 24
    ):
        raise ValueError("usable_steps must be an integer from 1 through 24")

    output = {"normal_replay": normal.copy()}
    indices_by_name = {
        "shift_earlier_4": np.roll(np.arange(usable_steps), -lag),
        "shift_later_4": np.roll(np.arange(usable_steps), lag),
        "shuffle_fixed": np.random.default_rng(shuffle_seed).permutation(
            usable_steps
        ),
    }
    for name, indices in indices_by_name.items():
        transformed = normal.copy()
        transformed[:usable_steps, :, 5] = normal[indices, :, 5]
        output[name] = transformed

    constant = normal.copy()
    constant[:usable_steps, :, 5] = normal[:usable_steps, :, 5].mean(
        axis=0, dtype=np.float64
    )
    output["constant_mean"] = constant

    validate_variant_invariants(normal, output, usable_steps=usable_steps)
    return output
