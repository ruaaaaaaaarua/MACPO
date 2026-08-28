"""Trust-region MACPO update for one shared system cost constraint.

This is a system-cost adaptation: all policies are constrained by the same
voltage trajectory cost, rather than claiming the local-cost theorem of MACPO.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree


ArrayObjective = Callable[[Any], jnp.ndarray]


def conjugate_gradient(
    matvec: Callable[[jnp.ndarray], jnp.ndarray],
    vector: jnp.ndarray,
    iterations: int,
    tolerance: float = 1e-10,
) -> jnp.ndarray:
    """Approximately solve ``Ax=b`` for a positive-definite implicit matrix."""
    solution = jnp.zeros_like(vector)
    residual = vector.copy()
    direction = residual.copy()
    residual_dot = float(jnp.vdot(residual, residual))
    for _ in range(iterations):
        if residual_dot <= tolerance:
            break
        matrix_direction = matvec(direction)
        denominator = float(jnp.vdot(direction, matrix_direction))
        if denominator <= tolerance:
            break
        step = residual_dot / denominator
        solution = solution + step * direction
        residual = residual - step * matrix_direction
        next_residual_dot = float(jnp.vdot(residual, residual))
        direction = residual + (next_residual_dot / residual_dot) * direction
        residual_dot = next_residual_dot
    return solution


@dataclass(frozen=True)
class SharedSystemMACPOUpdater:
    """CPO-style update with reward, global-cost, and KL sample checks."""

    max_kl: float = 0.01
    cg_iterations: int = 10
    damping: float = 1e-2
    max_backtracks: int = 10
    backtrack_ratio: float = 0.5
    tolerance: float = 1e-8

    def update(
        self,
        params: Any,
        *,
        reward_objective: ArrayObjective,
        cost_objective: ArrayObjective,
        kl_divergence: ArrayObjective,
        budget: float,
    ) -> tuple[Any, dict[str, float | bool | str]]:
        """Return a line-searched policy update for the shared system cost."""
        flat_params, unravel = ravel_pytree(params)
        reward_before = float(reward_objective(params))
        cost_before = float(cost_objective(params))
        kl_before = float(kl_divergence(params))
        reward_grad, _ = ravel_pytree(jax.grad(reward_objective)(params))
        cost_grad, _ = ravel_pytree(jax.grad(cost_objective)(params))
        _, kl_hessian_linear_map = jax.linearize(jax.grad(kl_divergence), params)

        def fisher_vector_product(vector: jnp.ndarray) -> jnp.ndarray:
            tangent = unravel(vector)
            hessian_vector, _ = ravel_pytree(
                kl_hessian_linear_map(tangent)
            )
            return hessian_vector + self.damping * vector

        inverse_reward_grad = conjugate_gradient(
            fisher_vector_product, reward_grad, self.cg_iterations
        )
        inverse_cost_grad = conjugate_gradient(
            fisher_vector_product, cost_grad, self.cg_iterations
        )
        cost_gap = cost_before - float(budget)
        reward_quadratic = float(jnp.vdot(reward_grad, inverse_reward_grad))
        reward_cost_cross = float(jnp.vdot(reward_grad, inverse_cost_grad))
        cost_quadratic = float(jnp.vdot(cost_grad, inverse_cost_grad))
        mode = "reward"

        if cost_gap > self.tolerance and cost_quadratic > self.tolerance:
            # The sampled policy is infeasible: spend the whole trust region on
            # reducing the shared voltage cost before attempting reward ascent.
            scale = (2.0 * self.max_kl / cost_quadratic) ** 0.5
            direction = -scale * inverse_cost_grad
            mode = "cost_recovery"
        elif reward_quadratic <= self.tolerance:
            direction = jnp.zeros_like(flat_params)
            mode = "stationary"
        else:
            reward_scale = (2.0 * self.max_kl / reward_quadratic) ** 0.5
            reward_direction = reward_scale * inverse_reward_grad
            predicted_cost = cost_gap + float(jnp.vdot(cost_grad, reward_direction))
            if predicted_cost <= self.tolerance:
                direction = reward_direction
            elif cost_quadratic <= self.tolerance:
                direction = jnp.zeros_like(flat_params)
                mode = "cost_stationary"
            else:
                # Intersection of the linearized cost boundary and KL ellipsoid.
                residual_kl = 2.0 * self.max_kl - cost_gap**2 / cost_quadratic
                denominator = reward_quadratic - reward_cost_cross**2 / cost_quadratic
                if residual_kl <= self.tolerance or denominator <= self.tolerance:
                    scale = (2.0 * self.max_kl / cost_quadratic) ** 0.5
                    direction = -scale * inverse_cost_grad
                    mode = "cost_recovery"
                else:
                    alpha = (residual_kl / denominator) ** 0.5
                    beta = (-cost_gap - alpha * reward_cost_cross) / cost_quadratic
                    direction = alpha * inverse_reward_grad + beta * inverse_cost_grad
                    mode = "constrained_reward"

        accepted_params = params
        accepted = False
        reward_after = reward_before
        cost_after = cost_before
        kl_after = kl_before
        for backtrack in range(self.max_backtracks):
            fraction = self.backtrack_ratio**backtrack
            candidate = unravel(flat_params + fraction * direction)
            candidate_reward = float(reward_objective(candidate))
            candidate_cost = float(cost_objective(candidate))
            candidate_kl = float(kl_divergence(candidate))
            if mode == "cost_recovery":
                cost_ok = candidate_cost <= cost_before + self.tolerance
                reward_ok = True
            else:
                cost_ok = candidate_cost <= float(budget) + self.tolerance
                reward_ok = candidate_reward >= reward_before - self.tolerance
            if candidate_kl <= self.max_kl + self.tolerance and cost_ok and reward_ok:
                accepted_params = candidate
                accepted = True
                reward_after = candidate_reward
                cost_after = candidate_cost
                kl_after = candidate_kl
                break

        return accepted_params, {
            "accepted": accepted,
            "mode": mode,
            "reward_before": reward_before,
            "reward_after": reward_after,
            "cost_before": cost_before,
            "cost_after": cost_after,
            "kl_before": kl_before,
            "kl_after": kl_after,
            "cost_gap": cost_gap,
        }
