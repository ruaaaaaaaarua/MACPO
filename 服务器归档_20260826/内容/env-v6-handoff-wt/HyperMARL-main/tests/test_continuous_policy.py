import unittest

import jax
import jax.numpy as jnp
import numpy as np

from baselines.MAPPO.mappo_ff_shared_weights import ActorCritic
from baselines.MAPPO.continuous_policy import (
    clamp_log_std,
    deterministic_action,
    sample_squashed_gaussian,
    squashed_log_prob,
)


class SquashedGaussianTest(unittest.TestCase):
    def test_log_std_is_clamped_to_configured_bounds(self):
        actual = clamp_log_std(
            jnp.asarray([-10.0, -1.0, 3.0]), minimum=-2.5, maximum=-0.5
        )
        np.testing.assert_allclose(actual, [-2.5, -1.0, -0.5])

    def test_log_std_cap_can_be_a_jitted_curriculum_scalar(self):
        @jax.jit
        def apply_curriculum_cap(maximum):
            return clamp_log_std(
                jnp.asarray([-3.0, -0.75, 0.0]), minimum=-2.5, maximum=maximum
            )

        np.testing.assert_allclose(
            apply_curriculum_cap(jnp.asarray(-1.0)), [-2.5, -1.0, -1.0]
        )

    def test_sample_is_bounded_and_log_prob_is_reproducible(self):
        mean = jnp.asarray([[0.0, 0.5, -0.5]], dtype=jnp.float32)
        log_std = jnp.asarray([-1.0, -1.0, -1.0], dtype=jnp.float32)
        action, log_prob = sample_squashed_gaussian(
            mean,
            log_std,
            jax.random.PRNGKey(7),
            log_std_min=-2.5,
            log_std_max=-0.5,
        )
        self.assertTrue(np.all(np.asarray(action) < 1.0))
        self.assertTrue(np.all(np.asarray(action) > -1.0))
        recomputed = squashed_log_prob(
            mean,
            log_std,
            action,
            log_std_min=-2.5,
            log_std_max=-0.5,
        )
        np.testing.assert_allclose(log_prob, recomputed, rtol=1e-5, atol=1e-5)
        self.assertTrue(np.all(np.isfinite(np.asarray(log_prob))))

    def test_near_boundary_actions_have_finite_log_prob(self):
        action = jnp.asarray([[0.999999, -0.999999]], dtype=jnp.float32)
        result = squashed_log_prob(
            jnp.zeros_like(action),
            jnp.zeros((2,)),
            action,
            log_std_min=-2.5,
            log_std_max=-0.5,
        )
        self.assertTrue(np.all(np.isfinite(np.asarray(result))))

    def test_deterministic_action_is_tanh_mean(self):
        mean = jnp.asarray([[0.0, 1.0, -1.0]])
        np.testing.assert_allclose(deterministic_action(mean), np.tanh(mean))

    def test_actor_critic_returns_effective_bounded_log_std(self):
        network = ActorCritic(
            action_dim=2,
            num_agents=2,
            observation_dim=3,
            actor_layers=[4],
            critic_layers=[4],
            is_continuous=True,
            log_std_init=1.0,
            log_std_min=-2.5,
            log_std_max=-0.5,
        )
        actor_obs = jnp.asarray(
            [[0.0, 0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 0.0, 1.0]]
        )
        critic_obs = jnp.zeros((2, 6))
        params = network.init(jax.random.PRNGKey(9), actor_obs, critic_obs)
        (_, effective_log_std), _ = network.apply(params, actor_obs, critic_obs)
        np.testing.assert_allclose(effective_log_std, [-0.5, -0.5])


if __name__ == "__main__":
    unittest.main()
