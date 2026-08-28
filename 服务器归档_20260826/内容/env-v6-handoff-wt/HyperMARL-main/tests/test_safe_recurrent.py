import unittest

import jax
import jax.numpy as jnp
import numpy as np

from baselines.MAPPO.safe_recurrent import (
    CentralGRUCritic,
    IndependentGRUActors,
    compute_gae,
    mask_cross_agent_channels,
    reset_actor_hidden,
    reset_global_hidden,
    update_lagrange_multiplier,
)


class SafeRecurrentUtilitiesTest(unittest.TestCase):
    def test_done_resets_per_environment_hidden_state(self):
        actor_hidden = jnp.ones((2, 4, 3))
        critic_hidden = jnp.ones((2, 5))
        dones = jnp.array([False, True])

        next_actor_hidden = reset_actor_hidden(actor_hidden, dones)
        next_critic_hidden = reset_global_hidden(critic_hidden, dones)
        np.testing.assert_allclose(next_actor_hidden[0], 1.0)
        np.testing.assert_allclose(next_actor_hidden[1], 0.0)
        np.testing.assert_allclose(next_critic_hidden[0], 1.0)
        np.testing.assert_allclose(next_critic_hidden[1], 0.0)

    def test_gae_respects_terminal_boundaries_and_has_rollout_shape(self):
        rewards = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        values = jnp.zeros_like(rewards)
        dones = jnp.array([[False, True], [True, True]])
        advantages, returns = compute_gae(
            rewards, values, jnp.array([5.0, 6.0]), dones, gamma=1.0, gae_lambda=1.0
        )

        self.assertEqual(advantages.shape, (2, 2))
        self.assertEqual(returns.shape, (2, 2))
        np.testing.assert_allclose(advantages, [[4.0, 2.0], [3.0, 4.0]])
        np.testing.assert_allclose(returns, advantages)

    def test_lagrange_multiplier_increases_when_system_cost_exceeds_budget(self):
        increased = update_lagrange_multiplier(0.2, cost_mean=0.3, budget=0.0, lr=0.5)
        decreased = update_lagrange_multiplier(0.2, cost_mean=0.0, budget=0.5, lr=0.5)

        self.assertGreater(increased, 0.2)
        self.assertEqual(decreased, 0.0)


    def test_independent_actor_and_central_critic_recurrent_shapes(self):
        batch_size, agent_count, obs_dim, action_dim, hidden_size = 2, 4, 5, 3, 7
        actor = IndependentGRUActors(
            num_agents=agent_count,
            action_dim=action_dim,
            hidden_size=hidden_size,
        )
        critic = CentralGRUCritic(hidden_size=hidden_size)
        local_obs = jnp.zeros((batch_size, agent_count, obs_dim))
        global_obs = jnp.zeros((batch_size, agent_count * obs_dim))
        actor_hidden = jnp.zeros((batch_size, agent_count, hidden_size))
        critic_hidden = jnp.zeros((batch_size, hidden_size))

        actor_params = actor.init(jax.random.PRNGKey(0), local_obs, actor_hidden)
        critic_params = critic.init(jax.random.PRNGKey(1), global_obs, critic_hidden)
        means, log_stds, next_actor_hidden = actor.apply(
            actor_params, local_obs, actor_hidden
        )
        values, next_critic_hidden = critic.apply(
            critic_params, global_obs, critic_hidden
        )

        self.assertEqual(means.shape, (batch_size, agent_count, action_dim))
        self.assertEqual(log_stds.shape, means.shape)
        self.assertEqual(next_actor_hidden.shape, actor_hidden.shape)
        self.assertEqual(values.shape, (batch_size,))
        self.assertEqual(next_critic_hidden.shape, critic_hidden.shape)
        for agent_index in range(agent_count):
            self.assertIn(f"actor_{agent_index}_gru", actor_params["params"])

    def test_two_stage_actor_broadcasts_all_current_intents_before_actions(self):
        actor = IndependentGRUActors(
            num_agents=4,
            action_dim=7,
            hidden_size=8,
            two_stage_intent=True,
            intent_dim=3,
        )
        local_obs = jnp.ones((2, 4, 5))
        hidden = jnp.zeros((2, 4, 8))
        params = actor.init(
            jax.random.PRNGKey(7), local_obs, hidden, return_intents=True
        )
        means, log_stds, next_hidden, intents = actor.apply(
            params, local_obs, hidden, return_intents=True
        )

        self.assertEqual(means.shape, (2, 4, 7))
        self.assertEqual(log_stds.shape, (2, 4, 7))
        self.assertEqual(next_hidden.shape, (2, 4, 8))
        self.assertEqual(intents.shape, (2, 4, 3))
        self.assertTrue(np.all(np.abs(np.asarray(intents)) <= 1.0))
        self.assertIn("actor_0_intent", params["params"])
        self.assertIn("actor_0_stage2_hidden", params["params"])

    def test_intent_bound_actor_keeps_physical_action_close_to_intent(self):
        actor = IndependentGRUActors(
            num_agents=4, action_dim=7, hidden_size=8,
            two_stage_intent=True, intent_dim=3, intent_residual_limit=0.25,
        )
        obs = jnp.ones((1, 4, 5))
        hidden = jnp.zeros((1, 4, 8))
        params = actor.init(jax.random.PRNGKey(9), obs, hidden, return_intents=True)
        means, _, _, intents = actor.apply(params, obs, hidden, return_intents=True)
        physical = jnp.tanh(means[:, :, [0, 1, 5]])
        self.assertLessEqual(float(jnp.max(jnp.abs(physical - intents))), 0.25001)

    def test_supply_message_has_action_intent_then_four_observed_facts(self):
        actor = IndependentGRUActors(
            num_agents=4, action_dim=7, hidden_size=8,
            two_stage_intent=True, intent_dim=3, supply_message_dim=4,
        )
        obs = jnp.ones((1, 4, 9))
        facts = jnp.array([[[.1, .2, .3, .4], [.5, .6, .7, .8], [.2, .3, .4, .5], [.6, .7, .8, .9]]])
        hidden = jnp.zeros((1, 4, 8))
        obs = obs.at[..., -4:].set(facts)
        params = actor.init(jax.random.PRNGKey(11), obs, hidden, return_intents=True, return_messages=True)
        _, _, _, _, message = actor.apply(params, obs, hidden, return_intents=True, return_messages=True)
        self.assertEqual(message.shape, (1, 4, 7))
        np.testing.assert_allclose(message[..., 3:], facts)

    def test_self_only_message_mask_preserves_only_observer_channel(self):
        messages = jnp.arange(1 * 4 * 7, dtype=jnp.float32).reshape(1, 4, 7) + 1
        masked = mask_cross_agent_channels(messages, observer_index=2, scope="self_only")
        np.testing.assert_allclose(masked[:, :2], 0.0)
        np.testing.assert_allclose(masked[:, 2], messages[:, 2])
        np.testing.assert_allclose(masked[:, 3], 0.0)
        np.testing.assert_allclose(
            mask_cross_agent_channels(messages, observer_index=2, scope="full"),
            messages,
        )

if __name__ == "__main__":
    unittest.main()
