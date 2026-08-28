import unittest

import jax.numpy as jnp

from baselines.MAPPO.shared_system_macpo import SharedSystemMACPOUpdater


class SharedSystemMACPOTest(unittest.TestCase):
    def test_infeasible_policy_uses_cost_recovery_with_kl_bound(self):
        initial = jnp.array([0.5])
        reward = lambda params: -jnp.square(params[0] - 1.0)
        cost = lambda params: jnp.square(params[0])
        kl = lambda params: 0.5 * jnp.square(params[0] - initial[0])
        updater = SharedSystemMACPOUpdater(
            max_kl=0.2, cg_iterations=10, damping=1e-3, max_backtracks=12
        )

        updated, metrics = updater.update(
            initial, reward_objective=reward, cost_objective=cost, kl_divergence=kl, budget=0.04
        )

        self.assertTrue(metrics["accepted"])
        self.assertEqual(metrics["mode"], "cost_recovery")
        self.assertLess(float(cost(updated)), float(cost(initial)))
        self.assertLessEqual(float(kl(updated)), 0.2 + 1e-6)

    def test_feasible_policy_preserves_budget_and_improves_surrogate_reward(self):
        initial = jnp.array([0.0])
        reward = lambda params: -jnp.square(params[0] - 0.2)
        cost = lambda params: jnp.square(params[0])
        kl = lambda params: 0.5 * jnp.square(params[0] - initial[0])
        updater = SharedSystemMACPOUpdater(max_kl=0.1, cg_iterations=10, damping=1e-3)

        updated, metrics = updater.update(
            initial, reward_objective=reward, cost_objective=cost, kl_divergence=kl, budget=0.05
        )

        self.assertTrue(metrics["accepted"])
        self.assertLessEqual(float(cost(updated)), 0.05 + 1e-6)
        self.assertGreaterEqual(float(reward(updated)), float(reward(initial)))


if __name__ == "__main__":
    unittest.main()
