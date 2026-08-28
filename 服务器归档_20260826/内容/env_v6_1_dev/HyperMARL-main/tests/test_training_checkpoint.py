import tempfile
import unittest
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training.train_state import TrainState

from baselines.utils.training_checkpoint import (
    load_jax_training_checkpoint,
    save_jax_training_checkpoint,
)


class _TinyModel:
    @staticmethod
    def apply(params, x):
        return x * params["weight"]


class JaxTrainingCheckpointTest(unittest.TestCase):
    def test_round_trip_restores_optimizer_rng_and_progress(self):
        tx = optax.adam(1e-2)
        state = TrainState.create(
            apply_fn=_TinyModel.apply,
            params={"weight": jnp.asarray([1.0, -2.0], dtype=jnp.float32)},
            tx=tx,
        )
        grads = {"weight": jnp.asarray([0.25, -0.5], dtype=jnp.float32)}
        state = state.apply_gradients(grads=grads)
        rng = jax.random.PRNGKey(123)

        template = TrainState.create(
            apply_fn=_TinyModel.apply,
            params={"weight": jnp.zeros(2, dtype=jnp.float32)},
            tx=tx,
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "state.msgpack"
            save_jax_training_checkpoint(
                path,
                train_state=state,
                rng=rng,
                update=7,
                episode=28,
                global_step=672,
            )
            restored = load_jax_training_checkpoint(path, template)

        self.assertEqual(restored.update, 7)
        self.assertEqual(restored.episode, 28)
        self.assertEqual(restored.global_step, 672)
        self.assertEqual(int(restored.train_state.step), int(state.step))
        np.testing.assert_array_equal(restored.rng, rng)
        np.testing.assert_allclose(
            restored.train_state.params["weight"], state.params["weight"]
        )
        expected_leaves = jax.tree_util.tree_leaves(state.opt_state)
        actual_leaves = jax.tree_util.tree_leaves(restored.train_state.opt_state)
        self.assertEqual(len(actual_leaves), len(expected_leaves))
        for actual, expected in zip(actual_leaves, expected_leaves):
            np.testing.assert_allclose(actual, expected)

    def test_disabled_checkpoint_does_not_write(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "state.msgpack"
            result = save_jax_training_checkpoint(
                path,
                train_state=None,
                rng=jax.random.PRNGKey(0),
                update=0,
                episode=0,
                global_step=0,
                enabled=False,
            )
            self.assertIsNone(result)
            self.assertFalse(path.exists())


if __name__ == "__main__":
    unittest.main()
