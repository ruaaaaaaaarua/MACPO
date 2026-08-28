# STAS-MAPPO

STAS-MAPPO keeps the existing microgrid environment and MAPPO training flow, then
adds a spatial-temporal reward decomposition model before advantage estimation.

The reward model is trained from full rollout chunks with the global discounted
episode return as its target. Its output is an agent-by-time credit reward matrix
that can be blended with the original environment reward for MAPPO GAE.

The implementation intentionally avoids importing PyTorch from inside JAX `jit`
functions. Rollouts are collected first, the STAS reward model runs in PyTorch on
NumPy arrays, and the redistributed reward is passed back to the JAX MAPPO update.

Run the minimal check from the repository root:

```bash
python baselines/STAS-MAPPO/smoke_test.py
```

The smoke test always checks the microgrid rollout and STAS credit output. It also
runs one MAPPO update when the JAX stack (`jax`, `flax`, `optax`, `distrax`) is
available in the active Python environment.
