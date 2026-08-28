# MATD3 a5 Timing Counterfactual Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add and run a fail-closed post-hoc evaluator that preserves MATD3's daily a5 order budget while changing only order timing.

**Architecture:** A two-pass evaluator first records deterministic MATD3 actions on each locked test day, then builds pure NumPy a5 timing variants. Existing fixed-scenario evaluation code replays the recorded a0-a4 and transformed a5 trajectories, preserving the established metrics and split checks. Results go to a new post-hoc directory and the locked `final/` files are hash-checked against mutation.

**Tech Stack:** Python 3.10, NumPy, PyTorch, existing `MicrogridVecEnv` and fixed-scenario evaluator, `unittest`.

## Global Constraints

- Use the seed-30 MATD3 checkpoint at 30,000 episodes.
- Use only test days 1, 7, 14, and 24 with seed 5200.
- Replay recorded a0-a4 exactly; only a5 may change.
- Transform only hours 0-19 and leave hours 20-23 unchanged.
- Preserve every agent's hours-0-19 a5 sum within `1e-6`.
- Apply one common time permutation to the four-agent a5 vector.
- Never overwrite the locked `final/` artifacts.
- Treat this as a single-seed post-hoc diagnostic, not model selection.

---

### Task 1: Pure a5 timing transformations

**Files:**
- Create: `scripts/evaluate_matd3_order_timing.py`
- Create: `tests/test_matd3_order_timing.py`

**Interfaces:**
- Produces: `build_a5_variants(actions: np.ndarray, *, usable_steps: int = 20, lag: int = 4, shuffle_seed: int = 20260715) -> dict[str, np.ndarray]`
- Produces: `validate_variant_invariants(normal: np.ndarray, variants: Mapping[str, np.ndarray], *, usable_steps: int = 20, atol: float = 1e-6) -> dict[str, object]`

- [ ] **Step 1: Write failing transformation tests**

```python
def test_variants_preserve_non_a5_actions_boundary_and_budget():
    actions = np.arange(24 * 4 * 6, dtype=np.float32).reshape(24, 4, 6) / 1000
    variants = timing.build_a5_variants(actions)
    for transformed in variants.values():
        np.testing.assert_array_equal(transformed[:, :, :5], actions[:, :, :5])
        np.testing.assert_array_equal(transformed[20:, :, 5], actions[20:, :, 5])
        np.testing.assert_allclose(
            transformed[:20, :, 5].sum(axis=0),
            actions[:20, :, 5].sum(axis=0),
            atol=1e-6,
        )

def test_shift_and_constant_mean_definitions():
    actions = np.zeros((24, 4, 6), dtype=np.float32)
    actions[:20, :, 5] = np.arange(20, dtype=np.float32)[:, None]
    variants = timing.build_a5_variants(actions)
    np.testing.assert_array_equal(
        variants["shift_earlier_4"][:20, 0, 5], np.roll(actions[:20, 0, 5], -4)
    )
    np.testing.assert_array_equal(
        variants["shift_later_4"][:20, 0, 5], np.roll(actions[:20, 0, 5], 4)
    )
    np.testing.assert_allclose(
        variants["constant_mean"][:20, :, 5], actions[:20, :, 5].mean(axis=0)
    )
```

- [ ] **Step 2: Run the focused tests and verify RED**

Run: `python -m unittest tests.test_matd3_order_timing -v`

Expected: import or attribute failure because the evaluator does not exist.

- [ ] **Step 3: Implement transformations and invariants**

```python
VARIANT_NAMES = (
    "normal_replay", "shift_earlier_4", "shift_later_4",
    "shuffle_fixed", "constant_mean",
)

def build_a5_variants(actions, *, usable_steps=20, lag=4, shuffle_seed=20260715):
    normal = np.asarray(actions, dtype=np.float32)
    if normal.shape != (24, 4, 6) or not np.isfinite(normal).all():
        raise ValueError("actions must be finite with shape (24, 4, 6)")
    output = {"normal_replay": normal.copy()}
    for name, indices in {
        "shift_earlier_4": np.roll(np.arange(usable_steps), -lag),
        "shift_later_4": np.roll(np.arange(usable_steps), lag),
        "shuffle_fixed": np.random.default_rng(shuffle_seed).permutation(usable_steps),
    }.items():
        transformed = normal.copy()
        transformed[:usable_steps, :, 5] = normal[indices, :, 5]
        output[name] = transformed
    constant = normal.copy()
    constant[:usable_steps, :, 5] = normal[:usable_steps, :, 5].mean(axis=0)
    output["constant_mean"] = constant
    validate_variant_invariants(normal, output, usable_steps=usable_steps)
    return output
```

- [ ] **Step 4: Run focused tests and verify GREEN**

Run: `python -m unittest tests.test_matd3_order_timing -v`

Expected: all transformation tests pass.

- [ ] **Step 5: Commit Task 1**

```bash
git add scripts/evaluate_matd3_order_timing.py tests/test_matd3_order_timing.py
git commit -m "test: define MATD3 order timing transformations"
```

### Task 2: Locked rollout, reporting, and fail-closed output

**Files:**
- Modify: `scripts/evaluate_matd3_order_timing.py`
- Modify: `tests/test_matd3_order_timing.py`

**Interfaces:**
- Consumes: `build_a5_variants` and `validate_variant_invariants` from Task 1.
- Produces: `collect_action_trajectory(action_fn, base_overrides, scenario) -> np.ndarray`
- Produces: `make_replay_policy(trajectories_by_day: Mapping[int, np.ndarray]) -> Callable`
- Produces CLI: `python scripts/evaluate_matd3_order_timing.py --root RESULT_ROOT`

- [ ] **Step 1: Add failing replay and output-safety tests**

```python
def test_replay_policy_selects_day_and_hour_without_mutation():
    trajectories = {1: np.zeros((24, 4, 6), np.float32), 7: np.ones((24, 4, 6), np.float32)}
    policy = timing.make_replay_policy(trajectories)
    context = SimpleNamespace(episode_step=3, config={"italian_day_indices": (7,)})
    np.testing.assert_array_equal(policy(np.empty((4, 19)), context), trajectories[7][3])

def test_existing_output_directory_is_rejected():
    with tempfile.TemporaryDirectory() as directory:
        output = Path(directory) / "posthoc"
        output.mkdir()
        with self.assertRaises(FileExistsError):
            timing.require_new_output_directory(output)
```

- [ ] **Step 2: Run focused tests and verify RED**

Run: `python -m unittest tests.test_matd3_order_timing -v`

Expected: failures for missing replay and output-safety functions.

- [ ] **Step 3: Implement the two-pass evaluator**

Implementation requirements:

```python
scenarios = build_scenarios(TEST_DAYS, TEST_NOISE_SEEDS)
normal_actions = {
    scenario.day: collect_action_trajectory(matd3_policy, overrides, scenario)
    for scenario in scenarios
}
variant_actions = {
    name: {day: build_a5_variants(actions)[name] for day, actions in normal_actions.items()}
    for name in VARIANT_NAMES
}
for name, trajectories in variant_actions.items():
    result = evaluate_contextual_policy(
        make_replay_policy(trajectories), overrides, scenarios,
        algorithm=f"MATD3-{name}", split_name="test",
    )
```

Before evaluation, hash every regular file under `<root>/final`. After writing
`summary.json`, `episodes.json`, and `report.md` to the new post-hoc directory,
rehash `final/` and require exact equality. Compare `normal_replay` against the
locked MATD3 per-day returns and aggregate metrics; fail if return or total cost
differs by more than `1e-4`.

- [ ] **Step 4: Run focused and full tests**

Run: `python -m unittest tests.test_matd3_order_timing -v`

Expected: all timing tests pass.

Run: `OMP_NUM_THREADS=1 python -m unittest discover -s tests -v`

Expected: existing suite and new timing tests all pass.

- [ ] **Step 5: Commit Task 2**

```bash
git add scripts/evaluate_matd3_order_timing.py tests/test_matd3_order_timing.py
git commit -m "feat: evaluate MATD3 hydrogen order timing"
```

### Task 3: Run the post-hoc diagnostic and verify results

**Files:**
- Create at runtime only: `/root/autodl-tmp/fair-stas-h2-action-order-20260715/posthoc_matd3_a5_timing_20260715/`

**Interfaces:**
- Consumes the Task 2 CLI and the existing 30k MATD3 checkpoint.
- Produces isolated `summary.json`, `episodes.json`, and `report.md` artifacts.

- [ ] **Step 1: Run the evaluator once**

```bash
cd /root/autodl-tmp/hypermarl-stas-corrected-20260711/HyperMARL-main
OMP_NUM_THREADS=1 PYTHONPATH=. python scripts/evaluate_matd3_order_timing.py \
  --root /root/autodl-tmp/fair-stas-h2-action-order-20260715
```

Expected: five variants complete on four test days and the script exits zero.

- [ ] **Step 2: Verify artifacts and invariants**

Run a read-only verifier that asserts:

```python
assert set(summary["variants"]) == set(VARIANT_NAMES)
assert summary["scenarios"] == {"days": [1, 7, 14, 24], "seed": 5200}
assert summary["normal_replay_matches_locked"] is True
assert summary["final_artifacts_unchanged"] is True
assert all(item["budget_preserved"] for item in summary["invariants"].values())
```

Expected: all assertions pass, every metric is finite, and each variant has four
24-step episodes.

- [ ] **Step 3: Interpret timing sensitivity**

Compare each transformed variant against `normal_replay` for return, total cost,
external H2, realized requested quantity, delivered H2, and t+4 correlation.
State whether timing has a material effect, distinguish timing from realized
clipping/matching changes, and preserve the single-seed post-hoc caveat.

- [ ] **Step 4: Commit any report-format correction only if verification exposed one**

If no code correction is needed, do not create an empty commit. If needed:

```bash
git add scripts/evaluate_matd3_order_timing.py tests/test_matd3_order_timing.py
git commit -m "fix: harden MATD3 timing diagnostic report"
```
