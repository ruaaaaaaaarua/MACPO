# MATD3 a5 Timing Counterfactual Design

## Objective

Determine whether MATD3's benefit from internal hydrogen ordering comes from the
timing of its `a5` sequence, rather than only from submitting a large daily order
budget. This is a post-hoc diagnostic and must not alter model selection or the
locked final-test artifacts.

## Locked scenarios and policy

- Use the trained seed-30 MATD3 checkpoint at 30,000 episodes.
- Evaluate only locked test days 1, 7, 14, and 24 with scenario seed 5200.
- Use deterministic actions with exploration noise disabled.
- Write results under a new post-hoc output directory; never overwrite `final/`.

## Two-pass counterfactual

For each test day, first run the normal policy and record its `(24, 4, 6)` action
trajectory. Counterfactual rollouts replay the recorded `a0` through `a4` exactly
at each hour and replace only `a5`. The policy is not queried again on the
counterfactual state, so downstream adaptation cannot contaminate the timing
test.

The usable order window is hours 0 through 19 because an order must satisfy
`t + 4 < 24`. Hours 20 through 23 remain unchanged in every variant. Transform
the four-agent `a5` vector with one common time permutation so cross-agent
coordination at a given source hour is preserved.

Variants:

1. `normal_replay`: recorded action trajectory unchanged.
2. `shift_earlier_4`: circularly move usable-window vectors four hours earlier.
3. `shift_later_4`: circularly move usable-window vectors four hours later.
4. `shuffle_fixed`: apply one deterministic permutation, seeded with 20260715,
   to the usable-window vectors.
5. `constant_mean`: replace each agent's usable-window `a5` values with that
   agent's mean over hours 0 through 19.

Circular shifts are restricted to the usable 20-hour window. They therefore
preserve each agent's normalized `a5` sum without moving values across the
horizon-clipped boundary.

## Invariants and measurements

Before rollout, require every variant to preserve each agent's usable-window
`a5` sum within `1e-6`; because `Qmax` is static and the action-to-quantity map is
linear, this also preserves the potential requested order budget.

Record per day and aggregate across four days:

- return, total/base cost, and external hydrogen purchases;
- internal hydrogen trade, requested/effective/clipped orders, pending, and
  delivered hydrogen;
- terminal SOC/H2 and order-versus-t+4-load correlation;
- potential order budget per agent and realized requested order quantity.

The normal replay must reproduce the locked MATD3 test result within numerical
tolerance. All values must be finite, every episode must contain exactly 24
steps, and existing final artifacts must remain byte-for-byte untouched.

## Interpretation

- If shuffled/constant/shifted variants are materially worse while preserving
  the order budget, timing contributes to MATD3's performance.
- If they are similar to normal replay, the gain primarily comes from maintaining
  an internal-order pipeline or from total order volume, not precise t+4 timing.
- A single shift direction winning does not by itself prove forecasting; inspect
  day-level results, realized clipping, delivery, and t+4 correlation.
- This remains a single-seed post-hoc diagnostic and is not a statistical claim.
