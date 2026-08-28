"""Did giving the critics 8 gradient steps per update actually fit the cost signal?

N6 is N1 with ``critic_epochs`` 1 -> 8 and nothing else changed, so the comparison
is one-variable.  Three questions, in order of how much they matter:

1. Out-of-sample.  ``cost_critic_loss`` is measured on a freshly sampled rollout
   *before* any of that update's critic steps, so it is comparable across the two
   runs and across every run recorded before ``critic_epochs`` existed.
2. Generalization gap.  ``cost_critic_loss_last`` is the in-sample fit after the 8
   passes.  A large gap between it and the next update's out-of-sample loss means
   the critic is memorising the batch rather than learning the cost function --
   which would mean the residual is irreducible given the observation, and the
   next lever is features (electrical lookahead), not more epochs.
3. Did it transfer to the thing we care about -- RMSE relative to the size of the
   cost signal itself?  The 2026-07-30 diagnosis found RMSE/signal ~ 100% on every
   run; that ratio is the number the whole hypothesis is about.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

BASE = Path("/root/autodl-tmp/env_v6_1_runs")
N6 = BASE / "n6_criticfit/v6q_nocomm_gru_macpo_criticfit.metrics.jsonl"
N1 = BASE / "n1_q/v6q_nocomm_gru_macpo_bigbatch.metrics.jsonl"


def load(p: Path) -> list[dict]:
    if not p.exists():
        return []
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else float("nan")


def window(rows, lo, hi, key):
    return mean([r.get(key) for r in rows[lo:hi]])


def main() -> None:
    print(__doc__)
    n6, n1 = load(N6), load(N1)
    print(f"N6 updates: {len(n6)}   N1 updates: {len(n1)}")
    if not n6 or not n1:
        return

    print("\n### 1. out-of-sample cost_critic_loss at matched updates")
    print(f"{'updates':>12} {'N1':>12} {'N6':>12} {'ratio N1/N6':>13}")
    upto = min(len(n6), len(n1))
    bounds = [(0, 50), (50, 100), (100, 200), (200, 400), (400, 600), (600, 800), (800, 1000)]
    for lo, hi in bounds:
        if lo >= upto:
            break
        hi = min(hi, upto)
        a, b = window(n1, lo, hi, "cost_critic_loss"), window(n6, lo, hi, "cost_critic_loss")
        r = a / b if b and b == b and b != 0 else float("nan")
        print(f"{f'{lo}-{hi}':>12} {a:>12.2f} {b:>12.2f} {r:>13.2f}")

    print("\n### 2. N6 in-sample (after 8 passes) vs out-of-sample (next fresh rollout)")
    print(f"{'updates':>12} {'in-sample':>12} {'out-of-sample':>14} {'gap x':>8}")
    for lo, hi in bounds:
        if lo >= len(n6):
            break
        hi = min(hi, len(n6))
        ins = window(n6, lo, hi, "cost_critic_loss_last")
        out = window(n6, lo, hi, "cost_critic_loss")
        g = out / ins if ins and ins == ins and ins != 0 else float("nan")
        print(f"{f'{lo}-{hi}':>12} {ins:>12.2f} {out:>14.2f} {g:>8.2f}")

    print("\n### 3. RMSE vs the size of the cost signal (last 50 updates)")
    print(f"{'run':>6} {'cost_before':>12} {'critic RMSE':>12} {'RMSE/signal':>12}")
    for label, rows in (("N1", n1), ("N6", n6)):
        sig = window(rows, len(rows) - 50, len(rows), "cost_before")
        loss = window(rows, len(rows) - 50, len(rows), "cost_critic_loss")
        rmse = math.sqrt(loss) if loss == loss and loss >= 0 else float("nan")
        print(f"{label:>6} {sig:>12.3f} {rmse:>12.3f} {rmse / sig if sig else float('nan'):>11.0%}")

    print("\n### 4. training-side safety and economics (last 50 updates)")
    print(f"{'run':>6} {'vcost_raw':>11} {'econ':>12} {'mode mix (last 50)':>24}")
    for label, rows in (("N1", n1), ("N6", n6)):
        tail = rows[-50:]
        modes = {}
        for r in tail:
            modes[r.get("mode")] = modes.get(r.get("mode"), 0) + 1
        v = window(rows, len(rows) - 50, len(rows), "daily_voltage_cost_raw")
        e = window(rows, len(rows) - 50, len(rows), "daily_economic_cost")
        if e != e:
            e = window(rows, len(rows) - 50, len(rows), "econ")
        print(f"{label:>6} {v:>11.4f} {e:>12.4g} {str(modes):>24}")


if __name__ == "__main__":
    main()
