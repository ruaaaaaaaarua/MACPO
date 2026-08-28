"""Offline diagnosis: how optimistic is the CPO linearised cost surrogate?

Motivation
----------
Round 1 refuted the textbook CPO feasibility branch in this environment: with
``constrained_reward`` allowed, the sampled cost diverged even though the
surrogate kept claiming large cost drops (one update claimed ``cost_after``
48.4 while the freshly sampled ``cost_before`` was 183.8).  The conservative
branch ("infeasible -> spend the whole trust region on cost") works, but it is a
blunt fix: it distrusts the surrogate unconditionally.  The principled fix is the
classical trust-region ratio test, which distrusts the surrogate *in proportion
to how wrong it just was*:

    rho = actual cost decrease / predicted cost decrease

with ``max_kl`` shrunk when rho is small and grown when rho is near 1 and the
step is saturating.  This script measures whether rho is actually informative
here before any of that gets implemented.

Confound (important)
--------------------
``cost_before[t]`` and ``cost_after[t]`` are both surrogate values on rollout
batch *t*.  The only unbiased measurement of the post-update cost is
``cost_before[t+1]``, which is sampled on a *fresh* rollout drawn over different
Italian day types.  Across-day variance of the daily voltage cost spans 0 to
~1.9, i.e. far larger than a single update's intended step, so per-update rho is
extremely noisy.  Windowed rho (actual drop across W updates vs the sum of
predicted drops over the same W) cancels most of that day-sampling noise and is
the number to trust.  Both are reported.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

RUNS = {
    "E1 (textbook branch, diverged)": (
        "/root/autodl-tmp/env_v6_1_runs/v6_nocomm_gru_macpo_bigbatch.metrics.jsonl"
    ),
    "E2 (textbook + Q, diverged)": (
        "/root/autodl-tmp/env_v6_1_runs/v6q_nocomm_gru_macpo_bigbatch.metrics.jsonl"
    ),
    "retry1 (conservative branch)": (
        "/root/autodl-tmp/env_v6_1_runs/retry1/v6_nocomm_gru_macpo_bigbatch.metrics.jsonl"
    ),
    "N1 (+Q action)": (
        "/root/autodl-tmp/env_v6_1_runs/n1_q/v6q_nocomm_gru_macpo_bigbatch.metrics.jsonl"
    ),
    "N3 (+tight budget)": (
        "/root/autodl-tmp/env_v6_1_runs/n3_tight/v6_nocomm_gru_macpo_tightbudget.metrics.jsonl"
    ),
    "N6 (critic_epochs=8)": (
        "/root/autodl-tmp/env_v6_1_runs/n6_criticfit/v6q_nocomm_gru_macpo_criticfit.metrics.jsonl"
    ),
}

WINDOW = 20
MIN_PREDICTED_DROP = 1e-6


def load(path: str) -> list[dict]:
    p = Path(path)
    if not p.exists():
        return []
    return [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]


def describe(name: str, rows: list[dict]) -> None:
    if len(rows) < WINDOW + 2:
        print(f"\n### {name}: only {len(rows)} rows, skipped")
        return

    before = np.array([r["cost_before"] for r in rows], dtype=np.float64)
    after = np.array([r["cost_after"] for r in rows], dtype=np.float64)
    kl = np.array([r.get("kl_after") or 0.0 for r in rows], dtype=np.float64)
    modes = [r["mode"] for r in rows]

    # predicted drop on batch t; actual drop measured by the next fresh rollout
    predicted = before[:-1] - after[:-1]
    actual = before[:-1] - before[1:]
    ok = predicted > MIN_PREDICTED_DROP

    print(f"\n### {name}   ({len(rows)} updates)")
    print(f"  mode counts: {dict((m, modes.count(m)) for m in sorted(set(modes)))}")
    print(
        f"  cost_before: first50 mean {before[:50].mean():.3f}"
        f" -> last50 mean {before[-50:].mean():.3f}"
    )
    print(
        f"  updates with a predicted cost drop: {int(ok.sum())}/{len(predicted)}"
        f"  (median predicted drop {np.median(predicted[ok]):.3f})"
        if ok.any()
        else "  no update predicted a cost drop"
    )
    if not ok.any():
        return

    rho = actual[ok] / predicted[ok]
    print(
        f"  per-update rho: median {np.median(rho):+.3f}, "
        f"mean {rho.mean():+.3f}, IQR [{np.percentile(rho, 25):+.3f}, "
        f"{np.percentile(rho, 75):+.3f}], frac(rho<0.25) {np.mean(rho < 0.25):.2f}, "
        f"frac(rho<0) {np.mean(rho < 0):.2f}"
    )

    # windowed rho: cancels day-sampling noise
    w_rho = []
    for s in range(0, len(predicted) - WINDOW, WINDOW):
        pred_w = predicted[s : s + WINDOW].sum()
        act_w = before[s] - before[s + WINDOW]
        if pred_w > MIN_PREDICTED_DROP:
            w_rho.append(act_w / pred_w)
    if w_rho:
        w = np.array(w_rho)
        print(
            f"  windowed rho (W={WINDOW}, n={len(w)}): median {np.median(w):+.3f}, "
            f"IQR [{np.percentile(w, 25):+.3f}, {np.percentile(w, 75):+.3f}], "
            f"frac(<0.25) {np.mean(w < 0.25):.2f}"
        )

    # does the surrogate's optimism grow with the step size it took?
    kl_ok = kl[:-1][ok]
    if np.ptp(kl_ok) > 0:
        edges = np.percentile(kl_ok, [0, 25, 50, 75, 100])
        print("  per-update rho by kl_after quartile (step size -> reliability):")
        for i in range(4):
            lo, hi = edges[i], edges[i + 1]
            sel = (kl_ok >= lo) & (kl_ok <= hi if i == 3 else kl_ok < hi)
            if sel.sum() >= 5:
                print(
                    f"    kl in [{lo:.4f},{hi:.4f}]  n={int(sel.sum()):4d}  "
                    f"median rho {np.median(rho[sel]):+.3f}  "
                    f"frac(<0.25) {np.mean(rho[sel] < 0.25):.2f}"
                )

    # absolute optimism: claimed post-step cost vs the cost actually sampled next
    claimed = after[:-1]
    realised = before[1:]
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = realised / np.maximum(claimed, 1e-9)
    finite = np.isfinite(ratio) & (claimed > 1e-3)
    if finite.any():
        print(
            f"  realised/claimed post-step cost: median {np.median(ratio[finite]):.2f}x "
            f"(>1 means the surrogate under-predicted the true cost), "
            f"90th pct {np.percentile(ratio[finite], 90):.2f}x"
        )


def main() -> None:
    print(__doc__)
    for name, path in RUNS.items():
        describe(name, load(path))
    print(
        "\nReading guide: rho ~ 1 means the surrogate is trustworthy and an adaptive\n"
        "trust region would mostly leave max_kl alone; rho persistently << 1 (or the\n"
        "quartile table showing rho degrading as kl grows) means step size is the\n"
        "controlling variable and the ratio test would add real value over the\n"
        "current unconditional conservatism."
    )


if __name__ == "__main__":
    sys.exit(main())
