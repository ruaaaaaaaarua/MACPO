"""Aggregate the sharded 200-day evaluation into a single report."""

from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path

import numpy as np

OUT = Path("/root/autodl-tmp/env_v6_diag_20260726/out")
RAW_BUDGET = 0.02


def wilson(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval -- valid near 0 and 1, unlike the normal approximation."""
    if total == 0:
        return (0.0, 0.0)
    p = successes / total
    denom = 1.0 + z * z / total
    centre = (p + z * z / (2 * total)) / denom
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def describe(summaries: list[dict], label: str) -> dict:
    n = len(summaries)
    vcost = np.array([s["daily_voltage_cost"] for s in summaries])
    vmin = np.array([s["voltage_min_pu"] for s in summaries], dtype=float)
    vmax = np.array([s["voltage_max_pu"] for s in summaries], dtype=float)
    econ = np.array([s["economic_cost"] for s in summaries])
    safe = sum(1 for s in summaries if s["safe_day"])
    strict = sum(1 for s in summaries if s["strictly_within_limits"])
    pf = sum(1 for s in summaries if s["pf_failure_rate"] > 0)
    lo, hi = wilson(safe, n)
    slo, shi = wilson(strict, n)
    hours = Counter()
    for s in summaries:
        hours.update(s["violating_hours"])
    print(f"\n===== {label}  (n={n} days) =====")
    print(
        f"  safe days (vcost<={RAW_BUDGET}) : {safe}/{n} = {safe/n:6.1%}   "
        f"95% CI [{lo:.1%}, {hi:.1%}]"
    )
    print(
        f"  strictly inside [0.95,1.05]    : {strict}/{n} = {strict/n:6.1%}   "
        f"95% CI [{slo:.1%}, {shi:.1%}]"
    )
    print(f"  power-flow failures            : {pf}/{n}")
    print(
        f"  raw daily voltage cost         : mean={vcost.mean():.4f}  "
        f"median={np.median(vcost):.4f}  p90={np.percentile(vcost,90):.4f}  "
        f"max={vcost.max():.4f}   (budget {RAW_BUDGET})"
    )
    print(f"  mean / budget                  : {vcost.mean()/RAW_BUDGET:.1f}x")
    print(
        f"  voltage_min_pu                 : mean={vmin.mean():.4f}  "
        f"p10={np.percentile(vmin,10):.4f}  min={vmin.min():.4f}"
    )
    print(
        f"  voltage_max_pu                 : mean={vmax.mean():.4f}  "
        f"p90={np.percentile(vmax,90):.4f}  max={vmax.max():.4f}  (limit 1.05)"
    )
    print(
        f"  economic cost                  : mean={econ.mean():.4g}  "
        f"std={econ.std():.4g}"
    )
    if hours:
        top = sorted(hours.items(), key=lambda kv: -kv[1])[:10]
        print("  violating-hour histogram (top): " + ", ".join(f"h{h}:{c}" for h, c in top))
    return {
        "label": label,
        "n": n,
        "safe_days": safe,
        "safe_rate": safe / n,
        "safe_ci95": [lo, hi],
        "strict_days": strict,
        "strict_rate": strict / n,
        "strict_ci95": [slo, shi],
        "pf_failure_days": pf,
        "vcost_mean": float(vcost.mean()),
        "vcost_median": float(np.median(vcost)),
        "vcost_p90": float(np.percentile(vcost, 90)),
        "vcost_max": float(vcost.max()),
        "vmin_mean": float(vmin.mean()),
        "vmin_min": float(vmin.min()),
        "vmax_max": float(vmax.max()),
        "econ_mean": float(econ.mean()),
        "violating_hour_histogram": dict(sorted(hours.items())),
    }


def main() -> None:
    shards = sorted(OUT.glob("wide_*.json"))
    merged: dict[str, list[dict]] = {"deterministic": [], "stochastic": []}
    for shard in shards:
        data = json.loads(shard.read_text())
        for mode, payload in data["results"].items():
            merged[mode].extend(payload["summaries"])
    report = {}
    for mode, summaries in merged.items():
        summaries.sort(key=lambda s: s["seed"])
        report[mode] = describe(summaries, mode)

    det = {s["seed"]: s for s in merged["deterministic"]}
    worst = sorted(merged["deterministic"], key=lambda s: -s["daily_voltage_cost"])[:12]
    print("\n----- worst deterministic days -----")
    print("  seed   vcost    vmin    vmax     econ       violating hours")
    for s in worst:
        print(
            f"  {s['seed']:4d}  {s['daily_voltage_cost']:7.4f}  {s['voltage_min_pu']:.4f}  "
            f"{s['voltage_max_pu']:.4f}  {s['economic_cost']:9.4g}   {s['violating_hours']}"
        )
    report["worst_deterministic_seeds"] = [s["seed"] for s in worst]

    print("\n----- the three published evaluation days, in context -----")
    n = len(merged["deterministic"])
    ranks = sorted(merged["deterministic"], key=lambda s: s["daily_voltage_cost"])
    order = {s["seed"]: i for i, s in enumerate(ranks)}
    for seed in (30, 31, 32):
        if seed in det:
            s = det[seed]
            print(
                f"  seed {seed}: vcost={s['daily_voltage_cost']:.4f}  "
                f"percentile among {n} days = {100*order[seed]/(n-1):.0f}th "
                f"(0th = safest)"
            )

    path = OUT / "wide_eval_summary.json"
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
