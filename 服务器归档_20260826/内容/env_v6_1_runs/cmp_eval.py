"""Compare per-day evaluations across the v6.1 experiment series.

Prints the headline safe-day counts, the economics, and -- the thing the round-3
hypotheses actually hinge on -- which day types each variant fixed relative to the
others.  The six "hard" day types (1, 4, 15, 17, 23, 24) were identified in round 2
as physically infeasible on real power alone.
"""

import json
from pathlib import Path

BASE = Path("/root/autodl-tmp/env_v6_1_runs/eval")
RUNS = [
    ("baseline", "baseline_final.json"),
    ("retry1", "retry1_final.json"),
    ("N1 +Q1200", "n1_q_final.json"),
    ("N3 tightbudget", "n3_tight_final.json"),
    ("N4 Q+tight", "n4_combo_final.json"),
    ("N5 Q2400", "n5_qcap_final.json"),
    ("N6 criticfit", "n6_criticfit_final.json"),
    ("N7 Q2400+tight", "n7_qcap2400_tight_final.json"),
    ("N9 Q4800", "n9_qcap4800_final.json"),
    ("N10 allthree", "n10_allthree_final.json"),
    ("N11 Q9600", "n11_qcap9600_final.json"),
    ("N12 Q4800+tight", "n12_qcap4800_tight_final.json"),
    ("N13 Q4800+all", "n13_q4800_allthree_final.json"),
    ("N14 Q9600+tight", "n14_q9600_tight_final.json"),
    ("N15 Q4800+bud.25", "n15_q4800_budget025_final.json"),
    ("N16 Q4800+soft", "n16_q4800_softmargin_final.json"),
    ("N17 soft seed31", "n17_soft_seed31_final.json"),
    ("N18 Q2400+soft", "n18_q2400_soft_final.json"),
    ("N19 soft+critic8", "n19_soft_criticfit_final.json"),
    ("N20 softc8 s31", "n20_softc8_s31_final.json"),
    ("N21 softc8 s32", "n21_softc8_s32_final.json"),
    ("N22 Q2400soft s31", "n22_q2400_soft_s31_final.json"),
    ("N23 knee.955", "n23_knee955_final.json"),
    ("N24 knee.97", "n24_knee97_final.json"),
    ("N25 knee.98", "n25_knee98_final.json"),
    ("N26 knee.97 s31", "n26_knee97_s31_final.json"),
    ("N27 knee.97 s32", "n27_knee97_s32_final.json"),
    ("N28 knee.97 Q2400", "n28_knee97_q2400_final.json"),
    ("B1 mappo (no safety)", "b1_mappo_knee97_final.json"),
    ("B2 penalty c=1", "b2_mappopen1_knee97_final.json"),
    ("B3 penalty c=10", "b3_mappopen10_knee97_final.json"),
    ("B4 lagrangian", "b4_lagr_knee97_final.json"),
    ("B5 penalty c=0.1", "b5_mappopen01_knee97_final.json"),
    ("B6 penalty c=0.3", "b6_mappopen03_knee97_final.json"),
    ("B7 penalty c=0.03", "b7_mappopen003_knee97_final.json"),
    ("B8 penalty c=0.01", "b8_mappopen001_knee97_final.json"),
    ("B9 c=0.01 s31", "b9_mappopen001_s31_final.json"),
    ("B10 c=0.01 s32", "b10_mappopen001_s32_final.json"),
]
HARD = {1, 4, 15, 17, 23, 24}


def load(name):
    p = BASE / name
    if not p.exists():
        return None
    return json.loads(p.read_text())


def days(doc):
    """Return {day_index: record} regardless of the exact container key."""
    for key in ("days", "per_day", "results", "day_results"):
        if key in doc:
            v = doc[key]
            if isinstance(v, dict):
                return {int(k): rec for k, rec in v.items()}
            return {int(rec.get("day", rec.get("day_index", i))): rec for i, rec in enumerate(v)}
    raise KeyError(f"no per-day container; top-level keys = {sorted(doc)}")


def is_safe(rec):
    for k in ("safe_all_seeds", "safe", "is_safe", "safe_day"):
        if k in rec:
            return bool(rec[k])
    raise KeyError(f"no safe flag; keys = {sorted(rec)}")


def vcost(rec):
    for k in ("voltage_cost_raw", "raw_voltage_cost", "daily_voltage_cost_raw", "vcost_raw", "vcost_mean"):
        if k in rec:
            return float(rec[k])
    return float("nan")


def econ(rec):
    for k in ("economic_cost", "econ", "daily_econ_cost", "economic", "econ_mean"):
        if k in rec:
            return float(rec[k])
    return float("nan")


def main():
    table = {}
    for label, fname in RUNS:
        doc = load(fname)
        if doc is None:
            continue
        d = days(doc)
        table[label] = d

    if not table:
        print("no eval files found in", BASE)
        return

    print(f"{'variant':<18} {'safe':>7} {'hard-safe':>10} {'vcost mean':>11} {'vcost max':>10} {'econ mean':>11}")
    for label, d in table.items():
        safe = [k for k, r in d.items() if is_safe(r)]
        hs = sorted(set(safe) & HARD)
        vs = [vcost(r) for r in d.values()]
        es = [econ(r) for r in d.values()]
        es = [e for e in es if e == e]
        print(
            f"{label:<18} {len(safe):>3}/{len(d):<3} {str(hs):>10} "
            f"{sum(vs)/len(vs):>11.4f} {max(vs):>10.4f} "
            f"{(sum(es)/len(es) if es else float('nan')):>11.3e}"
        )

    print("\nsafe-day sets:")
    for label, d in table.items():
        print(f"  {label:<18} {sorted(k for k, r in d.items() if is_safe(r))}")

    print("\nper-day raw voltage cost on the six hard day types:")
    hdr = "  day  " + "".join(f"{l[:12]:>14}" for l in table)
    print(hdr)
    for day in sorted(HARD):
        cells = ""
        for d in table.values():
            r = d.get(day)
            cells += f"{vcost(r):>14.4f}" if r else f"{'-':>14}"
        print(f"  {day:>3}  {cells}")


if __name__ == "__main__":
    main()
