"""Does the soft margin improve the cost critic's fit?

Round 4 established that the cost critic underfits: out-of-sample RMSE was the same order
as the cost values themselves.  Round 7 established that the pure hinge gives zero gradient
before violation.  These two findings predict something that has never been tested: the
hinge's flat region also makes most samples carry NO information about the cost function, so
moving the knee inward should make the critic's regression target informative on more samples
and therefore reduce the RELATIVE fit error.

N9 and N16 are a clean single-variable pair -- same capacity (4800 kVA), same nominal budget
(raw 0.02), differing only in the training band (0.95/1.05 vs 0.96/1.04).  So this compares
them directly.

The comparison must be RELATIVE, not absolute: N16's band is narrower, so its raw cost values
are inherently larger and an absolute loss comparison would be meaningless.  We report
sqrt(cost_critic_loss) / |cost| , i.e. RMSE divided by the magnitude of what is being predicted.

N13/N19-style critic_epochs=8 runs are shown for reference but are NOT part of the controlled
comparison (they change the optimizer schedule, not the cost landscape).
"""

import glob
import json
import math

BASE = "/root/autodl-tmp/env_v6_1_runs"

PAIR = [
    ("N9  Q4800 band .95", "n9_qcap4800", False),
    ("N16 Q4800 band .96", "n16_q4800_softmargin", True),
]
REFERENCE = [
    ("N11 Q9600 band .95", "n11_qcap9600", False),
    ("N12 Q4800 x0.5", "n12_qcap4800_tight", False),
    ("N13 Q4800 x0.5 c8", "n13_q4800_allthree", False),
    ("N14 Q9600 x0.5", "n14_q9600_tight", False),
    ("N15 Q4800 x0.25", "n15_q4800_budget025", False),
]


def stats(run_dir):
    files = glob.glob(f"{BASE}/{run_dir}/*.metrics.jsonl")
    if not files:
        return None
    rows = [json.loads(line) for line in open(files[0])]
    out = {}
    for label, window in (("u200-400", slice(200, 400)), ("u800-1000", slice(800, 1000))):
        seg = rows[window]
        loss = [r["cost_critic_loss"] for r in seg if r.get("cost_critic_loss") is not None]
        cost = [abs(r["cost_before"]) for r in seg if r.get("cost_before") is not None]
        frac_nonzero = None
        cb = [r["cost_before"] for r in seg if r.get("cost_before") is not None]
        if cb:
            frac_nonzero = sum(1 for x in cb if x > 1e-9) / len(cb)
        if not loss or not cost:
            continue
        rmse = math.sqrt(sum(loss) / len(loss))
        mag = sum(cost) / len(cost)
        out[label] = (rmse, mag, rmse / mag if mag > 0 else float("nan"), frac_nonzero)
    return out


def show(rows, title):
    print(f"\n{title}")
    print(f"{'run':22s} {'window':10s} {'critic RMSE':>12} {'|cost|':>10} {'RMSE/|cost|':>12} {'frac cost>0':>12}")
    for name, d, _soft in rows:
        s = stats(d)
        if s is None:
            print(f"{name:22s} (missing)")
            continue
        for w, (rmse, mag, rel, frac) in s.items():
            f = "n/a" if frac is None else f"{frac:.3f}"
            print(f"{name:22s} {w:10s} {rmse:12.4f} {mag:10.4f} {rel:12.3f} {f:>12}")


show(PAIR, "CONTROLLED PAIR (only the training band differs)")
show(REFERENCE, "reference (not controlled -- capacity/budget/critic_epochs also differ)")

print(
    "\nPrediction being tested: if the hinge's flat region is what starves the cost critic,\n"
    "then N16 (knee at 0.96) should show a LOWER RMSE/|cost| than N9 (knee at 0.95), and a\n"
    "HIGHER fraction of updates with nonzero sampled cost.  A null result would mean the\n"
    "critic's difficulty is not caused by the flat region, and the soft margin helps purely\n"
    "through the policy gradient rather than through better cost prediction."
)
