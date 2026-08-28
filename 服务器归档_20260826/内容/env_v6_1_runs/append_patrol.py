"""Append one patrol row for the round-5 runs to the table at the end of PLAN.md."""
import json
import subprocess
import sys
from pathlib import Path

BASE = Path("/root/autodl-tmp/env_v6_1_runs")
PLAN = BASE / "PLAN.md"
RUNS = [
    ("n24_ext3000", "v6q_nocomm_gru_macpo_softc8_knee97"),
    ("b8_ext3000", "v6q_nocomm_gru_mappopen001_knee97"),
    ("b5_ext3000", "v6q_nocomm_gru_mappopen01_knee97"),
    ("b2_ext3000", "v6q_nocomm_gru_mappopen1_knee97"),
    ("b4_ext3000", "v6q_nocomm_gru_lagr_knee97"),
]


def rows(run: str, variant: str) -> list[dict]:
    p = BASE / run / f"{variant}.metrics.jsonl"
    if not p.exists():
        return []
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


def tail_mean(rs: list[dict], key: str, n: int = 50) -> str:
    v = [r.get(key) for r in rs[-n:] if r.get(key) is not None]
    return f"{sum(v)/len(v):.4f}" if v else "-"


def modes(rs: list[dict], n: int = 50) -> str:
    if not rs:
        return "-"
    c: dict[str, int] = {}
    for r in rs[-n:]:
        m = str(r.get("mode"))
        c[m] = c.get(m, 0) + 1
    return "/".join(f"{k[:4]}{v}" for k, v in sorted(c.items()))


def main() -> None:
    note = sys.argv[1] if len(sys.argv) > 1 else ""
    when = subprocess.run(["date", "+%m-%d %H:%M"], capture_output=True, text=True).stdout.strip()
    cells = []
    for run, variant in RUNS:
        rs = rows(run, variant)
        cells.append((len(rs), tail_mean(rs, "daily_voltage_cost_raw"), modes(rs)))
    row = f"| {when} | " + " | ".join(
        f"{n} | {v} | {m}" for n, v, m in cells
    ) + f" | {note} |\n"
    with PLAN.open("a", encoding="utf-8") as fh:
        fh.write(row)
    print(row, end="")


if __name__ == "__main__":
    main()
