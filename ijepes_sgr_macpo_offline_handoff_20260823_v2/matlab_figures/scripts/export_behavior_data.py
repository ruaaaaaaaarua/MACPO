"""Export paper-authoritative Env-v6 evaluation data to compact CSV files.

Only the archived behavior report is read.  The script performs no training and
does not alter the manuscript or any archived result.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
FIGURE_DIR = SCRIPT_DIR.parent
WORKSPACE_DIR = FIGURE_DIR.parent.parent
REPORT_PATH = (
    WORKSPACE_DIR
    / "服务器归档_20260826"
    / "内容"
    / "env_v6_swiss_runs"
    / "long"
    / "env_v6_behavior_report.json"
)
DATA_DIR = FIGURE_DIR / "data"

VARIANTS = (
    ("v6_nocomm_gru_mappo", "GRU-MAPPO"),
    ("v6_nocomm_gru_mappo_penalty", "Fixed-penalty GRU-MAPPO"),
    ("v6_nocomm_gru_macpo", "SGR-MACPO"),
)


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    report = json.loads(REPORT_PATH.read_text(encoding="utf-8"))

    voltage_rows: list[dict[str, object]] = []
    h2_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []

    for variant, label in VARIANTS:
        result = report["variants"][variant]
        summary_rows.append(
            {
                "method": label,
                "safe_days": result["safe_days"],
                "evaluation_days": 3,
                "voltage_cost": result["three_day_voltage_cost"],
                "economic_cost_m_cny": sum(
                    result["summaries"][str(seed)]["economic_cost"]
                    for seed in (30, 31, 32)
                )
                / 1.0e6,
            }
        )

        rollout = result["rollouts"]["32"]["steps"]
        for step in rollout:
            voltage_rows.append(
                {
                    "method": label,
                    "hour": int(step["step"]) + 1,
                    "aggregate_pcc_p_mw": sum(step["pcc_p_kw"]) / 1000.0,
                    "aggregate_pcc_q_mvar": sum(step["pcc_q_kvar"]) / 1000.0,
                    "voltage_min_pu": step["voltage_min_pu"],
                    "voltage_max_pu": step["voltage_max_pu"],
                    "hourly_voltage_cost": step["voltage_cost"],
                }
            )

        for seed in (30, 31, 32):
            day = result["summaries"][str(seed)]
            h2_rows.append(
                {
                    "method": label,
                    "seed": seed,
                    "planned_mwh": day["planned_h2_order"] / 1000.0,
                    "emergency_mwh": day["emergency_h2_buy"] / 1000.0,
                    "late_mwh": day["late_h2_order"] / 1000.0,
                    "undelivered_mwh": day["pending_h2_energy"] / 1000.0,
                }
            )

    write_csv(
        DATA_DIR / "seed32_voltage_case.csv",
        [
            "method",
            "hour",
            "aggregate_pcc_p_mw",
            "aggregate_pcc_q_mvar",
            "voltage_min_pu",
            "voltage_max_pu",
            "hourly_voltage_cost",
        ],
        voltage_rows,
    )
    write_csv(
        DATA_DIR / "hydrogen_service_days.csv",
        ["method", "seed", "planned_mwh", "emergency_mwh", "late_mwh", "undelivered_mwh"],
        h2_rows,
    )
    write_csv(
        DATA_DIR / "paper_main_results.csv",
        ["method", "safe_days", "evaluation_days", "voltage_cost", "economic_cost_m_cny"],
        summary_rows,
    )

    counterfactual_path = DATA_DIR / "seed32_delivery_counterfactual.json"
    counterfactual_csv = None
    if counterfactual_path.is_file():
        counterfactual = json.loads(counterfactual_path.read_text(encoding="utf-8"))
        delivery_rows: list[dict[str, object]] = []
        for mode, label in (("delayed", "Delayed delivery"), ("instant", "Instant delivery")):
            for step in counterfactual[mode]["steps"]:
                delivery_rows.append(
                    {
                        "scenario": label,
                        "hour": int(step["step"]) + 1,
                        "h2_load_mwh": sum(step["e_h2_load"]) / 1000.0,
                        "electrolyzer_power_mw": sum(step["p_el"]) / 1000.0,
                        "planned_order_mwh": sum(step["h2_planned_external_order_energy"]) / 1000.0,
                        "emergency_supply_mwh": sum(step["h2_emergency_buy_energy"]) / 1000.0,
                        "late_order_mwh": sum(step["h2_late_order_energy"]) / 1000.0,
                        "tank_energy_mwh": sum(step["h2_level"]) * 33.33 / 1000.0,
                        "pending_energy_mwh": step["pending_h2_energy_total"] / 1000.0,
                        "voltage_min_pu": step["voltage_min_pu"],
                    }
                )
        counterfactual_csv = "seed32_delivery_counterfactual.csv"
        write_csv(
            DATA_DIR / counterfactual_csv,
            [
                "scenario",
                "hour",
                "h2_load_mwh",
                "electrolyzer_power_mw",
                "planned_order_mwh",
                "emergency_supply_mwh",
                "late_order_mwh",
                "tank_energy_mwh",
                "pending_energy_mwh",
                "voltage_min_pu",
            ],
            delivery_rows,
        )

    manifest = {
        "source": str(REPORT_PATH),
        "authority": "Current manuscript values take precedence over server-side logs.",
        "evaluation_seed_for_hourly_case": 32,
        "raw_daily_voltage_budget": report["raw_voltage_budget"],
        "files": [
            "seed32_voltage_case.csv",
            "hydrogen_service_days.csv",
            "paper_main_results.csv",
        ] + ([counterfactual_csv] if counterfactual_csv else []),
    }
    (DATA_DIR / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
