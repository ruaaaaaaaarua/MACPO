import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts.screen_swiss_mv import (
    assign_agents_to_pcc,
    discover_case_dirs,
    dynamic_gate,
    rank_static_candidates,
    select_pcc_sets,
    select_passing_candidate,
)


class SwissMVScreeningTest(unittest.TestCase):
    def test_discovers_only_complete_matpower_case_directories(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            complete = root / "10_2"
            incomplete = root / "11_0"
            complete.mkdir()
            incomplete.mkdir()
            for suffix in ("bus_data.csv", "branch_data.csv", "generator_data.csv"):
                (complete / f"10_2_{suffix}").touch()
            (incomplete / "11_0_bus_data.csv").touch()

            self.assertEqual(discover_case_dirs(root), [complete])

    def test_pcc_sets_are_deterministic_with_electrical_distance_tie_breaks(self):
        bus_ids = np.arange(10, 130, 10, dtype=np.int64)
        sensitivity = {int(bus): float(index) for index, bus in enumerate(bus_ids)}
        native_pd = {int(bus): 1.0 for bus in bus_ids}
        native_pd.update({20: 5.0, 60: 5.0, 100: 5.0})
        distance = {
            (int(first), int(second)): abs(float(first) - float(second))
            for first in bus_ids
            for second in bus_ids
        }

        first = select_pcc_sets(bus_ids, sensitivity, native_pd, distance)
        second = select_pcc_sets(bus_ids[::-1], sensitivity, native_pd, distance)

        self.assertEqual(first, second)
        self.assertEqual(
            first,
            [(20, 40, 10, 30), (60, 80, 50, 70), (100, 120, 90, 110)],
        )

    def test_largest_import_agent_is_assigned_to_strongest_bus(self):
        assigned = assign_agents_to_pcc(
            pcc_bus_ids=(20, 40, 10, 30),
            sensitivity={10: 0.1, 20: 0.2, 30: 0.3, 40: 0.4},
            agent_import_p95=[1.0, 4.0, 2.0, 3.0],
        )

        self.assertEqual(assigned, (40, 10, 30, 20))

    def test_four_eligible_load_buses_still_produce_one_pcc_set(self):
        buses = [10, 20, 30, 40]
        sensitivity = {bus: float(index) for index, bus in enumerate(buses)}
        native_pd = {10: 1.0, 20: 5.0, 30: 1.0, 40: 1.0}
        distance = {
            (first, second): abs(float(first) - float(second))
            for first in buses
            for second in buses
        }

        self.assertEqual(
            select_pcc_sets(buses, sensitivity, native_pd, distance),
            [(20, 40, 10, 30)],
        )

    def test_dynamic_gate_uses_raw_point_zero_two_budget(self):
        nominal = [
            {"daily_voltage_cost": cost, "pf_converged": True, "steps": 24}
            for cost in (0.021, 0.02, 0.03)
        ]
        reference = [
            {"daily_voltage_cost": cost, "pf_converged": True, "steps": 24}
            for cost in (0.0, 0.01, 0.02)
        ]

        self.assertTrue(dynamic_gate(nominal, reference)["passed"])
        nominal[2]["daily_voltage_cost"] = 0.02
        self.assertFalse(dynamic_gate(nominal, reference)["passed"])
        nominal[2]["daily_voltage_cost"] = 0.03
        reference[0]["pf_converged"] = False
        self.assertFalse(dynamic_gate(nominal, reference)["passed"])

    def test_static_ranking_prefers_proxy_gap_then_native_load_then_ids(self):
        candidates = [
            {"grid_id": "b", "pcc_bus_ids": [4, 3, 2, 1], "proxy_gap": 0.5, "native_total_mw": 8.0},
            {"grid_id": "a", "pcc_bus_ids": [1, 2, 3, 4], "proxy_gap": 0.6, "native_total_mw": 6.0},
            {"grid_id": "c", "pcc_bus_ids": [1, 2, 3, 4], "proxy_gap": 0.5, "native_total_mw": 9.0},
        ]

        ranked = rank_static_candidates(candidates, limit=2)

        self.assertEqual([row["grid_id"] for row in ranked], ["a", "c"])

    def test_passing_selector_uses_controllability_gap_then_native_load(self):
        def candidate(grid_id, nominal, reference, native):
            return {
                "grid_id": grid_id,
                "pcc_bus_ids": [1, 2, 3, 4],
                "native_total_mw": native,
                "nominal_days": [
                    {"daily_voltage_cost": value, "pf_converged": True, "steps": 24}
                    for value in nominal
                ],
                "reference_days": [
                    {"daily_voltage_cost": value, "pf_converged": True, "steps": 24}
                    for value in reference
                ],
            }

        selected = select_passing_candidate(
            [
                candidate("a", [0.025, 0.025, 0.0], [0.0, 0.0, 0.0], 8.0),
                candidate("b", [0.04, 0.04, 0.0], [0.01, 0.01, 0.01], 10.0),
                candidate("c", [0.04, 0.04, 0.0], [0.01, 0.01, 0.01], 11.0),
            ]
        )

        self.assertEqual(selected["grid_id"], "c")
        self.assertAlmostEqual(selected["controllability_gap"], 0.05)


if __name__ == "__main__":
    unittest.main()
