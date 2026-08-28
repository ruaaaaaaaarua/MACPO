import unittest
from unittest.mock import patch
from pathlib import Path
import tempfile

import numpy as np

from envs.microgrid.power_flow import IEEE33PowerFlow, SwissMVPowerFlow, build_power_flow


BUS_HEADER = "BUS_I,BUS_TYPE,Pd,Qd,Gs,Bs,area,Vm,Va,baseKV,zone,Vmax,Vmin\n"
BRANCH_HEADER = (
    "F_BUS,T_BUS,BR_R,BR_X,BR_B,RATE_A,RATE_B,RATE_C,TAP,SHIFT,"
    "BR_STATUS,ANGMIN,ANGMAX\n"
)
GEN_HEADER = "GEN_BUS,PG,QG,QMAX,QMIN,VG,MBASE,GEN_STATUS,PMAX,PMIN\n"


def write_swiss_case(case_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Write a small radial MATPOWER case with non-contiguous external BUS IDs."""
    bus = np.asarray(
        [
            [10, 3, 0.0, 0.0, 0, 0, 1, 1, 0, 20, 1, 1.05, 0.95],
            [30, 1, 0.4, 0.15, 0, 0, 1, 1, 0, 20, 1, 1.05, 0.95],
            [77, 1, 0.3, 0.12, 0, 0, 1, 1, 0, 20, 1, 1.05, 0.95],
            [99, 1, 0.2, 0.08, 0, 0, 1, 1, 0, 20, 1, 1.05, 0.95],
            [120, 1, 0.1, 0.04, 0, 0, 1, 1, 0, 20, 1, 1.05, 0.95],
        ],
        dtype=np.float64,
    )
    branch = np.asarray(
        [
            [10, 30, 0.0020, 0.0010, 0, 0, 0, 0, 0, 0, 1, -90, 90],
            [30, 77, 0.0020, 0.0010, 0, 0, 0, 0, 0, 0, 1, -90, 90],
            [77, 99, 0.0020, 0.0010, 0, 0, 0, 0, 0, 0, 1, -90, 90],
            [99, 120, 0.0020, 0.0010, 0, 0, 0, 0, 0, 0, 1, -90, 90],
        ],
        dtype=np.float64,
    )
    generator = np.asarray(
        [[10, 0, 0, 25, -25, 1, 100, 1, 25, -25]], dtype=np.float64
    )
    np.savetxt(case_dir / "demo_bus_data.csv", bus, delimiter=",", header=BUS_HEADER.strip(), comments="")
    np.savetxt(
        case_dir / "demo_branch_data.csv",
        branch,
        delimiter=",",
        header=BRANCH_HEADER.strip(),
        comments="",
    )
    np.savetxt(
        case_dir / "demo_generator_data.csv",
        generator,
        delimiter=",",
        header=GEN_HEADER.strip(),
        comments="",
    )
    return bus, branch, generator


class SwissMVPowerFlowTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.case_dir = Path(self.temporary.name)
        self.bus, self.branch, self.generator = write_swiss_case(self.case_dir)

    def tearDown(self):
        self.temporary.cleanup()

    def config(self):
        return {
            "power_flow_model": "swiss_mv",
            "power_flow_case_dir": str(self.case_dir),
            "power_flow_pcc_bus_ids": [30, 77, 99, 120],
            "power_flow_background_load_scale": 1.0,
            "power_flow_pcc_injection_scale": 1.0,
            "power_flow_failure_cost": 1.0,
        }

    def test_loads_non_contiguous_bus_ids_without_mutating_native_case(self):
        flow = SwissMVPowerFlow(self.config())

        np.testing.assert_allclose(flow._case["bus"], self.bus)
        np.testing.assert_allclose(flow._case["branch"], self.branch)
        np.testing.assert_allclose(flow._case["gen"][:, :10], self.generator)
        np.testing.assert_array_equal(flow.agent_bus_ids, [30, 77, 99, 120])
        np.testing.assert_array_equal(flow.agent_bus_rows, [1, 2, 3, 4])
        self.assertEqual(flow.base_mva, 100.0)
        self.assertEqual(flow.base_kv, 20.0)

    def test_pcc_kw_is_injected_directly_as_mw_at_the_selected_external_bus(self):
        flow = SwissMVPowerFlow(self.config())
        native = flow.solve(np.zeros(4), np.zeros(4))
        loaded = flow.solve(np.array([1000.0, 0.0, 0.0, 0.0]), np.zeros(4))

        self.assertTrue(native["pf_converged"])
        self.assertTrue(loaded["pf_converged"])
        self.assertLess(loaded["pcc_voltages_pu"][0], native["pcc_voltages_pu"][0])

    def test_screening_can_preview_an_arbitrary_native_bus_without_rebuilding(self):
        flow = SwissMVPowerFlow(self.config())
        native = flow.solve_bus_injections([], [], [])
        loaded = flow.solve_bus_injections([99], [1000.0], [0.0])

        self.assertTrue(native["pf_converged"])
        self.assertTrue(loaded["pf_converged"])
        self.assertLess(loaded["voltages_pu"][3], native["voltages_pu"][3])

    def test_env_v6_rejects_background_or_pcc_scaling(self):
        for key, value in (
            ("power_flow_background_load_scale", 0.9),
            ("power_flow_pcc_injection_scale", 0.9),
        ):
            config = self.config()
            config[key] = value
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "must be 1.0"):
                SwissMVPowerFlow(config)

    def test_factory_keeps_ieee33_backward_compatibility(self):
        self.assertIsInstance(build_power_flow({"power_flow_model": "ieee33"}), IEEE33PowerFlow)
        self.assertIsInstance(build_power_flow(self.config()), SwissMVPowerFlow)


class IEEE33PowerFlowTest(unittest.TestCase):
    def setUp(self):
        self.flow = IEEE33PowerFlow(
            {
                "elec_lmp_agent_bus_indices": [4, 12, 23, 32],
                "power_flow_base_mva": 10.0,
                "power_flow_base_kv": 12.66,
                "power_flow_load_power_factor": 0.95,
                "power_flow_failure_cost": 1.0,
            }
        )

    def test_added_pcc_load_reduces_its_voltage(self):
        no_microgrid_load = self.flow.solve(
            pcc_p_kw=np.zeros(4), pcc_q_kvar=np.zeros(4)
        )
        additional_tail_load = self.flow.solve(
            pcc_p_kw=np.array([0.0, 0.0, 0.0, 1000.0]),
            pcc_q_kvar=np.zeros(4),
        )

        self.assertTrue(no_microgrid_load["pf_converged"])
        self.assertTrue(additional_tail_load["pf_converged"])
        self.assertLess(
            additional_tail_load["voltages_pu"][32],
            no_microgrid_load["voltages_pu"][32],
        )

    def test_cost_is_zero_inside_limits_and_positive_for_violation(self):
        safe = self.flow.safety_metrics(np.full(33, 1.0))
        unsafe = self.flow.safety_metrics(
            np.array([0.94, 1.06] + [1.0] * 31)
        )

        self.assertEqual(safe["voltage_cost"], 0.0)
        self.assertAlmostEqual(unsafe["voltage_cost"], 0.02)
        self.assertAlmostEqual(unsafe["voltage_violation_area"], 0.02)
        self.assertAlmostEqual(unsafe["voltage_max_violation"], 0.01)

    def test_solver_failure_returns_deterministic_safety_cost(self):
        with patch("envs.microgrid.power_flow.runpf", side_effect=ArithmeticError):
            result = self.flow.solve(np.zeros(4), np.zeros(4))

        self.assertFalse(result["pf_converged"])
        self.assertEqual(result["voltage_cost"], 1.0)
        self.assertEqual(result["voltage_violation_area"], 1.0)
        self.assertEqual(result["voltage_max_violation"], 1.0)
        self.assertTrue(np.isnan(result["voltage_min_pu"]))
        self.assertEqual(len(result["voltages_pu"]), 33)

    def test_background_load_scale_only_scales_static_ieee33_load(self):
        half = IEEE33PowerFlow(
            {
                "elec_lmp_agent_bus_indices": [4, 12, 23, 32],
                "power_flow_background_load_scale": 0.5,
            }
        )
        full = IEEE33PowerFlow(
            {
                "elec_lmp_agent_bus_indices": [4, 12, 23, 32],
                "power_flow_background_load_scale": 1.0,
            }
        )
        np.testing.assert_allclose(half._background_load_kw, 0.5 * full._background_load_kw)
        np.testing.assert_allclose(half._case["branch"], full._case["branch"])
        self.assertEqual(half.base_mva, full.base_mva)


if __name__ == "__main__":
    unittest.main()
