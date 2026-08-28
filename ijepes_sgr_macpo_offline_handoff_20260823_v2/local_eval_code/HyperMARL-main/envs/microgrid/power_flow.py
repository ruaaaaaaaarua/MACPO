"""IEEE-33 bus AC power-flow adapter for the microgrid environment."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from pypower.api import ppoption, runpf
from pypower.idx_brch import ANGMAX, ANGMIN, BR_R, BR_STATUS, BR_X, F_BUS, RATE_A, SHIFT, TAP, T_BUS
from pypower.idx_bus import BASE_KV, BUS_AREA, BUS_I, BUS_TYPE, PD, QD, REF, VA, VM, VMAX, VMIN, ZONE
from pypower.idx_gen import GEN_BUS, GEN_STATUS, MBASE, PG, PMAX, PMIN, QG, QMAX, QMIN, VG

from envs.microgrid.electric_lmp import IEEE33_EDGES, IEEE33_LOAD_KW, IEEE33_R, IEEE33_X


class IEEE33PowerFlow:
    """Run a silent Newton-Raphson AC power flow after each environment step."""

    def __init__(self, config: Mapping[str, Any]):
        self.base_mva = float(config.get("power_flow_base_mva", 10.0))
        self.base_kv = float(config.get("power_flow_base_kv", 12.66))
        self.load_power_factor = float(config.get("power_flow_load_power_factor", 0.95))
        self.vmin = float(config.get("power_flow_vmin_pu", 0.95))
        self.vmax = float(config.get("power_flow_vmax_pu", 1.05))
        self.failure_cost = float(config.get("power_flow_failure_cost", 1.0))
        self.background_load_scale = float(
            config.get("power_flow_background_load_scale", 1.0)
        )
        if not np.isfinite(self.background_load_scale) or self.background_load_scale < 0.0:
            raise ValueError("power_flow_background_load_scale must be finite and non-negative")
        self.agent_buses = np.asarray(
            config.get("elec_lmp_agent_bus_indices", [4, 12, 23, 32]),
            dtype=np.int64,
        )
        self.bus_count = int(config.get("elec_lmp_bus_count", 33))
        if self.bus_count != 33:
            raise ValueError("IEEE33PowerFlow requires exactly 33 buses")
        if self.agent_buses.shape != (4,):
            raise ValueError("power-flow adapter expects four PCC bus indices")
        if np.any(self.agent_buses < 0) or np.any(self.agent_buses >= self.bus_count):
            raise ValueError("PCC bus index is outside the IEEE-33 network")
        if not 0.0 < self.load_power_factor <= 1.0:
            raise ValueError("power_flow_load_power_factor must lie in (0, 1]")
        self._background_load_kw = self.background_load_scale * np.asarray(
            config.get("elec_lmp_background_load_kw", IEEE33_LOAD_KW),
            dtype=np.float64,
        )
        if self._background_load_kw.shape != (self.bus_count,):
            raise ValueError("background IEEE-33 load must contain 33 entries")
        self._q_per_p = float(np.tan(np.arccos(self.load_power_factor)))
        self._case = self._build_case()
        self._options = ppoption(VERBOSE=0, OUT_ALL=0, PF_ALG=1)

    def _build_case(self) -> dict[str, np.ndarray | float | str]:
        bus = np.zeros((self.bus_count, 13), dtype=np.float64)
        bus[:, BUS_I] = np.arange(self.bus_count)
        bus[:, BUS_TYPE] = 1
        bus[0, BUS_TYPE] = REF
        bus[:, PD] = self._background_load_kw / 1000.0
        bus[:, QD] = self._background_load_kw * self._q_per_p / 1000.0
        bus[:, BUS_AREA] = 1
        bus[:, VM] = 1.0
        bus[:, VA] = 0.0
        bus[:, BASE_KV] = self.base_kv
        bus[:, ZONE] = 1
        bus[:, VMAX] = self.vmax
        bus[:, VMIN] = self.vmin

        generator = np.zeros((1, 21), dtype=np.float64)
        generator[0, GEN_BUS] = 0
        generator[0, PG] = 0.0
        generator[0, QG] = 0.0
        generator[0, QMAX] = 100.0
        generator[0, QMIN] = -100.0
        generator[0, VG] = 1.0
        generator[0, MBASE] = self.base_mva
        generator[0, GEN_STATUS] = 1
        generator[0, PMAX] = 100.0
        generator[0, PMIN] = -100.0

        branches = np.zeros((len(IEEE33_EDGES), 13), dtype=np.float64)
        zbase_ohm = self.base_kv**2 / self.base_mva
        for index, ((source, target), resistance, reactance) in enumerate(
            zip(IEEE33_EDGES, IEEE33_R, IEEE33_X)
        ):
            branches[index, F_BUS] = source
            branches[index, T_BUS] = target
            branches[index, BR_R] = resistance / zbase_ohm
            branches[index, BR_X] = reactance / zbase_ohm
            branches[index, RATE_A] = 100.0
            branches[index, TAP] = 0.0
            branches[index, SHIFT] = 0.0
            branches[index, BR_STATUS] = 1
            branches[index, ANGMIN] = -360.0
            branches[index, ANGMAX] = 360.0
        return {
            "version": "2",
            "baseMVA": self.base_mva,
            "bus": bus,
            "gen": generator,
            "branch": branches,
        }

    def safety_metrics(self, voltages_pu: np.ndarray) -> dict[str, float]:
        voltages = np.asarray(voltages_pu, dtype=np.float64)
        lower = np.maximum(0.0, self.vmin - voltages)
        upper = np.maximum(0.0, voltages - self.vmax)
        violation = lower + upper
        violation_sum = float(np.sum(violation))
        return {
            "voltage_cost": violation_sum,
            "voltage_violation_area": violation_sum,
            "voltage_max_violation": (
                float(np.max(violation)) if violation.size else 0.0
            ),
            "voltage_min_pu": float(np.min(voltages)),
            "voltage_max_pu": float(np.max(voltages)),
        }

    def solve(self, pcc_p_kw: np.ndarray, pcc_q_kvar: np.ndarray) -> dict[str, Any]:
        pcc_p = np.asarray(pcc_p_kw, dtype=np.float64)
        pcc_q = np.asarray(pcc_q_kvar, dtype=np.float64)
        if pcc_p.shape != self.agent_buses.shape or pcc_q.shape != self.agent_buses.shape:
            raise ValueError("PCC injection arrays must have one value per microgrid")
        case = {
            key: value.copy() if isinstance(value, np.ndarray) else value
            for key, value in self._case.items()
        }
        for bus, p_kw, q_kvar in zip(self.agent_buses, pcc_p, pcc_q):
            case["bus"][bus, PD] += p_kw / 1000.0
            case["bus"][bus, QD] += q_kvar / 1000.0
        try:
            result, success = runpf(case, self._options)
        except (ArithmeticError, ValueError, FloatingPointError):
            result, success = case, False
        if not success:
            voltages = np.full(self.bus_count, np.nan, dtype=np.float64)
            return {
                "pf_converged": False,
                "voltages_pu": voltages,
                "pcc_voltages_pu": voltages[self.agent_buses],
                "voltage_cost": self.failure_cost,
                "voltage_violation_area": self.failure_cost,
                "voltage_max_violation": self.failure_cost,
                "voltage_min_pu": float("nan"),
                "voltage_max_pu": float("nan"),
            }
        voltages = np.asarray(result["bus"][:, VM], dtype=np.float64)
        metrics = self.safety_metrics(voltages)
        metrics.update(
            {
                "pf_converged": True,
                "voltages_pu": voltages,
                "pcc_voltages_pu": voltages[self.agent_buses],
            }
        )
        return metrics


class SwissMVPowerFlow:
    """AC power-flow adapter for one native Swiss-PDGs MV MATPOWER case."""

    def __init__(self, config: Mapping[str, Any]):
        self.base_mva = 100.0
        self.vmin = float(config.get("power_flow_vmin_pu", 0.95))
        self.vmax = float(config.get("power_flow_vmax_pu", 1.05))
        self.failure_cost = float(config.get("power_flow_failure_cost", 1.0))
        for key in (
            "power_flow_background_load_scale",
            "power_flow_pcc_injection_scale",
        ):
            value = float(config.get(key, 1.0))
            if not np.isfinite(value) or not np.isclose(value, 1.0):
                raise ValueError(f"{key} must be 1.0 for Swiss MV power flow")

        case_dir_value = config.get("power_flow_case_dir")
        if not case_dir_value:
            raise ValueError("power_flow_case_dir is required for Swiss MV power flow")
        self.case_dir = Path(str(case_dir_value)).expanduser().resolve()
        if not self.case_dir.is_dir():
            raise ValueError(f"Swiss MV case directory does not exist: {self.case_dir}")

        bus = self._load_unique_csv("*_bus_data.csv", columns=13)
        branch = self._load_unique_csv("*_branch_data.csv", columns=13)
        generator_source = self._load_unique_csv("*_generator_data.csv", columns=10)
        generator = np.zeros((generator_source.shape[0], 21), dtype=np.float64)
        generator[:, : generator_source.shape[1]] = generator_source
        self._case = {
            "version": "2",
            "baseMVA": self.base_mva,
            "bus": bus,
            "gen": generator,
            "branch": branch,
        }
        self.bus_count = int(bus.shape[0])
        base_kv_values = np.unique(bus[:, BASE_KV])
        if base_kv_values.size != 1 or not np.isclose(base_kv_values[0], 20.0):
            raise ValueError("Swiss MV case must use one native 20 kV voltage level")
        self.base_kv = float(base_kv_values[0])

        bus_ids = bus[:, BUS_I].astype(np.int64)
        if np.unique(bus_ids).size != bus_ids.size:
            raise ValueError("Swiss MV BUS_I values must be unique")
        row_by_bus_id = {int(bus_id): row for row, bus_id in enumerate(bus_ids)}
        self._row_by_bus_id = row_by_bus_id
        self.agent_bus_ids = np.asarray(
            config.get("power_flow_pcc_bus_ids", []), dtype=np.int64
        ).reshape(-1)
        if self.agent_bus_ids.shape != (4,) or np.unique(self.agent_bus_ids).size != 4:
            raise ValueError("power_flow_pcc_bus_ids must contain four unique BUS IDs")
        missing = [int(bus_id) for bus_id in self.agent_bus_ids if int(bus_id) not in row_by_bus_id]
        if missing:
            raise ValueError(f"Swiss MV PCC BUS IDs are absent from the case: {missing}")
        self.agent_bus_rows = np.asarray(
            [row_by_bus_id[int(bus_id)] for bus_id in self.agent_bus_ids],
            dtype=np.int64,
        )
        # Compatibility for the existing deterministic PCC refinement helper.
        self.agent_buses = self.agent_bus_ids.copy()
        self._options = ppoption(VERBOSE=0, OUT_ALL=0, PF_ALG=1)

    def _load_unique_csv(self, pattern: str, *, columns: int) -> np.ndarray:
        paths = sorted(self.case_dir.glob(pattern))
        if len(paths) != 1:
            raise ValueError(
                f"Swiss MV case requires exactly one {pattern}, found {len(paths)}"
            )
        values = np.genfromtxt(paths[0], delimiter=",", skip_header=1, ndmin=2)
        values = np.asarray(values, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != columns or not np.all(np.isfinite(values)):
            raise ValueError(f"invalid Swiss MV MATPOWER table: {paths[0]}")
        return values

    def safety_metrics(self, voltages_pu: np.ndarray) -> dict[str, float]:
        voltages = np.asarray(voltages_pu, dtype=np.float64)
        violation = np.maximum(0.0, self.vmin - voltages) + np.maximum(
            0.0, voltages - self.vmax
        )
        violation_sum = float(np.sum(violation))
        return {
            "voltage_cost": violation_sum,
            "voltage_violation_area": violation_sum,
            "voltage_max_violation": float(np.max(violation)) if violation.size else 0.0,
            "voltage_min_pu": float(np.min(voltages)),
            "voltage_max_pu": float(np.max(voltages)),
        }

    def solve(self, pcc_p_kw: np.ndarray, pcc_q_kvar: np.ndarray) -> dict[str, Any]:
        pcc_p = np.asarray(pcc_p_kw, dtype=np.float64)
        pcc_q = np.asarray(pcc_q_kvar, dtype=np.float64)
        if pcc_p.shape != (4,) or pcc_q.shape != (4,):
            raise ValueError("PCC injection arrays must have one value per microgrid")
        return self.solve_bus_injections(self.agent_bus_ids, pcc_p, pcc_q)

    def solve_bus_injections(
        self,
        bus_ids: Sequence[int] | np.ndarray,
        p_kw: Sequence[float] | np.ndarray,
        q_kvar: Sequence[float] | np.ndarray,
    ) -> dict[str, Any]:
        """Preview arbitrary native BUS injections for deterministic screening."""
        injection_bus_ids = np.asarray(bus_ids, dtype=np.int64).reshape(-1)
        injection_p = np.asarray(p_kw, dtype=np.float64).reshape(-1)
        injection_q = np.asarray(q_kvar, dtype=np.float64).reshape(-1)
        if not (
            injection_bus_ids.size == injection_p.size == injection_q.size
        ):
            raise ValueError("BUS IDs and P/Q injection arrays must have equal length")
        missing = [
            int(bus_id)
            for bus_id in injection_bus_ids
            if int(bus_id) not in self._row_by_bus_id
        ]
        if missing:
            raise ValueError(f"Swiss MV injection BUS IDs are absent from the case: {missing}")
        injection_rows = np.asarray(
            [self._row_by_bus_id[int(bus_id)] for bus_id in injection_bus_ids],
            dtype=np.int64,
        )
        case = {
            key: value.copy() if isinstance(value, np.ndarray) else value
            for key, value in self._case.items()
        }
        for row, p_value, q_value in zip(injection_rows, injection_p, injection_q):
            case["bus"][row, PD] += p_value / 1000.0
            case["bus"][row, QD] += q_value / 1000.0
        try:
            result, success = runpf(case, self._options)
        except (ArithmeticError, ValueError, FloatingPointError):
            result, success = case, False
        if not success:
            voltages = np.full(self.bus_count, np.nan, dtype=np.float64)
            return {
                "pf_converged": False,
                "voltages_pu": voltages,
                "pcc_voltages_pu": voltages[injection_rows],
                "voltage_cost": self.failure_cost,
                "voltage_violation_area": self.failure_cost,
                "voltage_max_violation": self.failure_cost,
                "voltage_min_pu": float("nan"),
                "voltage_max_pu": float("nan"),
            }
        result_bus = np.asarray(result["bus"], dtype=np.float64)
        voltage_by_bus_id = {
            int(row[BUS_I]): float(row[VM]) for row in result_bus
        }
        voltages = np.asarray(
            [voltage_by_bus_id[int(bus_id)] for bus_id in self._case["bus"][:, BUS_I]],
            dtype=np.float64,
        )
        pcc_voltages = np.asarray(
            [voltage_by_bus_id[int(bus_id)] for bus_id in injection_bus_ids],
            dtype=np.float64,
        )
        metrics = self.safety_metrics(voltages)
        metrics.update(
            {
                "pf_converged": True,
                "voltages_pu": voltages,
                "pcc_voltages_pu": pcc_voltages,
            }
        )
        return metrics


def build_power_flow(config: Mapping[str, Any]):
    """Construct the configured physical power-flow model."""
    model = str(config.get("power_flow_model", "ieee33"))
    if model == "ieee33":
        return IEEE33PowerFlow(config)
    if model == "swiss_mv":
        return SwissMVPowerFlow(config)
    raise ValueError("power_flow_model must be 'ieee33' or 'swiss_mv'")
