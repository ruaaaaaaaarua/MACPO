"""H2 delivery lag ablation overrides for MAPPO instant vs delayed experiments."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Dict

import copy
import importlib

from scripts.microgrid_experiment_overrides import MICROGRID_EXPERIMENT_OVERRIDES


def _fmt_override_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, list):
        inner = ",".join(_fmt_override_value(v) for v in value)
        return f"[{inner}]"
    return str(value)


def build_hydra_override_args(overrides: Dict[str, Any]) -> str:
    parts = [f"{key}:{_fmt_override_value(val)}" for key, val in overrides.items()]
    return "+MICROGRID_CONFIG_OVERRIDES={" + ",".join(parts) + "}"


# Shared baseline (FullCDA-ReserveDemand + Price30); lag-specific keys patched per env.
SHARED_BASE_OVERRIDES: Dict[str, Any] = copy.deepcopy(MICROGRID_EXPERIMENT_OVERRIDES)

INSTANT_LAG_PATCH: Dict[str, Any] = {
    "h2_market_lag_enable": False,
    "h2_delivery_lag": 0,
    "h2_pending_obs_enable": False,
    "h2_pending_obs_horizon": 0,
    "h2_pending_summary_obs_enable": False,
    "h2_delivery_reservation_enable": False,
    "h2_delivery_reservation_horizon": 0,
    "h2_cap_aware_buy_enable": False,
}

DELAYED_LAG_PATCH: Dict[str, Any] = {
    "h2_market_lag_enable": True,
    "h2_delivery_lag": 4,
    "h2_pending_obs_enable": True,
    "h2_pending_obs_horizon": 4,
    "h2_pending_summary_obs_enable": True,
    "h2_delivery_reservation_enable": True,
    "h2_delivery_reservation_horizon": 4,
    "h2_cap_aware_buy_enable": True,
}


def build_env_overrides(env_name: str) -> Dict[str, Any]:
    if env_name == "instant":
        out = copy.deepcopy(SHARED_BASE_OVERRIDES)
        out.update(INSTANT_LAG_PATCH)
        return out
    if env_name in ("lag4h", "delayed"):
        out = copy.deepcopy(SHARED_BASE_OVERRIDES)
        out.update(DELAYED_LAG_PATCH)
        return out
    raise ValueError(f"Unknown env_name: {env_name}")


INSTANT_OVERRIDES = build_env_overrides("instant")
LAG4H_OVERRIDES = build_env_overrides("lag4h")

INSTANT_HYDRA_OVERRIDE_ARGS = build_hydra_override_args(INSTANT_OVERRIDES)
LAG4H_HYDRA_OVERRIDE_ARGS = build_hydra_override_args(LAG4H_OVERRIDES)


def save_env_config_snapshots(out_dir: Path) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, overrides in (
        ("env_a_instant", INSTANT_OVERRIDES),
        ("env_b_lag4h", LAG4H_OVERRIDES),
    ):
        path = out_dir / f"{name}.json"
        path.write_text(
            json.dumps(overrides, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


if __name__ == "__main__":
    import sys

    save_env_config_snapshots(Path(sys.argv[1]) if len(sys.argv) > 1 else Path("."))
    print(INSTANT_HYDRA_OVERRIDE_ARGS)
    print(LAG4H_HYDRA_OVERRIDE_ARGS)
