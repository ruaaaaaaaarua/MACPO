"""Temporary H2 price unit experiment: config prices in yuan/kg."""

from __future__ import annotations

from scripts.microgrid_experiment_overrides import (
    MICROGRID_EXPERIMENT_OVERRIDES,
    build_hydra_override_args,
)

H2_KG_TMP_OVERRIDES = {
    **MICROGRID_EXPERIMENT_OVERRIDES,
    "h2_price_unit": "kg",
    "h2_price_min": 3.0,
    "h2_price_max": 30.0,
    "h2_price_init": 16.5,
    # lambda_h2_buy=30, lambda_h2_sell=3 inherited; semantic yuan/kg via h2_price_unit.
}

HYDRA_OVERRIDE_ARGS = build_hydra_override_args(H2_KG_TMP_OVERRIDES)
