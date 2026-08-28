"""Canonical configuration for the dense env-v3-safe experiment line.

This module intentionally does not call ``env_v2_overrides``: env-v2-sparse is
frozen as the historical STAS baseline, while this file is the explicit entry
point for dense economic rewards plus a separate global voltage constraint.
"""

from __future__ import annotations

import copy


ENV_V3_SAFE = {
    # One physical day per episode. Gamma=1 preserves undiscounted daily
    # economic returns and daily voltage-cost returns for constrained training.
    "episode_length": 24,
    "multi_day_episode_enable": False,
    "episode_days": 1,
    "day_boundary_interval": 24,
    "day_boundary_info_enable": True,
    "daily_truncation_enable": False,
    "gamma": 1.0,
    "reward_emission_mode": "dense",
    "italian_split_enable": True,
    "italian_split_name": "train",
    "italian_split_strategy": "manifest",
    # Keep the validated asymmetric v2 physical fleet, without inheriting its
    # sparse-reward entry point or credit-assignment-specific knobs.
    "pv_cap": [7500.0, 1500.0, 500.0, 2000.0],
    "wt_cap": [1500.0, 6000.0, 3000.0, 500.0],
    "load_h_peak": [750.0, 600.0, 2925.0, 3656.25],
    "h2_tank_init_ratio": 0.4,
    "h2_tank_power": [2000.0, 2000.0, 3400.0, 4200.0],
    "soc_init": 0.5,
    # Markets and physical hydrogen transport.
    "elec_internal_cda_enable": True,
    "h2_internal_cda_enable": True,
    "gas_network_enable": False,
    "gas_price_dynamic_enable": False,
    "gas_price_bidirectional_enable": False,
    "gas_price_obs_enable": False,
    "gas_pressure_obs_enable": False,
    "h2_transport_loss": 0.0,
    "h2_market_schedule_enable": False,
    "h2_market_lag_enable": True,
    "h2_delivery_lag": 4,
    "h2_pending_obs_enable": True,
    "h2_pending_obs_horizon": 10,
    "h2_pending_summary_obs_enable": True,
    "h2_cap_aware_buy_enable": False,
    "h2_delivery_reservation_enable": True,
    "h2_delivery_reservation_horizon": 10,
    "h2_delivery_reservation_ratio": 0.5,
    "h2_learnable_rolling_order_enable": True,
    "h2_learnable_rolling_order_active": True,
    "h2_action_order_max_peak_hours": 2.0,
    "h2_learnable_rolling_order_agent_indices": [0, 1, 2, 3],
    "h2_buyer_reservation_demand_enable": False,
    "h2_traffic_enable": True,
    "h2_route_action_enable": True,
    "h2_traffic_external_node_enable": True,
    "h2_traffic_eta_min": 4,
    "h2_traffic_eta_max": 10,
    "h2_traffic_truck_capacity_kg": 100.0,
    "h2_traffic_edge_capacity": 2.5,
    "h2_traffic_bpr_alpha": 0.15,
    "h2_traffic_bpr_beta": 4.0,
    "h2_traffic_background_base_min": 0.25,
    "h2_traffic_background_base_max": 0.45,
    "h2_traffic_morning_peak_amplitude": 1.0,
    "h2_traffic_evening_peak_amplitude": 1.1,
    "h2_traffic_peak_width_hours": 2.0,
    "h2_traffic_directional_phase_hours": 4.0,
    "h2_traffic_seed": 20260716,
    "h2_traffic_transit_loss_per_hour": 0.008,
    # Orders beyond the final physical hour stay paid but have zero terminal
    # asset value because the hydrogen is not physically in a local tank.
    "h2_order_horizon_clip_mode": "pay_and_lose",
    "h2_planned_external_order_enable": True,
    "h2_emergency_price_multiplier": 2.0,
    "external_h2_dependency_penalty_enable": False,
    "external_h2_dependency_penalty_kg": 0.0,
    # No historical inventory targets, action regularisation, or bonus shaping.
    "penalty_enable": False,
    "low_inventory_penalty_enable": False,
    "terminal_h2_floor_penalty_enable": False,
    "terminal_h2_shortfall_value_enable": False,
    "terminal_h2_settlement_in_reward_enable": False,
    "terminal_soc_floor_penalty_enable": False,
    "terminal_battery_salvage_enable": False,
    "stepwise_h2_floor_penalty_enable": False,
    "action_reg_enable": False,
    "h2_internal_trade_bonus_enable": False,
    # Safe env-v3 additions: the reward remains economic; voltage cost is
    # exposed separately as the unique system-level constrained signal.
    "power_flow_enable": True,
    "power_flow_base_mva": 10.0,
    "power_flow_base_kv": 12.66,
    "power_flow_load_power_factor": 0.95,
    "power_flow_vmin_pu": 0.95,
    "power_flow_vmax_pu": 1.05,
    "power_flow_failure_cost": 1.0,
    "terminal_economic_settlement_enable": True,
    "terminal_battery_value_yuan_per_kwh": 0.0,
    "terminal_h2_value_yuan_per_kwh": 0.0,
}


def env_v3_safe_overrides() -> dict:
    """Return a detached env-v3-safe configuration dictionary."""
    return copy.deepcopy(ENV_V3_SAFE)


def _hydra_value(value) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (list, tuple)):
        return "[" + ",".join(_hydra_value(item) for item in value) + "]"
    return str(value)


def hydra_override_arg(overrides: dict | None = None) -> str:
    """Render the safe line as a Hydra MICROGRID_CONFIG_OVERRIDES argument."""
    values = env_v3_safe_overrides() if overrides is None else overrides
    body = ",".join(f"{key}:{_hydra_value(value)}" for key, value in values.items())
    return "+MICROGRID_CONFIG_OVERRIDES={" + body + "}"
