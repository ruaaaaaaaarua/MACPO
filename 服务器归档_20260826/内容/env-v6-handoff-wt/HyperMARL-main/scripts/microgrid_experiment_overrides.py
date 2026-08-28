"""Shared microgrid experiment overrides (from run_ctde_40k.sh, unchanged)."""

MICROGRID_EXPERIMENT_OVERRIDES = {
    "episode_length": 24,
    "multi_day_episode_enable": False,
    "episode_days": 1,
    "day_boundary_interval": 24,
    "day_boundary_info_enable": True,
    "daily_truncation_enable": False,
    "italian_split_enable": True,
    "italian_split_name": "train",
    "terminal_h2_shortfall_value_enable": False,
    "lambda_h2": 16.5,
    "lambda_h2_buy": 30.0,
    "lambda_h2_sell": 3.0,
    "h2_price_min": 3.0,
    "h2_price_max": 30.0,
    "h2_price_init": 16.5,
    "pv_cap": [7500.0, 1500.0, 500.0, 2000.0],
    "wt_cap": [1500.0, 6000.0, 3000.0, 500.0],
    "load_h_peak": [750.0, 600.0, 2925.0, 3656.25],
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
    "h2_pending_obs_horizon": 4,
    "h2_pending_summary_obs_enable": True,
    "h2_cap_aware_buy_enable": True,
    "h2_delivery_reservation_enable": True,
    "h2_delivery_reservation_horizon": 4,
    "h2_delivery_reservation_ratio": 1.0,
    "h2_buyer_reservation_demand_enable": True,
    "h2_buyer_reservation_agent_indices": [2, 3],
    "h2_buyer_reservation_target_ratios": [0.0, 0.0, 0.35, 0.45],
    "h2_buyer_reservation_demand_gain": 1.0,
    "h2_buyer_reservation_max_order_fraction": 0.25,
}

HYDRA_OVERRIDE_ARGS = (
    "+MICROGRID_CONFIG_OVERRIDES={"
    "episode_length:24,multi_day_episode_enable:false,episode_days:1,day_boundary_interval:24,"
    "day_boundary_info_enable:true,daily_truncation_enable:false,italian_split_enable:true,"
    "italian_split_name:train,terminal_h2_shortfall_value_enable:false,"
    "lambda_h2:16.5,lambda_h2_buy:30.0,lambda_h2_sell:3.0,"
    "h2_price_min:3.0,h2_price_max:30.0,h2_price_init:16.5,"
    "pv_cap:[7500.0,1500.0,500.0,2000.0],"
    "wt_cap:[1500.0,6000.0,3000.0,500.0],load_h_peak:[750.0,600.0,2925.0,3656.25],"
    "elec_internal_cda_enable:true,h2_internal_cda_enable:true,gas_network_enable:false,"
    "gas_price_dynamic_enable:false,gas_price_bidirectional_enable:false,gas_price_obs_enable:false,"
    "gas_pressure_obs_enable:false,h2_transport_loss:0.0,h2_market_schedule_enable:false,"
    "h2_market_lag_enable:true,h2_delivery_lag:4,h2_pending_obs_enable:true,h2_pending_obs_horizon:4,"
    "h2_pending_summary_obs_enable:true,h2_cap_aware_buy_enable:true,h2_delivery_reservation_enable:true,"
    "h2_delivery_reservation_horizon:4,h2_delivery_reservation_ratio:1.0,"
    "h2_buyer_reservation_demand_enable:true,h2_buyer_reservation_agent_indices:[2,3],"
    "h2_buyer_reservation_target_ratios:[0.0,0.0,0.35,0.45],h2_buyer_reservation_demand_gain:1.0,"
    "h2_buyer_reservation_max_order_fraction:0.25}"
)
