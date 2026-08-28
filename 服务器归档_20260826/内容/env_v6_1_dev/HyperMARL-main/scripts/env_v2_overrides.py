"""Canonical environment overrides for the env v2 congestion experiments.

基线 = paper-sparse 线 (traffic-stas-paper-sparse-20260716-10k-v1) 的已验证
配置; V2_DIFF = 设计规格 (docs/superpowers/specs/2026-07-17-env-v2-congestion-
design.md) 的全部旋钮。Hydra 覆盖串由字典机械生成, 避免手工双副本漂移。
"""

from __future__ import annotations

import copy


# paper-sparse 线的环境配置 (稀疏终端奖励形态)。
PAPER_SPARSE_BASE = {
    "episode_length": 24,
    "multi_day_episode_enable": False,
    "episode_days": 1,
    "day_boundary_interval": 24,
    "day_boundary_info_enable": True,
    "daily_truncation_enable": False,
    "italian_split_enable": True,
    "italian_split_name": "train",
    "italian_split_strategy": "manifest",
    "terminal_h2_shortfall_value_enable": False,
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
    "h2_pending_obs_horizon": 6,
    "h2_pending_summary_obs_enable": True,
    "h2_cap_aware_buy_enable": True,
    "h2_delivery_reservation_enable": True,
    "h2_delivery_reservation_horizon": 6,
    "h2_delivery_reservation_ratio": 1.0,
    "h2_buyer_reservation_demand_enable": False,
    "h2_buyer_reservation_agent_indices": [2, 3],
    "h2_buyer_reservation_target_ratios": [0.0, 0.0, 0.35, 0.45],
    "h2_buyer_reservation_demand_gain": 1.0,
    "h2_buyer_reservation_max_order_fraction": 0.25,
    "lambda_h2": 0.4950495049504951,
    "lambda_h2_buy": 1.3501350135013501,
    "lambda_h2_sell": 0.09000900090009001,
    "h2_price_min": 0.09000900090009001,
    "h2_price_max": 0.9000900090009001,
    "h2_price_init": 0.4950495049504951,
    "external_h2_dependency_penalty_enable": True,
    "external_h2_dependency_penalty_kg": 15.0,
    "h2_learnable_rolling_order_enable": True,
    "h2_action_order_max_peak_hours": 1.0,
    "h2_learnable_rolling_order_agent_indices": [0, 1, 2, 3],
    "penalty_enable": False,
    "low_inventory_penalty_enable": False,
    "terminal_h2_floor_penalty_enable": False,
    "terminal_soc_floor_penalty_enable": False,
    "terminal_battery_salvage_enable": False,
    "stepwise_h2_floor_penalty_enable": False,
    "action_reg_enable": False,
    "h2_internal_trade_bonus_enable": False,
    "h2_traffic_enable": True,
    "h2_route_action_enable": True,
    "h2_traffic_min_eta": 4,
    "h2_traffic_max_eta": 6,
    "h2_traffic_truck_capacity_kg": 500.0,
    "h2_traffic_edge_capacity": 8.0,
    "h2_traffic_bpr_alpha": 0.15,
    "h2_traffic_bpr_beta": 4.0,
    "h2_traffic_background_base_min": 0.25,
    "h2_traffic_background_base_max": 0.45,
    "h2_traffic_morning_peak_amplitude": 1.0,
    "h2_traffic_evening_peak_amplitude": 1.1,
    "h2_traffic_peak_width_hours": 2.0,
    "h2_traffic_directional_phase_hours": 4.0,
    "h2_traffic_seed": 20260716,
    "terminal_h2_settlement_in_reward_enable": False,
    "reward_emission_mode": "terminal_total",
    "gamma": 1.0,
}

# 设计规格的 v2 差异 (逐条对应规格章节)。
V2_DIFF = {
    # 1a 外部订单入网 + 计划外购通道
    "h2_traffic_external_node_enable": True,
    "h2_planned_external_order_enable": True,
    # 1b 容量收紧 (制造拥堵博弈)
    "h2_traffic_truck_capacity_kg": 100.0,
    "h2_traffic_edge_capacity": 2.5,
    # 1c ETA 放宽
    "h2_traffic_eta_min": 4,
    "h2_traffic_eta_max": 10,
    "h2_pending_obs_horizon": 10,
    "h2_delivery_reservation_horizon": 10,
    # 1d 在途损耗 (boil-off 0.8%/h)
    "h2_traffic_transit_loss_per_hour": 0.008,
    # 2a 订单是真实承诺 + 溢出照付
    "h2_order_horizon_clip_mode": "pay_and_lose",
    "h2_cap_aware_buy_enable": False,
    # 2b 预留降级
    "h2_delivery_reservation_ratio": 0.5,
    # 2c 应急价 = 计划价 x2 (计划 45 元/kg, 应急 90 元/kg);
    #    依赖惩罚退役, 经济分层完全由价格表达。
    "h2_emergency_price_multiplier": 2.0,
    "external_h2_dependency_penalty_enable": False,
    "external_h2_dependency_penalty_kg": 0.0,
    # 冷启动校准: 罐从 5% 抬到 40% (约 1 个高峰小时), 消除 episode 开局
    # 管道未建立期的不可避免应急采购 (G4 爆表的主因), 电站带库存过夜
    # 物理合理; 罐容 << 日负荷, 不存在靠初始库存躺平的可能。
    "h2_tank_init_ratio": 0.4,
    # 订货速率上限 1->2 个高峰小时/小时: 爬坡覆盖需要的管道余量。
    "h2_action_order_max_peak_hours": 2.0,
    # 放电功率按峰值负荷选型 (峰值氢需求 = load_h_peak/boiler_eff =
    # [833,667,3250,4062]): 原 [2000,2000,1200,1500] 中 2/3 号阀门比峰值
    # 细一半, 满罐也只能放 1200/1500 每小时, 峰时应急是物理必然而非
    # 时机失误 —— 这是特权规则残余 20k 应急的真正来源。
    "h2_tank_power": [2000.0, 2000.0, 3400.0, 4200.0],
}


def env_v2_overrides(sparse: bool = True) -> dict:
    """Return the canonical env-v2 override dict (sparse terminal by default)."""
    overrides = copy.deepcopy(PAPER_SPARSE_BASE)
    overrides.update(copy.deepcopy(V2_DIFF))
    if not sparse:
        overrides.pop("reward_emission_mode", None)
        overrides.pop("gamma", None)
    return overrides


def _hydra_value(value) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (list, tuple)):
        return "[" + ",".join(_hydra_value(item) for item in value) + "]"
    return str(value)


def hydra_override_arg(overrides: dict) -> str:
    """Render a +MICROGRID_CONFIG_OVERRIDES={...} arg from an override dict."""
    body = ",".join(
        f"{key}:{_hydra_value(value)}" for key, value in overrides.items()
    )
    return "+MICROGRID_CONFIG_OVERRIDES={" + body + "}"
