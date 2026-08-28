"""Global configuration for the microgrid multi-agent environment."""

from pathlib import Path

import numpy as np


_MICROGRID_DIR = Path(__file__).resolve().parent
_ENVS_DIR = _MICROGRID_DIR.parent


MICROGRID_CONFIG = {
    # Core dimensions.
    "num_agents": 4,
    "episode_length": 24,
    "multi_day_episode_enable": False,
    "episode_days": 1,
    "day_boundary_interval": 24,
    "day_boundary_info_enable": True,
    "daily_truncation_enable": False,
    "dt": 1.0,
    # Base observation layout (13-d):
    # pv, wt, electric load, hydrogen load, SOC, H2 tank ratio,
    # previous electricity/H2 clearing prices, time sin/cos,
    # current external buy price, previous grid exchange, previous H2 exchange.
    "obs_dim": 13,
    # Action layout (5-d continuous):
    #   a0 P_el, a1 P_bat, a2 elec_bid_price, a3 h2_bid_price, a4 P_ht
    "action_dim": 5,
    "gamma": 0.99,
    "reward_scale": 200.0,
    "profile_source": "italian",
    "italian_data_path": str(_ENVS_DIR / "Italian_data.csv"),
    "italian_split_enable": True,
    "italian_split_strategy": "fixed_random",
    "italian_split_seed": 42,
    "italian_split_train_ratio": 0.70,
    "italian_split_manifest_path": str(_MICROGRID_DIR / "italian_day_splits.json"),
    "italian_split_name": "train",
    "italian_day_indices": None,
    "derive_heat_from_electric": True,
    "derived_heat_base_ratio": 0.15,
    "derived_heat_variable_ratio": 0.45,

    # Agent roles.
    "agent_types": ["mg", "mg", "mg", "mg"],

    # Asset capacities.
    "pv_cap": [5000.0, 1000.0, 500.0, 2000.0],      # kW
    "wt_cap": [1000.0, 4000.0, 3000.0, 500.0],      # kW
    "bat_cap": [5000.0, 3000.0, 4000.0, 2000.0],    # kWh
    "bat_power": [2000.0, 1200.0, 1500.0, 800.0],   # kW
    "el_cap": [2000.0, 3000.0, 500.0, 800.0],       # kW
    "el_eff": [0.70, 0.65, 0.60, 0.65],
    # Every agent now carries its own hydrogen tank (asymmetric sizing).
    "h2_tank_cap":   [500.0, 500.0, 300.0, 300.0],   # kg
    "h2_tank_power": [2000.0, 2000.0, 1200.0, 1500.0],  # kW_H2 equivalent
    "load_e_peak": [2250.0, 1875.0, 3000.0, 2625.0],  # kW (0.75x)
    "load_h_peak": [750.0, 600.0, 1500.0, 1875.0],   # kW_thermal

    # Battery constraints.
    "bat_eff_c": 0.95,
    "bat_eff_d": 0.95,
    "soc_min": 0.1,
    "soc_max": 0.9,
    "soc_init": 0.1,

    # Hydrogen storage constraints.
    "h2_tank_min_ratio": 0.05,
    "h2_tank_max_ratio": 0.95,
    "h2_tank_init_ratio": 0.05,
    "h2_eff_c": 1.0,
    "h2_eff_d": 1.0,

    # Physical constants.
    "LHV_H2": 33.33,   # kWh/kg
    "boiler_eff": 0.90,

    # Hydrogen demand is fixed: all thermal demand is represented as hydrogen load.
    "h2_thermal_share_default": 1.0,
    # 外部氢市场价格. 对称价 (lambda_h2_buy == lambda_h2_sell) 会让内部市场
    # 缺乏撮合动机; 非对称设置 (buy=1.00 高, sell=0.40 低) 在两者间留出价差,
    # 逼迫 agent 优先走内部 CDA 市场. 第四轮把 sell 从 0.30 提到 0.40, 让
    # producer 外卖机会成本接近其谷价产氢成本 0.429, 更倾向走内部撮合.
    # lambda_h2 保留作为历史默认值.
    "lambda_h2": 0.70,
    "lambda_h2_buy":  1.00,   # yuan / kWh_H2  向外部买氢 (贵)
    # 课程阶段 1: 先用 0.40 缓和 producer 探索亏损; 市场能成交后再降回 0.20
    # 验证非对称外部氢价下的稳定策略.
    "lambda_h2_sell": 0.50,   # yuan / kWh_H2  向外部卖氢 (训练阶段先缓和探索惩罚)
    "gas_network_enable": False,
    "gas_price_dynamic_enable": False,
    "gas_price_bidirectional_enable": False,
    "gas_price_obs_enable": False,
    "gas_pressure_obs_enable": False,
    "gas_network_model": "single",
    "gas_node_count": 4,
    "gas_agent_node_indices": [0, 1, 2, 3],
    "gas_node_pressure_init": [1.0, 1.0, 1.0, 1.0],
    "gas_node_pressure_ref": [1.0, 1.0, 1.0, 1.0],
    "gas_node_recovery_rate": 0.10,
    "gas_node_withdrawal_gain": 1.0e-5,
    "gas_node_injection_gain": 1.0e-5,
    "gas_node_exogenous_supply": [0.0, 0.0, 0.0, 0.0],
    "gas_node_exogenous_demand": [0.0, 0.0, 0.0, 0.0],
    "gas_line_edges": [[0, 1], [1, 2], [2, 3]],
    "gas_line_conductance": [0.04, 0.04, 0.04],
    "gas_pressure_init": 1.0,
    "gas_pressure_ref": 1.0,
    "gas_pressure_min": 0.6,
    "gas_pressure_max": 1.2,
    "gas_pressure_recovery_rate": 0.10,
    "gas_withdrawal_gain": 1.0e-5,
    "gas_injection_gain": 1.0e-5,
    "gas_exogenous_supply": 0.0,
    "gas_exogenous_demand": 0.0,
    "gas_price_base": 1.00,
    "gas_price_sensitivity": 0.80,
    "gas_price_min": 0.60,
    "gas_price_max": 1.80,
    "gas_sell_ratio": 0.50,

    # Electricity continuous double auction price bounds.
    "elec_price_min": 0.15,
    "elec_price_max": 1.00,
    "elec_price_init": 0.60,
    "elec_price_mode": "tou",
    "elec_lmp_bus_count": 33,
    "elec_lmp_slack_bus": 0,
    "elec_lmp_agent_bus_indices": [4, 12, 23, 32],
    "elec_lmp_agent_bus_one_indexed": False,
    "elec_lmp_background_load_scale": 1.0,
    "elec_lmp_line_capacity_kw": 9000.0,
    "elec_lmp_line_capacity_min_kw": 1000.0,
    "elec_lmp_congestion_threshold": 0.70,
    "elec_lmp_loss_coef": 0.08,
    "elec_lmp_congestion_coef": 0.35,
    "elec_lmp_depth_coef": 0.01,
    "elec_lmp_local_generation_credit_coef": 0.05,
    "elec_lmp_price_min": 0.10,
    "elec_lmp_price_max": 1.40,
    "elec_lmp_sell_mode": "ratio",
    "elec_lmp_sell_ratio": 0.55,
    "elec_lmp_sell_spread": 0.15,
    "elec_lmp_sell_price_min": 0.0,
    "elec_lmp_simple_topology": "chain",
    "elec_lmp_simple_agent_bus_indices": [1, 2, 3, 4],

    # Hydrogen continuous double auction price bounds.
    "h2_price_min": 0.50,
    "h2_price_max": 1.00,
    "h2_price_init": 0.75,

    # Backward-compatible aliases for the original single-market code path.
    "cda_price_min": 0.50,
    "cda_price_max": 1.00,
    "cda_price_init": 0.75,
    "elec_internal_cda_enable": True,
    "h2_internal_cda_enable": True,
    "elec_p2p_diagnostics_enable": False,
    "h2_p2p_diagnostics_enable": False,
    "elec_p2p_enable": False,
    "h2_p2p_enable": False,
    "elec_p2p_price_rule": "split_surplus",
    "h2_p2p_price_rule": "split_surplus",
    "elec_p2p_distance_fee_coef": 0.0,
    "h2_p2p_distance_fee_coef": 0.0,
    "elec_p2p_max_distance": 0.0,
    "h2_p2p_max_distance": 0.0,

    # Operation and shaping costs (kept for backward compat; not used in reward).
    "c_el": 0.005,
    "c_bat": 0.002,
    "cda_shaping_coef": 0.01,
    "h2_internal_trade_bonus_enable": False,
    "h2_internal_trade_bonus_coef": 0.0,
    "terminal_value_coef": 1.0,

    # === Running shaping on SOC / H2 tank level ===
    # 第五轮恢复最初的 "软终端约束" 形式:
    #   penalty_t = coef / (T - t) * sum_i max(0, |x_i - 0.5| - deadband)^2
    # 含义: 临近终端权重 1/(T-t) 无穷放大, 逼 agent 在 t=T-1 把 SOC / 氢罐
    # 收敛到 [0.5 - deadband, 0.5 + deadband] = [0.4, 0.6] 死区内. 期间 agent
    # 可以自由做套利 (冲到 0.9 或塌到 0.1), 但越靠近终端越必须往 0.5 归位.
    # H2 项仅对有电解槽的 producer (A0/A1) 生效, consumer (A2/A3) 的氢罐依靠
    # 氢市场, 强加目标不合理.
    # 系数演进历史:
    #   第一轮 α=500: 1 episode 累积惩罚 ~680 < 补电成本 2500, agent 算出
    #                 "宁可被罚也不补电" 最优, 全塌到 0.1.
    #   第二轮 α=500 + (t/T)^2: 形状不对, agent 没学.
    #   第三轮 α=150 + 时变 target curve: 惩罚占比 50%, 压过 base_cost 梯度.
    #   第四轮 α=60 + 时变 target + look-ahead obs: HyperMARL-MLP 学到 A1
    #                 尖峰套利, 但因 target curve 不是 0.5, 终端 SOC 塌到 0.1.
    #   第五轮 α=2500 + α/(T-t) + 死区 0.1: 累积惩罚 ~3400 > 补电成本 2500,
    #                 经济激励倾向终端达标.
    #   第六轮 α=350 + α/(T-t) + 死区 0.1 (producer) / 0.3 (consumer):
    #                 P0-a 修好后 c_h2 量级下来了, 2500 的惩罚占比会压过 base
    #                 梯度; 按最坏情况 sanity check, α=350 时总惩罚占 base_cost
    #                 约 20-25%, 不压梯度. Consumer 氢罐靠深度循环充放撬动 P2P,
    #                 死区由 0.1 放宽到 0.3 ([0.2, 0.8]), 允许 CDA 买氢入罐 /
    #                 烧氢放空的自然波动, 只在偏离 [0.2, 0.8] 时才罚.
    #   课程阶段 1-a: α=100, 先减弱终端库存约束, 让策略先学会产氢/买氢/供热链路.
    #   课程阶段 1-b: α=250 + 低库存持续惩罚, 抑制上一轮学到的"放空电池/氢罐"
    #                 取短期收益, 但仍比第六轮 α=350 更温和.
    "penalty_enable": True,
    "soc_penalty_coef": 5663.605646,
    "h2_penalty_coef":  5663.605646,
    "penalty_target_center": 0.5,     # 终端目标中心 (SOC 和 H2 同值)
    "soc_penalty_targets": [0.45, 0.45, 0.45, 0.45],
    "h2_penalty_targets": [0.40, 0.40, 0.50, 0.60],
    "penalty_deadband": 0.08,
    # 第六轮新增: consumer 氢罐专用死区, [0.2, 0.8] 允许深度循环充放.
    "consumer_h2_deadband": 0.3,
    "consumer_h2_deadband_agent_indices": [2, 3],
    # 防止策略把库存长期贴在物理下限:只在低于阈值时按平方罚,不要求库存一直回 0.5.
    "low_inventory_penalty_enable": True,
    "low_inventory_penalty_coef": 1000.0,
    "soc_low_threshold": 0.0,
    "h2_low_threshold": 0.15,
    "terminal_h2_floor_penalty_enable": True,
    "terminal_h2_floor_threshold": 0.20,
    "terminal_h2_floor_penalty_coef": 50000.0,
    "terminal_h2_floor_agent_indices": [2, 3],
    "terminal_h2_shortfall_value_enable": False,
    "terminal_h2_shortfall_value_targets": [0.40, 0.40, 0.50, 0.60],
    "terminal_h2_shortfall_value_coef": 0.0,
    "terminal_h2_shortfall_value_agent_indices": [2, 3],
    "terminal_soc_floor_penalty_enable": False,
    "terminal_soc_floor_threshold": 0.20,
    "terminal_soc_floor_penalty_coef": 8000.0,
    "terminal_soc_floor_agent_indices": [2],
    "terminal_battery_salvage_enable": False,
    "terminal_battery_salvage_value_coef": 3000.0,
    "terminal_battery_salvage_capacity_scaled_enable": False,
    "terminal_battery_salvage_reference_capacity": 4000.0,
    "terminal_battery_salvage_agent_indices": [2],
    "a2_late_soc_reserve_enable": False,
    "a2_late_soc_reserve_agent_indices": [2],
    "a2_late_soc_reserve_threshold": 0.20,
    "a2_late_soc_reserve_horizon": 4,
    "stepwise_h2_floor_penalty_enable": True,
    "stepwise_h2_floor_thresholds": [0.0, 0.0, 0.25, 0.35],
    "stepwise_h2_floor_weights": [0.0, 0.0, 1.5, 4.0],
    "stepwise_h2_floor_penalty_coef": 5000.0,
    "stepwise_h2_floor_urgency_gain": 3.0,
    "action_reg_enable": False,
    "action_reg_indices": [0, 1, 4],
    "action_magnitude_penalty_coef": 10.0,
    "action_delta_penalty_coef": 20.0,
    # soc_target_curve_* 保留但置 False, 仅作为历史代码路径的兼容开关
    "soc_target_curve_enable": False,
    "soc_target_lo": 0.2,
    "soc_target_hi": 0.8,
    "h2_target_lo":  0.2,
    "h2_target_hi":  0.8,

    # === Hydrogen market: agreement schedule + optional delivery lag ===
    # 课程阶段 1 默认关闭协议时刻和运输延迟, 让氢交易变成每步撮合 + 即时入罐,
    # 先恢复短反馈链. 后续可按课程逐步打开:
    #   stage 2: h2_market_schedule_enable=True, h2_market_lag_enable=False
    #   stage 3/4: h2_market_schedule_enable=True, h2_market_lag_enable=True,
    #              h2_delivery_lag=1/4
    # 第六轮: [6,10,14] -> [2,6,10,14,18]. 一天 5 次 agreement, 每 4 小时一次,
    # 覆盖早上第一个热峰 (t=2->t=6 送达, 覆盖 t=6..9) 和晚高峰 (t=14->t=18 送达).
    # 原 [6,10,14] 错过早上 t=5..9 热峰, 是结构性漏洞. 所有订单均在 episode
    # (T=24) 内到货, 不需处理跨边界.
    "h2_market_schedule_enable": False,
    "h2_market_lag_enable": True,
    "h2_market_schedule": [2, 6, 10, 14, 18],
    "h2_delivery_lag": 4,
    "h2_pending_obs_enable": True,
    "h2_pending_obs_horizon": 4,
    "h2_pending_summary_obs_enable": False,
    "h2_cap_aware_buy_enable": True,
    "h2_delivery_reservation_enable": False,
    "h2_delivery_reservation_horizon": 4,
    "h2_delivery_reservation_ratio": 1.0,
    "h2_buyer_reservation_demand_enable": False,
    "h2_buyer_reservation_agent_indices": [2, 3],
    "h2_buyer_reservation_target_ratios": [0.0, 0.0, 0.35, 0.45],
    "h2_buyer_reservation_demand_gain": 1.0,
    "h2_buyer_reservation_max_order_fraction": 0.25,
    "h2_transport_loss": 0.0,
}


def get_tou_price(step_idx):
    """Return the buy/sell tariff for an hourly step."""
    s = step_idx % 24
    if s <= 6 or s >= 23:
        return 0.30, 0.15
    if (7 <= s <= 9) or (15 <= s <= 17) or (21 <= s <= 22):
        return 0.60, 0.35
    return 1.00, 0.55


def build_tou_table(episode_length=None):
    """Build the time-of-use tariff tables."""
    if episode_length is None:
        episode_length = MICROGRID_CONFIG["episode_length"]

    buy_prices = np.zeros(episode_length, dtype=np.float32)
    sell_prices = np.zeros(episode_length, dtype=np.float32)
    for t in range(episode_length):
        buy_prices[t], sell_prices[t] = get_tou_price(t)
    return buy_prices, sell_prices
