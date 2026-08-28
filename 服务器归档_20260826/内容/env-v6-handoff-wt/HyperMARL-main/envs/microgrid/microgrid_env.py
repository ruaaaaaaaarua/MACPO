"""微电网多智能体强化学习环境。

动作默认为 5 维:[P_el, P_bat, elec_price, h2_price, P_ht]。启用
``h2_learnable_rolling_order_enable`` 后加入 a5；a5 是完整的前向 H2 买单量，
而不是当前净缺口之上的额外订单，也不是热负荷占比。
启用动态交通网和路径动作后再加入 a6；a6 只由成交后的 buyer 用来选择
直达/两条绕行路径，seller 或无成交 agent 的 a6 不产生作用。

- 热负荷固定 100% 由氢承担,不足部分由外部氢市场兜底,不再引入天然气分流动作。
- H2 CDA 支持协议时刻、延迟交割、pending headroom 与 DirectReserve。
- 训练奖励仅使用外部系统成本和保留的外部 H2 依赖 surcharge；库存、终端、
  动作和 bonus 项仅保留为诊断。

本文件所有跨时间步信息流均经由 self.* 状态变量,不依赖全局状态。
"""

import numpy as np

from envs.microgrid.cda_market import run_continuous_double_auction
from envs.microgrid.config import MICROGRID_CONFIG, build_tou_table
from envs.microgrid.data_generator import generate_daily_profiles
from envs.microgrid.electric_lmp import build_electric_price_tables
from envs.microgrid.h2_transport import H2TransportNetwork
from envs.microgrid.p2p_market import run_bilateral_p2p_market, summarize_pairwise_trades


EPS = 1e-6


def _json_safe_voltage(value):
    values = np.asarray(value, dtype=np.float64)
    if values.ndim == 0:
        scalar = float(values)
        return scalar if np.isfinite(scalar) else None
    return [float(entry) if np.isfinite(entry) else None for entry in values.flat]


class MicrogridEnv:
    """多智能体微电网环境,所有智能体共享同一个系统级奖励。"""

    def __init__(self, config_overrides=None):
        # 基础维度与时间尺度,训练脚本会直接依赖 obs_dim/action_dim。
        self.cfg = dict(MICROGRID_CONFIG)
        if config_overrides:
            self.cfg.update(dict(config_overrides))
        self.agent_num = self.cfg["num_agents"]
        self.h2_traffic_enable = bool(self.cfg.get("h2_traffic_enable", False))
        self.h2_route_action_enable = bool(
            self.cfg.get("h2_route_action_enable", self.h2_traffic_enable)
        )
        if self.h2_route_action_enable and not self.h2_traffic_enable:
            raise ValueError("h2_route_action_enable requires h2_traffic_enable")
        # v2 键 (h2_traffic_eta_min/max) 优先于 legacy 键; 与 H2TransportNetwork
        # 的解析保持一致, 使 pending 观测窗与订单越界判定跟随实际 ETA 上界。
        self.h2_traffic_min_eta = int(
            self.cfg.get("h2_traffic_eta_min", self.cfg.get("h2_traffic_min_eta", 4))
        )
        self.h2_traffic_max_eta = int(
            self.cfg.get("h2_traffic_eta_max", self.cfg.get("h2_traffic_max_eta", 6))
        )
        if (
            self.h2_traffic_min_eta <= 0
            or self.h2_traffic_max_eta < self.h2_traffic_min_eta
        ):
            raise ValueError("H2 traffic ETA bounds must define a positive range")
        self.gas_price_obs_enable = bool(
            self.cfg.get("gas_price_obs_enable", False)
        )
        self.gas_pressure_obs_enable = bool(
            self.cfg.get("gas_pressure_obs_enable", False)
        )
        self.h2_pending_obs_enable = bool(
            self.cfg.get("h2_pending_obs_enable", False)
        )
        self.h2_pending_obs_horizon = max(
            0, int(self.cfg.get("h2_pending_obs_horizon", 0))
        )
        self.h2_pending_obs_auto_expand_to_eta = bool(
            self.cfg.get("h2_pending_obs_auto_expand_to_eta", True)
        )
        if self.h2_traffic_enable:
            self.h2_pending_obs_enable = True
            if self.h2_pending_obs_auto_expand_to_eta:
                self.h2_pending_obs_horizon = max(
                    self.h2_pending_obs_horizon, self.h2_traffic_max_eta
                )
        self.h2_pending_summary_obs_enable = bool(
            self.cfg.get("h2_pending_summary_obs_enable", self.h2_pending_obs_enable)
        )
        if self.h2_traffic_enable:
            self.h2_pending_summary_obs_enable = True
        self.h2_pending_obs_extra_dim = (
            2
            if self.h2_pending_obs_enable and self.h2_pending_summary_obs_enable
            else 0
        )
        self.h2_day_ahead_forecast_enable = bool(
            self.cfg.get("h2_day_ahead_forecast_enable", False)
        )
        self.h2_day_ahead_forecast_horizons = tuple(
            sorted({int(hour) for hour in self.cfg.get("h2_day_ahead_forecast_horizons", [4, 6, 10]) if int(hour) > 0})
        )
        self.h2_day_ahead_forecast_dim = (
            2 * len(self.h2_day_ahead_forecast_horizons)
            if self.h2_day_ahead_forecast_enable else 0
        )
        self.h2_supply_intent_message_enable = bool(
            self.cfg.get("h2_supply_intent_message_enable", False)
        )
        local_supply_facts = self.cfg.get("h2_local_supply_facts_enable")
        self.h2_local_supply_facts_enable = (
            self.h2_supply_intent_message_enable
            if local_supply_facts is None
            else bool(local_supply_facts)
        )
        self.h2_supply_intent_message_dim = (
            4 if self.h2_local_supply_facts_enable else 0
        )
        self.obs_dim = int(self.cfg["obs_dim"]) + (
            int(self.gas_price_obs_enable) + int(self.gas_pressure_obs_enable)
            + (self.h2_pending_obs_horizon if self.h2_pending_obs_enable else 0)
            + self.h2_pending_obs_extra_dim
            + (3 if self.h2_traffic_enable else 0)
            + self.h2_day_ahead_forecast_dim
            + self.h2_supply_intent_message_dim
        )
        # Historical config name retained for checkpoint/config compatibility.
        # Its current semantics are full action-controlled H2 buy ordering.
        self.h2_action_controlled_order_enable = bool(
            self.cfg.get("h2_learnable_rolling_order_enable", False)
        )
        self.h2_learnable_rolling_order_enable = (
            self.h2_action_controlled_order_enable
        )
        if self.h2_route_action_enable and not self.h2_action_controlled_order_enable:
            raise ValueError(
                "H2 route action requires action-controlled H2 ordering"
            )
        self.h2_action_order_max_peak_hours = float(
            self.cfg.get("h2_action_order_max_peak_hours", 1.0)
        )
        if self.h2_action_controlled_order_enable and (
            not np.isfinite(self.h2_action_order_max_peak_hours)
            or self.h2_action_order_max_peak_hours <= 0.0
        ):
            raise ValueError(
                "h2_action_order_max_peak_hours must be finite and strictly positive"
            )
        self.h2_learnable_rolling_order_active = bool(
            self.cfg.get(
                "h2_learnable_rolling_order_active",
                self.h2_learnable_rolling_order_enable,
            )
        )
        self.action_dim = int(self.cfg["action_dim"]) + int(
            self.h2_action_controlled_order_enable
            and int(self.cfg["action_dim"]) < 6
        )
        self.action_dim += int(
            self.h2_route_action_enable and self.action_dim < 7
        )

        self.dt = self.cfg["dt"]
        self.T = self.cfg["episode_length"]
        self.reward_scale = self.cfg["reward_scale"]
        self.reward_emission_mode = str(
            self.cfg.get("reward_emission_mode", "dense")
        )
        if self.reward_emission_mode not in {"dense", "terminal_total"}:
            raise ValueError(
                "reward_emission_mode must be 'dense' or 'terminal_total'"
            )
        self.day_boundary_interval = max(
            1, int(self.cfg.get("day_boundary_interval", 24))
        )
        self.day_boundary_info_enable = bool(
            self.cfg.get("day_boundary_info_enable", True)
        )
        self.daily_truncation_enable = bool(
            self.cfg.get("daily_truncation_enable", False)
        )

        # 分时电价表:buy 用于外部购电,sell 用于向外部电网售电。
        self.tou_buy, self.tou_sell = build_tou_table(self.T)

        # 智能体类型只用于区分产氢者/用氢者的设备能力与分析标记。
        # 当前氢能市场的买卖方向由净氢需求 net_h2_demand 的正负决定。
        self.agent_types = self.cfg["agent_types"]
        self.is_producer = np.array([role == "producer" for role in self.agent_types])
        self.is_consumer = np.array([role == "consumer" for role in self.agent_types])

        # 将配置中的标量列表转为 numpy 数组,便于逐智能体向量化计算。
        self.pv_cap = np.array(self.cfg["pv_cap"], dtype=np.float32)
        self.wt_cap = np.array(self.cfg["wt_cap"], dtype=np.float32)
        self.bat_cap = np.array(self.cfg["bat_cap"], dtype=np.float32)
        self.bat_power = np.array(self.cfg["bat_power"], dtype=np.float32)
        self.el_cap = np.array(self.cfg["el_cap"], dtype=np.float32)
        self.el_eff = np.array(self.cfg["el_eff"], dtype=np.float32)
        self.h2_tank_cap = np.array(self.cfg["h2_tank_cap"], dtype=np.float32)
        self.h2_tank_power = np.array(self.cfg["h2_tank_power"], dtype=np.float32)
        self.load_e_peak = np.array(self.cfg["load_e_peak"], dtype=np.float32)
        self.load_h_peak = np.array(self.cfg["load_h_peak"], dtype=np.float32)
        # One static peak heat-demand hour, expressed as kWh_H2. This is the
        # sole scale for a5 and deliberately uses no current/future profile.
        self.h2_order_qmax = (
            self.load_h_peak
            / max(float(self.cfg["boiler_eff"]), EPS)
            * self.dt
            * self.h2_action_order_max_peak_hours
        ).astype(np.float32)

        # 储氢罐上下限,内部状态 h2_level 的单位为 kg。
        self.h2_min = self.h2_tank_cap * self.cfg["h2_tank_min_ratio"]
        self.h2_max = self.h2_tank_cap * self.cfg["h2_tank_max_ratio"]

        # 氢能与市场价格参数。
        # lambda_h2 是兼容性默认值; 若 config 提供 lambda_h2_buy / lambda_h2_sell
        # 则启用"非对称外部氢价"——外部买氢贵、外部卖氢贱,迫使 agent 优先走内部市场。
        self.lambda_h2 = float(self.cfg["lambda_h2"])
        self.lambda_h2_buy = float(self.cfg.get("lambda_h2_buy", self.lambda_h2))
        self.lambda_h2_sell = float(self.cfg.get("lambda_h2_sell", self.lambda_h2))
        # 应急外购乘数只作用于"当小时未满足负荷"的瞬时平衡采购 (e_h2_ext > 0),
        # 不影响计划性采购的计价; 默认 1.0 保持历史行为。
        self.h2_emergency_price_multiplier = float(
            self.cfg.get("h2_emergency_price_multiplier", 1.0)
        )
        if self.h2_emergency_price_multiplier <= 0.0:
            raise ValueError("h2_emergency_price_multiplier must be positive")
        # 订单越界语义: free_cancel = 免费取消 (历史行为); pay_and_lose =
        # 照常入市、成交付款、越界交付作废 (设计规格 2a)。
        self.h2_order_horizon_clip_mode = str(
            self.cfg.get("h2_order_horizon_clip_mode", "free_cancel")
        )
        if self.h2_order_horizon_clip_mode not in {"free_cancel", "pay_and_lose"}:
            raise ValueError(
                "h2_order_horizon_clip_mode must be 'free_cancel' or 'pay_and_lose'"
            )
        self.gas_network_enable = bool(self.cfg.get("gas_network_enable", False))
        self.gas_price_dynamic_enable = bool(
            self.cfg.get("gas_price_dynamic_enable", False)
        )
        self.gas_price_bidirectional_enable = bool(
            self.cfg.get("gas_price_bidirectional_enable", False)
        )
        self.gas_pressure_init = float(self.cfg.get("gas_pressure_init", 1.0))
        self.gas_pressure_ref = float(self.cfg.get("gas_pressure_ref", 1.0))
        self.gas_pressure_min = float(self.cfg.get("gas_pressure_min", 0.6))
        self.gas_pressure_max = float(self.cfg.get("gas_pressure_max", 1.2))
        self.gas_pressure_recovery_rate = float(
            self.cfg.get("gas_pressure_recovery_rate", 0.1)
        )
        self.gas_withdrawal_gain = float(self.cfg.get("gas_withdrawal_gain", 1.0e-5))
        self.gas_injection_gain = float(self.cfg.get("gas_injection_gain", 1.0e-5))
        self.gas_exogenous_supply = float(self.cfg.get("gas_exogenous_supply", 0.0))
        self.gas_exogenous_demand = float(self.cfg.get("gas_exogenous_demand", 0.0))
        self.gas_price_base = float(self.cfg.get("gas_price_base", self.lambda_h2_buy))
        self.gas_price_sensitivity = float(
            self.cfg.get("gas_price_sensitivity", 0.0)
        )
        self.gas_price_min = float(self.cfg.get("gas_price_min", self.lambda_h2_sell))
        self.gas_price_max = float(self.cfg.get("gas_price_max", self.lambda_h2_buy))
        self.gas_sell_ratio = float(self.cfg.get("gas_sell_ratio", 0.5))
        self.gas_network_model = str(self.cfg.get("gas_network_model", "single"))
        self.gas_multinode_enable = self.gas_network_model in (
            "linear_multinode",
            "multinode",
            "linear",
        )
        self.gas_node_count = max(1, int(self.cfg.get("gas_node_count", self.agent_num)))
        self.gas_agent_node_indices = np.asarray(
            self.cfg.get("gas_agent_node_indices", list(range(self.agent_num))),
            dtype=np.int64,
        )
        if self.gas_agent_node_indices.size != self.agent_num:
            self.gas_agent_node_indices = np.arange(self.agent_num, dtype=np.int64)
        self.gas_agent_node_indices = np.clip(
            self.gas_agent_node_indices, 0, self.gas_node_count - 1
        )
        self.gas_node_pressure_init = self._gas_node_array(
            "gas_node_pressure_init", self.gas_pressure_init
        )
        self.gas_node_pressure_ref = self._gas_node_array(
            "gas_node_pressure_ref", self.gas_pressure_ref
        )
        self.gas_node_recovery_rate = float(
            self.cfg.get("gas_node_recovery_rate", self.gas_pressure_recovery_rate)
        )
        self.gas_node_withdrawal_gain = float(
            self.cfg.get("gas_node_withdrawal_gain", self.gas_withdrawal_gain)
        )
        self.gas_node_injection_gain = float(
            self.cfg.get("gas_node_injection_gain", self.gas_injection_gain)
        )
        self.gas_node_exogenous_supply = self._gas_node_array(
            "gas_node_exogenous_supply", 0.0
        )
        self.gas_node_exogenous_demand = self._gas_node_array(
            "gas_node_exogenous_demand", 0.0
        )
        self.gas_line_edges = np.asarray(
            self.cfg.get("gas_line_edges", []), dtype=np.int64
        )
        if self.gas_line_edges.size == 0:
            self.gas_line_edges = np.zeros((0, 2), dtype=np.int64)
        self.gas_line_edges = self.gas_line_edges.reshape(-1, 2)
        self.gas_line_conductance = np.asarray(
            self.cfg.get("gas_line_conductance", []), dtype=np.float32
        )
        if self.gas_line_conductance.size != self.gas_line_edges.shape[0]:
            self.gas_line_conductance = np.full(
                self.gas_line_edges.shape[0], 0.0, dtype=np.float32
            )
        # h2_thermal_share 现在由动作决定,这里的 default 仅用于 obs 归一化分母。
        self.h2_thermal_share_default = float(
            np.clip(self.cfg.get("h2_thermal_share_default", 1.0), 0.0, 1.0)
        )
        self.h2_eff_c = float(self.cfg.get("h2_eff_c", 1.0))
        self.h2_eff_d = float(self.cfg.get("h2_eff_d", 1.0))

        self.elec_price_min = float(self.cfg["elec_price_min"])
        self.elec_price_max = float(self.cfg["elec_price_max"])
        self.elec_price_init = float(self.cfg["elec_price_init"])
        self.elec_price_mode = str(self.cfg.get("elec_price_mode", "tou"))
        self.elec_lmp_price_min = float(
            self.cfg.get("elec_lmp_price_min", self.elec_price_min)
        )
        self.elec_lmp_price_max = float(
            self.cfg.get("elec_lmp_price_max", self.elec_price_max)
        )
        self.h2_price_min = float(self.cfg["h2_price_min"])
        self.h2_price_max = float(self.cfg["h2_price_max"])
        self.h2_price_init = float(self.cfg["h2_price_init"])

        self.cda_shaping_coef = float(self.cfg["cda_shaping_coef"])
        self.elec_internal_cda_enable = bool(
            self.cfg.get("elec_internal_cda_enable", True)
        )
        self.h2_internal_cda_enable = bool(
            self.cfg.get("h2_internal_cda_enable", True)
        )
        self.elec_p2p_enable = bool(self.cfg.get("elec_p2p_enable", False))
        self.h2_p2p_enable = bool(self.cfg.get("h2_p2p_enable", False))
        self.elec_p2p_diagnostics_enable = bool(
            self.cfg.get("elec_p2p_diagnostics_enable", False)
        )
        self.h2_p2p_diagnostics_enable = bool(
            self.cfg.get("h2_p2p_diagnostics_enable", False)
        )
        self.elec_p2p_price_rule = str(
            self.cfg.get("elec_p2p_price_rule", "split_surplus")
        )
        self.h2_p2p_price_rule = str(
            self.cfg.get("h2_p2p_price_rule", "split_surplus")
        )
        self.elec_p2p_distance_fee_coef = float(
            self.cfg.get("elec_p2p_distance_fee_coef", 0.0)
        )
        self.h2_p2p_distance_fee_coef = float(
            self.cfg.get("h2_p2p_distance_fee_coef", 0.0)
        )
        self.elec_p2p_max_distance = float(self.cfg.get("elec_p2p_max_distance", 0.0))
        self.h2_p2p_max_distance = float(self.cfg.get("h2_p2p_max_distance", 0.0))
        self.h2_internal_trade_bonus_enable = bool(
            self.cfg.get("h2_internal_trade_bonus_enable", False)
        )
        self.h2_internal_trade_bonus_coef = float(
            self.cfg.get("h2_internal_trade_bonus_coef", 0.0)
        )
        self.external_h2_dependency_penalty_enable = bool(
            self.cfg.get("external_h2_dependency_penalty_enable", False)
        )
        self.external_h2_dependency_penalty_coef = float(
            self.cfg.get("external_h2_dependency_penalty_coef", 0.0)
        )
        external_h2_dependency_penalty_kg = float(
            self.cfg.get("external_h2_dependency_penalty_kg", 0.0)
        )
        if (
            self.external_h2_dependency_penalty_coef <= 0.0
            and external_h2_dependency_penalty_kg > 0.0
        ):
            self.external_h2_dependency_penalty_coef = (
                external_h2_dependency_penalty_kg
                / max(float(self.cfg["LHV_H2"]), EPS)
            )
        self.terminal_value_coef = float(self.cfg["terminal_value_coef"])

        # 逐步惩罚超参 (第五轮: 软终端约束 α/(T-t) * max(0,|x-0.5|-db)^2)。
        self.penalty_enable = bool(self.cfg.get("penalty_enable", True))
        self.soc_penalty_coef = float(self.cfg.get("soc_penalty_coef", 0.0))
        self.h2_penalty_coef = float(self.cfg.get("h2_penalty_coef", 0.0))
        self.penalty_target_center = float(
            self.cfg.get("penalty_target_center", 0.5)
        )
        self.penalty_deadband = float(self.cfg.get("penalty_deadband", 0.1))
        self.consumer_h2_deadband = float(
            self.cfg.get("consumer_h2_deadband", self.penalty_deadband)
        )
        self.consumer_h2_deadband_agent_indices = np.array(
            self.cfg.get("consumer_h2_deadband_agent_indices", []),
            dtype=np.int64,
        )
        self.soc_penalty_targets = np.asarray(
            self.cfg.get(
                "soc_penalty_targets",
                [self.penalty_target_center] * self.agent_num,
            ),
            dtype=np.float32,
        )
        if self.soc_penalty_targets.size != self.agent_num:
            self.soc_penalty_targets = np.full(
                self.agent_num, self.penalty_target_center, dtype=np.float32
            )
        self.h2_penalty_targets = np.asarray(
            self.cfg.get(
                "h2_penalty_targets",
                [self.penalty_target_center] * self.agent_num,
            ),
            dtype=np.float32,
        )
        if self.h2_penalty_targets.size != self.agent_num:
            self.h2_penalty_targets = np.full(
                self.agent_num, self.penalty_target_center, dtype=np.float32
            )
        self.low_inventory_penalty_enable = bool(
            self.cfg.get("low_inventory_penalty_enable", False)
        )
        self.low_inventory_penalty_coef = float(
            self.cfg.get("low_inventory_penalty_coef", 0.0)
        )
        self.soc_low_threshold = float(self.cfg.get("soc_low_threshold", 0.0))
        self.h2_low_threshold = float(self.cfg.get("h2_low_threshold", 0.0))
        self.terminal_h2_floor_penalty_enable = bool(
            self.cfg.get("terminal_h2_floor_penalty_enable", False)
        )
        self.terminal_h2_floor_threshold = float(
            self.cfg.get("terminal_h2_floor_threshold", 0.0)
        )
        self.terminal_h2_floor_penalty_coef = float(
            self.cfg.get("terminal_h2_floor_penalty_coef", 0.0)
        )
        self.terminal_h2_floor_agent_indices = np.array(
            self.cfg.get(
                "terminal_h2_floor_agent_indices",
                list(range(self.agent_num)),
            ),
            dtype=np.int64,
        )
        self.terminal_h2_shortfall_value_enable = bool(
            self.cfg.get("terminal_h2_shortfall_value_enable", False)
        )
        self.terminal_h2_shortfall_value_targets = np.asarray(
            self.cfg.get(
                "terminal_h2_shortfall_value_targets",
                [self.penalty_target_center] * self.agent_num,
            ),
            dtype=np.float32,
        )
        if self.terminal_h2_shortfall_value_targets.size != self.agent_num:
            self.terminal_h2_shortfall_value_targets = np.full(
                self.agent_num, self.penalty_target_center, dtype=np.float32
            )
        self.terminal_h2_shortfall_value_coef = float(
            self.cfg.get("terminal_h2_shortfall_value_coef", 0.0)
        )
        self.terminal_h2_shortfall_value_agent_indices = np.array(
            self.cfg.get(
                "terminal_h2_shortfall_value_agent_indices",
                list(range(self.agent_num)),
            ),
            dtype=np.int64,
        )
        self.terminal_h2_settlement_in_reward_enable = bool(
            self.cfg.get("terminal_h2_settlement_in_reward_enable", False)
        )
        self.terminal_soc_floor_penalty_enable = bool(
            self.cfg.get("terminal_soc_floor_penalty_enable", False)
        )
        self.terminal_soc_floor_threshold = float(
            self.cfg.get("terminal_soc_floor_threshold", 0.0)
        )
        self.terminal_soc_floor_penalty_coef = float(
            self.cfg.get("terminal_soc_floor_penalty_coef", 0.0)
        )
        self.terminal_soc_floor_agent_indices = np.array(
            self.cfg.get(
                "terminal_soc_floor_agent_indices",
                list(range(self.agent_num)),
            ),
            dtype=np.int64,
        )
        self.terminal_battery_salvage_enable = bool(
            self.cfg.get("terminal_battery_salvage_enable", False)
        )
        self.terminal_battery_salvage_value_coef = float(
            self.cfg.get("terminal_battery_salvage_value_coef", 0.0)
        )
        self.terminal_battery_salvage_capacity_scaled_enable = bool(
            self.cfg.get("terminal_battery_salvage_capacity_scaled_enable", False)
        )
        self.terminal_battery_salvage_reference_capacity = float(
            self.cfg.get("terminal_battery_salvage_reference_capacity", 4000.0)
        )
        self.terminal_battery_salvage_agent_indices = np.array(
            self.cfg.get(
                "terminal_battery_salvage_agent_indices",
                list(range(self.agent_num)),
            ),
            dtype=np.int64,
        )
        self.a2_late_soc_reserve_enable = bool(
            self.cfg.get("a2_late_soc_reserve_enable", False)
        )
        self.a2_late_soc_reserve_agent_indices = np.array(
            self.cfg.get("a2_late_soc_reserve_agent_indices", [2]),
            dtype=np.int64,
        )
        self.a2_late_soc_reserve_threshold = float(
            self.cfg.get("a2_late_soc_reserve_threshold", 0.0)
        )
        self.a2_late_soc_reserve_horizon = int(
            self.cfg.get("a2_late_soc_reserve_horizon", 0)
        )
        self.stepwise_h2_floor_penalty_enable = bool(
            self.cfg.get("stepwise_h2_floor_penalty_enable", False)
        )
        self.stepwise_h2_floor_thresholds = np.asarray(
            self.cfg.get("stepwise_h2_floor_thresholds", [0.0] * self.agent_num),
            dtype=np.float32,
        )
        if self.stepwise_h2_floor_thresholds.size != self.agent_num:
            self.stepwise_h2_floor_thresholds = np.zeros(
                self.agent_num, dtype=np.float32
            )
        self.stepwise_h2_floor_weights = np.asarray(
            self.cfg.get("stepwise_h2_floor_weights", [1.0] * self.agent_num),
            dtype=np.float32,
        )
        if self.stepwise_h2_floor_weights.size != self.agent_num:
            self.stepwise_h2_floor_weights = np.ones(
                self.agent_num, dtype=np.float32
            )
        self.stepwise_h2_floor_penalty_coef = float(
            self.cfg.get("stepwise_h2_floor_penalty_coef", 0.0)
        )
        self.stepwise_h2_floor_urgency_gain = float(
            self.cfg.get("stepwise_h2_floor_urgency_gain", 0.0)
        )
        self.action_reg_enable = bool(self.cfg.get("action_reg_enable", False))
        self.action_reg_indices = np.array(
            self.cfg.get("action_reg_indices", []), dtype=np.int64
        )
        self.action_magnitude_penalty_coef = float(
            self.cfg.get("action_magnitude_penalty_coef", 0.0)
        )
        self.action_delta_penalty_coef = float(
            self.cfg.get("action_delta_penalty_coef", 0.0)
        )

        # soc_target_curve_* 是第三/四轮时变曲线的历史路径, 第五轮起默认关闭.
        # 当 enable=False 时不再使用 curve, 完全走软终端约束. 保留字段与代码
        # 仅为未来回滚做兼容 (例如需要对比 ablation 时再打开).
        self.soc_target_curve_enable = bool(
            self.cfg.get("soc_target_curve_enable", False)
        )
        if self.soc_target_curve_enable:
            tou_lo = float(self.tou_buy.min())
            tou_hi = float(self.tou_buy.max())
            if tou_hi > tou_lo:
                rel = (self.tou_buy - tou_lo) / (tou_hi - tou_lo)
            else:
                rel = np.zeros_like(self.tou_buy)
            soc_lo = float(self.cfg.get("soc_target_lo", 0.2))
            soc_hi = float(self.cfg.get("soc_target_hi", 0.8))
            h2_lo = float(self.cfg.get("h2_target_lo", 0.2))
            h2_hi = float(self.cfg.get("h2_target_hi", 0.8))
            self.soc_target_curve = (soc_hi - (soc_hi - soc_lo) * rel).astype(
                np.float32
            )
            self.h2_target_curve = (h2_hi - (h2_hi - h2_lo) * rel).astype(
                np.float32
            )
        else:
            # 兼容性: 保留数组但全部 = center, 下游若误引用也不会走偏.
            self.soc_target_curve = np.full(
                self.T, self.penalty_target_center, dtype=np.float32
            )
            self.h2_target_curve = np.full(
                self.T, self.penalty_target_center, dtype=np.float32
            )

        # 氢市场协议时刻与运输延迟分开控制, 便于做课程学习。
        self.h2_market_lag_enable = bool(self.cfg.get("h2_market_lag_enable", True))
        self.h2_market_schedule_enable = bool(
            self.cfg.get("h2_market_schedule_enable", self.h2_market_lag_enable)
        )
        schedule = self.cfg.get("h2_market_schedule", list(range(self.T)))
        self.h2_market_schedule = set(int(s) for s in schedule)
        self.h2_delivery_lag = int(self.cfg.get("h2_delivery_lag", 0))
        self.h2_cap_aware_buy_enable = bool(
            self.cfg.get("h2_cap_aware_buy_enable", False)
        )
        # 计划性外购通道 (设计规格 1a): 买单在内部市场未撮合的剩余量按外部
        # 计划价即时付款、延迟交付; False = v1 行为 (剩余量消失, 仅应急平衡)。
        self.h2_planned_external_order_enable = bool(
            self.cfg.get("h2_planned_external_order_enable", False)
        )
        if self.h2_planned_external_order_enable and not (
            self.h2_market_lag_enable and self.h2_delivery_lag > 0
        ):
            raise ValueError(
                "h2_planned_external_order_enable requires lagged H2 delivery"
            )
        self.h2_delivery_reservation_enable = bool(
            self.cfg.get("h2_delivery_reservation_enable", False)
        )
        self.h2_delivery_reservation_horizon = max(
            0, int(self.cfg.get("h2_delivery_reservation_horizon", self.h2_delivery_lag))
        )
        if self.h2_traffic_enable:
            self.h2_delivery_reservation_horizon = max(
                self.h2_delivery_reservation_horizon,
                self.h2_traffic_max_eta,
            )
        self.h2_delivery_reservation_ratio = float(
            self.cfg.get("h2_delivery_reservation_ratio", 1.0)
        )
        self.h2_learnable_rolling_order_agent_indices = np.array(
            self.cfg.get("h2_learnable_rolling_order_agent_indices", []),
            dtype=np.int64,
        )
        self.h2_learnable_rolling_order_max_fraction = float(
            self.cfg.get("h2_learnable_rolling_order_max_fraction", 0.25)
        )
        self.h2_buyer_reservation_demand_enable = bool(
            self.cfg.get("h2_buyer_reservation_demand_enable", False)
        )
        if (
            self.h2_action_controlled_order_enable
            and self.h2_buyer_reservation_demand_enable
        ):
            raise ValueError("H2 action-controlled ordering and heuristic buyer reservation cannot both be enabled")
        self.h2_buyer_reservation_agent_indices = np.array(
            self.cfg.get("h2_buyer_reservation_agent_indices", []),
            dtype=np.int64,
        )
        self.h2_buyer_reservation_target_ratios = np.asarray(
            self.cfg.get(
                "h2_buyer_reservation_target_ratios",
                np.zeros(self.agent_num, dtype=np.float32),
            ),
            dtype=np.float32,
        )
        if self.h2_buyer_reservation_target_ratios.size < self.agent_num:
            self.h2_buyer_reservation_target_ratios = np.pad(
                self.h2_buyer_reservation_target_ratios,
                (0, self.agent_num - self.h2_buyer_reservation_target_ratios.size),
                mode="constant",
            )
        self.h2_buyer_reservation_target_ratios = (
            self.h2_buyer_reservation_target_ratios[: self.agent_num]
        )
        self.h2_buyer_reservation_demand_gain = float(
            self.cfg.get("h2_buyer_reservation_demand_gain", 1.0)
        )
        self.h2_buyer_reservation_max_order_fraction = float(
            self.cfg.get("h2_buyer_reservation_max_order_fraction", 0.25)
        )
        self.h2_transport_loss = float(self.cfg.get("h2_transport_loss", 0.0))

        # 观测归一化分母。容量为 0 的设备使用 1.0 作为安全分母,避免除零。
        self.pv_cap_safe = np.where(self.pv_cap > 0, self.pv_cap, 1.0)
        self.wt_cap_safe = np.where(self.wt_cap > 0, self.wt_cap, 1.0)
        self.load_e_peak_safe = np.where(self.load_e_peak > 0, self.load_e_peak, 1.0)
        self.load_h_peak_safe = np.where(self.load_h_peak > 0, self.load_h_peak, 1.0)
        self.h2_tank_cap_safe = np.where(self.h2_tank_cap > 0, self.h2_tank_cap, 1.0)
        self.p_grid_scale = np.maximum(
            self.load_e_peak + self.el_cap + self.bat_power + self.pv_cap + self.wt_cap,
            1.0,
        )
        # 氢外部交互的归一化分母按"最大可能单步氢流"估算。
        self.h2_exchange_scale = np.maximum(
            self.load_h_peak * self.dt / max(self.cfg["boiler_eff"], EPS)
            + self.h2_tank_power * self.dt
            + self.el_cap * self.el_eff * self.dt,
            1.0,
        )
        # 第六轮新增: pending 在途氢归一化分母 (kWh_H2).
        # 使用单个 agent 的完整储氢罐能量容量,实际观测用 clip(..., 0, 1) 截断.
        self.pending_scale = np.maximum(
            self.h2_tank_cap * self.cfg["LHV_H2"],
            1.0,
        ).astype(np.float32)

        # 环境自有随机数生成器。seed() 会替换该对象,确保曲线可复现。
        self._rng = np.random.RandomState()
        self._seed_value = 0
        self.h2_transport_network = (
            H2TransportNetwork(self.cfg) if self.h2_traffic_enable else None
        )
        self.power_flow_enable = bool(self.cfg.get("power_flow_enable", False))
        self.power_flow = None
        self.power_flow_reactive_per_active_load = float(
            np.tan(np.arccos(float(self.cfg.get("power_flow_load_power_factor", 0.95))))
        )
        self.power_flow_pcc_injection_scale = float(
            self.cfg.get("power_flow_pcc_injection_scale", 1.0)
        )
        if (
            not np.isfinite(self.power_flow_pcc_injection_scale)
            or self.power_flow_pcc_injection_scale <= 0.0
        ):
            raise ValueError("power_flow_pcc_injection_scale must be finite and positive")
        if self.power_flow_enable:
            from envs.microgrid.power_flow import build_power_flow

            self.power_flow = build_power_flow(self.cfg)
        if (
            self.h2_planned_external_order_enable
            and self.h2_transport_network is not None
            and not self.h2_transport_network.external_node_enable
        ):
            raise ValueError(
                "planned external orders with traffic require "
                "h2_traffic_external_node_enable"
            )

        # reset() 中初始化的运行态变量。
        self.t = 0
        self.soc = None
        self.h2_level = None
        self.last_p_grid = None
        self.last_e_h2_ext = None
        self.last_elec_clearing_price = self.elec_price_init
        self.last_h2_clearing_price = self.h2_price_init
        self.elec_agent_bus_indices = np.arange(self.agent_num, dtype=np.int64)
        self.elec_node_buy_prices = np.tile(self.tou_buy.reshape(-1, 1), (1, self.agent_num))
        self.elec_node_sell_prices = np.tile(self.tou_sell.reshape(-1, 1), (1, self.agent_num))
        self.elec_agent_buy_prices = np.tile(self.tou_buy.reshape(-1, 1), (1, self.agent_num))
        self.elec_agent_sell_prices = np.tile(self.tou_sell.reshape(-1, 1), (1, self.agent_num))
        self.elec_line_loading = np.zeros((self.T, 0), dtype=np.float32)
        self.elec_line_loading_max = np.zeros(self.T, dtype=np.float32)
        self.elec_lmp_congestion_count = np.zeros(self.T, dtype=np.float32)
        self.elec_lmp_slack_import = np.zeros(self.T, dtype=np.float32)
        self.elec_lmp_price_spread = np.zeros(self.T, dtype=np.float32)
        self.elec_lmp_status_code = np.zeros(self.T, dtype=np.float32)
        self.gas_pressure = self.gas_pressure_init
        self.gas_pressure_prev = self.gas_pressure_init
        self.gas_node_pressure = self.gas_node_pressure_init.copy()
        self.gas_node_pressure_prev = self.gas_node_pressure_init.copy()
        self.h2_external_buy_price = self.lambda_h2_buy
        self.h2_external_sell_price = self.lambda_h2_sell
        self.h2_external_buy_prices = np.full(
            self.agent_num, self.lambda_h2_buy, dtype=np.float32
        )
        self.h2_external_sell_prices = np.full(
            self.agent_num, self.lambda_h2_sell, dtype=np.float32
        )
        self.profiles = None
        self.last_reg_actions = None
        # pending_h2_deliveries: list of dicts
        #   {"deliver_at": int, "buyer_id": int, "quantity": float (kWh_H2), "price": float}
        self.pending_h2_deliveries = []
        self.last_h2_transport_shipments = []
        self.episode_total_cost = 0.0
        self.terminal_economic_settlement_enable = bool(
            self.cfg.get("terminal_economic_settlement_enable", False)
        )
        self._initial_terminal_asset_value = 0.0

    def seed(self, seed):
        """设置环境随机种子,影响日内曲线和评估时的确定性复现。"""
        self._seed_value = int(seed)
        self._rng = np.random.RandomState(self._seed_value)

    def reset(self):
        """重置一个 episode,初始化储能状态并生成新的日内外生曲线。"""
        self.t = 0
        self.soc = np.full(
            self.agent_num,
            float(self.cfg.get("soc_init", 0.1)),
            dtype=np.float32,
        )
        # h2_tank_init_ratio 缺省时回退 min_ratio (与历史默认 0.05 相同,
        # v1 行为不变); v2 用 0.4 消除冷启动应急伪影。
        self.h2_level = (
            self.h2_tank_cap
            * float(self.cfg.get("h2_tank_init_ratio", self.cfg["h2_tank_min_ratio"]))
        ).astype(np.float32)
        self.last_p_grid = np.zeros(self.agent_num, dtype=np.float32)
        self.last_e_h2_ext = np.zeros(self.agent_num, dtype=np.float32)
        self.last_elec_clearing_price = self.elec_price_init
        self.last_h2_clearing_price = self.h2_price_init
        self.gas_pressure = float(
            np.clip(
                self.gas_pressure_init,
                self.gas_pressure_min,
                self.gas_pressure_max,
            )
        )
        self.gas_pressure_prev = self.gas_pressure
        self.gas_node_pressure = np.clip(
            self.gas_node_pressure_init.astype(np.float32),
            self.gas_pressure_min,
            self.gas_pressure_max,
        )
        self.gas_node_pressure_prev = self.gas_node_pressure.copy()
        self.h2_external_buy_price, self.h2_external_sell_price = (
            self._current_external_h2_prices()
        )
        self.h2_external_buy_prices, self.h2_external_sell_prices = (
            self._current_external_h2_price_vectors()
        )
        self.profiles = generate_daily_profiles(self.cfg, self._rng)
        if self.h2_transport_network is not None:
            self.h2_transport_network.reset(
                day_index=int(self.profiles.get("_italian_day_index", 0)),
                seed=self._seed_value,
            )
        self._set_electric_price_tables()
        self.last_reg_actions = None
        self.pending_h2_deliveries = []
        self.last_h2_transport_shipments = []
        self.episode_total_cost = 0.0
        self._initial_terminal_asset_value = self._terminal_asset_values()["terminal_asset_value"]
        return self._get_obs()

    def _set_electric_price_tables(self):
        tables = build_electric_price_tables(
            self.cfg, self.profiles, self.tou_buy, self.tou_sell
        )
        self.elec_price_mode = str(tables["mode"])
        self.elec_agent_bus_indices = np.asarray(
            tables["agent_bus_indices"], dtype=np.int64
        )
        self.elec_node_buy_prices = np.asarray(
            tables["node_buy_prices"], dtype=np.float32
        )
        self.elec_node_sell_prices = np.asarray(
            tables["node_sell_prices"], dtype=np.float32
        )
        self.elec_agent_buy_prices = np.asarray(
            tables["agent_buy_prices"], dtype=np.float32
        )
        self.elec_agent_sell_prices = np.asarray(
            tables["agent_sell_prices"], dtype=np.float32
        )
        self.elec_line_loading = np.asarray(
            tables["line_loading"], dtype=np.float32
        )
        self.elec_line_loading_max = np.asarray(
            tables["line_loading_max"], dtype=np.float32
        )
        self.elec_lmp_congestion_count = np.asarray(
            tables["congestion_count"], dtype=np.float32
        )
        self.elec_lmp_slack_import = np.asarray(
            tables["slack_import"], dtype=np.float32
        )
        self.elec_lmp_price_spread = np.asarray(
            tables["price_spread"], dtype=np.float32
        )
        self.elec_lmp_status_code = np.asarray(
            tables["status_code"], dtype=np.float32
        )

    def _current_electric_agent_prices(self, t):
        idx = int(np.clip(t, 0, self.T - 1))
        return self.elec_agent_buy_prices[idx], self.elec_agent_sell_prices[idx]

    def _scale_price(self, action_value, low, high):
        """将 [-1, 1] 的报价动作线性映射到市场价格区间。"""
        return ((action_value + 1.0) / 2.0) * (high - low) + low

    def _gas_node_array(self, key, default):
        values = np.asarray(self.cfg.get(key, default), dtype=np.float32)
        if values.ndim == 0:
            values = np.full(self.gas_node_count, float(values), dtype=np.float32)
        if values.size != self.gas_node_count:
            fill = float(values.reshape(-1)[0]) if values.size > 0 else float(default)
            values = np.full(self.gas_node_count, fill, dtype=np.float32)
        return values.reshape(self.gas_node_count).astype(np.float32)

    def _price_from_pressure(self, pressure, pressure_ref):
        if not (self.gas_network_enable and self.gas_price_dynamic_enable):
            return np.full_like(
                np.asarray(pressure, dtype=np.float32),
                self.lambda_h2_buy,
                dtype=np.float32,
            )
        pressure_shortfall = pressure_ref - pressure
        if not self.gas_price_bidirectional_enable:
            pressure_shortfall = np.maximum(0.0, pressure_shortfall)
        buy_price = self.gas_price_base + self.gas_price_sensitivity * pressure_shortfall
        return np.clip(buy_price, self.gas_price_min, self.gas_price_max)

    def _current_external_h2_prices(self):
        if self.gas_multinode_enable:
            buy_prices, sell_prices = self._current_external_h2_price_vectors()
            return float(np.mean(buy_prices)), float(np.mean(sell_prices))
        if self.gas_network_enable and self.gas_price_dynamic_enable:
            buy_price = float(
                self._price_from_pressure(self.gas_pressure, self.gas_pressure_ref)
            )
            sell_price = max(0.0, min(self.gas_sell_ratio * buy_price, buy_price - EPS))
            return buy_price, float(sell_price)
        return self.lambda_h2_buy, self.lambda_h2_sell

    def _current_external_h2_price_vectors(self):
        if self.gas_multinode_enable:
            node_buy = self._price_from_pressure(
                self.gas_node_pressure, self.gas_node_pressure_ref
            )
            node_buy = np.asarray(node_buy, dtype=np.float32)
            node_sell = np.minimum(
                self.gas_sell_ratio * node_buy, node_buy - EPS
            )
            node_sell = np.maximum(0.0, node_sell).astype(np.float32)
            return (
                node_buy[self.gas_agent_node_indices].astype(np.float32),
                node_sell[self.gas_agent_node_indices].astype(np.float32),
            )
        buy_price, sell_price = self._current_external_h2_prices()
        return (
            np.full(self.agent_num, buy_price, dtype=np.float32),
            np.full(self.agent_num, sell_price, dtype=np.float32),
        )

    def _update_gas_network(self, external_withdrawal, external_injection):
        self.gas_pressure_prev = float(self.gas_pressure)
        self.gas_node_pressure_prev = self.gas_node_pressure.copy()
        if self.gas_network_enable:
            if self.gas_multinode_enable:
                withdrawal = np.zeros(self.gas_node_count, dtype=np.float32)
                injection = np.zeros(self.gas_node_count, dtype=np.float32)
                external_withdrawal = np.asarray(external_withdrawal, dtype=np.float32)
                external_injection = np.asarray(external_injection, dtype=np.float32)
                for agent_id, node_id in enumerate(self.gas_agent_node_indices):
                    withdrawal[node_id] += max(0.0, float(external_withdrawal[agent_id]))
                    injection[node_id] += max(0.0, float(external_injection[agent_id]))
                line_flow = np.zeros(self.gas_node_count, dtype=np.float32)
                for edge_id, (u, v) in enumerate(self.gas_line_edges):
                    if (
                        u < 0
                        or v < 0
                        or u >= self.gas_node_count
                        or v >= self.gas_node_count
                    ):
                        continue
                    flow = float(self.gas_line_conductance[edge_id]) * (
                        float(self.gas_node_pressure[v])
                        - float(self.gas_node_pressure[u])
                    )
                    line_flow[u] += flow
                    line_flow[v] -= flow
                pressure_next = (
                    self.gas_node_pressure
                    + self.gas_node_recovery_rate
                    * (self.gas_node_pressure_ref - self.gas_node_pressure)
                    + line_flow
                    + self.gas_node_injection_gain * injection
                    - self.gas_node_withdrawal_gain * withdrawal
                    + self.gas_node_exogenous_supply
                    - self.gas_node_exogenous_demand
                )
                self.gas_node_pressure = np.clip(
                    pressure_next, self.gas_pressure_min, self.gas_pressure_max
                ).astype(np.float32)
                self.gas_pressure = float(np.mean(self.gas_node_pressure))
            else:
                pressure_next = (
                    self.gas_pressure
                    + self.gas_pressure_recovery_rate
                    * (self.gas_pressure_ref - self.gas_pressure)
                    + self.gas_injection_gain * max(0.0, float(external_injection))
                    - self.gas_withdrawal_gain * max(0.0, float(external_withdrawal))
                    + self.gas_exogenous_supply
                    - self.gas_exogenous_demand
                )
                self.gas_pressure = float(
                    np.clip(pressure_next, self.gas_pressure_min, self.gas_pressure_max)
                )
                self.gas_node_pressure = np.full(
                    self.gas_node_count, self.gas_pressure, dtype=np.float32
                )
        self.h2_external_buy_price, self.h2_external_sell_price = (
            self._current_external_h2_prices()
        )
        self.h2_external_buy_prices, self.h2_external_sell_prices = (
            self._current_external_h2_price_vectors()
        )
        return self.gas_pressure

    def _summarize_agent_results(self, market_result):
        """复制市场模块返回的逐智能体成交结果,避免外部误改内部字典。"""
        return [dict(result) for result in market_result["agent_results"]]

    def _store_h2_delivery(self, buyer, qty):
        """把到货氢气写入买方储氢罐,返回(实际入罐量,溢出量),单位 kWh_H2。"""
        buyer = int(buyer)
        qty = float(qty)
        if qty <= 0.0:
            return 0.0, 0.0
        if self.h2_tank_cap[buyer] <= 0:
            return 0.0, qty

        lhv = self.cfg["LHV_H2"]
        kg_delta = qty / lhv
        headroom_kg = self.h2_max[buyer] - self.h2_level[buyer]
        accept_kg = max(0.0, min(kg_delta, headroom_kg))
        self.h2_level[buyer] = float(
            np.clip(
                self.h2_level[buyer] + accept_kg,
                self.h2_min[buyer],
                self.h2_max[buyer],
            )
        )
        accepted_energy = accept_kg * lhv
        overflow_energy = max(0.0, (kg_delta - accept_kg) * lhv)
        return accepted_energy, overflow_energy

    def _deliver_pending_h2(self, current_t=None):
        """处理到达 self.t 的 pending 氢交易,把 kWh_H2 折算为 kg 入罐。

        Returns:
            tuple[np.ndarray, float]: (per-agent 入库 kWh_H2, 超容外售总 kWh_H2)。
            超容部分视作被迫在外部市场按 λ_h2 卖出,产生负成本(收益),
            与 c_h2 的正负号约定一致(正=购入=成本,负=外售=收益)。
        """
        if not self.pending_h2_deliveries:
            return np.zeros(self.agent_num, dtype=np.float32), 0.0

        delivered_energy = np.zeros(self.agent_num, dtype=np.float32)
        remaining: list = []
        overflow_energy = 0.0
        check_t = int(self.t if current_t is None else current_t)
        for record in self.pending_h2_deliveries:
            if record["deliver_at"] > check_t:
                remaining.append(record)
                continue
            buyer = int(record["buyer_id"])
            qty = float(record["quantity"])  # kWh_H2
            accepted, overflow = self._store_h2_delivery(buyer, qty)
            delivered_energy[buyer] += accepted
            overflow_energy += overflow
        self.pending_h2_deliveries = remaining
        return delivered_energy, overflow_energy

    def _pending_h2_arrival_buckets(self):
        horizon = max(0, int(self.h2_pending_obs_horizon))
        buckets = np.zeros((self.agent_num, horizon), dtype=np.float32)
        per_agent = np.zeros(self.agent_num, dtype=np.float32)
        if horizon <= 0:
            return buckets, per_agent
        for record in self.pending_h2_deliveries:
            buyer = int(record["buyer_id"])
            if buyer < 0 or buyer >= self.agent_num:
                continue
            qty = max(0.0, float(record["quantity"]))
            per_agent[buyer] += qty
            hours_until_arrival = int(record["deliver_at"]) - int(self.t)
            if 1 <= hours_until_arrival <= horizon:
                buckets[buyer, hours_until_arrival - 1] += qty
        return buckets, per_agent

    def _pending_adjusted_h2_headroom(self, pending_h2_energy_agent):
        h2_headroom_energy = np.maximum(
            (self.h2_max - self.h2_level) * self.cfg["LHV_H2"],
            0.0,
        ).astype(np.float32)
        return h2_headroom_energy - pending_h2_energy_agent

    def _supply_intent_facts(self, agent_id, t, pending_h2_arrival_buckets):
        """Return four local, current-day-known supply facts for Env-v5.

        Deficits refer only to future hours of the present episode; no profile
        element after the day boundary can leak into the actor observation.
        The final value is the external supplier's shortest ETA under only the
        background traffic observable at the current hour.
        """
        facts = []
        for horizon in (4, 6, 10):
            stop = min(self.T, int(t) + int(horizon) + 1)
            future_h2_need = float(np.sum(
                self.profiles["load_h"][agent_id, int(t) + 1:stop]
            ) * self.dt / max(float(self.cfg["boiler_eff"]), EPS))
            scale = max(
                float(self.load_h_peak_safe[agent_id]) * self.dt * horizon
                / max(float(self.cfg["boiler_eff"]), EPS),
                EPS,
            )
            pending = float(np.sum(
                pending_h2_arrival_buckets[agent_id, :min(horizon, self.h2_pending_obs_horizon)]
            ))
            tank_energy = max(0.0, float(self.h2_level[agent_id]) * float(self.cfg["LHV_H2"]))
            facts.append(np.clip(max(0.0, future_h2_need - tank_energy - pending) / scale, 0.0, 1.0))
        facts.append(self._external_min_eta_normalized(agent_id, t))
        return facts

    def _external_min_eta_normalized(self, agent_id, t):
        """Current-background shortest external route ETA, normalized to [0, 1]."""
        external_min_eta = 0.0
        network = self.h2_transport_network
        if network is not None and network.external_node_id is not None:
            current_background = network.background_utilization(int(t))
            eta = min(
                network._route_eta(path, current_background)[0]
                for path in network.route_options(network.external_node_id, agent_id)
            )
            eta_width = max(float(network.max_eta - network.min_eta), 1.0)
            external_min_eta = np.clip(
                (float(eta) - float(network.min_eta)) / eta_width,
                0.0,
                1.0,
            )
        return float(external_min_eta)

    def _pending_h2_reservation_energy(self):
        horizon = max(0, int(self.h2_delivery_reservation_horizon))
        reserved = np.zeros(self.agent_num, dtype=np.float32)
        if horizon <= 0 or not self.pending_h2_deliveries:
            return reserved
        for record in self.pending_h2_deliveries:
            buyer = int(record["buyer_id"])
            if buyer < 0 or buyer >= self.agent_num:
                continue
            hours_until_arrival = int(record["deliver_at"]) - int(self.t)
            if 1 <= hours_until_arrival <= horizon:
                reserved[buyer] += max(0.0, float(record["quantity"]))
        return (
            reserved * max(0.0, float(self.h2_delivery_reservation_ratio))
        ).astype(np.float32)

    def _terminal_asset_values(self):
        """Value only end-of-day assets that are physically available on site."""
        battery_price = float(self.cfg.get("terminal_battery_value_yuan_per_kwh", 0.0))
        h2_price = float(self.cfg.get("terminal_h2_value_yuan_per_kwh", 0.0))
        if battery_price <= 0.0:
            battery_price = float(self.tou_buy[0])
        if h2_price <= 0.0:
            h2_price = self.lambda_h2_buy
        battery_energy = float(
            np.sum(np.maximum(self.soc - self.cfg["soc_min"], 0.0) * self.bat_cap)
        )
        h2_energy = float(
            np.sum(np.maximum(self.h2_level - self.h2_min, 0.0) * self.cfg["LHV_H2"])
        )
        pending_h2_energy = float(
            sum(
                max(0.0, float(record["quantity"]))
                for record in self.pending_h2_deliveries
            )
        )
        battery_value = battery_energy * battery_price
        h2_value = h2_energy * h2_price
        return {
            "terminal_battery_asset_value": battery_value,
            "terminal_h2_asset_value": h2_value,
            # Orders have already been paid for, but hydrogen that has not arrived
            # cannot be salvaged or refunded at the episode boundary.
            "terminal_pending_h2_asset_value": 0.0,
            "terminal_undelivered_h2_energy": pending_h2_energy,
            "terminal_asset_value": battery_value + h2_value,
        }

    def step(self, actions):
        """推进一个时间步。

        Args:
            actions: shape=(num_agents, action_dim) 的连续动作,每维 ∈ [-1, 1]。

        Returns:
            [obs_list, reward_list, done_list, info_list]。
        """
        t = self.t
        dt = self.dt
        elec_buy_prices_t, elec_sell_prices_t = self._current_electric_agent_prices(t)
        h2_external_buy_price_t, h2_external_sell_price_t = (
            self._current_external_h2_prices()
        )
        h2_external_buy_prices_t, h2_external_sell_prices_t = (
            self._current_external_h2_price_vectors()
        )
        self.h2_external_buy_price = h2_external_buy_price_t
        self.h2_external_sell_price = h2_external_sell_price_t
        self.h2_external_buy_prices = h2_external_buy_prices_t
        self.h2_external_sell_prices = h2_external_sell_prices_t

        # 读取当前时刻外生量,单位均与配置保持一致:功率为 kW,热负荷为 kW_thermal。
        pv_now = self.profiles["pv"][:, t]
        wt_now = self.profiles["wt"][:, t]
        le_now = self.profiles["load_e"][:, t]
        lh_now = self.profiles["load_h"][:, t]

        # 防御性检查动作维度,避免策略网络维度和环境 action_dim 不一致时静默出错。
        raw_actions = np.asarray(actions, dtype=np.float32)
        if raw_actions.shape != (self.agent_num, self.action_dim):
            raise ValueError(
                f"Expected actions with shape {(self.agent_num, self.action_dim)}, "
                f"got {raw_actions.shape}"
            )
        raw_actions = np.clip(raw_actions, -1.0, 1.0)
        current_reg_actions = raw_actions[:, self.action_reg_indices]
        h2_route_actions = np.full(self.agent_num, -1.0, dtype=np.float32)
        if self.h2_route_action_enable and raw_actions.shape[1] >= 7:
            h2_route_actions = raw_actions[:, 6].astype(np.float32)

        p_el = np.zeros(self.agent_num, dtype=np.float32)
        p_bat = np.zeros(self.agent_num, dtype=np.float32)
        elec_bid_price = np.zeros(self.agent_num, dtype=np.float32)
        h2_bid_price = np.zeros(self.agent_num, dtype=np.float32)
        p_ht = np.zeros(self.agent_num, dtype=np.float32)
        h2_thermal_share = np.full(
            self.agent_num, self.h2_thermal_share_default, dtype=np.float32
        )

        # 动作反归一化 (6-d):
        # a0 -> 电解槽功率,       a1 -> 电池功率,
        # a2 -> 电力市场报价,     a3 -> 氢能市场报价,
        # a4 -> 储氢罐充放功率,   a5 -> 完整的前向 H2 CDA 买单量
        # a5 在订单阶段按 [0, qmax_i] 映射；卖方始终忽略 a5。
        for agent_id in range(self.agent_num):
            action = raw_actions[agent_id]
            p_el[agent_id] = ((action[0] + 1.0) / 2.0) * self.el_cap[agent_id]
            p_bat[agent_id] = action[1] * self.bat_power[agent_id]
            elec_bid_price[agent_id] = self._scale_price(
                action[2], elec_sell_prices_t[agent_id], elec_buy_prices_t[agent_id]
            )
            h2_bid_price[agent_id] = self._scale_price(
                action[3],
                self.h2_price_min, self.h2_price_max,
            )
            p_ht[agent_id] = action[4] * self.h2_tank_power[agent_id]

        # ------------------------------------------------------------------
        # 处理本步到货的 pending 氢交易。到货后 h2_level 才被更新,所以
        # 在下面的储氢罐限幅逻辑之前执行。
        # ------------------------------------------------------------------
        delivered_energy = np.zeros(self.agent_num, dtype=np.float32)
        delivery_overflow_energy = 0.0

        soc_min = self.cfg["soc_min"]
        soc_max = self.cfg["soc_max"]
        eta_c = self.cfg["bat_eff_c"]
        eta_d = self.cfg["bat_eff_d"]
        a2_soc_reserve_discharge_clip = np.zeros(self.agent_num, dtype=np.float32)
        a2_soc_reserve_agent_set = set(
            int(i)
            for i in self.a2_late_soc_reserve_agent_indices
            if 0 <= int(i) < self.agent_num
        )

        # 电池功率根据 SOC 可行区间二次限幅,然后更新 SOC。
        # 正功率为充电,负功率为放电。
        for agent_id in range(self.agent_num):
            max_charge = (
                (soc_max - self.soc[agent_id]) * self.bat_cap[agent_id] / (eta_c * dt)
            )
            max_discharge = (
                (self.soc[agent_id] - soc_min) * self.bat_cap[agent_id] * eta_d / dt
            )
            if p_bat[agent_id] < 0.0:
                local_electric_deficit_power = max(
                    0.0,
                    le_now[agent_id] + p_el[agent_id] - pv_now[agent_id] - wt_now[agent_id],
                )
                max_discharge = min(max_discharge, local_electric_deficit_power)
            if (
                self.a2_late_soc_reserve_enable
                and self.a2_late_soc_reserve_horizon > 0
                and t >= self.T - self.a2_late_soc_reserve_horizon
                and agent_id in a2_soc_reserve_agent_set
                and p_bat[agent_id] < 0.0
                and self.bat_cap[agent_id] > 0
            ):
                requested_discharge = max(0.0, -float(p_bat[agent_id]))
                allowed_before_reserve = min(requested_discharge, float(max_discharge))
                reserve_soc = max(soc_min, self.a2_late_soc_reserve_threshold)
                reserve_max_discharge = (
                    max(0.0, float(self.soc[agent_id] - reserve_soc))
                    * self.bat_cap[agent_id]
                    * eta_d
                    / dt
                )
                max_discharge = min(max_discharge, reserve_max_discharge)
                allowed_after_reserve = min(requested_discharge, float(max_discharge))
                a2_soc_reserve_discharge_clip[agent_id] = (
                    max(0.0, allowed_before_reserve - allowed_after_reserve) * dt
                )
            p_bat[agent_id] = np.clip(p_bat[agent_id], -max_discharge, max_charge)

            charge = max(0.0, p_bat[agent_id])
            discharge = max(0.0, -p_bat[agent_id])
            if self.bat_cap[agent_id] > 0:
                delta_soc = (
                    (eta_c * charge - discharge / eta_d) / self.bat_cap[agent_id] * dt
                )
                self.soc[agent_id] = np.clip(
                    self.soc[agent_id] + delta_soc, soc_min, soc_max
                )

        lhv = self.cfg["LHV_H2"]
        boiler_eff = self.cfg["boiler_eff"]
        e_h2_prod = np.zeros(self.agent_num, dtype=np.float32)

        # 电解槽消耗电功率 p_el,并按效率转换为氢能产出 e_h2_prod (kWh_H2)。
        for agent_id in range(self.agent_num):
            p_el[agent_id] = np.clip(p_el[agent_id], 0.0, self.el_cap[agent_id])
            if self.el_cap[agent_id] > 0:
                e_h2_prod[agent_id] = p_el[agent_id] * self.el_eff[agent_id] * dt

        e_h2_load = np.zeros(self.agent_num, dtype=np.float32)
        for agent_id in range(self.agent_num):
            e_thermal = lh_now[agent_id] * dt
            share = float(h2_thermal_share[agent_id])
            # 氢供热部分折算为 kWh_H2 入 net_h2_demand。
            e_h2_load[agent_id] = share * e_thermal / boiler_eff

        h2_delivery_reserved_energy = np.zeros(self.agent_num, dtype=np.float32)
        h2_delivery_reservation_charge_clip = np.zeros(self.agent_num, dtype=np.float32)
        h2_delivery_reservation_margin = np.zeros(self.agent_num, dtype=np.float32)
        if (
            self.h2_delivery_reservation_enable
            and self.h2_market_lag_enable
            and self.h2_delivery_lag > 0
        ):
            h2_delivery_reserved_energy = self._pending_h2_reservation_energy()

        # 储氢罐动作限幅 + 更新 h2_level (kg)。
        # p_ht > 0 充氢,p_ht < 0 放氢。
        for agent_id in range(self.agent_num):
            if self.h2_tank_cap[agent_id] <= 0:
                p_ht[agent_id] = 0.0
                continue
            max_charge = (
                (self.h2_max[agent_id] - self.h2_level[agent_id]) * lhv
                / (self.h2_eff_c * dt)
            )
            max_discharge = (
                (self.h2_level[agent_id] - self.h2_min[agent_id]) * lhv
                * self.h2_eff_d / dt
            )
            if p_ht[agent_id] < 0.0:
                local_h2_deficit_power = max(
                    0.0, e_h2_load[agent_id] - e_h2_prod[agent_id]
                ) / dt
                max_discharge = min(max_discharge, local_h2_deficit_power)
            if h2_delivery_reserved_energy[agent_id] > 0.0:
                base_clipped_charge = min(max(0.0, float(p_ht[agent_id])), max_charge)
                current_headroom_energy = max(
                    0.0,
                    float((self.h2_max[agent_id] - self.h2_level[agent_id]) * lhv),
                )
                available_charge_energy = max(
                    0.0,
                    current_headroom_energy
                    - float(h2_delivery_reserved_energy[agent_id]),
                )
                reserved_max_charge = available_charge_energy / (
                    self.h2_eff_c * dt
                )
                h2_delivery_reservation_charge_clip[agent_id] = max(
                    0.0,
                    base_clipped_charge - min(base_clipped_charge, reserved_max_charge),
                ) * self.h2_eff_c * dt
                max_charge = min(max_charge, reserved_max_charge)
            p_ht[agent_id] = np.clip(p_ht[agent_id], -max_discharge, max_charge)

            charge = max(0.0, p_ht[agent_id])
            discharge = max(0.0, -p_ht[agent_id])
            delta_h2_energy = (
                self.h2_eff_c * charge - discharge / self.h2_eff_d
            ) * dt
            self.h2_level[agent_id] = np.clip(
                self.h2_level[agent_id] + delta_h2_energy / lhv,
                self.h2_min[agent_id],
                self.h2_max[agent_id],
            )
            h2_delivery_reservation_margin[agent_id] = (
                (self.h2_max[agent_id] - self.h2_level[agent_id]) * lhv
                - h2_delivery_reserved_energy[agent_id]
            )

        net_electric_demand = np.zeros(self.agent_num, dtype=np.float32)
        net_h2_demand = np.zeros(self.agent_num, dtype=np.float32)

        # 计算电力净需求和氢能净需求。
        # 净需求为正=需买入,为负=有富余可卖出。
        for agent_id in range(self.agent_num):
            bat_charge = max(0.0, p_bat[agent_id])
            bat_discharge = max(0.0, -p_bat[agent_id])

            net_electric_demand[agent_id] = (
                le_now[agent_id]
                + p_el[agent_id]
                + bat_charge
                - pv_now[agent_id]
                - wt_now[agent_id]
                - bat_discharge
            )

            net_h2_demand[agent_id] = (
                e_h2_load[agent_id] + p_ht[agent_id] * dt - e_h2_prod[agent_id]
            )

        # AC power flow sees physical PCC net demand, not financial CDA flows.
        pcc_p_kw = net_electric_demand.astype(np.float64)
        pcc_q_kvar = (
            np.maximum(le_now, 0.0) * self.power_flow_reactive_per_active_load
        ).astype(np.float64)
        power_flow_pcc_p_kw = pcc_p_kw * self.power_flow_pcc_injection_scale
        power_flow_pcc_q_kvar = pcc_q_kvar * self.power_flow_pcc_injection_scale
        if self.power_flow is not None:
            voltage_diagnostics = self.power_flow.solve(
                power_flow_pcc_p_kw, power_flow_pcc_q_kvar
            )
        else:
            voltage_diagnostics = {
                "pf_converged": False,
                "voltages_pu": np.array([], dtype=np.float64),
                "pcc_voltages_pu": np.array([], dtype=np.float64),
                "voltage_cost": 0.0,
                "voltage_violation_area": 0.0,
                "voltage_max_violation": 0.0,
                "voltage_min_pu": float("nan"),
                "voltage_max_pu": float("nan"),
            }

        # ------------------------------------------------------------------
        # 电力市场 CDA:每步撮合,订单量 = |net_electric_demand| * dt (全量报)。
        # ------------------------------------------------------------------
        elec_orders = []
        elec_order_quantity = np.zeros(self.agent_num, dtype=np.float32)
        elec_order_side = [None] * self.agent_num

        if self.elec_internal_cda_enable or self.elec_p2p_enable:
            for agent_id in range(self.agent_num):
                elec_order_quantity[agent_id] = abs(net_electric_demand[agent_id]) * dt

                if net_electric_demand[agent_id] > EPS and elec_order_quantity[agent_id] > EPS:
                    elec_order_side[agent_id] = "buy"
                elif net_electric_demand[agent_id] < -EPS and elec_order_quantity[agent_id] > EPS:
                    elec_order_side[agent_id] = "sell"

                if elec_order_side[agent_id] is not None:
                    elec_orders.append({
                        "agent_id": agent_id,
                        "side": elec_order_side[agent_id],
                        "price": float(elec_bid_price[agent_id]),
                        "quantity": float(elec_order_quantity[agent_id]),
                    })

        if self.elec_p2p_enable:
            elec_result = run_bilateral_p2p_market(
                elec_orders,
                default_price=self.last_elec_clearing_price,
                agent_count=self.agent_num,
                agent_locations=self.elec_agent_bus_indices,
                distance_fee_coef=self.elec_p2p_distance_fee_coef,
                max_distance=self.elec_p2p_max_distance,
                price_rule=self.elec_p2p_price_rule,
            )
            elec_market_mechanism = "p2p_bilateral"
        else:
            elec_result = run_continuous_double_auction(
                elec_orders, default_price=self.last_elec_clearing_price
            )
            elec_market_mechanism = "cda"
        self.last_elec_clearing_price = elec_result["clearing_price"]
        elec_p2p_diag = summarize_pairwise_trades(
            elec_result["trades"] if (
                self.elec_p2p_enable or self.elec_p2p_diagnostics_enable
            ) else [],
            self.agent_num,
            agent_locations=self.elec_agent_bus_indices,
            distance_fee_coef=self.elec_p2p_distance_fee_coef,
        )
        elec_total_traded = float(
            sum(trade["quantity"] for trade in elec_result["trades"])
        )
        elec_cda_paid = np.zeros(self.agent_num, dtype=np.float32)
        elec_cda_received = np.zeros(self.agent_num, dtype=np.float32)
        for trade in elec_result["trades"]:
            trade_cost = float(trade["price"]) * float(trade["quantity"])
            elec_cda_paid[int(trade["buyer_id"])] += trade_cost
            elec_cda_received[int(trade["seller_id"])] += trade_cost

        # 电力市场成交后,剩余缺口/富余由外部电网平衡。
        p_grid = np.zeros(self.agent_num, dtype=np.float32)
        for agent_id in range(self.agent_num):
            elec_bought = elec_result["buy_matched"].get(agent_id, 0.0)
            elec_sold = elec_result["sell_matched"].get(agent_id, 0.0)
            p_grid[agent_id] = (
                net_electric_demand[agent_id]
                - elec_bought / dt
                + elec_sold / dt
            )

        # ------------------------------------------------------------------
        # 氢市场 CDA:可配置为每步撮合或只在协议时刻撮合。成交的"卖方"当步立刻出货(实际 net 已经
        # 包含 p_ht 消耗,对卖方无需额外扣减);"买方"物理货延迟 lag 小时到达,
        # 在此之前其 net_h2_demand 全量挂给外部市场(λ_h2 成本)。
        # ------------------------------------------------------------------
        is_agreement_step = (
            (not self.h2_market_schedule_enable)
            or (int(t) in self.h2_market_schedule)
        )
        h2_order_quantity = np.zeros(self.agent_num, dtype=np.float32)
        h2_order_quantity_raw = np.zeros(self.agent_num, dtype=np.float32)
        h2_action_requested_buy_quantity = np.zeros(self.agent_num, dtype=np.float32)
        h2_action_effective_buy_quantity = np.zeros(self.agent_num, dtype=np.float32)
        h2_order_source = ["none"] * self.agent_num
        h2_buy_future_headroom = np.zeros(self.agent_num, dtype=np.float32)
        h2_buy_clip_amount = np.zeros(self.agent_num, dtype=np.float32)
        h2_buy_horizon_clip_amount = np.zeros(self.agent_num, dtype=np.float32)
        h2_late_order_energy = np.zeros(self.agent_num, dtype=np.float32)
        h2_buyer_reservation_shortfall = np.zeros(self.agent_num, dtype=np.float32)
        h2_buyer_reservation_extra_order = np.zeros(self.agent_num, dtype=np.float32)
        h2_traffic_active = bool(
            self.h2_traffic_enable
            and self.h2_market_lag_enable
            and self.h2_delivery_lag > 0
        )
        h2_order_delivery_horizon = (
            self.h2_traffic_max_eta if h2_traffic_active else self.h2_delivery_lag
        )
        action_ordering_active = (
            self.h2_action_controlled_order_enable
            and self.h2_learnable_rolling_order_active
            and raw_actions.shape[1] >= 6
        )
        if action_ordering_active:
            h2_action_requested_buy_quantity = (
                ((raw_actions[:, 5] + 1.0) / 2.0) * self.h2_order_qmax
            ).astype(np.float32)
        # Deprecated compatibility alias: now means the full requested a5 buy.
        h2_learnable_rolling_order_extra = h2_action_requested_buy_quantity.copy()
        h2_order_side = [None] * self.agent_num
        h2_orders: list = []
        _, pending_h2_energy_agent_before_order = self._pending_h2_arrival_buckets()
        pending_adjusted_h2_headroom_before_order = self._pending_adjusted_h2_headroom(
            pending_h2_energy_agent_before_order
        )
        h2_buyer_reservation_agent_set = set(
            int(i)
            for i in self.h2_buyer_reservation_agent_indices
            if 0 <= int(i) < self.agent_num
        )

        if (self.h2_internal_cda_enable or self.h2_p2p_enable) and is_agreement_step:
            for agent_id in range(self.agent_num):
                net_demand = float(net_h2_demand[agent_id])
                if net_demand < 0.0:
                    raw_quantity = -net_demand
                    h2_order_source[agent_id] = "physical_surplus"
                elif action_ordering_active:
                    raw_quantity = float(
                        h2_action_requested_buy_quantity[agent_id]
                    )
                    if raw_quantity > EPS:
                        h2_order_source[agent_id] = "action_buy"
                else:
                    raw_quantity = max(0.0, net_demand)
                    if raw_quantity > EPS:
                        h2_order_source[agent_id] = "automatic_deficit"
                    if (
                        self.h2_buyer_reservation_demand_enable
                        and agent_id in h2_buyer_reservation_agent_set
                        and self.h2_tank_cap[agent_id] > 0
                    ):
                        target_energy = (
                            float(self.h2_buyer_reservation_target_ratios[agent_id])
                            * float(self.h2_tank_cap[agent_id])
                            * lhv
                        )
                        effective_inventory_energy = (
                            float(self.h2_level[agent_id]) * lhv
                            + float(pending_h2_energy_agent_before_order[agent_id])
                        )
                        shortfall = max(0.0, target_energy - effective_inventory_energy)
                        max_extra = max(
                            0.0,
                            float(self.h2_buyer_reservation_max_order_fraction)
                            * float(self.h2_tank_cap[agent_id])
                            * lhv,
                        )
                        extra_order = min(
                            shortfall * max(0.0, float(self.h2_buyer_reservation_demand_gain)),
                            max_extra,
                        )
                        h2_buyer_reservation_shortfall[agent_id] = shortfall
                        h2_buyer_reservation_extra_order[agent_id] = extra_order
                        raw_quantity += extra_order
                        if extra_order > EPS:
                            h2_order_source[agent_id] = "heuristic_reservation"
                h2_order_quantity_raw[agent_id] = raw_quantity
                h2_order_quantity[agent_id] = raw_quantity

                if net_demand >= 0.0 and h2_order_quantity[agent_id] > EPS:
                    h2_order_side[agent_id] = "buy"
                    delayed_delivery = (
                        self.h2_market_lag_enable and self.h2_delivery_lag > 0
                    )
                    deliverable_within_episode = (
                        not delayed_delivery
                        or int(t) + int(h2_order_delivery_horizon) < int(self.T)
                    )
                    if not deliverable_within_episode:
                        if self.h2_order_horizon_clip_mode == "pay_and_lose":
                            # 订单是真实承诺: 照常入市, 成交部分按 CDA 价付款,
                            # 越界交付永不到罐, 损失由买方承担。
                            h2_late_order_energy[agent_id] = raw_quantity
                        else:
                            h2_order_quantity[agent_id] = 0.0
                            h2_buy_clip_amount[agent_id] = raw_quantity
                            h2_buy_horizon_clip_amount[agent_id] = raw_quantity
                    elif self.h2_cap_aware_buy_enable:
                        if self.h2_market_lag_enable and self.h2_delivery_lag > 0:
                            deliverable_within_episode = (
                                int(t) + int(h2_order_delivery_horizon) < int(self.T)
                            )
                            if deliverable_within_episode:
                                future_headroom = max(
                                    0.0,
                                    float(pending_adjusted_h2_headroom_before_order[agent_id]),
                                )
                            else:
                                future_headroom = 0.0
                        else:
                            deliverable_within_episode = True
                            current_headroom = max(
                                0.0,
                                float((self.h2_max[agent_id] - self.h2_level[agent_id]) * lhv),
                            )
                            future_headroom = max(0.0, net_demand) + current_headroom
                        h2_buy_future_headroom[agent_id] = future_headroom
                        h2_order_quantity[agent_id] = min(raw_quantity, future_headroom)
                        h2_buy_clip_amount[agent_id] = max(
                            0.0,
                            raw_quantity - h2_order_quantity[agent_id],
                        )
                        if (
                            self.h2_market_lag_enable
                            and self.h2_delivery_lag > 0
                            and not deliverable_within_episode
                        ):
                            h2_buy_horizon_clip_amount[agent_id] = (
                                h2_buy_clip_amount[agent_id]
                            )
                    if h2_order_source[agent_id] == "action_buy":
                        h2_action_effective_buy_quantity[agent_id] = (
                            h2_order_quantity[agent_id]
                        )
                elif net_demand < 0.0 and h2_order_quantity[agent_id] > EPS:
                    h2_order_side[agent_id] = "sell"

                if h2_order_side[agent_id] is not None and h2_order_quantity[agent_id] > EPS:
                    h2_orders.append({
                        "agent_id": agent_id,
                        "side": h2_order_side[agent_id],
                        "price": float(h2_bid_price[agent_id]),
                        "quantity": float(h2_order_quantity[agent_id]),
                    })

        h2_agent_locations = (
            self.gas_agent_node_indices
            if self.gas_multinode_enable
            else np.arange(self.agent_num, dtype=np.float32)
        )
        if self.h2_p2p_enable:
            h2_result = run_bilateral_p2p_market(
                h2_orders,
                default_price=self.last_h2_clearing_price,
                agent_count=self.agent_num,
                agent_locations=h2_agent_locations,
                distance_fee_coef=self.h2_p2p_distance_fee_coef,
                max_distance=self.h2_p2p_max_distance,
                price_rule=self.h2_p2p_price_rule,
            )
            h2_market_mechanism = "p2p_bilateral"
        else:
            h2_result = run_continuous_double_auction(
                h2_orders, default_price=self.last_h2_clearing_price
            )
            h2_market_mechanism = "cda"
        self.last_h2_clearing_price = h2_result["clearing_price"]
        h2_p2p_diag = summarize_pairwise_trades(
            h2_result["trades"] if (
                self.h2_p2p_enable or self.h2_p2p_diagnostics_enable
            ) else [],
            self.agent_num,
            agent_locations=h2_agent_locations,
            distance_fee_coef=self.h2_p2p_distance_fee_coef,
        )
        h2_total_traded = float(
            sum(trade["quantity"] for trade in h2_result["trades"])
        )
        h2_buy_order_quantity_total = float(
            sum(
                float(h2_order_quantity[i])
                for i in range(self.agent_num)
                if h2_order_side[i] == "buy"
            )
        )
        h2_sell_order_quantity_total = float(
            sum(
                float(h2_order_quantity[i])
                for i in range(self.agent_num)
                if h2_order_side[i] == "sell"
            )
        )
        h2_buy_prices = [
            float(h2_bid_price[i])
            for i in range(self.agent_num)
            if h2_order_side[i] == "buy" and h2_order_quantity[i] > EPS
        ]
        h2_sell_prices = [
            float(h2_bid_price[i])
            for i in range(self.agent_num)
            if h2_order_side[i] == "sell" and h2_order_quantity[i] > EPS
        ]
        h2_best_buy_price = max(h2_buy_prices) if h2_buy_prices else 0.0
        h2_best_sell_price = min(h2_sell_prices) if h2_sell_prices else 0.0
        h2_bid_cross = bool(
            h2_buy_prices
            and h2_sell_prices
            and h2_best_buy_price + EPS >= h2_best_sell_price
        )
        h2_cross_matchable_quantity = (
            min(h2_buy_order_quantity_total, h2_sell_order_quantity_total)
            if h2_bid_cross
            else 0.0
        )

        # 氢市场平衡 (第六轮 P0-a 修复):
        # 老师电力市场逻辑: "中标走 CDA 价, 未中标走外部价".
        # - Seller (net<0): e_ext_sell = net + h2_sold
        #     卖方物理上当步即出货, net 富余 + h2_sold 趋近 0 (甚至转正).
        # - Buyer  (net>0): e_ext_buy  = net - h2_bought    <-- 新增扣减
        #     买方 CDA 中标部分按 CDA 成交价记账, 剩余未中标才走外部高价.
        #     物理交割仍 lag 4h 后入罐 (pending_h2_deliveries), 但账面当步结清.
        # - CDA 成交额 price * qty: buyer 付 c_h2 += cost, seller 收 c_h2 -= cost.
        #     shared reward 下 sum(paid) == sum(received), 净 0 转账, 但 info
        #     里可以看到各 agent 的 cda_paid / cda_received, 便于 debug.
        cda_paid = np.zeros(self.agent_num, dtype=np.float32)
        cda_received = np.zeros(self.agent_num, dtype=np.float32)
        for trade in h2_result["trades"]:
            trade_cost = float(trade["price"]) * float(trade["quantity"])
            cda_paid[int(trade["buyer_id"])] += trade_cost
            cda_received[int(trade["seller_id"])] += trade_cost

        # v2 计划性外购 (设计规格 1a): 未撮合剩余量按计划价即时付款、延迟交付,
        # 有路网时从 EXT 供应站节点发运, 与内部交易共享道路容量。
        h2_planned_external_order_energy = np.zeros(self.agent_num, dtype=np.float32)
        h2_planned_external_order_cost = 0.0
        planned_external_trades: list = []
        if self.h2_planned_external_order_enable:
            external_seller_id = (
                self.h2_transport_network.external_node_id
                if h2_traffic_active and self.h2_transport_network is not None
                else -1
            )
            for agent_id in range(self.agent_num):
                if h2_order_side[agent_id] != "buy":
                    continue
                matched = float(h2_result["buy_matched"].get(agent_id, 0.0))
                remainder = max(0.0, float(h2_order_quantity[agent_id]) - matched)
                if remainder <= EPS:
                    continue
                planned_price = float(h2_external_buy_prices_t[agent_id])
                h2_planned_external_order_energy[agent_id] = remainder
                h2_planned_external_order_cost += planned_price * remainder
                planned_external_trades.append({
                    "seller_id": int(external_seller_id),
                    "buyer_id": int(agent_id),
                    "quantity": float(remainder),
                    "price": planned_price,
                })

        h2_transport_shipments = []
        routable_trades = list(h2_result["trades"]) + planned_external_trades
        if h2_traffic_active and self.h2_transport_network is not None:
            h2_transport_shipments = self.h2_transport_network.assign_shipments(
                routable_trades,
                h2_route_actions,
                dispatch_t=int(t),
                transport_loss=self.h2_transport_loss,
            )
            for shipment in h2_transport_shipments:
                if float(shipment["net_quantity"]) > EPS:
                    self.pending_h2_deliveries.append(dict(shipment))
        elif self.h2_market_lag_enable and self.h2_delivery_lag > 0:
            for trade in routable_trades:
                delivered_qty = float(trade["quantity"]) * (
                    1.0 - np.clip(self.h2_transport_loss, 0.0, 1.0)
                )
                if delivered_qty > EPS:
                    self.pending_h2_deliveries.append({
                        "deliver_at": int(t) + int(self.h2_delivery_lag),
                        "buyer_id": int(trade["buyer_id"]),
                        "seller_id": int(trade["seller_id"]),
                        "quantity": delivered_qty,
                        "price": float(trade["price"]),
                    })
        self.last_h2_transport_shipments = [
            dict(shipment) for shipment in h2_transport_shipments
        ]

        e_h2_ext = np.zeros(self.agent_num, dtype=np.float32)
        immediate_h2_stored_energy = np.zeros(self.agent_num, dtype=np.float32)
        immediate_h2_overflow_energy = np.zeros(self.agent_num, dtype=np.float32)
        h2_no_lag_demand_offset = np.zeros(self.agent_num, dtype=np.float32)
        h2_no_lag_conservation_residual = np.zeros(
            self.agent_num, dtype=np.float32
        )
        lagged_delivery = self.h2_market_lag_enable and self.h2_delivery_lag > 0
        for agent_id in range(self.agent_num):
            net_demand = float(net_h2_demand[agent_id])
            h2_sold = float(h2_result["sell_matched"].get(agent_id, 0.0))
            h2_bought = float(h2_result["buy_matched"].get(agent_id, 0.0))
            if net_demand < 0.0:
                e_h2_ext[agent_id] = net_demand + h2_sold
            elif lagged_delivery:
                # A delayed trade cannot serve the current step's load.
                e_h2_ext[agent_id] = net_demand
            else:
                # Immediate H2 first serves current demand. Only the excess is
                # stored, and any tank-clipped overflow is never external resale.
                demand_offset = min(net_demand, h2_bought)
                excess_buy = max(0.0, h2_bought - demand_offset)
                accepted, overflow = self._store_h2_delivery(
                    agent_id, excess_buy
                )
                h2_no_lag_demand_offset[agent_id] = demand_offset
                immediate_h2_stored_energy[agent_id] = accepted
                immediate_h2_overflow_energy[agent_id] = overflow
                h2_no_lag_conservation_residual[agent_id] = (
                    h2_bought - demand_offset - accepted - overflow
                )
                delivered_energy[agent_id] += accepted
                e_h2_ext[agent_id] = max(0.0, net_demand - demand_offset)

        self.last_p_grid = p_grid.astype(np.float32)
        self.last_e_h2_ext = e_h2_ext.astype(np.float32)
        if self.h2_market_lag_enable and self.h2_delivery_lag > 0:
            delivered_energy, delivery_overflow_energy = self._deliver_pending_h2(
                current_t=int(t) + 1
            )

        # ------------------------------------------------------------------
        # 成本核算。
        # ------------------------------------------------------------------
        c_grid = 0.0
        for agent_id in range(self.agent_num):
            if p_grid[agent_id] > 0:
                c_grid += p_grid[agent_id] * elec_buy_prices_t[agent_id] * dt
            else:
                c_grid += p_grid[agent_id] * elec_sell_prices_t[agent_id] * dt

        # 外部氢成本: 买卖分开计价. e_h2_ext[i] > 0 = 向外部买, < 0 = 向外部卖.
        # delivery_overflow_energy > 0 = 罐溢出被动外卖 (视作向外部卖出).
        # 第六轮 P0-a: 未中标部分才走外部 (e_ext_buy / e_ext_sell); 中标部分按
        # CDA 成交价在 cda_transfer 里转账, shared reward 下净 0.
        e_ext_np = np.asarray(e_h2_ext, dtype=np.float32)
        e_ext_buy = float(np.sum(np.maximum(e_ext_np, 0.0)))   # 未中标 buyer 走外部高价
        e_ext_sell = float(np.sum(np.minimum(e_ext_np, 0.0)))  # 未中标 seller 走外部低价
        cda_transfer = float(np.sum(cda_paid) - np.sum(cda_received))  # shared reward 下 = 0
        # 应急采购 = 当小时未满足负荷的瞬时平衡外购, 按乘数计价 (设计规格 2c)。
        h2_emergency_buy_energy_agent = np.maximum(e_ext_np, 0.0)
        h2_emergency_buy_cost = float(
            np.sum(
                h2_external_buy_prices_t
                * self.h2_emergency_price_multiplier
                * h2_emergency_buy_energy_agent
            )
        )
        c_h2 = float(
            h2_emergency_buy_cost
            + h2_planned_external_order_cost
            + np.sum(h2_external_sell_prices_t * np.minimum(e_ext_np, 0.0))
            - h2_external_sell_price_t * float(delivery_overflow_energy)
        )
        gas_external_withdrawal = e_ext_buy
        gas_external_injection = max(0.0, -e_ext_sell) + float(delivery_overflow_energy)
        gas_external_withdrawal_agent = np.maximum(e_ext_np, 0.0)
        gas_external_injection_agent = np.maximum(-e_ext_np, 0.0)
        if delivery_overflow_energy > 0.0:
            gas_external_injection_agent += (
                float(delivery_overflow_energy) / max(self.agent_num, 1)
            )
        gas_pressure_before = float(self.gas_pressure)
        if self.gas_multinode_enable:
            gas_pressure_after = self._update_gas_network(
                gas_external_withdrawal_agent, gas_external_injection_agent
            )
        else:
            gas_pressure_after = self._update_gas_network(
                gas_external_withdrawal, gas_external_injection
            )
        base_cost = c_grid + c_h2
        total_market_traded = elec_total_traded + h2_total_traded
        market_bonus = 0.0
        h2_internal_trade_bonus = 0.0
        external_h2_dependency_penalty_term = 0.0
        if (
            self.h2_internal_trade_bonus_enable
            and self.h2_internal_trade_bonus_coef > 0.0
        ):
            h2_internal_trade_bonus = (
                self.h2_internal_trade_bonus_coef * h2_total_traded
            )
            market_bonus += h2_internal_trade_bonus
        if (
            self.external_h2_dependency_penalty_enable
            and self.external_h2_dependency_penalty_coef > 0.0
        ):
            external_h2_dependency_penalty_term = (
                self.external_h2_dependency_penalty_coef * e_ext_buy
            )

        penalty_value = 0.0
        soc_penalty_term = 0.0
        h2_penalty_term = 0.0
        low_inventory_penalty_term = 0.0
        terminal_h2_floor_penalty_term = 0.0
        terminal_h2_shortfall_value_term = 0.0
        terminal_h2_shortfall_energy = 0.0
        terminal_soc_floor_penalty_term = 0.0
        terminal_battery_salvage_value_term = 0.0
        stepwise_h2_floor_penalty_term = 0.0
        action_magnitude_penalty_term = 0.0
        action_delta_penalty_term = 0.0
        if self.penalty_enable:
            weight = 1.0 / max(self.T - t, 1)
            soc_sq = 0.0
            for agent_id in range(self.agent_num):
                if self.bat_cap[agent_id] > 0:
                    target = float(self.soc_penalty_targets[agent_id])
                    dev = abs(float(self.soc[agent_id] - target))
                    soc_sq += max(0.0, dev - self.penalty_deadband) ** 2
            h2_sq = 0.0
            for agent_id in range(self.agent_num):
                if self.h2_tank_cap[agent_id] > 0:
                    ratio = float(self.h2_level[agent_id] / self.h2_tank_cap[agent_id])
                    target = float(self.h2_penalty_targets[agent_id])
                    deadband = self.penalty_deadband
                    if agent_id in self.consumer_h2_deadband_agent_indices:
                        deadband = self.consumer_h2_deadband
                    dev = abs(ratio - target)
                    h2_sq += max(0.0, dev - deadband) ** 2
            soc_penalty_term = self.soc_penalty_coef * weight * soc_sq
            h2_penalty_term = self.h2_penalty_coef * weight * h2_sq
            penalty_value = soc_penalty_term + h2_penalty_term

        if self.low_inventory_penalty_enable and self.low_inventory_penalty_coef > 0.0:
            low_sq = 0.0
            for agent_id in range(self.agent_num):
                if self.bat_cap[agent_id] > 0:
                    low_sq += max(0.0, self.soc_low_threshold - self.soc[agent_id]) ** 2
                if self.h2_tank_cap[agent_id] > 0:
                    ratio = self.h2_level[agent_id] / self.h2_tank_cap[agent_id]
                    low_sq += max(0.0, self.h2_low_threshold - ratio) ** 2
            low_inventory_penalty_term = self.low_inventory_penalty_coef * low_sq
            penalty_value += low_inventory_penalty_term

        if (
            self.stepwise_h2_floor_penalty_enable
            and self.stepwise_h2_floor_penalty_coef > 0.0
        ):
            floor_sq = 0.0
            urgency = 1.0
            if self.T > 1:
                urgency += self.stepwise_h2_floor_urgency_gain * t / (self.T - 1)
            for agent_id in range(self.agent_num):
                if self.h2_tank_cap[agent_id] > 0:
                    ratio = self.h2_level[agent_id] / self.h2_tank_cap[agent_id]
                    threshold = float(self.stepwise_h2_floor_thresholds[agent_id])
                    weight = float(self.stepwise_h2_floor_weights[agent_id])
                    floor_sq += weight * max(0.0, threshold - ratio) ** 2
            stepwise_h2_floor_penalty_term = (
                self.stepwise_h2_floor_penalty_coef * urgency * floor_sq
            )
            penalty_value += stepwise_h2_floor_penalty_term

        if (
            self.terminal_h2_floor_penalty_enable
            and self.terminal_h2_floor_penalty_coef > 0.0
            and t >= self.T - 1
        ):
            floor_sq = 0.0
            for agent_id in self.terminal_h2_floor_agent_indices:
                if agent_id < 0 or agent_id >= self.agent_num:
                    continue
                if self.h2_tank_cap[agent_id] > 0:
                    ratio = self.h2_level[agent_id] / self.h2_tank_cap[agent_id]
                    floor_sq += max(0.0, self.terminal_h2_floor_threshold - ratio) ** 2
            terminal_h2_floor_penalty_term = (
                self.terminal_h2_floor_penalty_coef * floor_sq
            )
            penalty_value += terminal_h2_floor_penalty_term

        if (
            self.terminal_h2_shortfall_value_enable
            and t >= self.T - 1
        ):
            coef = self.terminal_h2_shortfall_value_coef
            if coef <= 0.0:
                coef = self.lambda_h2_buy
            for agent_id in self.terminal_h2_shortfall_value_agent_indices:
                if agent_id < 0 or agent_id >= self.agent_num:
                    continue
                if self.h2_tank_cap[agent_id] > 0:
                    ratio = float(self.h2_level[agent_id] / self.h2_tank_cap[agent_id])
                    target = float(self.terminal_h2_shortfall_value_targets[agent_id])
                    terminal_h2_shortfall_energy += (
                        max(0.0, target - ratio)
                        * float(self.h2_tank_cap[agent_id])
                        * lhv
                    )
            terminal_h2_shortfall_value_term = (
                coef * terminal_h2_shortfall_energy
            )
            penalty_value += terminal_h2_shortfall_value_term

        if (
            self.terminal_soc_floor_penalty_enable
            and self.terminal_soc_floor_penalty_coef > 0.0
            and t >= self.T - 1
        ):
            floor_sq = 0.0
            for agent_id in self.terminal_soc_floor_agent_indices:
                if agent_id < 0 or agent_id >= self.agent_num:
                    continue
                if self.bat_cap[agent_id] > 0:
                    floor_sq += max(
                        0.0,
                        self.terminal_soc_floor_threshold - float(self.soc[agent_id]),
                    ) ** 2
            terminal_soc_floor_penalty_term = (
                self.terminal_soc_floor_penalty_coef * floor_sq
            )
            penalty_value += terminal_soc_floor_penalty_term

        if (
            self.terminal_battery_salvage_enable
            and self.terminal_battery_salvage_value_coef > 0.0
            and t >= self.T - 1
        ):
            salvage_value = 0.0
            for agent_id in self.terminal_battery_salvage_agent_indices:
                if agent_id < 0 or agent_id >= self.agent_num:
                    continue
                if self.bat_cap[agent_id] > 0:
                    salvage_unit = max(0.0, float(self.soc[agent_id] - soc_min))
                    if self.terminal_battery_salvage_capacity_scaled_enable:
                        salvage_unit *= float(self.bat_cap[agent_id]) / max(
                            EPS,
                            self.terminal_battery_salvage_reference_capacity,
                        )
                    salvage_value += salvage_unit
            terminal_battery_salvage_value_term = (
                self.terminal_battery_salvage_value_coef * salvage_value
            )
            penalty_value -= terminal_battery_salvage_value_term

        if (
            self.action_reg_enable
            and self.action_reg_indices.size > 0
            and (
                self.action_magnitude_penalty_coef > 0.0
                or self.action_delta_penalty_coef > 0.0
            )
        ):
            action_magnitude_penalty_term = (
                self.action_magnitude_penalty_coef
                * float(np.sum(current_reg_actions ** 2))
            )
            if self.last_reg_actions is not None:
                delta_actions = current_reg_actions - self.last_reg_actions
                action_delta_penalty_term = (
                    self.action_delta_penalty_coef
                    * float(np.sum(delta_actions ** 2))
                )
            penalty_value += action_magnitude_penalty_term + action_delta_penalty_term

        # Historical penalties stay diagnostic-only. The optional terminal
        # H2 settlement enters reward only on the terminal step.
        terminal_h2_settlement_cost_term = (
            terminal_h2_shortfall_value_term
            if self.terminal_h2_settlement_in_reward_enable
            else 0.0
        )
        terminal_asset_values = self._terminal_asset_values()
        terminal_settlement_cost = 0.0
        if self.terminal_economic_settlement_enable and t >= self.T - 1:
            terminal_settlement_cost = (
                self._initial_terminal_asset_value
                - float(terminal_asset_values["terminal_asset_value"])
            )
        total_cost = (
            base_cost
            + external_h2_dependency_penalty_term
            + terminal_h2_settlement_cost_term
            + terminal_settlement_cost
        )
        self.episode_total_cost += float(total_cost)

        self.t += 1
        self.last_reg_actions = current_reg_actions.copy()
        done = self.t >= self.T
        reward_emitted = self.reward_emission_mode == "dense" or done
        if self.reward_emission_mode == "terminal_total":
            reward = (
                -self.episode_total_cost / self.reward_scale
                if done
                else 0.0
            )
        else:
            reward = -total_cost / self.reward_scale
        day_boundary = (
            self.day_boundary_info_enable
            and self.day_boundary_interval > 0
            and (self.t % self.day_boundary_interval == 0 or done)
        )

        terminal_value = 0.0
        if done and self.pending_h2_deliveries:
            # 终端时处理仍未送达的 pending 氢:买方未收到货,系统层视为违约
            # (买方已付款但没拿到货)。这部分作为 penalty-free 记录留作分析,
            # 不再调整 c_h2,避免双重计账。
            pass

        pending_h2_arrival_buckets, pending_h2_energy_agent = (
            self._pending_h2_arrival_buckets()
        )
        pending_adjusted_h2_headroom = self._pending_adjusted_h2_headroom(
            pending_h2_energy_agent
        )
        pending_h2_energy_agent_norm = np.clip(
            pending_h2_energy_agent / self.pending_scale,
            0.0,
            1.0,
        )
        pending_adjusted_h2_headroom_norm = np.clip(
            pending_adjusted_h2_headroom / self.pending_scale,
            -1.0,
            1.0,
        )
        pending_h2_energy_total = float(np.sum(pending_h2_energy_agent))
        h2_route_rank = np.full(self.agent_num, -1, dtype=np.int64)
        for shipment in h2_transport_shipments:
            buyer = int(shipment["buyer_id"])
            if 0 <= buyer < self.agent_num:
                h2_route_rank[buyer] = int(shipment["route_rank"])
        if self.h2_transport_network is not None:
            traffic_route_features = [
                self.h2_transport_network.route_features(agent_id, t).tolist()
                for agent_id in range(self.agent_num)
            ]
            traffic_edge_utilization = {
                f"{source}->{target}": float(value)
                for (source, target), value in (
                    self.h2_transport_network.last_edge_utilization.items()
                )
            }
            traffic_background_utilization = {
                f"{source}->{target}": float(value)
                for (source, target), value in (
                    self.h2_transport_network.last_background_utilization.items()
                )
            }
        else:
            traffic_route_features = []
            traffic_edge_utilization = {}
            traffic_background_utilization = {}
        traffic_etas = [int(shipment["eta"]) for shipment in h2_transport_shipments]
        traffic_mean_eta = float(np.mean(traffic_etas)) if traffic_etas else 0.0
        traffic_delayed_quantity = float(sum(
            float(shipment["gross_quantity"])
            for shipment in h2_transport_shipments
            if int(shipment["eta"]) > self.h2_traffic_min_eta
        ))

        # step 后生成下一时刻观测;最后一个时间步会用 T-1 的外生曲线索引保护。
        obs_list = self._get_obs()
        reward_list = [[reward] for _ in range(self.agent_num)]
        done_list = [done] * self.agent_num

        elec_buy_cost = float(sum(elec_result["buy_cost"].values()))
        elec_sell_revenue = float(sum(elec_result["sell_revenue"].values()))
        h2_buy_cost = float(sum(h2_result["buy_cost"].values()))
        h2_sell_revenue = float(sum(h2_result["sell_revenue"].values()))

        info = {
            "episode_step": int(self.t),
            "day_index": int(t // self.day_boundary_interval),
            "hour_of_day": int(t % self.day_boundary_interval),
            "day_boundary": bool(day_boundary),
            "C_grid": float(c_grid),
            "C_h2": float(c_h2),
            "C_grid_ext": float(c_grid),
            "C_h2_ext": float(c_h2),
            "base_cost": float(base_cost),
            "economic_cost": float(base_cost),
            "total_cost": float(total_cost),
            "step_total_cost": float(total_cost),
            "episode_total_cost": float(self.episode_total_cost),
            "voltage_cost": float(voltage_diagnostics["voltage_cost"]),
            "voltage_violation_area": float(voltage_diagnostics["voltage_violation_area"]),
            "voltage_max_violation": float(voltage_diagnostics["voltage_max_violation"]),
            "voltage_min_pu": _json_safe_voltage(voltage_diagnostics["voltage_min_pu"]),
            "voltage_max_pu": _json_safe_voltage(voltage_diagnostics["voltage_max_pu"]),
            "voltages_pu": _json_safe_voltage(voltage_diagnostics["voltages_pu"]),
            "pcc_voltages_pu": _json_safe_voltage(voltage_diagnostics["pcc_voltages_pu"]),
            "pcc_p_kw": pcc_p_kw.astype(float).tolist(),
            "pcc_q_kvar": pcc_q_kvar.astype(float).tolist(),
            "power_flow_pcc_p_kw": power_flow_pcc_p_kw.astype(float).tolist(),
            "power_flow_pcc_q_kvar": power_flow_pcc_q_kvar.astype(float).tolist(),
            "power_flow_pcc_injection_scale": float(self.power_flow_pcc_injection_scale),
            "pf_converged": bool(voltage_diagnostics["pf_converged"]),
            "reward_emitted": bool(reward_emitted),
            "penalty_inv": penalty_value,
            "penalty_total": penalty_value,
            "penalty_soc": soc_penalty_term,
            "penalty_h2": h2_penalty_term,
            "penalty_low_inventory": low_inventory_penalty_term,
            "penalty_terminal_h2_floor": terminal_h2_floor_penalty_term,
            "penalty_terminal_h2_shortfall_value": terminal_h2_shortfall_value_term,
            "terminal_h2_shortfall_kg": float(
                terminal_h2_shortfall_energy / max(lhv, EPS)
            ),
            "terminal_h2_settlement_cost": float(
                terminal_h2_settlement_cost_term
            ),
            "terminal_economic_settlement_enable": bool(self.terminal_economic_settlement_enable),
            "terminal_settlement_cost": float(terminal_settlement_cost),
            "initial_terminal_asset_value": float(self._initial_terminal_asset_value),
            "terminal_battery_asset_value": float(terminal_asset_values["terminal_battery_asset_value"]),
            "terminal_h2_asset_value": float(terminal_asset_values["terminal_h2_asset_value"]),
            "terminal_pending_h2_asset_value": float(terminal_asset_values["terminal_pending_h2_asset_value"]),
            "terminal_undelivered_h2_energy": float(terminal_asset_values["terminal_undelivered_h2_energy"]),
            "terminal_asset_value": float(terminal_asset_values["terminal_asset_value"]),
            "effective_external_h2_cost_yuan_per_kg": float(
                (
                    h2_external_buy_price_t
                    + (
                        self.external_h2_dependency_penalty_coef
                        if self.external_h2_dependency_penalty_enable
                        else 0.0
                    )
                )
                * lhv
            ),
            "penalty_terminal_soc_floor": terminal_soc_floor_penalty_term,
            "penalty_stepwise_h2_floor": stepwise_h2_floor_penalty_term,
            "penalty_action": action_magnitude_penalty_term + action_delta_penalty_term,
            "penalty_action_magnitude": action_magnitude_penalty_term,
            "penalty_action_delta": action_delta_penalty_term,
            "terminal_battery_salvage_value": terminal_battery_salvage_value_term,
            "elec_clearing_price": self.last_elec_clearing_price,
            "elec_price_mode": self.elec_price_mode,
            "elec_lmp_status_code": float(self.elec_lmp_status_code[t]),
            "elec_lmp_line_loading_max": float(self.elec_line_loading_max[t]),
            "elec_lmp_congestion_count": float(self.elec_lmp_congestion_count[t]),
            "elec_lmp_slack_import": float(self.elec_lmp_slack_import[t]),
            "elec_lmp_price_spread": float(self.elec_lmp_price_spread[t]),
            "h2_clearing_price": self.last_h2_clearing_price,
            "h2_external_buy_price": h2_external_buy_price_t,
            "h2_external_sell_price": h2_external_sell_price_t,
            "elec_agent_bus_indices": self.elec_agent_bus_indices.tolist(),
            "elec_node_buy_prices": self.elec_node_buy_prices[t].tolist(),
            "elec_node_sell_prices": self.elec_node_sell_prices[t].tolist(),
            "elec_agent_buy_prices": elec_buy_prices_t.tolist(),
            "elec_agent_sell_prices": elec_sell_prices_t.tolist(),
            "elec_line_loading": self.elec_line_loading[t].tolist(),
            "h2_external_buy_prices": self.h2_external_buy_prices.tolist(),
            "h2_external_sell_prices": self.h2_external_sell_prices.tolist(),
            "elec_internal_cda_enable": bool(self.elec_internal_cda_enable),
            "h2_internal_cda_enable": bool(self.h2_internal_cda_enable),
            "elec_p2p_enable": bool(self.elec_p2p_enable),
            "h2_p2p_enable": bool(self.h2_p2p_enable),
            "elec_p2p_diagnostics_enable": bool(self.elec_p2p_diagnostics_enable),
            "h2_p2p_diagnostics_enable": bool(self.h2_p2p_diagnostics_enable),
            "elec_market_mechanism": elec_market_mechanism,
            "h2_market_mechanism": h2_market_mechanism,
            "gas_pressure": gas_pressure_after,
            "gas_pressure_prev": gas_pressure_before,
            "gas_node_pressure": self.gas_node_pressure.tolist(),
            "gas_node_pressure_prev": self.gas_node_pressure_prev.tolist(),
            "gas_external_withdrawal": gas_external_withdrawal,
            "gas_external_injection": gas_external_injection,
            "gas_external_withdrawal_agent": gas_external_withdrawal_agent.tolist(),
            "gas_external_injection_agent": gas_external_injection_agent.tolist(),
            "elec_market_traded": elec_total_traded,
            "h2_market_traded": h2_total_traded,
            "h2_buy_order_quantity_total": h2_buy_order_quantity_total,
            "h2_sell_order_quantity_total": h2_sell_order_quantity_total,
            "h2_best_buy_price": h2_best_buy_price,
            "h2_best_sell_price": h2_best_sell_price,
            "h2_bid_cross": h2_bid_cross,
            "h2_cross_matchable_quantity": h2_cross_matchable_quantity,
            "cda_total_traded": total_market_traded,
            "elec_market_buy_cost": elec_buy_cost,
            "elec_market_sell_revenue": elec_sell_revenue,
            "h2_market_buy_cost": h2_buy_cost,
            "h2_market_sell_revenue": h2_sell_revenue,
            "elec_p2p_pair_count": int(elec_p2p_diag["pair_count"]),
            "h2_p2p_pair_count": int(h2_p2p_diag["pair_count"]),
            "elec_p2p_mean_trade_distance": float(elec_p2p_diag["mean_trade_distance"]),
            "h2_p2p_mean_trade_distance": float(h2_p2p_diag["mean_trade_distance"]),
            # 第六轮新增: CDA 成交价按 trade 累加到 agent 的付/收账面, 用于 debug
            "elec_cda_paid": elec_cda_paid.tolist(),
            "elec_cda_received": elec_cda_received.tolist(),
            "h2_cda_paid": cda_paid.tolist(),
            "h2_cda_received": cda_received.tolist(),
            "h2_cda_transfer": cda_transfer,
            "cda_buy_cost": elec_buy_cost + h2_buy_cost,
            "cda_sell_revenue": elec_sell_revenue + h2_sell_revenue,
            "market_bonus": market_bonus,
            "cda_bonus": market_bonus,
            "h2_internal_trade_bonus": h2_internal_trade_bonus,
            "external_h2_dependency_penalty": external_h2_dependency_penalty_term,
            "external_h2_dependency_penalty_coef": self.external_h2_dependency_penalty_coef,
            "terminal_value": terminal_value,
            "p_grid": self.last_p_grid.tolist(),
            "soc": self.soc.tolist(),
            "h2_level": self.h2_level.tolist(),
            "h2_level_ratio": (
                self.h2_level / self.h2_tank_cap_safe
            ).astype(float).tolist(),
            "e_h2_ext": self.last_e_h2_ext.tolist(),
            "h2_emergency_buy_energy": h2_emergency_buy_energy_agent.astype(float).tolist(),
            "h2_emergency_buy_cost": h2_emergency_buy_cost,
            "h2_emergency_price_multiplier": float(self.h2_emergency_price_multiplier),
            "h2_planned_external_order_energy": h2_planned_external_order_energy.astype(float).tolist(),
            "h2_planned_external_order_cost": h2_planned_external_order_cost,
            "p_el": p_el.tolist(),
            "p_bat": p_bat.tolist(),
            "a2_soc_reserve_discharge_clip": a2_soc_reserve_discharge_clip.tolist(),
            "p_ht": p_ht.tolist(),
            "elec_bid_price": elec_bid_price.tolist(),
            "h2_bid_price": h2_bid_price.tolist(),
            "net_electric_demand": net_electric_demand.tolist(),
            "net_h2_demand": net_h2_demand.tolist(),
            "h2_order_qmax": self.h2_order_qmax.tolist(),
            "h2_action_order_qmax": self.h2_order_qmax.tolist(),
            "h2_action_requested_buy_quantity": h2_action_requested_buy_quantity.tolist(),
            "h2_action_effective_buy_quantity": h2_action_effective_buy_quantity.tolist(),
            "h2_traffic_enable": bool(self.h2_traffic_enable),
            "h2_route_action_enable": bool(self.h2_route_action_enable),
            "h2_route_action": h2_route_actions.tolist(),
            "h2_route_rank": h2_route_rank.tolist(),
            "h2_traffic_min_eta": int(self.h2_traffic_min_eta),
            "h2_traffic_max_eta": int(self.h2_traffic_max_eta),
            "h2_traffic_route_features": traffic_route_features,
            "h2_traffic_edge_utilization": traffic_edge_utilization,
            "h2_traffic_background_utilization": traffic_background_utilization,
            "h2_traffic_mean_eta": traffic_mean_eta,
            "h2_external_min_eta_normalized": [
                self._external_min_eta_normalized(agent_id, t)
                for agent_id in range(self.agent_num)
            ],
            "h2_traffic_delayed_quantity": traffic_delayed_quantity,
            "h2_transport_shipments": [
                dict(shipment) for shipment in h2_transport_shipments
            ],
            "h2_order_source": list(h2_order_source),
            "h2_order_quantity_raw": h2_order_quantity_raw.tolist(),
            "h2_order_quantity": h2_order_quantity.tolist(),
            # Deprecated compatibility alias for requested action quantity.
            "h2_learnable_rolling_order_extra": h2_learnable_rolling_order_extra.tolist(),
            "h2_buyer_reservation_shortfall": h2_buyer_reservation_shortfall.tolist(),
            "h2_buyer_reservation_extra_order": h2_buyer_reservation_extra_order.tolist(),
            "h2_buy_future_headroom": h2_buy_future_headroom.tolist(),
            "h2_buy_clip_amount": h2_buy_clip_amount.tolist(),
            "h2_buy_horizon_clip_amount": h2_buy_horizon_clip_amount.tolist(),
            "h2_late_order_energy": h2_late_order_energy.tolist(),
            "h2_order_horizon_clip_mode": self.h2_order_horizon_clip_mode,
            "h2_delivery_reserved_energy": h2_delivery_reserved_energy.tolist(),
            "h2_delivery_reservation_margin": h2_delivery_reservation_margin.tolist(),
            "h2_delivery_reservation_charge_clip": h2_delivery_reservation_charge_clip.tolist(),
            "e_h2_load": e_h2_load.tolist(),
            "e_h2_prod": e_h2_prod.tolist(),
            "h2_thermal_share": h2_thermal_share.tolist(),
            "h2_no_lag_demand_offset": h2_no_lag_demand_offset.tolist(),
            "h2_no_lag_conservation_residual": (
                h2_no_lag_conservation_residual.tolist()
            ),
            "h2_immediate_stored_energy": immediate_h2_stored_energy.tolist(),
            "h2_immediate_overflow_energy": immediate_h2_overflow_energy.tolist(),
            "delivered_h2_energy": delivered_energy.tolist(),
            "delivery_overflow_energy": float(delivery_overflow_energy),
            "h2_delivery_received": delivered_energy.tolist(),
            "h2_delivery_overflow": float(delivery_overflow_energy),
            "h2_is_agreement_step": bool(is_agreement_step),
            "h2_pending_count": len(self.pending_h2_deliveries),
            "pending_h2_energy_total": pending_h2_energy_total,
            "pending_h2_energy_agent": pending_h2_energy_agent.tolist(),
            "pending_h2_arrival_buckets": pending_h2_arrival_buckets.tolist(),
            "pending_h2_total": pending_h2_energy_agent.tolist(),
            "pending_h2_total_norm": pending_h2_energy_agent_norm.tolist(),
            "pending_h2_by_eta": pending_h2_arrival_buckets.tolist(),
            "pending_adjusted_h2_headroom": pending_adjusted_h2_headroom.tolist(),
            "pending_adjusted_h2_headroom_norm": pending_adjusted_h2_headroom_norm.tolist(),
            "elec_market_trades": list(elec_result["trades"]),
            "h2_market_trades": list(h2_result["trades"]),
            "elec_market_agent_results": self._summarize_agent_results(elec_result),
            "h2_market_agent_results": self._summarize_agent_results(h2_result),
            "elec_p2p_pair_summary": list(elec_p2p_diag["pair_summary"]),
            "h2_p2p_pair_summary": list(h2_p2p_diag["pair_summary"]),
            "elec_p2p_pair_quantity_matrix": list(elec_p2p_diag["pair_quantity_matrix"]),
            "h2_p2p_pair_quantity_matrix": list(h2_p2p_diag["pair_quantity_matrix"]),
            "elec_p2p_pair_value_matrix": list(elec_p2p_diag["pair_value_matrix"]),
            "h2_p2p_pair_value_matrix": list(h2_p2p_diag["pair_value_matrix"]),
            "elec_p2p_pair_distance_matrix": list(elec_p2p_diag["pair_distance_matrix"]),
            "h2_p2p_pair_distance_matrix": list(h2_p2p_diag["pair_distance_matrix"]),
            "elec_open_buy_orders": list(elec_result["open_buy_orders"]),
            "elec_open_sell_orders": list(elec_result["open_sell_orders"]),
            "h2_open_buy_orders": list(h2_result["open_buy_orders"]),
            "h2_open_sell_orders": list(h2_result["open_sell_orders"]),
        }
        info_list = [dict(info) for _ in range(self.agent_num)]

        return [obs_list, reward_list, done_list, info_list]

    def _normalize_price(self, price, lower, upper):
        """把市场出清价映射到 [0, 1],作为观测的一部分。"""
        if upper <= lower:
            return 0.5
        return np.clip((price - lower) / (upper - lower), 0.0, 1.0)

    def _get_obs(self):
        """构造每个智能体的局部观测向量。

        观测包含本地出力/负荷、储能状态、两个市场上一时刻出清价、
        时间编码,以及上一时刻与外部电/氢市场的交互量。
        """
        t = min(self.t, self.T - 1)

        pv_now = self.profiles["pv"][:, t]
        wt_now = self.profiles["wt"][:, t]
        le_now = self.profiles["load_e"][:, t]
        lh_now = self.profiles["load_h"][:, t]

        elec_clearing_norm = self._normalize_price(
            self.last_elec_clearing_price, self.elec_price_min, self.elec_price_max
        )
        h2_clearing_norm = self._normalize_price(
            self.last_h2_clearing_price, self.h2_price_min, self.h2_price_max
        )

        hour_of_day = t % self.day_boundary_interval
        sin_t = np.sin(2 * np.pi * hour_of_day / self.day_boundary_interval)
        cos_t = np.cos(2 * np.pi * hour_of_day / self.day_boundary_interval)
        pending_h2_arrival_buckets, pending_h2_energy_agent = (
            self._pending_h2_arrival_buckets()
        )
        pending_adjusted_h2_headroom = self._pending_adjusted_h2_headroom(
            pending_h2_energy_agent
        )
        obs_list = []
        for agent_id in range(self.agent_num):
            if self.gas_multinode_enable:
                node_id = int(self.gas_agent_node_indices[agent_id])
                gas_price_buy_norm = self._normalize_price(
                    self.h2_external_buy_prices[agent_id],
                    self.gas_price_min,
                    self.gas_price_max,
                )
                gas_pressure_norm = self._normalize_price(
                    self.gas_node_pressure[node_id],
                    self.gas_pressure_min,
                    self.gas_pressure_max,
                )
            else:
                gas_price_buy_norm = self._normalize_price(
                    self.h2_external_buy_price,
                    self.gas_price_min,
                    self.gas_price_max,
                )
                gas_pressure_norm = self._normalize_price(
                    self.gas_pressure,
                    self.gas_pressure_min,
                    self.gas_pressure_max,
                )
            # 外部交互量可能为正也可能为负,因此归一化后保留 [-1, 1] 符号信息。
            p_grid_norm = np.clip(
                self.last_p_grid[agent_id] / self.p_grid_scale[agent_id],
                -1.0,
                1.0,
            )
            e_h2_ext_norm = np.clip(
                self.last_e_h2_ext[agent_id] / self.h2_exchange_scale[agent_id],
                -1.0,
                1.0,
            )
            h2_ratio = self.h2_level[agent_id] / self.h2_tank_cap_safe[agent_id]
            if self.elec_price_mode == "tou":
                elec_buy_norm = self.elec_agent_buy_prices[t, agent_id] / max(
                    float(np.max(self.tou_buy)), 1.0
                )
            else:
                elec_buy_norm = self._normalize_price(
                    self.elec_agent_buy_prices[t, agent_id],
                    self.elec_lmp_price_min,
                    self.elec_lmp_price_max,
                )
            obs_values = [
                pv_now[agent_id] / self.pv_cap_safe[agent_id],
                wt_now[agent_id] / self.wt_cap_safe[agent_id],
                le_now[agent_id] / self.load_e_peak_safe[agent_id],
                lh_now[agent_id] / self.load_h_peak_safe[agent_id],
                self.soc[agent_id],
                h2_ratio,
                elec_clearing_norm,
                h2_clearing_norm,
                sin_t,
                cos_t,
                elec_buy_norm,
                p_grid_norm,
                e_h2_ext_norm,
            ]
            if self.gas_price_obs_enable:
                obs_values.append(gas_price_buy_norm)
            if self.gas_pressure_obs_enable:
                obs_values.append(gas_pressure_norm)
            if self.h2_pending_obs_enable:
                for bucket_id in range(self.h2_pending_obs_horizon):
                    obs_values.append(np.clip(
                        pending_h2_arrival_buckets[agent_id, bucket_id]
                        / self.pending_scale[agent_id],
                        0.0,
                        1.0,
                    ))
                if self.h2_pending_summary_obs_enable:
                    obs_values.append(np.clip(
                        pending_h2_energy_agent[agent_id]
                        / self.pending_scale[agent_id],
                        0.0,
                        1.0,
                    ))
                    obs_values.append(np.clip(
                        pending_adjusted_h2_headroom[agent_id]
                        / self.pending_scale[agent_id],
                        -1.0,
                        1.0,
                    ))
            if self.h2_traffic_enable and self.h2_transport_network is not None:
                obs_values.extend(
                    self.h2_transport_network.route_features(agent_id, t).tolist()
                )
            if self.h2_day_ahead_forecast_enable:
                for horizon in self.h2_day_ahead_forecast_horizons:
                    stop = min(self.T, t + horizon + 1)
                    future_heat = self.profiles["load_h"][agent_id, t + 1:stop]
                    forecast_energy = float(np.sum(future_heat) * self.dt)
                    forecast_scale = max(
                        float(self.load_h_peak_safe[agent_id]) * self.dt * horizon,
                        EPS,
                    )
                    normalized_heat = np.clip(forecast_energy / forecast_scale, 0.0, 1.0)
                    pending_energy = float(np.sum(
                        pending_h2_arrival_buckets[agent_id, :min(horizon, self.h2_pending_obs_horizon)]
                    ))
                    available_proxy = (
                        float(h2_ratio) * forecast_scale + pending_energy
                    )
                    deficit = max(0.0, forecast_energy - available_proxy)
                    obs_values.extend((normalized_heat, np.clip(deficit / forecast_scale, 0.0, 1.0)))
            if self.h2_local_supply_facts_enable:
                obs_values.extend(self._supply_intent_facts(
                    agent_id, t, pending_h2_arrival_buckets
                ))
            obs = np.array(obs_values, dtype=np.float32)
            obs = np.nan_to_num(obs, nan=0.0, posinf=1.0, neginf=0.0)
            obs_list.append(obs)

        return obs_list

    def close(self):
        pass

    def render(self, mode="rgb_array"):
        return None
