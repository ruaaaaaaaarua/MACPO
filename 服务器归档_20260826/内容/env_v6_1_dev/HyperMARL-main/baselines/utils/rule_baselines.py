"""Deterministic physical and diagnostic microgrid rule policies."""

from __future__ import annotations

import numpy as np


EPS = 1e-8


def _action_width(config):
    return 7 if bool(config.get("h2_route_action_enable", False)) else 6


def _config_array(config, key, agent_count):
    values = np.asarray(config[key], dtype=np.float32)
    if values.shape != (agent_count,):
        raise ValueError(f"config {key!r} must have shape {(agent_count,)}, got {values.shape}")
    return values


def _current_physics(obs, config):
    obs = np.asarray(obs, dtype=np.float32)
    if obs.ndim != 2 or obs.shape[1] < 4:
        raise ValueError("microgrid observations must have shape (agents, >=4)")
    count = obs.shape[0]
    pv = obs[:, 0] * _config_array(config, "pv_cap", count)
    wt = obs[:, 1] * _config_array(config, "wt_cap", count)
    load_e = obs[:, 2] * _config_array(config, "load_e_peak", count)
    load_h = obs[:, 3] * _config_array(config, "load_h_peak", count)
    return pv, wt, load_e, load_h


def _projected_deficit(pv, wt, load_e, load_h, config):
    count = len(pv)
    el_cap = _config_array(config, "el_cap", count)
    el_eff = _config_array(config, "el_eff", count)
    dt = float(config["dt"])
    boiler_eff = max(float(config["boiler_eff"]), EPS)
    p_el = np.minimum(np.maximum(pv + wt - load_e, 0.0), el_cap).astype(np.float32)
    h2_load = np.asarray(load_h, dtype=np.float32) * dt / boiler_eff
    h2_production = p_el * el_eff * dt
    projected_net = (h2_load - h2_production).astype(np.float32)
    deficit = np.maximum(projected_net, 0.0).astype(np.float32)
    return p_el, projected_net, deficit


def _inverse_nonnegative_action(quantity, maximum):
    maximum = np.asarray(maximum, dtype=np.float32)
    safe = np.maximum(maximum, EPS)
    clipped = np.clip(np.asarray(quantity, dtype=np.float32), 0.0, maximum)
    action = 2.0 * clipped / safe - 1.0
    return np.where(maximum > EPS, action, -1.0).astype(np.float32)


def _side_aware_bid(net_demand):
    net_demand = np.asarray(net_demand, dtype=np.float32)
    return np.where(
        net_demand > EPS,
        1.0,
        np.where(net_demand < -EPS, -1.0, 0.0),
    ).astype(np.float32)


def physical_idle(obs, context):
    """Keep physical controls idle and submit no H2 forward buy order."""
    count = np.asarray(obs).shape[0]
    action = np.zeros((count, _action_width(context["config"])), dtype=np.float32)
    action[:, 0] = -1.0
    action[:, 5] = -1.0
    if action.shape[1] >= 7:
        action[:, 6] = -1.0
    return action


physical_idle.privileged_diagnostic = False


def current_deficit_rule(obs, context):
    """Use only local current observations to cover the current H2 deficit."""
    config = context["config"]
    pv, wt, load_e, load_h = _current_physics(obs, config)
    p_el, projected_h2_net, deficit = _projected_deficit(
        pv, wt, load_e, load_h, config
    )
    count = len(pv)
    action = np.zeros((count, _action_width(config)), dtype=np.float32)
    action[:, 0] = _inverse_nonnegative_action(
        p_el, _config_array(config, "el_cap", count)
    )
    # Battery and tank actions intentionally remain at zero.
    electric_net = load_e + p_el - pv - wt
    action[:, 2] = _side_aware_bid(electric_net)
    action[:, 3] = _side_aware_bid(projected_h2_net)
    qmax = (
        _config_array(config, "load_h_peak", count)
        / max(float(config["boiler_eff"]), EPS)
        * float(config["dt"])
    )
    action[:, 5] = _inverse_nonnegative_action(deficit, qmax)
    if action.shape[1] >= 7:
        action[:, 6] = -1.0
    return action


current_deficit_rule.privileged_diagnostic = False


def privileged_t4_rule(obs, context):
    """Keep current controls exact, but set a5 from the exact t+4 profile."""
    action = current_deficit_rule(obs, context).copy()
    config = context["config"]
    future_t = int(context["episode_step"]) + 4
    horizon = int(config["episode_length"])
    if future_t >= horizon:
        action[:, 5] = -1.0
        return action

    profiles = context["profiles"]
    pv = np.asarray(profiles["pv"], dtype=np.float32)[:, future_t]
    wt = np.asarray(profiles["wt"], dtype=np.float32)[:, future_t]
    load_e = np.asarray(profiles["load_e"], dtype=np.float32)[:, future_t]
    load_h = np.asarray(profiles["load_h"], dtype=np.float32)[:, future_t]
    _, _, deficit = _projected_deficit(pv, wt, load_e, load_h, config)
    count = len(pv)
    qmax = (
        _config_array(config, "load_h_peak", count)
        / max(float(config["boiler_eff"]), EPS)
        * float(config["dt"])
    )
    action[:, 5] = _inverse_nonnegative_action(deficit, qmax)
    return action


privileged_t4_rule.privileged_diagnostic = True


def _delivery_horizon_hours(config):
    """Expected hours until an order placed now arrives (mean ETA or lag)."""
    if bool(config.get("h2_traffic_enable", False)):
        eta_min = int(
            config.get("h2_traffic_eta_min", config.get("h2_traffic_min_eta", 4))
        )
        eta_max = int(
            config.get("h2_traffic_eta_max", config.get("h2_traffic_max_eta", 6))
        )
        return max(1, int(round((eta_min + eta_max) / 2.0)))
    return max(1, int(config.get("h2_delivery_lag", 4)))


def _order_qmax(config, count):
    return (
        _config_array(config, "load_h_peak", count)
        / max(float(config["boiler_eff"]), EPS)
        * float(config["dt"])
        * float(config.get("h2_action_order_max_peak_hours", 1.0))
    )


def make_base_stock_rule(
    safety_hours: float = 2.0,
    target_mult: float = 1.0,
    privileged_forecast: bool = False,
):
    """Base-stock (order-up-to) H2 ordering rule (设计规格 3a/3b).

    每小时把「库存位置 = 罐内库存 + 自有在途」补到目标水平:
        target = target_mult * 未来 (交付期+safety_hours) 小时的净缺口预测
    非特权版用当前缺口的持续性外推做预测; 特权版直接读真值 profile
    (仅作诊断, 不参与排名)。在途量由规则自己的订单台账估计, 不依赖
    观测归一化解码, 也因此对撮合/外购渠道拆分保持不可知。
    """

    state = {"episode_step": -1, "ledger": []}

    def rule(obs, context):
        config = context["config"]
        step = int(context["episode_step"])
        # 计步归零或回绕 => 新 episode, 清空自有订单台账。
        if step == 0 or step <= state["episode_step"]:
            state["ledger"] = []
        state["episode_step"] = step

        pv, wt, load_e, load_h = _current_physics(obs, config)
        count = len(pv)
        p_el, projected_h2_net, current_deficit = _projected_deficit(
            pv, wt, load_e, load_h, config
        )
        action = np.zeros((count, _action_width(config)), dtype=np.float32)
        action[:, 0] = _inverse_nonnegative_action(
            p_el, _config_array(config, "el_cap", count)
        )
        electric_net = load_e + p_el - pv - wt
        action[:, 2] = _side_aware_bid(electric_net)
        action[:, 3] = _side_aware_bid(projected_h2_net)
        # 库存必须被使用才有价值: 满额请求放氢服务当前负荷。环境把放氢
        # 钳制到本地缺口与罐内存量, 不存在过放风险。
        action[:, 4] = -1.0
        if action.shape[1] >= 7:
            action[:, 6] = -1.0

        horizon = _delivery_horizon_hours(config)
        window = float(horizon) + float(safety_hours)

        if privileged_forecast:
            profiles = context["profiles"]
            episode_length = int(config["episode_length"])
            forecast = np.zeros(count, dtype=np.float32)
            for offset in range(1, int(round(window)) + 1):
                future_t = step + offset
                if future_t >= episode_length:
                    break
                f_pv = np.asarray(profiles["pv"], dtype=np.float32)[:, future_t]
                f_wt = np.asarray(profiles["wt"], dtype=np.float32)[:, future_t]
                f_le = np.asarray(profiles["load_e"], dtype=np.float32)[:, future_t]
                f_lh = np.asarray(profiles["load_h"], dtype=np.float32)[:, future_t]
                _, _, future_deficit = _projected_deficit(
                    f_pv, f_wt, f_le, f_lh, config
                )
                forecast += future_deficit
        else:
            # 持续性预测: 以当前净缺口速率外推整个窗口。
            forecast = current_deficit * window

        obs_arr = np.asarray(obs, dtype=np.float32)
        tank_cap = _config_array(config, "h2_tank_cap", count)
        lhv = float(config.get("LHV_H2", 33.33))
        stock_energy = np.clip(obs_arr[:, 5], 0.0, 1.0) * tank_cap * lhv
        # 目标库存位置不超过可储上限, 避免必然溢出的订购。
        storable_energy = tank_cap * lhv * float(
            config.get("h2_tank_max_ratio", 0.9)
        )
        target = np.minimum(float(target_mult) * forecast, storable_energy)

        pending_energy = np.zeros(count, dtype=np.float32)
        remaining_ledger = []
        for dispatch_step, quantities in state["ledger"]:
            if step - dispatch_step < horizon:
                pending_energy += quantities
                remaining_ledger.append((dispatch_step, quantities))
        state["ledger"] = remaining_ledger

        order = np.maximum(0.0, target - stock_energy - pending_energy)
        qmax = _order_qmax(config, count)
        executed = np.clip(order, 0.0, qmax).astype(np.float32)
        action[:, 5] = _inverse_nonnegative_action(executed, qmax)
        if np.any(executed > EPS):
            state["ledger"].append((step, executed.copy()))
        return action

    rule.privileged_diagnostic = bool(privileged_forecast)
    rule.parameters = {
        "safety_hours": float(safety_hours),
        "target_mult": float(target_mult),
        "privileged_forecast": bool(privileged_forecast),
    }
    return rule


base_stock_rule = make_base_stock_rule()
base_stock_rule.__name__ = "base_stock_rule"
base_stock_privileged = make_base_stock_rule(privileged_forecast=True)
base_stock_privileged.__name__ = "base_stock_privileged"


RULE_BASELINES = {
    "physical_idle": physical_idle,
    "current_deficit_rule": current_deficit_rule,
    "privileged_t4_rule": privileged_t4_rule,
    "base_stock_rule": base_stock_rule,
    "base_stock_privileged": base_stock_privileged,
}
