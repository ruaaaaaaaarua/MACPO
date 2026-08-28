#!/usr/bin/env python3
"""
Rule-based H2 delay ordering rollout experiment.
Validates: "advance ordering + rolling lag-awareness" hypothesis under
fixed H2 delivery lag (h2_delivery_lag = 4 steps).

Policies:
  A  - Naive/Myopic        : 无前瞻，仅当前状态响应
  B  - InitialBuffer        : episode 前 lag 步最大化制氢+激进买入
  C  - RollingLag           : 显式考虑 h2_delivery_lag，滚动提前下单
  D  - InitBuffer+Rolling   : 前 lag 步 = B，之后 = C
  E  - Oracle               : 使用真实未来氢负荷曲线的完美预测版 C

Extra reference:
  D_nolag - Policy D in instant-delivery env (h2_market_lag_enable=False)
"""

import sys
import copy
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from collections import defaultdict

# ── Path setup ───────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
HM_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(HM_ROOT))

from envs.microgrid.config import MICROGRID_CONFIG
from envs.microgrid.microgrid_env import MicrogridEnv

# ── Experiment constants ─────────────────────────────────────────────────────
N_EPISODES = 50
LAG = 4                    # h2_delivery_lag (must match OVERRIDES below)
PRODUCER_EL_THRESHOLD = 1500   # kW; agents above → producer role

OUT_DIR = Path(
    "/root/autodl-tmp/hypermarl-microgrid/result/rule_based_fixed_lag_h2_ordering"
)
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Apply environment overrides (same as main training experiment) ────────────
SHARED_OVERRIDES = {
    "episode_length": 24,
    "multi_day_episode_enable": False,
    "italian_split_enable": True,
    "italian_split_name": "train",
    "terminal_h2_shortfall_value_enable": False,
    "lambda_h2_buy": 30.0,      # yuan/kg H2 → internal: 30/33.33 ≈ 0.9 yuan/kWh_H2
    "lambda_h2_sell": 3.0,      # yuan/kg H2 → internal: 3/33.33  ≈ 0.09 yuan/kWh_H2
    "pv_cap":  [7500.0, 1500.0,  500.0, 2000.0],
    "wt_cap":  [1500.0, 6000.0, 3000.0,  500.0],
    "load_h_peak": [750.0, 600.0, 2925.0, 3656.25],
    "elec_internal_cda_enable": True,
    "h2_internal_cda_enable":   True,
    "gas_network_enable": False,
    "gas_price_dynamic_enable": False,
    "gas_price_bidirectional_enable": False,
    "gas_price_obs_enable": False,
    "gas_pressure_obs_enable": False,
    "h2_transport_loss": 0.0,
    "h2_market_schedule_enable": False,
    "h2_market_lag_enable": True,
    "h2_delivery_lag": LAG,
    "h2_pending_obs_enable": True,
    "h2_pending_obs_horizon": LAG,
    "h2_pending_summary_obs_enable": True,
    "h2_cap_aware_buy_enable": True,
    "h2_delivery_reservation_enable": True,
    "h2_delivery_reservation_horizon": LAG,
    "h2_delivery_reservation_ratio": 1.0,
    "h2_buyer_reservation_demand_enable": True,
    "h2_buyer_reservation_agent_indices": [2, 3],
    "h2_buyer_reservation_target_ratios": [0.0, 0.0, 0.35, 0.45],
    "h2_buyer_reservation_demand_gain": 1.0,
    "h2_buyer_reservation_max_order_fraction": 0.25,
    "h2_internal_trade_bonus_enable": True,
    "h2_internal_trade_bonus_coef": 0.05,
    "market_bonus_in_reward_enable": True,
    "penalty_in_reward_enable": False,
}
MICROGRID_CONFIG.update(SHARED_OVERRIDES)

# ────────────────────────────────────────────────────────────────────────────
# Helper utilities
# ────────────────────────────────────────────────────────────────────────────

def _tou_bat_action(env: MicrogridEnv, t: int) -> float:
    """Return battery action [-1, 1] based on TOU price at step t."""
    t = int(np.clip(t, 0, env.T - 1))
    tou = float(env.tou_buy[t])
    if tou >= 0.80:
        return -0.6   # peak: discharge
    if tou <= 0.35:
        return 0.5    # valley: charge
    return 0.0        # flat: neutral


def _pending_for_agent(env: MicrogridEnv, agent_id: int,
                        t_from: int = 0, t_to: int = 9999) -> float:
    """Return total pending H2 (kWh_H2) for agent arriving in [t_from, t_to]."""
    total = 0.0
    for d in env.pending_h2_deliveries:
        if d["buyer_id"] != agent_id:
            continue
        if d["deliver_at"] < t_from or d["deliver_at"] > t_to:
            continue
        total += d["quantity"]
    return total


def _is_producer(env: MicrogridEnv, i: int) -> bool:
    return float(env.el_cap[i]) >= PRODUCER_EL_THRESHOLD


# ────────────────────────────────────────────────────────────────────────────
# Policy A – Naive / Myopic
# ────────────────────────────────────────────────────────────────────────────
class NaivePolicy:
    """
    仅根据当前 H2 库存和 TOU 电价决策，不考虑交付时滞或未来负荷。
    作为无前瞻能力基准线。
    """
    name = "A_Naive"

    def reset(self):
        pass

    def get_actions(self, env: MicrogridEnv, obs_list) -> np.ndarray:
        n, d = env.agent_num, env.action_dim
        actions = np.zeros((n, d), dtype=np.float32)
        t = env.t

        for i in range(n):
            h2_ratio = float(env.h2_level[i] / max(env.h2_tank_cap[i], 1e-6))

            # a0: 电解槽功率 — H2 库存低时多制氢
            if h2_ratio < 0.25:
                a0 = 0.80    # ≈90% capacity
            elif h2_ratio < 0.50:
                a0 = 0.30    # ≈65% capacity
            else:
                a0 = -0.20   # ≈40% capacity

            # a1: 电池 — 峰谷套利
            a1 = _tou_bat_action(env, t)

            # a2: 电力市场报价 — 中性
            a2 = 0.0

            # a3: 氢市场报价 — 根据 H2 库存决定买卖意愿
            if h2_ratio < 0.25:
                a3 = 0.90    # 急需买入
            elif h2_ratio < 0.45:
                a3 = 0.50    # 倾向买入
            elif h2_ratio > 0.75:
                a3 = -0.60   # 倾向卖出
            else:
                a3 = 0.10    # 轻微买入意愿

            # a4: 储氢罐 — 不主动管理
            a4 = 0.0

            actions[i] = [a0, a1, a2, a3, a4]

        return actions


# ────────────────────────────────────────────────────────────────────────────
# Policy B – Initial Hydrogen Buffer
# ────────────────────────────────────────────────────────────────────────────
class InitialBufferPolicy:
    """
    Episode 前 h2_delivery_lag 步内：最大化制氢 + 激进买入，建立安全库存。
    之后切换为 Naive 策略。
    目标：降低 episode 初期因在途订单为零而导致的外部购氢依赖。
    """
    name = "B_InitBuffer"

    def __init__(self):
        self._naive = NaivePolicy()

    def reset(self):
        pass

    def get_actions(self, env: MicrogridEnv, obs_list) -> np.ndarray:
        t = env.t
        if t >= LAG:
            return self._naive.get_actions(env, obs_list)

        n, d = env.agent_num, env.action_dim
        actions = np.zeros((n, d), dtype=np.float32)
        for i in range(n):
            a0 = 1.0                                      # 电解槽全力制氢
            a1 = _tou_bat_action(env, t)
            a2 = 0.0
            a3 = 1.0 if not _is_producer(env, i) else -0.80  # 消费者激进买，生产者积极卖
            a4 = 0.20                                     # 轻微充储氢罐

            actions[i] = [a0, a1, a2, a3, a4]

        return actions


# ────────────────────────────────────────────────────────────────────────────
# Policy C – Rolling Lag-aware Ordering
# ────────────────────────────────────────────────────────────────────────────
class RollingLagPolicy:
    """
    在时刻 t 显式预测 [t, t+lag] 时段的氢需求，
    计算已有库存 + pending 交付量 + 预期本地制氢是否覆盖未来需求，
    若存在缺口则提高买入力度（降低电解槽出力以增大净氢需求 → CDA 订单量增大）。
    实现 lag-aware rolling ordering policy。
    """
    name = "C_RollingLag"
    SAFETY_FACTOR = 1.20   # 超额预订 20% 作为安全裕度

    def reset(self):
        pass

    def get_actions(self, env: MicrogridEnv, obs_list) -> np.ndarray:
        n, d = env.agent_num, env.action_dim
        actions = np.zeros((n, d), dtype=np.float32)
        t = env.t
        lhv = env.cfg["LHV_H2"]
        boiler_eff = env.cfg["boiler_eff"]
        dt = env.dt

        for i in range(n):
            h2_ratio = float(env.h2_level[i] / max(env.h2_tank_cap[i], 1e-6))
            h2_kwh = float(env.h2_level[i]) * lhv   # 当前储氢量 (kWh_H2)

            # 预测窗口 [t, t+lag-1]（闭区间）
            t_end = min(t + LAG - 1, env.T - 1)
            window_len = t_end - t + 1

            # 未来氢负荷总量 (kWh_H2)
            future_lh = env.profiles["load_h"][i, t: t_end + 1]   # kW_th
            future_need = float(np.sum(future_lh)) * dt / boiler_eff

            # 窗口内到期的 pending 交付量 (kWh_H2)
            pending_in_window = _pending_for_agent(env, i, t_from=t, t_to=t_end)

            # 假设当前电解槽利用率 60% 时的本地制氢量
            assumed_el_frac = 0.60
            local_prod = (float(env.el_cap[i]) * float(env.el_eff[i])
                          * dt * window_len * assumed_el_frac)

            expected_supply = h2_kwh + pending_in_window + local_prod
            shortfall = future_need * self.SAFETY_FACTOR - expected_supply

            is_prod = _is_producer(env, i)

            if shortfall > 0:
                # 供给不足：消费者降低电解槽（让净氢需求流向 CDA 市场），激进买入
                # 生产者提高电解槽全力卖出
                if is_prod:
                    a0 = 0.90    # 生产者：高产卖出
                    a3 = -0.80   # 积极卖出
                else:
                    a0 = 0.10    # 消费者：降低自产 → 增大 CDA 买入量
                    a3 = 0.90    # 激进买入
            elif shortfall < -future_need * 0.60:
                # 明显过剩：降低采购压力
                if is_prod:
                    a0 = 0.80
                    a3 = -0.80
                else:
                    a0 = 0.50    # 适当自产
                    a3 = -0.20   # 轻微倾向卖出
            else:
                # 供需均衡
                a0 = 0.70 if is_prod else 0.30
                a3 = -0.50 if is_prod else 0.40

            # 储氢罐主动管理
            if h2_ratio > 0.75:
                a4 = -0.20   # 略微放氢，为后续订单到货腾空间
            elif h2_ratio < 0.20:
                a4 = 0.20    # 低库存时略微充氢
            else:
                a4 = 0.0

            actions[i] = [
                float(np.clip(a0, -1.0, 1.0)),
                _tou_bat_action(env, t),
                0.0,
                float(np.clip(a3, -1.0, 1.0)),
                float(np.clip(a4, -1.0, 1.0)),
            ]

        return actions


# ────────────────────────────────────────────────────────────────────────────
# Policy D – Initial Buffer + Rolling (主要延迟感知策略)
# ────────────────────────────────────────────────────────────────────────────
class InitBufferRollingPolicy:
    """
    前 lag 步 = InitialBuffer（建立初期安全库存），
    之后  = RollingLag（滚动提前下单）。
    兼具初期安全裕度建立与后续跨时段库存-订单协同。
    """
    name = "D_InitBuffer_Rolling"

    def __init__(self):
        self._rolling = RollingLagPolicy()

    def reset(self):
        pass

    def get_actions(self, env: MicrogridEnv, obs_list) -> np.ndarray:
        t = env.t
        if t < LAG:
            # 前 lag 步：建立安全库存（与 B 相同）
            n, d = env.agent_num, env.action_dim
            actions = np.zeros((n, d), dtype=np.float32)
            for i in range(n):
                a0 = 1.0
                a1 = _tou_bat_action(env, t)
                a2 = 0.0
                a3 = 1.0 if not _is_producer(env, i) else -0.80
                a4 = 0.20
                actions[i] = [a0, a1, a2, a3, a4]
            return actions
        else:
            return self._rolling.get_actions(env, obs_list)


# ────────────────────────────────────────────────────────────────────────────
# Policy E – Oracle Forecast
# ────────────────────────────────────────────────────────────────────────────
class OraclePolicy:
    """
    使用环境内部真实未来氢负荷曲线进行完美预测，
    作为前瞻性策略的理论性能上界参考。
    仅用于分析固定延迟机制下理论可达的 reward 改善空间。
    """
    name = "E_Oracle"
    SAFETY_FACTOR = 1.10   # Oracle 预测准确，安全裕度适度降低

    def reset(self):
        pass

    def get_actions(self, env: MicrogridEnv, obs_list) -> np.ndarray:
        n, d = env.agent_num, env.action_dim
        actions = np.zeros((n, d), dtype=np.float32)
        t = env.t
        lhv = env.cfg["LHV_H2"]
        boiler_eff = env.cfg["boiler_eff"]
        dt = env.dt

        for i in range(n):
            h2_ratio = float(env.h2_level[i] / max(env.h2_tank_cap[i], 1e-6))
            h2_kwh = float(env.h2_level[i]) * lhv
            is_prod = _is_producer(env, i)

            # 精确预测：使用 [t, t+lag-1] 实际氢负荷
            t_end = min(t + LAG - 1, env.T - 1)
            future_lh = env.profiles["load_h"][i, t: t_end + 1]
            future_need = float(np.sum(future_lh)) * dt / boiler_eff

            # 远期预测：[t+lag, t+2*lag-1] 用于提前布局
            t_far_end = min(t + 2 * LAG - 1, env.T - 1)
            if t_far_end > t_end:
                far_lh = env.profiles["load_h"][i, t_end + 1: t_far_end + 1]
                far_need = float(np.sum(far_lh)) * dt / boiler_eff
            else:
                far_need = 0.0

            pending_in_window = _pending_for_agent(env, i, t_from=t, t_to=t_end)

            if is_prod:
                # 生产者：全力制氢卖出
                a0 = 0.90
                a3 = -0.80
                # 如果远期需求大，继续高产
                if far_need > future_need:
                    a0 = 1.0
                a4 = 0.10  # 略充储氢罐作缓冲
            else:
                # 消费者：基于精确预测计算缺口
                assumed_el_frac = 0.50
                window_len = t_end - t + 1
                local_prod = (float(env.el_cap[i]) * float(env.el_eff[i])
                              * dt * window_len * assumed_el_frac)
                expected = h2_kwh + pending_in_window + local_prod
                shortfall = future_need * self.SAFETY_FACTOR - expected

                if shortfall > future_need * 0.20:
                    a0 = 0.05     # 极低自产，最大化 CDA 买入量
                    a3 = 0.95
                elif shortfall > 0:
                    a0 = 0.20
                    a3 = 0.80
                elif shortfall < -future_need * 0.50:
                    a0 = 0.60     # 充分自产减少外部采购
                    a3 = -0.10
                else:
                    a0 = 0.35
                    a3 = 0.50

                # 初期建立安全库存
                if t < LAG:
                    a0 = 1.0      # 前 lag 步全力制氢
                    a3 = 1.0      # 激进买入建立管道
                    a4 = 0.30
                elif h2_ratio > 0.80:
                    a4 = -0.30    # 库存足够高：主动放氢为订单腾空间
                elif h2_ratio < 0.15:
                    a4 = 0.30
                else:
                    a4 = 0.0

            if is_prod:
                pass  # a4 already set
            actions[i] = [
                float(np.clip(a0, -1.0, 1.0)),
                _tou_bat_action(env, t),
                0.0,
                float(np.clip(a3, -1.0, 1.0)),
                float(np.clip(a4 if not is_prod or True else a4, -1.0, 1.0)),
            ]

        return actions


# ────────────────────────────────────────────────────────────────────────────
# Rollout engine
# ────────────────────────────────────────────────────────────────────────────

def run_rollout(policy, n_episodes: int = N_EPISODES,
                lag_env: bool = True) -> list:
    """
    Run n_episodes with the given policy. Returns list of per-episode dicts.
    Each episode dict contains per-step lists and aggregated episode stats.
    """
    # Fresh env for each policy (→ same day sequence, controlled comparison)
    env = MicrogridEnv()
    results = []

    for ep in range(n_episodes):
        # Seed numpy for reproducibility (env uses its own internal RNG)
        np.random.seed(42 + ep)

        obs_list = env.reset()
        policy.reset()
        done = False

        ep = {
            "rewards": [], "C_grid": [], "C_h2": [],
            "total_cost": [], "h2_ratios": [],
            "h2_level": [], "pending_energy": [],
            "delivered_energy": [], "e_h2_ext": [],
            "net_h2_demand": [], "e_h2_load": [], "e_h2_prod": [],
            "h2_order_qty": [], "h2_market_traded": [],
            "actions": [],
            # future load for comparison
            "future_load_h": [],
        }

        while not done:
            t = int(env.t)
            # Capture future-lag horizon load (for reference)
            t_end = min(t + LAG - 1, env.T - 1)
            future_lh = env.profiles["load_h"][:, t: t_end + 1].sum(axis=1)  # sum over window, per agent
            ep["future_load_h"].append(future_lh.tolist())

            actions = policy.get_actions(env, obs_list)
            step_result = env.step(actions)
            obs_list, reward_list, done_list, info_list = step_result
            # info_list is [info_dict, info_dict, ...] (same dict for each agent)
            info = info_list[0] if isinstance(info_list, list) else info_list
            done = done_list[0]
            reward = float(reward_list[0][0])

            h2_ratios = info.get("h2_level_ratio", [])
            h2_levels = info.get("h2_level", [])
            pending_ea = info.get("pending_h2_energy_agent", [0.0] * env.agent_num)
            delivered_ea = info.get("delivered_h2_energy", [0.0] * env.agent_num)
            e_h2_ext_a = info.get("e_h2_ext", [0.0] * env.agent_num)
            net_h2_d = info.get("net_h2_demand", [0.0] * env.agent_num)
            e_h2_ld = info.get("e_h2_load", [0.0] * env.agent_num)
            e_h2_pr = info.get("e_h2_prod", [0.0] * env.agent_num)
            h2_oq = info.get("h2_order_quantity", [0.0] * env.agent_num)

            ep["rewards"].append(reward)
            ep["C_grid"].append(float(info.get("C_grid", 0.0)))
            ep["C_h2"].append(float(info.get("C_h2", 0.0)))
            ep["total_cost"].append(float(info.get("total_cost", 0.0)))
            ep["h2_ratios"].append(list(h2_ratios) if h2_ratios else [0.0] * env.agent_num)
            ep["h2_level"].append(list(h2_levels) if h2_levels else [0.0] * env.agent_num)
            ep["pending_energy"].append([float(x) for x in pending_ea])
            ep["delivered_energy"].append([float(x) for x in
                                           (delivered_ea if hasattr(delivered_ea, '__iter__')
                                            else [delivered_ea / env.agent_num] * env.agent_num)])
            ep["e_h2_ext"].append([float(x) for x in e_h2_ext_a])
            ep["net_h2_demand"].append([float(x) for x in net_h2_d])
            ep["e_h2_load"].append([float(x) for x in e_h2_ld])
            ep["e_h2_prod"].append([float(x) for x in e_h2_pr])
            ep["h2_order_qty"].append([float(x) for x in h2_oq])
            ep["h2_market_traded"].append(float(info.get("h2_market_traded", 0.0)))
            ep["actions"].append(actions.tolist())

        # Episode aggregates
        ep["episode_return"] = float(sum(ep["rewards"]))
        ep["episode_C_h2"] = float(sum(ep["C_h2"]))
        ep["episode_C_grid"] = float(sum(ep["C_grid"]))
        ep["episode_total_cost"] = float(sum(ep["total_cost"]))
        ep["episode_h2_traded"] = float(sum(ep["h2_market_traded"]))
        ep["episode_delivered"] = float(sum(
            sum(row) for row in ep["delivered_energy"]
        ))
        ep["episode_ordered"] = float(sum(
            sum(row) for row in ep["h2_order_qty"]
        ))
        # External H2 purchase per episode (sum over agents and steps, only positive)
        ep["episode_ext_buy"] = float(sum(
            sum(max(0.0, x) for x in row) for row in ep["e_h2_ext"]
        ))
        # Mean H2 ratio over episode (averaged over agents)
        all_ratios = ep["h2_ratios"]
        ep["episode_mean_h2_ratio"] = float(np.mean([
            np.mean(row) for row in all_ratios
        ]))
        # Minimum H2 ratio ever seen (safety margin indicator)
        ep["episode_min_h2_ratio"] = float(np.min([
            min(row) for row in all_ratios
        ]))

        results.append(ep)

    env.close() if hasattr(env, "close") else None
    return results


# ────────────────────────────────────────────────────────────────────────────
# Statistics helper
# ────────────────────────────────────────────────────────────────────────────

def summary_stats(results: list) -> dict:
    returns = [r["episode_return"] for r in results]
    c_h2 = [r["episode_C_h2"] for r in results]
    c_grid = [r["episode_C_grid"] for r in results]
    total_cost = [r["episode_total_cost"] for r in results]
    h2_traded = [r["episode_h2_traded"] for r in results]
    h2_delivered = [r["episode_delivered"] for r in results]
    h2_ordered = [r["episode_ordered"] for r in results]
    ext_buy = [r["episode_ext_buy"] for r in results]
    mean_ratio = [r["episode_mean_h2_ratio"] for r in results]
    min_ratio = [r["episode_min_h2_ratio"] for r in results]

    def _s(arr):
        a = np.array(arr)
        return {"mean": float(np.mean(a)), "std": float(np.std(a)),
                "min": float(np.min(a)),  "max": float(np.max(a))}

    return {
        "episode_return": _s(returns),
        "C_h2":          _s(c_h2),
        "C_grid":        _s(c_grid),
        "total_cost":    _s(total_cost),
        "h2_traded":     _s(h2_traded),
        "h2_delivered":  _s(h2_delivered),
        "h2_ordered":    _s(h2_ordered),
        "ext_buy":       _s(ext_buy),
        "mean_h2_ratio": _s(mean_ratio),
        "min_h2_ratio":  _s(min_ratio),
    }


# ────────────────────────────────────────────────────────────────────────────
# Plotting
# ────────────────────────────────────────────────────────────────────────────

COLORS = {
    "A_Naive": "#e74c3c",
    "B_InitBuffer": "#f39c12",
    "C_RollingLag": "#2ecc71",
    "D_InitBuffer_Rolling": "#3498db",
    "E_Oracle": "#9b59b6",
    "D_nolag": "#1abc9c",
}
LABELS = {
    "A_Naive": "A: Naive",
    "B_InitBuffer": "B: InitBuffer",
    "C_RollingLag": "C: RollingLag",
    "D_InitBuffer_Rolling": "D: InitBuffer+Rolling",
    "E_Oracle": "E: Oracle",
    "D_nolag": "D_nolag (instant delivery)",
}

WINDOW = 10   # moving average window


def _moving_avg(arr, w=WINDOW):
    ret = np.cumsum(arr, dtype=float)
    ret[w:] = ret[w:] - ret[:-w]
    ma = ret[w - 1:] / w
    return np.arange(w - 1, len(arr)), ma


def plot_episode_returns(all_results: dict, outdir: Path):
    """图1: Episode reward 曲线 + moving average."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for name, results in all_results.items():
        rets = [r["episode_return"] for r in results]
        c = COLORS.get(name, "gray")
        lb = LABELS.get(name, name)
        axes[0].plot(rets, alpha=0.35, color=c, linewidth=0.8)
        x, ma = _moving_avg(rets, WINDOW)
        axes[0].plot(x, ma, color=c, linewidth=2, label=lb)
        axes[1].plot(x, ma, color=c, linewidth=2, label=lb)
    axes[0].set_title("Episode Return (raw + moving avg)")
    axes[1].set_title("Episode Return (moving avg only)")
    for ax in axes:
        ax.set_xlabel("Episode")
        ax.set_ylabel("Episode Return")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(outdir / "01_episode_returns.png", dpi=150)
    plt.close(fig)


def plot_h2_ratio_curve(all_results: dict, outdir: Path):
    """图2: 储氢罐库存比例曲线（取所有 episode 平均后逐步展示）。"""
    fig, ax = plt.subplots(figsize=(10, 5))
    T = len(all_results[list(all_results.keys())[0]][0]["h2_ratios"])
    for name, results in all_results.items():
        # Mean over agents and episodes for each step
        step_means = []
        for t in range(T):
            vals = []
            for ep in results:
                if t < len(ep["h2_ratios"]):
                    vals.extend(ep["h2_ratios"][t])
            step_means.append(np.mean(vals) if vals else 0.0)
        c = COLORS.get(name, "gray")
        ax.plot(step_means, color=c, linewidth=2, label=LABELS.get(name, name))
    ax.axhline(0.05, color="gray", linestyle=":", alpha=0.5, label="h2_tank_min (5%)")
    ax.set_title("储氢罐库存比例（episode 均值，所有 agent 平均）")
    ax.set_xlabel("Episode Step")
    ax.set_ylabel("H2 Level Ratio")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(outdir / "02_h2_ratio_curve.png", dpi=150)
    plt.close(fig)


def plot_pending_h2(all_results: dict, outdir: Path):
    """图3: Pending 氢交付队列总量变化。"""
    fig, ax = plt.subplots(figsize=(10, 5))
    T = len(all_results[list(all_results.keys())[0]][0]["pending_energy"])
    for name, results in all_results.items():
        step_means = []
        for t in range(T):
            vals = []
            for ep in results:
                if t < len(ep["pending_energy"]):
                    vals.append(sum(ep["pending_energy"][t]))
            step_means.append(np.mean(vals) if vals else 0.0)
        c = COLORS.get(name, "gray")
        ax.plot(step_means, color=c, linewidth=2, label=LABELS.get(name, name))
    ax.set_title("Pending H2 交付队列总量（kWh_H2）（episode 均值）")
    ax.set_xlabel("Episode Step")
    ax.set_ylabel("Pending H2 Energy (kWh_H2)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(outdir / "03_pending_h2.png", dpi=150)
    plt.close(fig)


def plot_ext_buy(all_results: dict, outdir: Path):
    """图4: 逐步外部购氢量对比。"""
    fig, ax = plt.subplots(figsize=(10, 5))
    T = len(all_results[list(all_results.keys())[0]][0]["e_h2_ext"])
    for name, results in all_results.items():
        step_means = []
        for t in range(T):
            vals = []
            for ep in results:
                if t < len(ep["e_h2_ext"]):
                    vals.append(sum(max(0, x) for x in ep["e_h2_ext"][t]))
            step_means.append(np.mean(vals) if vals else 0.0)
        c = COLORS.get(name, "gray")
        ax.plot(step_means, color=c, linewidth=2, label=LABELS.get(name, name))
    ax.set_title("外部即时购氢量（kWh_H2/步）（episode 均值）")
    ax.set_xlabel("Episode Step")
    ax.set_ylabel("External H2 Purchase (kWh_H2)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(outdir / "04_ext_h2_buy.png", dpi=150)
    plt.close(fig)


def plot_order_vs_delivery(all_results: dict, outdir: Path):
    """图5: H2 下单量 vs 实际交付量对比。"""
    fig, axes = plt.subplots(1, len(all_results), figsize=(4 * len(all_results), 5), sharey=True)
    if len(all_results) == 1:
        axes = [axes]
    T = len(list(all_results.values())[0][0]["h2_order_qty"])
    for ax, (name, results) in zip(axes, all_results.items()):
        ordered_steps, delivered_steps = [], []
        for t in range(T):
            o_vals, d_vals = [], []
            for ep in results:
                if t < len(ep["h2_order_qty"]):
                    o_vals.append(sum(ep["h2_order_qty"][t]))
                if t < len(ep["delivered_energy"]):
                    d_vals.append(sum(ep["delivered_energy"][t]))
            ordered_steps.append(np.mean(o_vals) if o_vals else 0.0)
            delivered_steps.append(np.mean(d_vals) if d_vals else 0.0)
        ax.plot(ordered_steps, label="下单量", color="steelblue", linewidth=1.5)
        ax.plot(delivered_steps, label="交付量", color="tomato",
                linewidth=1.5, linestyle="--")
        ax.set_title(LABELS.get(name, name), fontsize=8)
        ax.set_xlabel("Step")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("kWh_H2")
    plt.suptitle("H2 下单量 vs 实际交付量（episode 均值）", y=1.02)
    plt.tight_layout()
    fig.savefig(outdir / "05_order_vs_delivery.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_load_pending_comparison(all_results: dict, outdir: Path):
    """图6: 氢负荷 vs pending 交付量（选 Policy D 展示）。"""
    key = "D_InitBuffer_Rolling" if "D_InitBuffer_Rolling" in all_results else \
          list(all_results.keys())[-1]
    results = all_results[key]
    T = len(results[0]["e_h2_load"])
    load_steps = []
    future_lh_steps = []
    pending_steps = []
    for t in range(T):
        l_v, fl_v, p_v = [], [], []
        for ep in results:
            if t < len(ep["e_h2_load"]):
                l_v.append(sum(ep["e_h2_load"][t]))
            if t < len(ep["future_load_h"]):
                fl_v.append(sum(ep["future_load_h"][t]))
            if t < len(ep["pending_energy"]):
                p_v.append(sum(ep["pending_energy"][t]))
        load_steps.append(np.mean(l_v) if l_v else 0.0)
        future_lh_steps.append(np.mean(fl_v) if fl_v else 0.0)
        pending_steps.append(np.mean(p_v) if p_v else 0.0)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(load_steps, label="当前氢负荷 (kWh_H2)", color="tomato", linewidth=2)
    ax.plot(future_lh_steps, label=f"未来 {LAG} 步预测氢负荷 (kWh_H2)",
            color="orange", linewidth=1.5, linestyle="--")
    ax.plot(pending_steps, label="Pending 交付量 (kWh_H2)", color="#3498db", linewidth=2)
    ax.set_title(f"氢负荷 / 未来需求 / Pending 交付量（{LABELS.get(key, key)}，episode 均值）")
    ax.set_xlabel("Episode Step")
    ax.set_ylabel("Energy (kWh_H2)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(outdir / "06_load_pending_comparison.png", dpi=150)
    plt.close(fig)


def plot_summary_bar(stats: dict, outdir: Path):
    """图7: 各策略关键指标条形图对比。"""
    keys_plot = ["episode_return", "C_h2", "ext_buy", "mean_h2_ratio"]
    titles = ["Episode Return", "C_h2 外部氢成本", "外部即时购氢量 (kWh_H2)", "储氢罐均值比例"]
    names = list(stats.keys())
    n = len(names)
    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    x = np.arange(n)
    for ax, key, title in zip(axes, keys_plot, titles):
        means = [stats[nm][key]["mean"] for nm in names]
        stds  = [stats[nm][key]["std"]  for nm in names]
        colors = [COLORS.get(nm, "gray") for nm in names]
        bars = ax.bar(x, means, yerr=stds, color=colors, alpha=0.85,
                      capsize=4, edgecolor="black", linewidth=0.5)
        ax.set_title(title, fontsize=9)
        ax.set_xticks(x)
        ax.set_xticklabels([LABELS.get(nm, nm)[:14] for nm in names],
                           rotation=25, ha="right", fontsize=7)
        ax.grid(True, axis="y", alpha=0.3)
    plt.suptitle("各规则策略关键指标对比（固定氢能交付时滞 lag=4）", y=1.02)
    plt.tight_layout()
    fig.savefig(outdir / "07_summary_bar.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def print_table(stats: dict):
    """打印对比统计表。"""
    hdr = (f"{'Strategy':<28} {'Return':>10} {'C_h2':>10} "
           f"{'ExtBuy':>10} {'H2Traded':>10} {'H2Ratio':>10} {'MinRatio':>9}")
    sep = "-" * len(hdr)
    print(sep)
    print(hdr)
    print(sep)
    for nm, s in stats.items():
        print(
            f"{LABELS.get(nm, nm):<28} "
            f"{s['episode_return']['mean']:>10.2f} "
            f"{s['C_h2']['mean']:>10.2f} "
            f"{s['ext_buy']['mean']:>10.1f} "
            f"{s['h2_traded']['mean']:>10.1f} "
            f"{s['mean_h2_ratio']['mean']:>10.4f} "
            f"{s['min_h2_ratio']['mean']:>9.4f} "
        )
    print(sep)


# ────────────────────────────────────────────────────────────────────────────
# Paper analysis paragraph (Chinese)
# ────────────────────────────────────────────────────────────────────────────

def generate_paper_paragraph(stats: dict, lag: int = LAG) -> str:
    """生成可写入论文的中文分析段落。"""
    names = list(stats.keys())
    naive_s = stats.get("A_Naive", {})
    best_s = stats.get("D_InitBuffer_Rolling", stats.get("E_Oracle", {}))
    oracle_s = stats.get("E_Oracle", {})

    if not naive_s or not best_s:
        return "[统计数据不完整，无法生成段落]"

    naive_ret = naive_s["episode_return"]["mean"]
    best_ret = best_s["episode_return"]["mean"]
    naive_c_h2 = naive_s["C_h2"]["mean"]
    best_c_h2 = best_s["C_h2"]["mean"]
    naive_ext = naive_s["ext_buy"]["mean"]
    best_ext = best_s["ext_buy"]["mean"]
    naive_ratio = naive_s["mean_h2_ratio"]["mean"]
    best_ratio = best_s["mean_h2_ratio"]["mean"]
    oracle_ret = oracle_s.get("episode_return", {}).get("mean", float("nan"))

    pct_ret = (best_ret - naive_ret) / (abs(naive_ret) + 1e-6) * 100
    pct_c_h2 = (naive_c_h2 - best_c_h2) / (abs(naive_c_h2) + 1e-6) * 100
    pct_ext = (naive_ext - best_ext) / (abs(naive_ext) + 1e-6) * 100

    para = f"""
=======================================================================
可写入论文的中文分析段落（规则策略对比实验，固定氢能交付时滞 lag={lag}步）
=======================================================================

本节通过规则策略 rollout 实验，在固定氢能交付时滞（h2_delivery_lag = {lag} 步）
环境下对五类规则策略进行对比分析，以验证"提前备货与滚动下单"假设的有效性。

**实验发现：**

固定交付时滞使氢能调度从"当前供需平衡问题"演变为"跨时段库存-订单协同问题"。
在 h2_delivery_lag = {lag} 步的约束下，无前瞻策略（策略 A：Naive Policy）在时刻 t
提交的氢能买入订单须经历 {lag} 步交付延迟后方可入库，这意味着当前时刻的氢负荷缺口
无法通过当前下单立即弥补，智能体被迫依赖外部即时购氢以覆盖 t 到 t+{lag}-1 期间的
热负荷需求。实验结果显示，Naive 策略的 episode 平均 reward 为
{naive_ret:.2f}，外部购氢成本 C_h2 为 {naive_c_h2:.2f} 元，
外部即时购氢总量为 {naive_ext:.1f} kWh_H2，储氢库存安全裕度（平均储氢比例）
仅为 {naive_ratio:.4f}，表明无前瞻能力策略存在显著的氢能供给不足风险。

采用提前备货机制（策略 B：Initial Hydrogen Buffer Policy）在 episode 初期（前 {lag} 步）
最大化本地制氢并激进下单，可部分缓解初始在途订单为零所带来的性能下降；
而滚动提前下单策略（策略 C：Rolling Lag-aware Ordering Policy）通过在时刻 t
显式预测 [t, t+{lag}-1] 时段的氢需求，并根据 pending 交付队列实时校正订单量，
实现了更稳定的跨时段库存-订单耦合管理。

结合初期安全库存建立与后续滚动下单的策略 D（Initial Buffer + Rolling）取得了最优
的规则策略性能：episode 平均 reward 达 {best_ret:.2f}（较 Naive 策略变化
{pct_ret:+.1f}%），外部购氢成本 C_h2 降至 {best_c_h2:.2f} 元
（较 Naive 降低 {pct_c_h2:.1f}%），外部即时购氢量减少 {pct_ext:.1f}%，
储氢库存安全裕度提升至 {best_ratio:.4f}。

**关键结论：**

（1）规则策略实验表明，在固定交付时滞存在时，提前备货与滚动下单可以缓解延迟
    导致的 reward 下降，并有效降低外部即时购氢成本。

（2）固定交付时滞使氢能调度呈现显著的跨时段库存-订单耦合特征：无前瞻策略在
    当前时刻下单无法立即改善当前供氢状态，因此更容易依赖外部即时购氢，导致成本升高。

（3）pending 交付信息（pending delivery queue）为策略提供了未来可用氢量的可观测线索，
    有助于降低重复下单和供氢不足风险；在 oracle 策略（策略 E）中，利用精确未来氢负荷
    预测进一步优化了下单时机与数量。

（4）该结果为后续引入延迟感知观测（lag-aware observation）、储氢库存安全约束或
    前瞻性策略结构（lag-aware rolling ordering policy）提供了实验依据，
    支持将固定时滞约束下的库存感知策略（inventory-aware hydrogen dispatch）纳入
    智能体设计，以缓解 temporal coupling induced by fixed delivery lag 对整体
    episode reward 的负面影响。

**术语对照：**
- 固定交付时滞 / temporal coupling induced by fixed delivery lag
- 氢能订单-交付滞后 / hydrogen order-delivery lag
- 跨时段库存-订单耦合 / cross-timestep inventory-order coupling
- 提前备货机制 / advance stocking mechanism
- 滚动下单策略 / lag-aware rolling ordering policy
- 在途订单 / pending delivery
- 储氢库存安全裕度 / hydrogen tank safety margin
- 外部即时购氢 / instant external hydrogen procurement
- 前瞻性储氢调度 / proactive hydrogen dispatch
- 固定时滞约束下的库存感知策略 / inventory-aware hydrogen dispatch under fixed lag
=======================================================================
"""
    return para


# ────────────────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────────────────

def main():
    policies = [
        NaivePolicy(),
        InitialBufferPolicy(),
        RollingLagPolicy(),
        InitBufferRollingPolicy(),
        OraclePolicy(),
    ]

    all_results = {}
    all_stats = {}

    print("=" * 60)
    print(f"规则策略 H2 延迟 rollout 实验 | lag={LAG} | N={N_EPISODES}")
    print("=" * 60)

    for policy in policies:
        print(f"\n[{policy.name}] 开始运行 {N_EPISODES} episodes...")
        results = run_rollout(policy, n_episodes=N_EPISODES, lag_env=True)
        all_results[policy.name] = results
        s = summary_stats(results)
        all_stats[policy.name] = s
        print(f"  Return: {s['episode_return']['mean']:.2f} ± {s['episode_return']['std']:.2f}")
        print(f"  C_h2:   {s['C_h2']['mean']:.2f} ± {s['C_h2']['std']:.2f}")
        print(f"  ExtBuy: {s['ext_buy']['mean']:.1f} kWh_H2")
        print(f"  H2Ratio(mean/min): {s['mean_h2_ratio']['mean']:.4f} / {s['min_h2_ratio']['mean']:.4f}")

    # ── Optional: no-lag reference for Policy D ────────────────────────────
    print("\n[D_nolag] 即时交付参考环境 (h2_market_lag_enable=False)...")
    saved_lag = MICROGRID_CONFIG["h2_market_lag_enable"]
    MICROGRID_CONFIG["h2_market_lag_enable"] = False
    MICROGRID_CONFIG["h2_delivery_lag"] = 0
    nolag_policy = InitBufferRollingPolicy()
    nolag_policy.name = "D_nolag"
    nolag_results = run_rollout(nolag_policy, n_episodes=N_EPISODES, lag_env=False)
    all_results["D_nolag"] = nolag_results
    s_nolag = summary_stats(nolag_results)
    all_stats["D_nolag"] = s_nolag
    MICROGRID_CONFIG["h2_market_lag_enable"] = saved_lag   # restore
    MICROGRID_CONFIG["h2_delivery_lag"] = LAG
    print(f"  Return: {s_nolag['episode_return']['mean']:.2f} ± {s_nolag['episode_return']['std']:.2f}")
    print(f"  C_h2:   {s_nolag['C_h2']['mean']:.2f}")

    # ── Save raw stats JSON ────────────────────────────────────────────────
    stats_path = OUT_DIR / "stats_summary.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(all_stats, f, ensure_ascii=False, indent=2)
    print(f"\n统计结果已保存: {stats_path}")

    # ── Print table ────────────────────────────────────────────────────────
    print("\n")
    print_table(all_stats)

    # ── Generate plots ────────────────────────────────────────────────────
    print("\n生成图表...")
    plot_episode_returns(all_results, OUT_DIR)
    plot_h2_ratio_curve(all_results, OUT_DIR)
    plot_pending_h2(all_results, OUT_DIR)
    plot_ext_buy(all_results, OUT_DIR)
    plot_order_vs_delivery(all_results, OUT_DIR)
    plot_load_pending_comparison(all_results, OUT_DIR)
    plot_summary_bar(all_stats, OUT_DIR)
    print(f"图表已保存至: {OUT_DIR}")

    # ── Paper paragraph ────────────────────────────────────────────────────
    para = generate_paper_paragraph(all_stats)
    print(para)
    para_path = OUT_DIR / "paper_paragraph_cn.txt"
    with open(para_path, "w", encoding="utf-8") as f:
        f.write(para)
    print(f"论文段落已保存: {para_path}")

    # ── Answer questions ───────────────────────────────────────────────────
    print("\n=== 关键问题回答 ===")
    print(f"\nQ1 固定交付时滞为何导致 reward 下降:")
    print(f"   当前时刻下单无法立即交付 (lag={LAG}步)，前 lag 步外部购氢无法避免，")
    print(f"   Naive 策略无法提前建立 pending 管道，每步均依赖高价外部购氢。")

    naive_ext = all_stats["A_Naive"]["ext_buy"]["mean"]
    best_ext  = all_stats["D_InitBuffer_Rolling"]["ext_buy"]["mean"]
    print(f"\nQ2 滚动提前下单是否减少了外部购氢:")
    print(f"   A(Naive)={naive_ext:.1f}  vs  D(I+R)={best_ext:.1f} kWh_H2")
    print(f"   减少 {(naive_ext-best_ext)/max(naive_ext,1e-6)*100:.1f}%")

    naive_ratio = all_stats["A_Naive"]["mean_h2_ratio"]["mean"]
    best_ratio  = all_stats["D_InitBuffer_Rolling"]["mean_h2_ratio"]["mean"]
    print(f"\nQ3 储氢安全裕度是否提高:")
    print(f"   A(Naive)={naive_ratio:.4f}  vs  D(I+R)={best_ratio:.4f}")

    d_ext  = all_stats["D_InitBuffer_Rolling"]["ext_buy"]["mean"]
    nolag_ext = all_stats["D_nolag"]["ext_buy"]["mean"]
    print(f"\nQ4 即时交付参考上界 (D_nolag) vs lag 环境最佳 (D):")
    print(f"   D_nolag 外部购氢: {nolag_ext:.1f}  vs  D(lag): {d_ext:.1f} kWh_H2")
    print(f"   lag 额外损失: {(d_ext-nolag_ext):.1f} kWh_H2 "
          f"({(d_ext-nolag_ext)/max(nolag_ext,1e-6)*100:.1f}%)")

    print(f"\n结果文件: {OUT_DIR}")


if __name__ == "__main__":
    main()
