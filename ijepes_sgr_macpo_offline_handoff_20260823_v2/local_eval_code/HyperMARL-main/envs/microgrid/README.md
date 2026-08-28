# Microgrid Multi-Agent Reinforcement Learning Environment

> Path: `HyperMARL-main/envs/microgrid/`
> Version: Italian data + linear multi-node GasNet + H2 4h delay/pending observations + capacity/horizon-aware H2 buy clipping (2026-05-03)

---

## Table of Contents

1. [Environment Overview](#1-environment-overview)
2. [File Structure](#2-file-structure)
3. [Agent Configuration](#3-agent-configuration)
4. [Action Space](#4-action-space)
5. [Observation Space](#5-observation-space)
6. [Reward Function](#6-reward-function)
7. [Penalty System](#7-penalty-system)
8. [Market Mechanism](#8-market-mechanism)
9. [Time-of-Use Tariff](#9-time-of-use-tariff)
10. [Exogenous Profile Generation](#10-exogenous-profile-generation)
11. [Anti-Arbitrage Constraints](#11-anti-arbitrage-constraints)
12. [Key Parameter Reference](#12-key-parameter-reference)
13. [Quick Start](#13-quick-start)
14. [Training with HyperMARL](#14-training-with-hypermarl)
15. [Deterministic Rollout & Evaluation](#15-deterministic-rollout--evaluation)
16. [Changelog](#16-changelog)

---

## 1. Environment Overview

This environment simulates a community energy system with **4 microgrid agents** that jointly manage:

- **Local electricity generation** (PV, wind turbines)
- **Electricity storage** (batteries)
- **Hydrogen production** (electrolysers)
- **Hydrogen storage** (H2 tanks)
- **Heat supply** (hydrogen-powered boilers)
- **Peer-to-peer electricity market** (continuous double auction, every step)
- **Peer-to-peer hydrogen market** (continuous double auction, every step)
- **External grid** (fallback at TOU prices)
- **External hydrogen market** (asymmetric prices: buy expensive, sell cheap)

### Key Properties

| Property | Value |
|---|---|
| Episode length | 24 hours |
| Time step (dt) | 1 hour |
| Number of agents | 4 |
| Observation dimension | 17 by current default; optional GasNet observations add dimensions |
| Action dimension | 5 |
| Reward type | Shared (cooperative) |
| Discount factor (gamma) | 0.99 |
| Reward scale | 200.0 |

All agents receive the **same scalar reward** each step: `reward = -base_cost / reward_scale`. The goal is cooperative minimization of realized operating cost across the entire microgrid community.

---

## 2. File Structure

```
envs/microgrid/
├── config.py                   # All hyperparameters and physical constants (MICROGRID_CONFIG dict)
├── microgrid_env.py            # Core environment class MicrogridEnv (reset / step / _get_obs)
├── microgrid_continuous_env.py # Gymnasium-compatible wrapper for HyperMARL training
├── cda_market.py               # Price-time priority continuous double auction engine
├── data_generator.py           # Per-episode exogenous profile generation (PV/wind/load)
└── README.md                   # This file
```

### Module Responsibilities

| File | Description |
|---|---|
| `config.py` | Single source of truth for all physical, economic, and penalty parameters. Exports `MICROGRID_CONFIG` dict, `get_tou_price()`, and `build_tou_table()`. |
| `microgrid_env.py` | `MicrogridEnv` class implementing the full simulation loop: action denormalization, physical constraints, SOC/H2 updates, market clearing, cost calculation, penalty computation, and observation construction. |
| `microgrid_continuous_env.py` | `MicrogridContinuousEnv` wraps `MicrogridEnv` with `action_space` / `observation_space` attributes and stacks outputs into NumPy arrays for HyperMARL's vectorized environment interface. |
| `cda_market.py` | Stateless CDA engine shared by both electricity and hydrogen markets. Implements `run_continuous_double_auction(orders, default_price)` with price-time priority matching. |
| `data_generator.py` | Generates daily PV, wind, electric load, and heat load profiles. Supports two modes: **synthetic** (parametric curves) and **Italian** (real-data from CSV). Heat load can be derived from electric load profiles. |

---

## 3. Agent Configuration

All 4 agents have type `"mg"` (microgrid). Roles are distinguished by asset endowment:

| ID | Role | PV (kW) | Wind (kW) | Battery (kWh / kW) | Electrolyser (kW / eff) | H2 Tank (kg / kW_H2) | Load_e peak (kW) | Load_h peak (kW_th) |
|---|---|---|---|---|---|---|---|---|
| **A0** | Producer | 5000 | 1000 | 5000 / 2000 | 2000 / 0.70 | 500 / 2000 | 2250 | 750 |
| **A1** | Producer | 1000 | 4000 | 3000 / 1200 | 3000 / 0.65 | 500 / 2000 | 1875 | 600 |
| **A2** | Consumer | 500 | 3000 | 4000 / 1500 | 500 / 0.60 | 300 / 1200 | 3000 | 1500 |
| **A3** | Consumer | 2000 | 500 | 2000 / 800 | 800 / 0.65 | 300 / 1500 | 2625 | 1875 |

### Role Notes

- **All agents** have electrolysers, batteries, H2 tanks, and both electric and heat loads.
- **Producers (A0/A1)** have larger electrolysers and PV/wind capacity; they are net hydrogen exporters.
- **Consumers (A2/A3)** have higher heat loads; they are net hydrogen importers.
- The heat load is 100% supplied by hydrogen (via boiler with 90% efficiency). There is no gas bypass action; `h2_thermal_share` is fixed at 1.0.

---

## 4. Action Space

Each agent outputs a **5-dimensional continuous vector**, normalized to `[-1, 1]` during training.

| Dim | Symbol | Physical Meaning | Denormalization |
|---|---|---|---|
| a0 | `P_el` | Electrolyser power | `(a0 + 1) / 2 × el_cap` → `[0, el_cap]` kW |
| a1 | `P_bat` | Battery charge/discharge | `a1 × bat_power` → `[-bat_power, +bat_power]` kW (positive = charge) |
| a2 | `elec_bid_price` | Electricity market bid price | Linear map `[-1,1]` → `[TOU_sell, TOU_buy]` of current step |
| a3 | `h2_bid_price` | Hydrogen market bid price | Linear map `[-1,1]` → `[lambda_h2_sell, lambda_h2_buy]` = `[0.50, 1.00]` |
| a4 | `P_ht` | H2 tank charge/discharge | `a4 × h2_tank_power` → `[-h2_tank_power, +h2_tank_power]` kW_H2 (positive = charge) |

### Physical Clipping

After denormalization, battery and H2 tank actions are **further clipped** by the environment:

- **Battery**: Charge limited by `(soc_max - soc) × bat_cap / (eta_c × dt)`. Discharge limited by `(soc - soc_min) × bat_cap × eta_d / dt` **and** by local electric deficit (anti-arbitrage, see Section 11).
- **H2 Tank**: Charge limited by headroom to `h2_max`. Discharge limited by stored energy above `h2_min` **and** by local H2 deficit (anti-arbitrage, see Section 11).

### Market Order Generation

Market orders are **automatically derived** from net demand:

- **Electricity market**: Order quantity = `|net_electric_demand| × dt`. Side = buy if demand > 0, sell if < 0.
- **Hydrogen market**: Raw order quantity = `|net_h2_demand|`. Side = buy if demand > 0, sell if < 0. When H2 delivery lag and `h2_cap_aware_buy_enable=True` are active, buy orders are clipped by future tank headroom after pending deliveries and by episode-delivery feasibility.

Agents only control **price** (a2, a3); the quantity is determined by their physical state.

---

## 5. Observation Space

The core local observation has **16 dimensions** (`MICROGRID_CONFIG["obs_dim"] = 16`). Runtime `env.obs_dim` is expanded by optional gas-network and pending-H2 features:

- **Current default**: 22 dimensions = 16 core + 4 pending arrival buckets + 2 pending summary features.
- **Current GasNet H2-delay experiment**: 24 dimensions = 16 core + gas buy-price + gas pressure + 4 pending buckets + 2 pending summary features.

| Dim | Name | Description | Normalization |
|---|---|---|---|
| o0 | `pv_ratio` | Current PV output | `÷ pv_cap` (safe denom if cap=0) |
| o1 | `wt_ratio` | Current wind output | `÷ wt_cap` |
| o2 | `load_e_ratio` | Current electric load | `÷ load_e_peak` |
| o3 | `load_h_ratio` | Current heat load | `÷ load_h_peak` |
| o4 | `bat_soc` | Battery state of charge | Direct value in `[0.1, 0.9]` |
| o5 | `h2_level_ratio` | H2 tank fill ratio | `h2_level ÷ h2_tank_cap` |
| o6 | `elec_clearing_norm` | Last electricity market clearing price | `(price - 0.15) / (1.00 - 0.15)` → `[0, 1]` |
| o7 | `h2_clearing_norm` | Last hydrogen market clearing price | `(price - 0.50) / (1.00 - 0.50)` → `[0, 1]` |
| o8 | `sin_t` | Time encoding sin | `sin(2πt/T)` ∈ `[-1, 1]` |
| o9 | `cos_t` | Time encoding cos | `cos(2πt/T)` ∈ `[-1, 1]` |
| o10 | `tou_buy_norm` | Current TOU buy price | `÷ max(TOU_buy)` = `÷ 1.0` |
| o11 | `last_p_grid_norm` | Last step grid exchange | `clip(p_grid ÷ scale, -1, 1)` (positive = bought) |
| o12 | `last_e_h2_ext_norm` | Last step external H2 exchange | `clip(e_h2_ext ÷ scale, -1, 1)` (positive = bought) |
| o13 | `soc_dev` | SOC deviation from per-agent target | `soc - soc_penalty_targets[i]` |
| o14 | `h2_dev` | H2 ratio deviation from per-agent target | `h2_ratio - h2_penalty_targets[i]` |
| o15 | `time_pressure` | Terminal urgency signal | `1 / max(T - t, 1)` (rises from 0.042 to 1.0) |

### Optional GasNet Observations

When `gas_price_obs_enable=True` and `gas_pressure_obs_enable=True`, two local gas-network features are appended after the 16 core dimensions:

| Dim in GasNet layout | Name | Description |
|---|---|---|
| o16 | `h2_external_buy_price_norm` | Local node external H2 buy price normalized by configured gas price bounds |
| o17 | `gas_pressure_norm` | Local node gas pressure normalized by configured gas pressure bounds |

### Optional Pending-H2 Observations

When `h2_pending_obs_enable=True`, each agent receives normalized pending deliveries addressed to itself. With GasNet observations enabled, these occupy `o18`–`o23`; without GasNet observations, the same fields start at `o16`.

| GasNet Dim | Name | Description | Normalization |
|---|---|---|---|
| o18 | `pending_h2_arrive_1h_norm` | H2 scheduled to arrive in 1 hour | `÷ (h2_tank_cap × LHV_H2)`, clipped to `[0, 1]` |
| o19 | `pending_h2_arrive_2h_norm` | H2 scheduled to arrive in 2 hours | Same |
| o20 | `pending_h2_arrive_3h_norm` | H2 scheduled to arrive in 3 hours | Same |
| o21 | `pending_h2_arrive_4h_norm` | H2 scheduled to arrive in 4 hours | Same |
| o22 | `pending_h2_total_norm` | Total pending H2 for this buyer | Same |
| o23 | `pending_adjusted_h2_headroom_norm` | Future tank headroom after subtracting pending deliveries | `÷ (h2_tank_cap × LHV_H2)`, clipped to `[-1, 1]` |

---

## 6. Reward Function

All agents share a single scalar reward each step:

```
reward_t = -base_cost / reward_scale
```

Where `reward_scale = 200.0`.

### 6.1 Base Cost

```
base_cost = C_grid + C_h2
```

| Component | Formula |
|---|---|
| `C_grid` | `Σ_i [ p_grid_i > 0 ? p_grid_i × TOU_buy × dt : p_grid_i × TOU_sell × dt ]` |
| `C_h2` | `lambda_h2_buy × Σ max(e_h2_ext, 0) + lambda_h2_sell × Σ min(e_h2_ext, 0)` |

**Note**: Internal CDA trades transfer cost between agents, but under shared reward these cancel out (net transfer = 0). However, they reduce reliance on the expensive external market, which is the key incentive for P2P trading.

## 7. Penalty System

The environment still computes a **layered penalty system** for diagnostics in `info`, but these penalty terms are not added to the training reward. The current reward depends only on `base_cost`.

### 7.1 Soft Terminal Penalty (SOC + H2 target deviation)

```
penalty_soc = soc_penalty_coef × (1 / max(T-t, 1)) × Σ_i max(0, |soc_i - soc_target_i| - deadband)²
penalty_h2  = h2_penalty_coef  × (1 / max(T-t, 1)) × Σ_i max(0, |h2_ratio_i - h2_target_i| - deadband_i)²
```

- **Coefficient**: `soc_penalty_coef = h2_penalty_coef = 5663.6`
- **SOC targets**: `soc_penalty_targets = [0.45, 0.45, 0.45, 0.45]`
- **H2 targets**: `h2_penalty_targets = [0.40, 0.40, 0.50, 0.60]`
- **Default deadband**: `penalty_deadband = 0.08`
- **Consumer H2 deadband**: `consumer_h2_deadband = 0.3`
- **Weight `1/(T-t)`**: Grows from 1/24 at t=0 to 1/1 at t=23, creating increasing urgency to return inventory to center as the episode ends.

**Purpose**: Prevents agents from arbitraging initial inventory by penalizing large deviations from calibrated per-agent SOC/H2 targets, especially near episode end.

### 7.2 Low Inventory Penalty (continuous)

```
penalty_low_inv = low_inventory_penalty_coef × Σ_i [
    max(0, soc_low_threshold - soc_i)² +
    max(0, h2_low_threshold - h2_ratio_i)²
]
```

- **Coefficient**: `low_inventory_penalty_coef = 1000.0`
- **SOC threshold**: `soc_low_threshold = 0.0` (effectively disabled for SOC)
- **H2 threshold**: `h2_low_threshold = 0.15`
- **Applied**: Every step, all agents
- **Purpose**: Prevents H2 tank from chronically sitting at the physical lower bound (0.05 ratio). Only activates when H2 ratio drops below 0.15.

### 7.3 Terminal H2 Floor Penalty (episode end only)

```
penalty_terminal_h2_floor = terminal_h2_floor_penalty_coef × Σ_{i ∈ targets} max(0, threshold - h2_ratio_i)²
```

- **Enabled**: `terminal_h2_floor_penalty_enable = True`
- **Coefficient**: `terminal_h2_floor_penalty_coef = 50000.0`
- **Threshold**: `terminal_h2_floor_threshold = 0.20`
- **Target agents**: `terminal_h2_floor_agent_indices = [2, 3]` (consumers only)
- **Applied**: Only at `t >= T-1` (the final step)
- **Purpose**: Prevents consumer agents from depleting their H2 tanks to the absolute minimum by episode end. Does not affect producer agents or mid-episode behavior.

### 7.4 Stepwise H2 Floor Penalty

```
penalty_stepwise_h2_floor = coef × urgency(t) × Σ_i weight_i × max(0, threshold_i - h2_ratio_i)²
```

- **Enabled**: `stepwise_h2_floor_penalty_enable = True`
- **Thresholds**: `[0.0, 0.0, 0.25, 0.35]`
- **Weights**: `[0.0, 0.0, 1.5, 4.0]`
- **Coefficient**: `stepwise_h2_floor_penalty_coef = 5000.0`
- **Urgency gain**: `stepwise_h2_floor_urgency_gain = 3.0`
- **Purpose**: Protects consumer H2 inventories during the episode, especially A3.

### 7.5 Action Regularization Penalty (currently disabled)

```
penalty_action_mag   = action_magnitude_penalty_coef × Σ a_reg²
penalty_action_delta = action_delta_penalty_coef × Σ (a_reg_t - a_reg_{t-1})²
```

- **Enabled**: `action_reg_enable = False` (currently off)
- **Regulated indices**: `action_reg_indices = [0, 1, 4]` (electrolyser, battery, H2 tank)
- **Coefficients**: magnitude = 10.0, delta = 20.0

### Penalty Summary

| Penalty | When | What | Coefficient | Threshold |
|---|---|---|---|---|
| SOC target deviation | Every step | All agents' SOC | 5663.6 × 1/(T-t) | per-agent targets |
| H2 target deviation | Every step | All agents' H2 ratio | 5663.6 × 1/(T-t) | per-agent targets |
| Low H2 inventory | Every step | All agents' H2 ratio | 1000.0 | ratio < 0.15 |
| Terminal H2 floor | Final step only | A2, A3 H2 ratio | 50000.0 | ratio < 0.20 |
| Stepwise H2 floor | Every step | A2, A3 H2 ratio | 5000.0 × urgency | ratio < [0.25, 0.35] |
| Action magnitude | Disabled | — | 10.0 | — |
| Action delta | Disabled | — | 20.0 | — |

---

## 8. Market Mechanism

### 8.1 Electricity Market (every step)

- **Timing**: CDA clears every time step.
- **Order quantity**: `|net_electric_demand_i| × dt` (full demand, auto-generated).
- **Side**: `net_electric_demand > 0` → buy order; `< 0` → sell order.
- **Matching**: Price-time priority CDA (see Section 8.3).
- **Residual**: Unmatched buy → external grid at TOU buy price. Unmatched sell → external grid at TOU sell price.
- **Price bounds**: Agent bid ∈ `[TOU_sell, TOU_buy]` of the current step (dynamic per hour).

### 8.2 Hydrogen Market with 4h Delivery Delay

Current configuration uses CDA every step with fixed delayed physical delivery:

- `h2_market_schedule_enable = False` → CDA runs every step.
- `h2_market_lag_enable = True` and `h2_delivery_lag = 4` → matched H2 buy orders enter `pending_h2_deliveries` and physically arrive 4 hours later.
- Matched buyers pay the CDA price immediately, but their H2 tank is not filled until delivery.
- Matched sellers are treated as shipping their surplus immediately through the market.
- At delivery time, accepted H2 is stored in the buyer tank; overflow is sold to the external market and recorded in diagnostics.

- **Raw order quantity**: `|net_h2_demand_i|` (kWh_H2).
- **Side**: `net_h2_demand > 0` → buy; `< 0` → sell.
- **Residual**: Unmatched buy → external H2 market at `lambda_h2_buy = 1.00 yuan/kWh_H2`. Unmatched sell → external at `lambda_h2_sell = 0.50 yuan/kWh_H2`.
- **Price bounds**: Agent bid ∈ `[0.50, 1.00]` yuan/kWh_H2.

### 8.2.1 Capacity/Horizon-Aware H2 Buy Clipping

When `h2_cap_aware_buy_enable=True`, H2 buy orders are clipped before entering the CDA:

```python
pending_total = sum(pending deliveries for buyer)
if t + h2_delivery_lag <= T:
    future_headroom = max(0, (h2_max - h2_level) * LHV_H2 - pending_total)
else:
    future_headroom = 0
final_buy_order_qty = min(raw_buy_order_qty, future_headroom)
```

This directly prevents a buyer from ordering more delayed H2 than its tank can receive after already scheduled pending deliveries, and prevents new buy orders that would arrive after the episode ends. The following diagnostics are exported in `info` and rollout JSON:

- `h2_order_quantity_raw`
- `h2_order_quantity`
- `h2_buy_future_headroom`
- `h2_buy_clip_amount`
- `h2_buy_horizon_clip_amount`

### 8.2.2 Direct Delivery Reservation

When `h2_delivery_reservation_enable=True`, pending H2 arrivals within `h2_delivery_reservation_horizon` reserve buyer-side tank headroom before local H2 tank charging is applied:

```python
reserved_pending = reservation_ratio * sum(pending deliveries due within horizon)
available_charge_headroom = max(0, current_headroom - reserved_pending)
max_p_ht_charge = min(max_p_ht_charge, available_charge_headroom / (h2_eff_c * dt))
```

This prevents a buyer from filling its own H2 tank with `p_ht > 0` while a delayed shipment is already scheduled to arrive soon. It complements capacity/horizon-aware buy clipping: buy clipping limits new H2 orders before CDA, while delivery reservation protects receiving capacity after orders have already become pending.

Additional diagnostics:

- `h2_delivery_reserved_energy`
- `h2_delivery_reservation_margin`
- `h2_delivery_reservation_charge_clip`

### 8.3 CDA Engine (`cda_market.py`)

The `run_continuous_double_auction(orders, default_price)` function implements:

1. Buy orders sorted by price **descending** (highest bid first).
2. Sell orders sorted by price **ascending** (lowest ask first).
3. Incoming buy order matches against lowest resting sell if `buy_price >= sell_price`, at the **resting order's price**.
4. Incoming sell order matches against highest resting buy if `sell_price <= buy_price`, at the **resting order's price**.
5. Partial fills are supported; residual quantity stays on the book for subsequent orders within the same step.

Returns: trades list, clearing price, per-agent matched quantities, costs, revenues, and residual order books.

---

## 9. Time-of-Use Tariff

The external grid uses a three-tier time-of-use pricing scheme:

| Period | Hours | TOU Buy (yuan/kWh) | TOU Sell (yuan/kWh) |
|---|---|---|---|
| **Valley** | t = 0–6, 23 | 0.30 | 0.15 |
| **Shoulder** | t = 7–9, 15–17, 21–22 | 0.60 | 0.35 |
| **Peak** | t = 10–14, 18–20 | 1.00 | 0.55 |

### Hydrogen Production Cost Reference

| Scenario | Calculation | Cost (yuan/kWh_H2) |
|---|---|---|
| Valley electrolysis (A0, η=0.70) | 0.30 / 0.70 | **0.429** |
| Valley electrolysis (A1, η=0.65) | 0.30 / 0.65 | **0.462** |
| Shoulder electrolysis (A0) | 0.60 / 0.70 | 0.857 |

The profitable CDA price window for H2 depends on electrolyzer production cost and external H2 buy/sell prices.

---

## 10. Exogenous Profile Generation

Profiles are generated per-episode by `data_generator.py` when `env.reset()` is called.

### 10.1 Profile Source Selection

Controlled by `config["profile_source"]`:

| Mode | `profile_source` | PV / Wind / Load_e | Load_h |
|---|---|---|---|
| Synthetic | `"synthetic"` | Parametric curves + noise | Parametric double-peak |
| **Italian (current)** | `"italian"` | Real data from CSV | Derived from electric load |

### 10.2 Italian Data Mode (current default)

- **Data file**: `/root/autodl-tmp/HyperMARL-main/envs/Italian_data.csv`
- **Format**: 672 rows × 31 columns (28 days × 24 hours). Columns: 8 PV (`Ppv*`), 6 wind (`Pw*`), 17 electric load (`PL*`).
- **Day selection**: Fixed random day split by seed. Default train uses 70% of complete 24-hour days; test uses the remaining 30%. `reset()` samples only from the selected split's fixed day pool.
- **Column assignment**: Columns are evenly split across 4 agents using `np.array_split`.
- **Scaling**: Each agent's profile is scaled to its configured capacity (`pv_cap`, `wt_cap`, `load_e_peak`).

### 10.3 Derived Heat Load

When `derive_heat_from_electric = True` (current default):

```python
load_h[i] = load_h_peak[i] × (base_ratio + variable_ratio × shape) × noise
```

Where:
- `shape` = normalized electric load profile of agent i (min-max scaled to [0, 1])
- `base_ratio = 0.15` (constant base heat demand)
- `variable_ratio = 0.45` (heat correlated with electric load)
- `noise = 1 + N(0, 0.02)` per step

This ensures heat load patterns are **correlated with but not identical to** electric load, providing realistic seasonal variation.

### 10.4 Synthetic Mode (fallback)

If `profile_source = "synthetic"`:

| Profile | Generation Method |
|---|---|
| **PV** | Half-sine (sunrise 6h, sunset 18h) × cloud factor (0.7–1.0) × step noise |
| **Wind** | Base 0.3 + daily sine variation + Gaussian noise (σ=0.08) |
| **Electric load** | Double-peak (morning 9h + evening 19h) Gaussian + 3% noise |
| **Heat load** | Double-peak (morning 7h + evening 19h) Gaussian + 3% noise |

All curves are multiplied by a daily random factor `∈ [0.9, 1.1]`.

### 10.5 Reproducibility

```python
env.seed(42)
obs = env.reset()  # deterministic profiles
```

---

## 11. Anti-Arbitrage Constraints

Two physical clipping rules prevent unrealistic energy arbitrage:

### 11.1 Battery Discharge Limit

```python
if p_bat < 0:  # discharge
    local_electric_deficit = max(0, load_e + p_el - pv - wt)
    max_discharge = min(max_discharge, local_electric_deficit)
```

**Effect**: Battery can only discharge up to the agent's **local electric deficit**. It cannot discharge surplus energy to sell to the grid or P2P market. This prevents:
- Draining initial SOC (0.5) for free revenue
- Pure "buy low → store → sell high" grid arbitrage via the battery

**What's still allowed**: Charging during valley hours and **self-consuming** during peak hours to reduce grid purchases (demand shifting).

### 11.2 H2 Tank Discharge Limit

```python
if p_ht < 0:  # discharge
    local_h2_deficit = max(0, e_h2_load - e_h2_prod) / dt
    max_discharge = min(max_discharge, local_h2_deficit)
```

**Effect**: H2 tank can only discharge up to the agent's **local hydrogen deficit** (heat load minus electrolyser production). Cannot discharge surplus H2 to sell externally. Same anti-arbitrage rationale as battery.

---

## 12. Key Parameter Reference

### Core

| Parameter | Value | Description |
|---|---|---|
| `num_agents` | 4 | Number of microgrid agents |
| `episode_length` | 24 | Steps per episode (hours) |
| `dt` | 1.0 | Time step duration (hours) |
| `obs_dim` | 16 base / 22 default / 24 GasNet H2-delay | Observation dimension per agent |
| `action_dim` | 5 | Action dimension per agent |
| `gamma` | 0.99 | Discount factor |
| `reward_scale` | 200.0 | Reward normalization divisor |

### Battery

| Parameter | Value |
|---|---|
| `bat_eff_c` / `bat_eff_d` | 0.95 / 0.95 |
| `soc_min` / `soc_max` | 0.1 / 0.9 |
| `soc_init` | 0.1 |

### Hydrogen

| Parameter | Value |
|---|---|
| `h2_tank_min_ratio` / `h2_tank_max_ratio` | 0.05 / 0.95 |
| `h2_tank_init_ratio` | 0.05 |
| `h2_eff_c` / `h2_eff_d` | 1.0 / 1.0 |
| `LHV_H2` | 33.33 kWh/kg |
| `boiler_eff` | 0.90 |
| `h2_market_lag_enable` | True |
| `h2_delivery_lag` | 4 |
| `h2_pending_obs_enable` | True |
| `h2_pending_obs_horizon` | 4 |
| `h2_pending_summary_obs_enable` | True |
| `h2_cap_aware_buy_enable` | True |
| `h2_transport_loss` | 0.0 |

### Market Prices

| Parameter | Value | Description |
|---|---|---|
| `lambda_h2_buy` | 1.00 | External H2 buy price (yuan/kWh_H2) |
| `lambda_h2_sell` | 0.50 | External H2 sell price (yuan/kWh_H2) |
| `elec_price_min` / `max` | 0.15 / 1.00 | Electricity CDA price bounds |
| `h2_price_min` / `max` | 0.50 / 1.00 | Hydrogen CDA price bounds |

### Penalty Coefficients

| Parameter | Value | Description |
|---|---|---|
| `soc_penalty_coef` | 5663.6 | SOC center deviation penalty α |
| `h2_penalty_coef` | 5663.6 | H2 center deviation penalty α |
| `soc_penalty_targets` | [0.45, 0.45, 0.45, 0.45] | Per-agent SOC targets |
| `h2_penalty_targets` | [0.40, 0.40, 0.50, 0.60] | Per-agent H2 targets |
| `penalty_deadband` | 0.08 | Default deadband half-width |
| `consumer_h2_deadband` | 0.3 | Wider H2 deadband for consumer agents |
| `low_inventory_penalty_coef` | 1000.0 | Low inventory continuous penalty |
| `h2_low_threshold` | 0.15 | H2 low inventory trigger |
| `terminal_h2_floor_penalty_coef` | 50000.0 | Terminal H2 floor penalty |
| `terminal_h2_floor_threshold` | 0.20 | Terminal H2 floor trigger |
| `terminal_h2_floor_agent_indices` | [2, 3] | Only penalize consumers |
| `stepwise_h2_floor_penalty_coef` | 5000.0 | Stepwise consumer H2 floor coefficient |

### Profile Source

| Parameter | Value | Description |
|---|---|---|
| `profile_source` | `"italian"` | Use real Italian data |
| `italian_data_path` | `/root/autodl-tmp/HyperMARL-main/envs/Italian_data.csv` | Path to Italian CSV |
| `italian_split_enable` | `True` | Use a fixed train/test day split |
| `italian_split_strategy` | `"fixed_random"` | Build split from `italian_split_seed` and `italian_split_train_ratio` |
| `italian_split_name` | `"train"` | Default split sampled during training |
| `italian_split_seed` | 42 | Fixed seed used once to build train/test day pools |
| `italian_split_train_ratio` | 0.70 | Fraction of complete days assigned to train |
| `derive_heat_from_electric` | `True` | Derive heat from electric profiles |
| `derived_heat_base_ratio` | 0.15 | Constant base heat fraction |
| `derived_heat_variable_ratio` | 0.45 | Load-correlated heat fraction |

---

## 13. Quick Start

### Direct Environment Usage

```python
import numpy as np
from envs.microgrid.microgrid_env import MicrogridEnv

env = MicrogridEnv()
env.seed(42)
obs_list = env.reset()        # list of 4 arrays; shape is usually (22,) by default

for t in range(24):
    actions = np.zeros((4, 5), dtype=np.float32)
    obs_list, rew_list, done_list, info_list = env.step(actions)
    info = info_list[0]       # all agents share the same info

    print(f"t={t:2d}  reward={rew_list[0][0]:+.3f}"
          f"  C_grid={info['C_grid']:+.1f}"
          f"  C_h2={info['C_h2']:+.1f}"
          f"  penalty={info['penalty_total']:+.1f}"
          f"  h2_traded={info['h2_market_traded']:.1f}")

    if done_list[0]:
        break
```

### Using the Gymnasium Wrapper

```python
from envs.microgrid.microgrid_continuous_env import MicrogridContinuousEnv

env = MicrogridContinuousEnv()
env.seed(42)
obs = env.reset()             # np.ndarray shape (4, env.signal_obs_dim)

actions = np.zeros((4, 5), dtype=np.float32)
obs, rewards, dones, infos = env.step(actions)
# obs: (4, env.signal_obs_dim), rewards: (4, 1), dones: (4,), infos: list of 4 dicts
```

### Important: `step()` Return Format

`MicrogridEnv.step()` returns a **list** `[obs, rew, done, info]`, not a tuple. Unpack carefully:

```python
result = env.step(actions)
obs_list, rew_list, done_list, info_list = result
```

---

## 14. Training with HyperMARL

### Smoke Test (960 steps, ~10 seconds)

```bash
cd /root/autodl-tmp/HyperMARL-main
python baselines/IPPO/ippo_ff_shared_weights_mlp_hypernets.py \
    --config-name=ippo_ff_shared_weights_mlp_hypernets_microgrid \
    TOTAL_TIMESTEPS=960 CHECKPOINT=False WANDB_MODE=offline \
    ENT_COEF=0.01 LOG_STD_INIT=-1.0 ALG="smoke-test"
```

### Full Training (200k steps, ~5 minutes on GPU)

```bash
python baselines/IPPO/ippo_ff_shared_weights_mlp_hypernets.py \
    --config-name=ippo_ff_shared_weights_mlp_hypernets_microgrid \
    TOTAL_TIMESTEPS=200000 CHECKPOINT=True WANDB_MODE=offline \
    ENT_COEF=0.01 LOG_STD_INIT=-1.0 ALG="HyperMARL-MLP-experiment"
```

### Transfer Learning (from checkpoint)

```bash
python baselines/IPPO/ippo_ff_shared_weights_mlp_hypernets.py \
    --config-name=ippo_ff_shared_weights_mlp_hypernets_microgrid \
    TOTAL_TIMESTEPS=200000 CHECKPOINT=True WANDB_MODE=offline \
    ALG="transfer-experiment" \
    +CHECKPOINT_LOAD_DIR="/path/to/checkpoint_dir"
```

Checkpoints are saved under `wandb/wandb/offline-run-<ID>/files/models/`.

---

## 15. Deterministic Rollout & Evaluation

Post-training rollout, plotting, and diagnostic scripts are not kept in the
training-code tree. This repository focuses on the microgrid environment and
training entry points; generate evaluation scripts separately from saved
checkpoints when needed.

### Key Diagnostic Fields

| Field | Description |
|---|---|
| `episode_return_mean` | Average return across agents |
| `total_cost` | Sum of all costs over 24 steps |
| `C_grid` / `C_h2` | Cost components |
| `penalty_total` | Sum of all penalty terms |
| `penalty_soc` / `penalty_h2` | Center deviation penalties |
| `penalty_low_inventory` | Low H2 continuous penalty |
| `penalty_terminal_h2_floor` | Terminal H2 floor penalty (only non-zero at t=23) |
| `h2_market_traded` | Total H2 traded on internal CDA |
| `elec_market_traded` | Total electricity traded on internal CDA |
| `h2_delivery_overflow` | Delayed H2 delivered beyond buyer tank capacity and sold externally |
| `pending_h2_energy_total` | Total undelivered pending H2 at the current step |

### Per-Step Vectors in `info`

| Key | Type | Description |
|---|---|---|
| `p_grid` | `[4]` | Grid exchange per agent (kW; positive=buy) |
| `e_h2_ext` | `[4]` | External H2 exchange per agent (kWh_H2; positive=buy) |
| `p_el` | `[4]` | Electrolyser power per agent (kW) |
| `p_bat` | `[4]` | Battery power per agent (kW; positive=charge) |
| `p_ht` | `[4]` | H2 tank power per agent (kW_H2; positive=charge) |
| `net_electric_demand` | `[4]` | Net electric demand (kW; positive=deficit) |
| `net_h2_demand` | `[4]` | Net H2 demand (kWh_H2; positive=deficit) |
| `h2_order_quantity_raw` | `[4]` | H2 order quantity before cap-aware clipping |
| `h2_order_quantity` | `[4]` | H2 order quantity actually submitted to CDA |
| `h2_buy_future_headroom` | `[4]` | Future H2 tank headroom used for buy clipping |
| `h2_buy_clip_amount` | `[4]` | H2 buy quantity removed by cap-aware clipping |
| `h2_buy_horizon_clip_amount` | `[4]` | H2 buy quantity removed because it would arrive after episode end |
| `h2_delivery_reserved_energy` | `[4]` | Pending H2 energy reserved against local H2 tank charging |
| `h2_delivery_reservation_margin` | `[4]` | Remaining tank headroom after direct delivery reservation |
| `h2_delivery_reservation_charge_clip` | `[4]` | Local `p_ht > 0` charge energy clipped to keep delivery headroom |
| `pending_h2_total_norm` | `[4]` | Per-agent total pending H2 normalized by tank energy capacity |
| `pending_adjusted_h2_headroom_norm` | `[4]` | Per-agent future headroom after pending deliveries |

---

## 16. Changelog

| Version | Date | Key Changes | obs_dim | action_dim |
|---|---|---|---|---|
| Round 1 | — | Baseline, 13-dim obs, α=500, symmetric H2 price | 13 | 6 |
| Round 2 | — | Added `(t/T)²` penalty shape | 13 | 6 |
| Round 3 | — | Time-varying target curve, α=150 | 13 | 6 |
| Round 4 | — | Look-ahead TOU obs (17-dim), α=60, λ_sell→0.40 | 17 | 6 |
| Round 5 | — | Soft terminal `α/(T-t)`, α=2500 | 17 | 6 |
| Round 6 | 2026-04-23 | CDA accounting fix, pending obs (18-dim), λ_sell→0.20, consumer deadband 0.3, schedule [2,6,10,14,18] | 18 | 6 |
| Curriculum 1-a | 2026-04-28 | α→100, simplified H2 market (no schedule/lag), action_dim→5 (removed h2_thermal_share) | 16 | 5 |
| Curriculum 1-b | 2026-04-29 | α→250, low inventory penalty (coef=1000, h2_threshold=0.15), λ_sell→0.50 | 16 | 5 |
| No-arbitrage | 2026-04-30 | Battery/H2 tank anti-arbitrage discharge limits, device switch costs (el_start=50, bat_switch=30, ramps) | 16 | 5 |
| **Italian data** | **2026-05-02** | Italian CSV profiles (PV/wind/load_e), derived heat from electric (base=0.15, var=0.45), α→5663.6, consumer terminal H2 floor (threshold=0.20, coef=50000, agents=[2,3]) | **16** | **5** |
| GasNet linear multi-node | 2026-05-02 | Local gas pressure dynamics, local H2 external prices, gas price/pressure observations | 18 | 5 |
| H2Delay-4h-PendingObs | 2026-05-02 | Fixed 4h H2 delivery queue, 4 pending buckets, pending total/headroom observations | 24 with GasNet | 5 |
| H2Delay-4h-CapAwareBuyHorizon-PendingObs | 2026-05-03 | Adds future-headroom and episode-horizon clipping to H2 CDA buy order quantity and logs raw/final/clip diagnostics | 24 with GasNet | 5 |
| H2Delay-4h-DirectReserve-PendingObs | 2026-05-03 | Reserves pending H2 delivery headroom by clipping local `p_ht > 0` tank charging before arrivals | 24 with GasNet | 5 |
