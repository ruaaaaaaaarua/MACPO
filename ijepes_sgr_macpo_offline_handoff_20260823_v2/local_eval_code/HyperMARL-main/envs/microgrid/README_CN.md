# 微电网多智能体强化学习环境说明

本文档说明 `/root/autodl-tmp/HyperMARL-main/envs/microgrid/` 中当前实际使用的微电网环境，以及它如何被 HyperMARL 训练脚本调用。

## 1. 总览

当前环境是一个 4-agent cooperative microgrid MARL 环境。每个 agent 都是一个完整微电网，包含本地可再生发电、电池、电解槽、储氢罐、电负荷和氢负荷。所有 agent 共享同一个全局 reward，目标是最小化一天 24 小时内的基础运行成本。

重要路径：

| 用途 | 路径 |
|---|---|
| 环境默认配置 | `/root/autodl-tmp/HyperMARL-main/envs/microgrid/config.py` |
| 环境核心逻辑 | `/root/autodl-tmp/HyperMARL-main/envs/microgrid/microgrid_env.py` |
| Gym/HyperMARL 包装 | `/root/autodl-tmp/HyperMARL-main/envs/microgrid/microgrid_continuous_env.py` |
| CDA 撮合引擎 | `/root/autodl-tmp/HyperMARL-main/envs/microgrid/cda_market.py` |
| 外生曲线生成 | `/root/autodl-tmp/HyperMARL-main/envs/microgrid/data_generator.py` |
| HyperMARL wrapper | `/root/autodl-tmp/HyperMARL-main/baselines/utils/microgrid_vec_env.py` |
| HyperMARL rollout | 训练后 rollout 脚本不再随训练源码保留 |

HyperMARL 训练并没有复制一份独立环境，而是通过 wrapper 调用这里的 `MicrogridEnv`。因此，训练和环境诊断都应以本目录代码为准。

## 2. Agent 与设备

共有 4 个 agent，配置中 `agent_types = ["mg", "mg", "mg", "mg"]`。通常将 A0/A1 称为 producer，A2/A3 称为 consumer，但所有 agent 都拥有完整设备，只是容量不同。

| Agent | 角色 | PV kW | WT kW | 电池 kWh/kW | 电解槽 kW/效率 | 储氢 kg/kW_H2 | 电负荷峰值 kW | 热负荷峰值 kW_th |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| A0 | producer | 5000 | 1000 | 5000 / 2000 | 2000 / 0.70 | 500 / 2000 | 2250 | 750 |
| A1 | producer | 1000 | 4000 | 3000 / 1200 | 3000 / 0.65 | 500 / 2000 | 1875 | 600 |
| A2 | consumer | 500 | 3000 | 4000 / 1500 | 500 / 0.60 | 300 / 1200 | 3000 | 1500 |
| A3 | consumer | 2000 | 500 | 2000 / 800 | 800 / 0.65 | 300 / 1500 | 2625 | 1875 |

热负荷当前固定由氢锅炉承担，`h2_thermal_share` 在环境内固定为 1.0，不再作为动作维度。

## 3. 动作空间

基础 `action_dim = 5`；启用 `h2_learnable_rolling_order_enable` 后为 6。每个动作维度由策略输出为 `[-1, 1]`，环境再映射为物理量：

| 维度 | 名称 | 含义 | 物理映射 |
|---|---|---|---|
| a0 | `P_el` | 电解槽功率 | `(a0 + 1) / 2 * el_cap` |
| a1 | `P_bat` | 电池充放电功率 | `a1 * bat_power`，正值充电、负值放电 |
| a2 | `elec_bid_price` | 电力 CDA 报价 | 映射到当前 TOU `[sell, buy]` |
| a3 | `h2_bid_price` | H2 CDA 报价 | 固定映射到内部 `[h2_price_min, h2_price_max]` |
| a4 | `P_ht` | 储氢罐充放功率 | `a4 * h2_tank_power`，正值充氢、负值放氢 |
| a5 | `h2_buy_quantity` | 完整的前向 H2 CDA 买单量 | `(a5 + 1) / 2 * qmax_i` |

其中 `qmax_i = load_h_peak_i / boiler_eff * dt`，即一个静态热负荷峰值小时折算的 H2 能量。a5 是完整买单量，不会叠加当前净 H2 缺口或启发式 reserve；它也不是热负荷比例。物理富余的卖方忽略 a5。

## 4. 观测空间

`config.py` 中基础 `obs_dim = 16`。运行时观测维度会根据 gas price/pressure 和 pending H2 配置扩展。

基础 16 维：

| 维度 | 名称 | 含义 |
|---|---|---|
| o0 | `pv_ratio` | 当前 PV 输出 / PV 容量 |
| o1 | `wt_ratio` | 当前风电输出 / WT 容量 |
| o2 | `load_e_ratio` | 当前电负荷 / 峰值电负荷 |
| o3 | `load_h_ratio` | 当前热负荷 / 峰值热负荷 |
| o4 | `bat_soc` | 电池 SOC |
| o5 | `h2_level_ratio` | 储氢罐填充率 |
| o6 | `last_elec_price_norm` | 上一步电力 CDA 出清价归一化 |
| o7 | `last_h2_price_norm` | 上一步 H2 CDA 出清价归一化 |
| o8/o9 | `sin_t` / `cos_t` | 时间编码 |
| o10 | `tou_buy_norm` | 当前外部购电价归一化 |
| o11 | `last_p_grid_norm` | 上一步外部电网交互归一化 |
| o12 | `last_e_h2_ext_norm` | 上一步外部 H2 交互归一化 |
| o13 | `soc_dev` | SOC 相对目标值偏差 |
| o14 | `h2_dev` | H2 ratio 相对目标值偏差 |
| o15 | `time_pressure` | 临近终端的紧迫度 |

可选扩展：

- `gas_price_obs_enable=True` 时加入 local H2 external buy price 观测。
- `gas_pressure_obs_enable=True` 时加入 gas pressure 观测。
- `h2_pending_obs_enable=True` 时加入未来 1-4 小时 pending H2 到货桶。
- `h2_pending_summary_obs_enable=True` 时加入 total pending H2 和 pending-adjusted headroom。

因此常见维度为：

| 配置 | 观测维度 |
|---|---:|
| 基础 | 16 |
| 基础 + pending 4 桶 + summary 2 维 | 22 |
| GasNet price/pressure + pending/summary | 24 |

## 5. 外生数据和 Italian split

当前默认 `profile_source = "italian"`，数据来自：

`/root/autodl-tmp/HyperMARL-main/envs/Italian_data.csv`

`data_generator.py` 会从 Italian 数据中抽取 24 小时日曲线，并按 agent 容量缩放 PV、风电和电负荷。`derive_heat_from_electric=True` 时，热负荷从电负荷形状派生：

```text
load_h = load_h_peak * (base_ratio + variable_ratio * normalized_load_shape) * noise
```

训练和 rollout 脚本可通过 `MICROGRID_CONFIG_OVERRIDES` 覆盖：

- `italian_split_enable:true`
- `italian_split_name:train|validation|test`
- `italian_day_indices:[...]`

因此 README 中的默认配置不等于某次训练的完整环境；必须同时查看训练 shell 脚本里的覆盖项。

## 6. 电力市场 CDA

每一步都运行电力 continuous double auction。

电力净需求：

```text
net_electric_demand = load_e + P_el + battery_charge - PV - WT - battery_discharge
```

规则：

- `net_electric_demand > 0`：生成买单。
- `net_electric_demand < 0`：生成卖单。
- 数量为 `abs(net_electric_demand) * dt`。
- 报价来自动作 `a2` 映射后的 `elec_bid_price`。
- 未成交买单走外部电网 buy price。
- 未成交卖单走外部电网 sell price。

`cda_market.py` 使用价格时间优先：最高买价和最低卖价可交叉时成交，支持部分成交。

## 7. H2 市场 CDA、延迟交割和 DirectReserve

H2 净需求：

```text
net_h2_demand = e_h2_load + P_ht * dt - e_h2_prod
```

其中：

- `e_h2_load = heat_load / boiler_eff`
- `e_h2_prod = P_el * el_eff * dt`
- `P_ht > 0` 表示本地充罐，会增加当前 H2 需求。
- `P_ht < 0` 表示本地放罐，会减少当前 H2 需求。

H2 CDA 订单规则：

- 启用 action-controlled ordering 时，`net_h2_demand >= 0` 的完整买单量只由 a5 决定：-1/0/+1 分别对应 0/0.5/1.0 倍 `qmax_i`。
- 当前净缺口和启发式 reserve 不会自动加到 a5 数量上；action ordering 与 heuristic buyer reservation 同时启用会在构造环境时抛出 `ValueError`。
- `net_h2_demand < 0` 时忽略 a5，卖单量严格为 `-net_h2_demand`。
- 净需求恰好为 0 时仍可由 a5 提交买单。
- 报价来自动作 a3，始终映射到内部 `[h2_price_min, h2_price_max]`。

当前主线支持 4 小时延迟交割：

- `h2_market_lag_enable=True`
- `h2_delivery_lag=4`
- 成交后写入 `pending_h2_deliveries`。
- 买方不是当步入罐，而是在 `t + h2_delivery_lag` 到货。
- 到货时调用 `_deliver_pending_h2()`，能入罐的部分计入 `delivered_h2_energy`，超出容量的部分计入 overflow。
- no-lag 时，成交买量先抵消当前正净需求，超出部分立即入买方储氢罐；它不会变成负的外部 H2 交换或转售。

容量感知买单限幅：

- `h2_cap_aware_buy_enable=True` 时，lagged 买单按未来 headroom 减去 pending H2 限幅；no-lag 买单按当前需求加储罐 headroom 限幅。
- 到货必须满足 `t + h2_delivery_lag < T`；因此 `>= T` 的买单会被 horizon clipping 到 0。

DirectReserve：

- `h2_delivery_reservation_enable=True` 时，在途 H2 会预留买方储氢罐 headroom。
- 这样可以避免买方一边已有 pending H2，一边又用本地 `P_ht > 0` 把储氢罐充满，导致未来到货 overflow。
- 默认配置中该项可能为 False，但当前重要训练/rollout 常通过脚本覆盖为 True。

## 8. 成本和 reward

所有 agent 共享 reward：

```text
reward = -(C_grid + C_h2 + external_h2_dependency_penalty) / reward_scale
```

当前默认 `reward_scale = 200.0`。

成本：

```text
base_cost = C_grid + C_h2
total_cost = base_cost + external_h2_dependency_penalty
```

环境内部所有 H2 金额统一使用 yuan/kWh-H2；Group ABC 的 trial metadata 可以乘以 `LHV_H2` 后展示 yuan/kg，但传入环境的值仍是 yuan/kWh-H2。库存、终端、动作正则和 bonus 只保留原始诊断，不进入训练 reward。

主要项：

| 字段 | 含义 |
|---|---|
| `C_grid` | 外部电网买卖电成本，买电为正成本、卖电为负成本 |
| `C_h2` | 外部 H2 买卖成本，买 H2 为正成本、卖 H2 为负成本 |
| `penalty_total` | 诊断指标：SOC/H2 目标、低库存、终端 floor、动作正则等惩罚合计，不进入 reward |
| `market_bonus` | 诊断指标：市场相关奖励项合计，不进入 reward |

内部 CDA 转账在 shared reward 下买方支付和卖方收入相互抵消，所以不会直接改变 shared reward 的净成本。H2 内部成交奖励仍可在 `info` 中记录，但不再进入 reward：

| 配置 | 默认 | 含义 |
|---|---:|---|
| `h2_internal_trade_bonus_enable` | False | 是否启用内部 H2 成交量 bonus |
| `h2_internal_trade_bonus_coef` | 0.0 | 每 kWh_H2 内部成交给多少负成本 credit |

计算方式：

```text
h2_internal_trade_bonus = coef * h2_market_traded
```

该项仅用于诊断和机制分析，不改变当前训练目标。

## 9. 物理边界和反套利约束

电池放电不能直接创造外部卖电收益：

```text
if P_bat < 0:
    max_discharge <= local electric deficit
```

储氢罐放氢不能直接创造外部卖氢收益：

```text
if P_ht < 0:
    max_discharge <= local H2 deficit
```

这两个约束允许自用削峰，但禁止直接从初始库存中套取外部市场卖出收益。

## 10. 关键 rollout 诊断字段

环境 `info` 中会提供每步 action、observation 和市场诊断字段。H2 市场相关重点字段：

| 字段 | 含义 |
|---|---|
| `h2_market_traded` | 当步内部 H2 CDA 成交量 |
| `h2_buy_order_quantity_total` | 当步最终进入 CDA 的 H2 买单总量 |
| `h2_sell_order_quantity_total` | 当步最终进入 CDA 的 H2 卖单总量 |
| `h2_best_buy_price` | 当步最高 H2 买价 |
| `h2_best_sell_price` | 当步最低 H2 卖价 |
| `h2_bid_cross` | 是否存在 `best_buy >= best_sell` |
| `h2_cross_matchable_quantity` | 报价交叉时买卖量上界 `min(buy_qty, sell_qty)` |
| `h2_order_qmax` | 每个 agent 的静态峰值小时 H2 买单上限 |
| `h2_action_requested_buy_quantity` | a5 映射得到、clip 前的完整请求买量 |
| `h2_action_effective_buy_quantity` | cap/horizon clip 后的有效买量 |
| `h2_order_source` | `action_buy`、`physical_surplus` 或 `none` |
| `h2_order_quantity_raw` | cap-aware clipping 前订单量 |
| `h2_order_quantity` | 实际提交到 CDA 的订单量 |
| `h2_learnable_rolling_order_extra` | deprecated 兼容别名；等于 a5 请求买量，不再表示 extra |
| `h2_buy_clip_amount` | 因未来 headroom 不足被裁掉的买单量 |
| `h2_buy_horizon_clip_amount` | 因到货超过 episode 边界被裁掉的买单量 |
| `delivered_h2_energy` | 当步实际入罐的 immediate 或 delayed H2 |
| `pending_h2_total` | 每个 agent 当前在途 H2 总量 |
| `h2_delivery_overflow` | 到货时因容量不足溢出的 H2 |
| `h2_internal_trade_bonus` | 内部 H2 成交奖励，默认关闭时为 0 |

用于 100k/500k 对比的旧训练后分析脚本已从训练源码中移除。

默认输出：

- `h2_market_root_cause_summary.csv`
- `h2_market_root_cause_steps.csv`
- `H2_MARKET_ROOT_CAUSE_REPORT.md`

## 11. 当前 H2 市场已知问题

近期 500k NoGasDelay 训练中，H2 内部 CDA 活动远低于 100k。根据 rollout JSON 的 root-cause 诊断，问题不是完全没有买方，也不是 CDA 撮合代码不能成交，而是 500k 策略中的卖方 H2 富余几乎消失。

典型 split_test 现象：

- 100k main/fallback：卖单总量约 15,000 kWh_H2，内部成交约 3,700-3,800 kWh_H2。
- 500k main/fallback：买单仍有 17,000-24,000 kWh_H2，但卖单只有约 537 / 152 kWh_H2，内部成交接近 0。

因此 `delivery_ratio=0` 的直接原因是内部 H2 CDA 几乎没有成交；delayed delivery 只能来自内部成交后的 pending delivery，没有成交就没有到货。

修复方向应优先恢复 producer/seller 侧可卖氢富余，或给 shared reward 下的内部 H2 成交一个可见激励，而不是只推高 buyer 报价。

## 12. 训练和 rollout 注意事项

训练时常通过环境变量覆盖配置，例如：

```bash
MICROGRID_CONFIG_OVERRIDES='{italian_split_enable:true,italian_split_name:train,h2_delivery_reservation_enable:true,...}'
```

因此判断“当前训练环境”必须同时看：

1. `config.py` 默认值。
2. 训练 shell 脚本中的 `MICROGRID_CONFIG_OVERRIDES`。
3. rollout shell 脚本中的 `MICROGRID_CONFIG_OVERRIDES`。
4. 自行编写的 rollout/evaluation 脚本是否读取了对应诊断字段。

常用训练入口：

`/root/autodl-tmp/HyperMARL-main/baselines/IPPO/ippo_ff_shared_weights_mlp_hypernets.py`

deterministic rollout 脚本不再随训练源码保留；如需评估 checkpoint，请单独编写 evaluation 脚本。

## 13. 快速定位建议

如果 H2 market 低成交，按以下顺序查：

1. `h2_buy_order_quantity_total` 是否很大。
2. `h2_sell_order_quantity_total` 是否接近 0。
3. 如果买卖双方都有量，再看 `h2_best_buy_price >= h2_best_sell_price` 是否成立。
4. 如果报价交叉但成交仍为 0，看 `h2_cross_matchable_quantity` 是否接近 0。
5. 如果成交恢复但到货差，看 `pending_h2_total`、`delivered_h2_energy`、`h2_delivery_overflow` 和 `h2_delivery_reservation_charge_clip`。
6. 如果成交恢复但 reward 变差，看 `C_h2`、`penalty_h2`、A2/A3 H2 floor 和 action saturation。
