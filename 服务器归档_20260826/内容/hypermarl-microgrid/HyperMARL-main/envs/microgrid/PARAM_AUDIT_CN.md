# 微电网环境参数全面审查报告

**审查范围**: `config.py` 默认值 + `scripts/microgrid_experiment_overrides.py` 实验覆盖  
**聚焦实验**: `run_ctde_40k.sh` (MAPPO + MLP hypernets, CTDE, FullCDA-ReserveDemand-Price30)  
**症状**: 训练不稳定 / 动作饱和;HyperMARL 相比 MAPPO-IA baseline 无优势  
**审查日期**: 2026-06-23

---

## 问题清单 R1–R5

---

### R1 氢价单位写错 → 奖励量级爆炸(已修复)

**位置**: `config.py:92`

**问题描述**  
`"h2_price_unit": "kwh"` 意味着所有氢价参数单位是 **元/kWh_H2**。  
实验覆盖里 `lambda_h2_buy=30.0`、`lambda_h2_sell=3.0` 被当成 30/3 元/kWh_H2 使用。

但正确的物理含义是 **30 元/kg**、**3 元/kg** 的氢气价格。  
`microgrid_env.py:557-575` 已有 `h2_price_unit="kg"` 分支,会自动将 /kg 价格除以 `LHV_H2=33.33` 转为内部 /kWh_H2,但因默认值是 `"kwh"` 而从未触发。

**量级估算(修复前 vs 修复后)**

| 参数 | 修复前 ("kwh") | 修复后 ("kg") |
|---|---|---|
| `lambda_h2_buy` 内部值 | **30.0** 元/kWh_H2 | 30/33.33 = **0.900** 元/kWh_H2 |
| `lambda_h2_sell` 内部值 | **3.0** 元/kWh_H2 | 3/33.33 = **0.090** 元/kWh_H2 |
| A2 峰值 H2 缺口(~2000 kWh_H2/step) | 成本 **60,000** 元/step | 成本 **1,800** 元/step |
| 24 步 episode 平均外购成本(A2+A3 合计) | **~500,000** 元 | **~15,000** 元 |
| `reward_scale=200` 后 episode return | **~−2,500** | **~−75** |

修复前的 return 方差极大(量级 >1000),value network 目标爆炸,PPO clip 无法正常工作,训练发散。

**修复方式**: `config.py:92` `"h2_price_unit": "kwh"` → `"h2_price_unit": "kg"` ✅

---

### R2 惩罚项完全未进入 reward

**位置**: `microgrid_env.py:1629-1633`(旧版),修复后约第 1632 行

**问题描述**  
代码:
```python
total_cost = base_cost
reward = -base_cost / self.reward_scale
```

config 中精心调过的全套惩罚机制——

| 参数 | 值 |
|---|---|
| `soc_penalty_coef` | 5663 |
| `h2_penalty_coef` | 5663 |
| `terminal_h2_floor_penalty_coef` | 50,000 |
| `stepwise_h2_floor_penalty_coef` | 5,000 |
| `low_inventory_penalty_coef` | 1,000 |

——全部只写进 `info` 字典,不参与优化目标。agent 没有任何动机维持 SOC 和储氢库存,
最优策略是把库存放空以降低当步电解槽/储氢消耗,换取短期成本下降——这正是历史注释里反复出现的"全塌到 0.1"症状。

**建议**  
加 `penalty_in_reward_enable` 开关(默认 False,已加入 config 与 env),用 E3 实验验证。
推荐系数: `soc_penalty_coef=350`, `h2_penalty_coef=350`(量级 sanity check 见下方 E3 说明)。

**系数 sanity check**  
修复 R1 后:
- 24 步 episode 基础运行成本(估算) ≈ 400–2,000 元
- SOC 惩罚上界(worst case, 所有 agent SOC 偏离 0.5 且距终端 1 步): `α × 1/(T-1) × 4 × 0.5² ≈ 350 × (1/23) × 4 × 0.25 ≈ 15 元`
- 24 步累积惩罚上界 ≈ 120 元(约占 base_cost 的 6–30%)
- 结论: α=350 不会压过 base_cost 梯度,适合引导行为而不扭曲主目标。

---

### R3 报价动作维在 shared reward 下无梯度 → HyperMARL 无法体现优势

**位置**: 动作空间 a2(电力 CDA 报价)、a3(氢 CDA 报价)

**问题描述**  
4 个 agent 共享单一 reward:
```
reward = -base_cost / reward_scale
```
内部 CDA 成交: 买方付款 + 卖方收款 = 净转账。在 shared reward 下:
```
sum(cda_paid) == sum(cda_received)  →  cda_transfer = 0
```
(代码 `microgrid_env.py:1423` 已有注释 `# shared reward 下 = 0`)

因此 a2/a3 的价格决策对 shared reward 完全不产生梯度,这两维动作会在训练中随机游走直到饱和(贴 ±1)。HyperMARL 的核心优势是"为不同 agent 生成异质化参数",恰好在这两个最应体现差异化的维度上没有信号。

**建议(两个方向)**  
1. E4: 给内部成交量加 bonus 并计入 reward (`market_bonus_in_reward_enable=True`),让报价动作影响成交量 → 影响 reward。
2. 未来扩展: 将 shared reward 改为 local reward + penalty sharing,让每个 agent 的报价策略直接影响其自身支付/收入。

---

### R4 PPO 超参数设置偏不稳定

**位置**: `baselines/MAPPO/config/mappo_ff_shared_weights_mlp_hypernets_microgrid.yaml`

**问题一: `NUM_MINIBATCHES=1`(无 mini-batch)**  
每次 PPO 更新用完整 rollout buffer(4 envs × 24 steps × 4 agents = 384 个转换)做 8 个 epoch。  
等效于以完整数据反复更新 8 次,数据利用率高但方差大、容易过拟合当前批次——在高噪声环境(如修复前 return 方差 >1000)中尤为致命。  
建议: `NUM_MINIBATCHES=4`(每 epoch 用 96 个转换,方差更小)。

**问题二: `VF_COEF=1.0`(value loss 权重偏高)**  
在 return 量级不稳定时,value loss 主导了总 loss,策略梯度被淹没。  
建议: `VF_COEF=0.5`。

**问题三: `ANNEAL_LR=False`**  
固定 LR=3e-4 到训练结束,后期无法收敛到更精细的策略。  
建议: `ANNEAL_LR=True`。

**问题四: tanh log-prob 未做 Jacobian 修正**  
训练时 `env_action=tanh(action)`,但 `log_prob` 是用未经 tanh 的高斯分布计算的(`_select_action` 第 531–537 行)。  
正确的连续动作 PPO 应在 `log_prob` 中减去 `log(1 - tanh(a)²)` 的 sum。  
修正后 agent 会知道接近 ±1 的动作概率密度实际很低,从而更主动地避开饱和——这是动作饱和的根源之一。  
(此修正涉及修改 `mappo.py` 中 PPO loss 的计算,列为 stretch 目标。)

---

### R5 文档与实现不一致

**位置**: `README_CN.md` 第 52–73 行 vs `config.py` 与 `_get_obs`

**问题**: README 声称 `obs_dim=16`(含 soc_dev/h2_dev/time_pressure 三维),但实际:
- `config.py` `"obs_dim": 13`
- `_get_obs` 只构造 13 维(pv/wt/load_e/load_h/soc/h2_ratio/elec_price/h2_price/sin_t/cos_t/tou_buy/p_grid/e_h2_ext)
- 开 `h2_pending_obs_enable=True` + `h2_pending_summary_obs_enable=True` 后为 13+4+2=**19 维**

此不一致不影响训练(env 和 network 都用 `obs_dim` 属性,不用文档数值),但调参时会误导。

---

## 修复清单

| ID | 问题 | 修复方式 | 状态 |
|---|---|---|---|
| R1 | h2_price_unit 单位错误 | `config.py` `"kwh"` → `"kg"` | ✅ 已修复 |
| R2 | 惩罚项未进 reward | 加 `penalty_in_reward_enable` 开关 | ✅ 开关已加,E3 实验验证 |
| R3 | 报价维无梯度 | 加 `market_bonus_in_reward_enable` 开关 | ✅ 开关已加,E4 实验验证 |
| R4 | PPO 超参偏不稳定 | E5 实验验证最优组合 | 待 E5 |
| R5 | 文档不一致 | 修订 README_CN.md | 待办 |

---

## 消融实验设计 (E0–E5)

所有实验统一: `SEED=30`, `TOTAL_TIMESTEPS=120000`, `NUM_ENVS=10`, 进程级并行。

| 实验 | 与 E0 的差异 | 验证目标 |
|---|---|---|
| **E0** | `h2_price_unit=kwh`(故意保留 bug) | 复现坏 baseline,return 方差极大 |
| **E1** | `h2_price_unit=kg`(默认值修复) | R1 修复后 return 方差大幅下降、收敛 |
| **E2** | +`load_h_peak=[750,600,1500,1800]`(consumer 降半) | 排除物理不可行,验证供需平衡对成交的影响 |
| **E3** | +`penalty_in_reward_enable=true,soc_penalty_coef=350,h2_penalty_coef=350` | 库存维持行为改善(SOC/H2 终端值更接近目标) |
| **E4** | +`h2_internal_trade_bonus_enable=true,h2_internal_trade_bonus_coef=0.05,market_bonus_in_reward_enable=true` | 内部成交量上升、HyperMARL vs baseline gap 拉大 |
| **E5** | +`ANNEAL_LR=True,NUM_MINIBATCHES=4,VF_COEF=0.5` | 训练曲线平滑度提升、后期精细收敛 |

---

## E3 系数推导

修复 R1 后,episode 基础成本约 300–3000 元。

取 `α=350`:
- 终端步 SOC penalty upper bound: `350 × (1/1) × 4 × max(0, 0.4-0.08)² ≈ 350 × 4 × 0.1024 ≈ 143 元`
- 24 步累积上界(平均 1/(T-t)):约 350 × sum(1/(24-t), t=0..23) × 4 × 0.1024 ≈ 350 × 3.18 × 0.41 ≈ **456 元**
- base_cost 占比(低端): 456/300 ≈ 152% → 偏高,可能压过 base 梯度
- base_cost 占比(高端): 456/3000 ≈ 15% → 合适

建议初始值 `soc_penalty_coef=h2_penalty_coef=200`,若库存维持不足再提到 350。

## E4 bonus 系数推导

内部 H2 成交量期望: 参考 README §11 中 100k 实验 ~3700 kWh_H2/episode。

取 `coef=0.05` 元/kWh_H2:
- bonus ≈ 0.05 × 3700 = **185 元/episode**
- base_cost 占比(中段) ≈ 185/1500 ≈ 12% → 合适,能给报价维提供可见梯度
- `reward_scale=200` 后 return 贡献: 185/200 = **+0.925**

---

*本文档由代码审查自动生成。*
