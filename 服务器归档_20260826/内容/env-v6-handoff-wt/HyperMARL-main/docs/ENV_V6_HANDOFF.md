# HyperMARL 微电网项目交接文档

> 用途：让一个新的 Codex/AI 对话在几分钟内理解项目正在研究什么、已经做过什么、当前结果能说明什么，以及下一步应该怎么做。

更新时间：2026-07-26（Asia/Shanghai）

当前主线：Env-v6 Swiss MV、无跨 Agent 广播、shared-system GRU-MACPO
当前阶段：首轮单训练种子已经完成，下一步是多训练种子和跨网络验证

## 0. 新对话先记住的工作原则

- 主算法始终是现有 shared-system GRU-MACPO；没有实现 MASAL，也没有加入逐 Agent 顺序更新。
- 当前研究重点是“改环境、跑训练、看 Agent 是否形成合理策略”，不要把普通软件项目的复杂开发流程当成研究目标。
- Env-v2-sparse 和已有 Env-v3/v4/v5/v5.2 结果是历史对照，不能覆盖、重置或重新解释成 Env-v6 结果。
- Env-v6 使用 Swiss-PDGs 原始 MV 网络：不缩放 PCC、不缩放背景负荷、不修改阻抗。
- 训练前必须先确认物理工况可行；训练、性能或 smoke 门控失败时停止，不靠增加步数掩盖问题。
- 电压安全 cost 和经济 reward 分开记录。不要用混合 reward 代替经济、安全分项报告。
- 不要把“本轮预设验收通过”写成“已经获得全时段硬安全保证”。当前 MACPO 仍有一个评估日越限。

## 1. 项目到底在研究什么

环境是四个微电网 Agent 的 P2P/能源调度问题。每个 Agent 同时面对：

- 电解槽、电池、储氢和电力交易动作；
- 动态交通网络导致的氢气运输 ETA、拥堵和迟到订单；
- 微电网通过 PCC 接入配电网后，对节点电压和潮流的影响；
- 经济成本和电压安全约束之间的冲突。

研究主问题可以写成：

> 在交通与电网耦合、奖励存在延迟、且电压约束需要满足的多智能体环境中，GRU 记忆和 MACPO 是否能在不依赖显式 Agent 广播通信的情况下，学到兼顾经济性与配网安全的调度策略？

当前最可靠的阶段性答案是：

> 在选定的 Swiss MV 工况中，无通信 GRU-MACPO 比普通 GRU-MAPPO 显著降低电压越限，并比固定电压罚款 MAPPO 付出更少的经济代价；但它还没有在所有评估日达到零越限，因此结论是“安全性改善和合理折中”，不是“硬安全已解决”。

## 2. 两次老师讨论形成的决策链

### 2.1 7.19：从 STS/交通问题转向 GRU + 安全强化学习

原始讨论的核心证据和决策：

1. 交通网加入直达、绕路、拥堵和 ETA 后，决策不再只依赖当前瞬时状态；延迟订单和库存使历史信息重要。
2. STS/注意力式信用分配在稀疏奖励环境中没有比简单团队奖励均摊更好，反而干扰训练信号；因此冻结 STS 线，不再把它作为安全主线。
3. 用 GRU/LSTM 把历史状态、动作和运输信息编码进当前决策，以处理长期依赖和部分非马尔可夫性。
4. CTDE 是默认协同框架；如果各 Agent 都是独立 critic，才需要考虑通信来弥补信息差。
5. 电网要在环境内做 Newton-Raphson 潮流，节点电压安全区间为 `[0.95, 1.05] p.u.`。
6. 电压越限不是普通经济损失，而是独立安全 cost；允许动作先发生，再通过安全 cost 和约束优化纠偏。
7. 主安全算法选 MACPO，微网通过 PCC 接入配电网，电网拓扑和负荷量级必须匹配。
8. 订单迟到不再裁剪或退款：订单可以发生并付款，但氢不会凭空入罐，迟到和未送达要单独统计。

### 2.2 7.26：去广播、换 Swiss 数据、校准量级并加速 on-policy 训练

7.26 把研究范围进一步收窄：

1. 在 CTDE 中，集中式 critic 已经可以看到全局信息；之前的 intent 广播对策略影响很小，还可能引入隐私和语义混乱。因此 Env-v6 移除跨 Agent 即时/历史交易广播，保留本地物理观测、上一动作、pending、缺口和 ETA。
2. 电压安全与经济目标继续分开。电压 cost 用原始越限总和，训练 critic 使用 `raw_cost / 0.02`，因此内部安全预算固定为无量纲 `1.0`。
3. 作为对照保留固定惩罚 MAPPO，公式为 `reward_advantage - 1.0 * cost_advantage`；它不是主算法。
4. 原先的微网 PCC 功率与 IEEE-33 背景量级不匹配，继续缩放 PCC 不是最终方向。换用 Swiss-PDGs 原始 20 kV MV 网络，直接使用原始 `Pd/Qd/r/x` 和 `baseMVA=100`。
5. 训练采用 on-policy rollout，`2 个环境进程 × 24 小时 = 48 个 system transitions/update`，并把 actor、reward critic、cost critic 和采样合并为融合 JIT rollout kernel。
6. 参考论文可以借鉴安全 cost 分离、on-policy 小批量和并行思想，但不实现 MASAL，不修改 MACPO 更新器。

论文背景资料：

- [Network-constrained P2P trading: A safety-aware decentralized multi-agent reinforcement learning approach](https://github.com/aeonetos/Swiss-PDGs)
- [Swiss-PDGs 数据仓库](https://github.com/aeonetos/Swiss-PDGs)

## 3. 实验演进时间线

### Env-v2-sparse：历史 STS/稀疏奖励基线

- 24 小时中前 23 步不给经济 reward，末步给累计成本。
- 用来检验稀疏奖励和信用分配，结果显示 STS 没有优于简单奖励均摊。
- 该环境被冻结，不要为了 Env-v6 改它。

### Env-v3-safe：安全基础设施

- 逐步经济 reward 与独立 voltage cost；IEEE-33 AC 潮流每步计算。
- 全局唯一电压安全约束；MACPO、GRU-MAPPO 和 Lagrangian 基线。
- 早期发现：原始物理量级下策略经常持续越限，MACPO 长时间处于 cost-recovery。这首先是可行域问题，不应直接归因于算法失败。

### Env-v4：intent 两阶段与通信消融

- intent + 有限残差、full broadcast/no-broadcast、GRU 历史和本地 ETA/缺口。
- full broadcast 经济成本约 `2.449M`，no-broadcast 约 `2.787M`；full 的迟到订单更少。
- intent 与实际动作高度相关，说明动作约束生效；但两者都严重越限，不能说通信解决了安全问题。
- 关闭 GRU 历史后明显退化，支持保留时序记忆。

### Env-v5/v5.1/v5.2：供氢事实和 IEEE-33 可行域标定

- 增加本地 4/6/10 小时氢缺口、外部 ETA 等事实观测。
- 发现原始 PCC 负荷与 IEEE-33 背景负荷同量级，简单 PCC/background 缩放不能构成最终研究故事。
- v5.2 曾围绕 IEEE-33 的 `background=0.1、PCC=0.315` 做精细窗口门控；这条线保留作为历史记录，但被 7.26 的 Swiss 原生网络方案取代。

### Env-v6 Swiss：当前主线和最新结果

- 879 个 Swiss MV 网络完成静态筛查；前 30 个网络—PCC 组合完成三日动态门控。
- 选中 `347_1`，PCC 为 `[168, 147, 160, 183]`。
- 选网依据：原生潮流收敛、基准电压安全、变压器容量、1 MW/0.95 pf 电压灵敏度、PCC 电气分散性、名义/支撑控制差值。
- 三组训练：无通信 GRU-MAPPO、无通信固定惩罚 GRU-MAPPO、无通信 GRU-MACPO。
- 每组三个评估日使用 seeds `30/31/32`；训练本身是单 seed `30`。

## 4. Env-v6 的精确配置和语义

### 4.1 物理与电压

- 模型：`power_flow_model="swiss_mv"`
- 数据：`/root/autodl-tmp/datasets/Swiss-PDGs/grids/matpower_data/MV/347_1`
- baseMVA：`100`
- 电压上下限：`0.95/1.05 p.u.`
- PCC 注入比例：`1.0`
- 背景负荷比例：`1.0`
- 潮流失败 cost：`1.0`
- 原始 voltage cost：所有节点每小时 `max(0,0.95-V)+max(0,V-1.05)` 的总和
- 当前首轮只约束电压，不虚构线路容量约束。

### 4.2 Agent/critic

- Agent 数：4
- action dim：7
- actor：4 个独立 GRU，hidden size `128`
- actor 输入：本地 base observation `32` + 自己上一动作 `7` = `39`
- actor 不读取其他 Agent 即时交易、历史订单或 intent
- reward critic：集中式 GRU critic，读取四个 Agent 的拼接观测
- cost critic：集中式 GRU critic，读取同一全局观测
- 训练范式：CTDE；执行时 actor 本地、无显式通信

### 4.3 训练量纲

- 经济 reward scale：`3,151,704.8194007874 yuan`
- 训练 reward：环境原始经济 return 除以该 scale；日志同时保留 raw/normalized
- 训练 cost：`raw_voltage_cost / 0.02`
- MACPO 内部 budget：`1.0`，对应 raw budget `0.02`
- fixed penalty 系数：`1.0`
- MACPO KL 上限：`0.01`
- log-std 上限：前 300 updates 从 `-1.0` 退火到 `-2.3`
- rollout：2 process environments × 24 steps

### 4.4 末日结算和供氢

- 日末只估值电池和已经进入储罐的氢。
- 仍在途 pending H2 的资产价值为 0。
- 已付款但迟到/未送达订单不会退款，也不会凭空进罐。
- 报告分别记录经济成本、终端电池价值、罐内氢价值、未送达氢量、pending、计划/应急/迟到订单。

## 5. Env-v6 训练与评估结果

### 5.1 物理门控和性能门控

物理门控选中的 `347_1`：

- 原生总负荷约 `4.695 MW + 2.274 Mvar`
- 基准电压约 `[0.9826, 1.0] p.u.`
- reference 三天 voltage cost：`0、0.01936、0`
- nominal 至少两天超过 `0.02`

性能门控：

- legacy rollout：约 `10.82 transitions/s`
- fused + process rollout：约 `50.22 transitions/s`
- speedup：`5.55x`
- process/serial 在同一固定动作轨迹下的 24 小时观测、奖励、潮流和电压差异：`0`
- 融合 kernel 最大 float32 差异：`2.25e-4`（actor hidden）；动作和 log-prob 差异低于 `1e-5`

### 5.2 三组确定性评估

| 变体 | 安全天数 | 三日累计 raw voltage cost | 三日经济成本 | 潮流 |
|---|---:|---:|---:|---|
| 无通信 GRU-MAPPO | 1/3 | 7.757 | 约 294 万 | 全部收敛 |
| 无通信固定惩罚 MAPPO | 2/3 | 1.409 | 约 2353 万 | 全部收敛 |
| 无通信 GRU-MACPO | 2/3 | 0.685 | 约 1190 万 | 全部收敛 |

MACPO 的三个评估日：

| seed | daily voltage cost | 最低电压 | 经济成本 |
|---:|---:|---:|---:|
| 30 | `0` | `0.9798` | 约 345 万 |
| 31 | `0.685` | `0.9362` | 约 463 万 |
| 32 | `0` | `0.9572` | 约 382 万 |

训练末 200 updates：

- MACPO cost-recovery 占比：`0.80`，低于 `0.90` 门槛。
- 这只是满足首轮预设门控；末期随机 rollout 平均 raw voltage cost 仍约 `0.462`，远高于 `0.02`，所以不能宣称全时段硬安全。

### 5.3 反事实

MACPO seed30 基准经济成本约 `345 万`：

- 只清空 GRU hidden：约 `540 万`
- 只屏蔽上一动作：约 `786 万`
- ETA 增加 2 小时：约 `297 万`

hidden 和上一动作消融明显改变经济和供氢行为，支持时序信息被实际使用。ETA+2h 的经济成本下降，不应简单解释成“延迟越大性能越差”；它改变了采购规模、库存和应急购买组合，只能作为行为敏感性证据。

## 6. 当前可以写进论文的结论与不能写的结论

### 可以写

> 在 Swiss 原生 MV 配网、无显式 Agent 广播和集中式 reward/cost critic 条件下，shared-system GRU-MACPO 相比普通 GRU-MAPPO 显著降低电压越限，相比固定惩罚 MAPPO 取得更好的安全—经济折中。GRU hidden 和本地上一动作消融显示，交通 ETA、库存、延迟订单等历史信息参与了策略形成。

### 现在不能写

- 不能写 MACPO 已经保证所有时段零越限；seed31 仍有 `0.9362 p.u.`。
- 不能写显式通信完全没有价值；当前结果只说明在本环境和 CTDE critic 下，广播不是必要条件。
- 不能写对全部 Swiss 配网普遍有效；`347_1` 是经过可控性筛选的代表性工况。
- 不能用单个训练 seed 证明统计显著性；当前 `30/31/32` 是评估日，不是三个独立训练随机种子。
- 不能把 fixed penalty 的高安全成本和高经济成本简单说成算法失败，它展示的是固定权重过度保守的 trade-off。

## 7. 新对话上手路径

### 7.1 目录和数据

服务器原始活跃 worktree（脏，保留历史）：

```text
/root/autodl-tmp/env-v2-wt/HyperMARL-main
```

本次干净 handoff worktree：

```text
/root/autodl-tmp/env-v6-handoff-wt/HyperMARL-main
```

训练产物（服务器外部目录，不在 Git 代码目录内）：

```text
/root/autodl-tmp/env_v6_swiss_runs
```

Swiss 全量数据：

```text
/root/autodl-tmp/datasets/Swiss-PDGs/grids/matpower_data/MV
```

本地交接包：

```text
/Users/ruaaaaaaaa/Downloads/HyperMARL- CDA文档/GitHub/hypermarl-microgrid/tmp/two_stage_impl/env_v6_swiss
```

### 7.2 先做只读确认

```bash
cd /root/autodl-tmp/env-v6-handoff-wt/HyperMARL-main
git branch --show-current
git log -1 --oneline
python -m pytest -q tests/test_env_v6_training.py tests/test_safe_gru_trainer.py tests/test_power_flow.py tests/test_screen_swiss_mv.py
```

### 7.3 重新分析已有 1k 结果

不要在已有输出目录重新启动 launcher；直接分析：

```bash
cd /root/autodl-tmp/env-v6-handoff-wt/HyperMARL-main
python scripts/analyze_env_v6_swiss.py \
  --run-dir /root/autodl-tmp/env_v6_swiss_runs/long \
  --updates 1000 \
  --calibration /root/autodl-tmp/env_v6_swiss_runs/calibration.json
```

### 7.4 新建一轮独立实验

必须使用新的输出目录，不能覆盖已有结果：

```bash
cd /root/autodl-tmp/env-v6-handoff-wt/HyperMARL-main
python scripts/launch_env_v6_swiss.py \
  --run-dir /root/autodl-tmp/env_v6_swiss_runs_new \
  --calibration /root/autodl-tmp/env_v6_swiss_runs/calibration.json \
  --smoke-updates 100 \
  --long-updates 1000
```

当前 launcher 的三组配置名：

```text
v6_nocomm_gru_mappo
v6_nocomm_gru_mappo_penalty
v6_nocomm_gru_macpo
```

多训练种子还没有实现成独立矩阵；下一步先规划 seed `31/32` 的训练复制和跨网络 hold-out，不要直接复制当前结果当作统计实验。

## 8. 关键源码地图

| 文件 | 作用 |
|---|---|
| `envs/microgrid/power_flow.py` | IEEE-33 兼容和 Swiss MATPOWER CSV 潮流适配，非连续 BUS ID 映射 |
| `envs/microgrid/microgrid_env.py` | 交通、市场、氢气、PCC、经济 reward、电压 cost 和末日结算 |
| `envs/microgrid/config.py` | Env-v6 配置接口、scale/观测语义 |
| `baselines/MAPPO/safe_gru_trainer.py` | GRU actor、集中式 reward/cost critic、MAPPO/fixed penalty/MACPO、JIT rollout、评估反事实 |
| `baselines/MAPPO/safe_recurrent.py` | GRU、GAE、hidden reset 等基础函数 |
| `baselines/MAPPO/shared_system_macpo.py` | 现有 shared-system MACPO trust-region 更新器；本轮没有重构 |
| `baselines/utils/microgrid_vec_env.py` | serial/process 环境包装器 |
| `scripts/screen_swiss_mv.py` | 879 个 Swiss 网络筛查、PCC 选择、三日物理门控 |
| `scripts/run_env_v3_safe_matrix.py` | 三组配置、校准注入、checkpoint/resume runner |
| `scripts/benchmark_env_v6_rollout.py` | kernel parity、process/serial 轨迹、吞吐门控 |
| `scripts/launch_env_v6_swiss.py` | 物理→性能→smoke→1k 的硬门控启动器 |
| `scripts/analyze_env_v6_swiss.py` | seeds 30/31/32 确定性评估、反事实和成功标准 |

## 9. 下一步优先级

1. 不增加当前 seed30 的训练步数；先复现/整理报告。
2. 将三个训练种子 `30/31/32` 做成同一配置的独立实验，报告均值、标准差和逐日安全率。
3. 选择其他已经通过物理门控的 Swiss 网络做 hold-out，避免只在 `347_1` 上讲故事。
4. 深挖 MACPO seed31 的越限小时、PCC P/Q、P_el/P_bat、SOC、氢库存和 pending，确认是局部电压支撑不足还是运输行为引起的负荷变化。
5. 如果研究目标升级为硬安全，需要重新定义 per-day/per-step success gate 或改进可行域/安全恢复，而不是只继续堆 update。

## 10. Git 和安全说明

- 当前交接分支：`codex/env-v6-handoff-20260726`。
- 原始脏 worktree：`codex/env-v2-congestion-20260717`，没有 reset 或覆盖。
- 交接快照只提交主线源码、测试、报告、选中网络和文档；训练 checkpoint 保留在服务器外部产物目录，不提交 Git。
- SSH 主机和端口可以写入文档；密码不写入 Git、报告或脚本。
- 新对话如果看到旧实验目录或 `._*` 临时文件，先以本分支的 `README_HANDOFF.md` 和 `artifacts/env_v6_swiss/manifest.json` 为准，不要批量清理。
