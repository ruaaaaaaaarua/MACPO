# 环境 v2 设计规格：氢运输拥堵博弈 + 兜底清理 + 规则基线重做

日期: 2026-07-17　分支: codex/env-v2-congestion-20260717　状态: 待实现

## 背景与动机（一段话）

反事实机制验证显示 v1 路由无因果效果（forced_direct/permuted delta≈0）：
路网只承载内部 CDA 交易（0.7 卡车/天 vs 路容量 8/h）、无任何成本项、
延迟被 reservation+应急采购完全兜底。Phase 0 headroom 实验确定主战场为
24h 稀疏终端奖励（验证配对 headroom=136 分/42%），长时距已排除。
本改造目标：让路由和订购时机具有真实经济后果，并制造跨智能体的
拥堵外部性（喂给 STAS 信用分配的核心难题），全部改动以物理合理性为由。

## 改动 1：运输网络 v2（h2_transport.py + microgrid_env.py）

1a. 外部供应站入网：新增第 5 节点 EXT（id=4）。外部订单不再走固定
    h2_delivery_lag=4，改为经路网发货：dispatch 于下单小时，
    route_options(EXT, buyer) = 直达 + 经其余节点的 2 条绕行；
    deliver_at = t + eta。pending 观测/预留机制改从 shipment 队列取数
    （保持观测维度与语义不变，兼容旧 checkpoint 不是目标）。
    删除 v1 的两个硬断言（恰好 4 节点、ETA 恰好 4..6）。
1b. 容量收紧（制造博弈性）：h2_traffic_truck_capacity_kg 500→100；
    h2_traffic_edge_capacity 8→2.5。预期系统流量 15-20 车/天，
    高峰 edge-hour 自有流量利用率贡献 0.8-2.4。
1c. ETA 范围：clip [4,6]→[4,10]（新 config: h2_traffic_eta_min=4,
    h2_traffic_eta_max=10；旧 key 保留默认值兼容 v1 行为）。
1d. 在途损耗：新 config h2_traffic_transit_loss_per_hour（默认 0，
    实验用 0.008）。delivered = gross * max(0, 1 - rate*eta)。
    损耗量计入 transport_loss 指标（v1 该指标恒 0）。
1e. 发货凑单（可选，二期）：同一(卖,买)对累计 ≥0.5 车才发运。

## 改动 2：兜底清理（原则：兜物理可行性，不兜经济后果）

2a. cap-aware 裁剪取消：实验 overrides 置 h2_cap_aware_buy_enable=false。
    需核查 flag=false 的现行为：若到货超储罐容量存在静默截断，改为
    「溢出排空、货款照付」，并新增 overflow_kg / overflow_cost 指标。
2b. reservation 降级：实验 overrides 置 h2_delivery_reservation_ratio=0.5
    （消融含 0.0）。纯配置，无代码改动。
2c. 应急采购涨价：新 config h2_emergency_price_multiplier（默认 1.0，
    实验用 2.0 → 有效应急价 90 ¥/kg vs 计划价 45）。只作用于
    「负荷缺口触发的自动外购」，不影响动作下单的计划价。定位自动
    外购代码路径（physical_idle 的 82k 外购即此路径），新增
    emergency_buy_kwh / emergency_buy_cost 指标。
2d. 训练稳定性预案：若 500 集 smoke 发散，前 2000 集课程化
    （multiplier 1.33→2.0，reservation 0.7→0.5）。

## 改动 3：规则基线重做（baselines/utils/rule_baselines.py）

3a. base_stock_rule：order_t = max(0, S_target − stock − pending)，
    S_target = 未来 (交付期望+safety_hours) 小时负荷预测 × target_mult，
    预测用不开挂的同小时滑动均值（episode 内在线更新 + 日型先验）；
    价格分层：电/氢价高于 price_quantile 时只补安全线。
    参数网格：safety_hours∈{1,2,3}, price_quantile∈{0.3,0.5,0.7},
    target_mult∈{1.0,1.25}，验证日调优（复用 evaluate_rule_baselines）。
3b. base_stock_privileged：同 3a 但用真值未来负荷（诊断用，不参与排名）。
3c. 现有 current_deficit_rule / privileged_t4_rule 保留但退役出正文
    （历史红旗：idle > deficit_rule，见 07-17 分析）。

## Smoke 门控（全过才准进 Phase 2）

G1 机制门控：MAPPO 5k 集后 forced_direct_route 与 permuted_route 的
   验证 delta ≥ 15 return 单位且 4 日方向一致（v1 为 ≈0）。
G2 基线门控：调优后 base_stock_rule 显著优于 physical_idle（v1 中
   deficit_rule 反而更差，为审稿红旗）。
G3 稳定门控：稀疏终端 500 集 smoke 无发散（否则触发 2d 课程化）。
G4 经济量级：路由/时序敏感成本（损耗+应急价差+溢出）合计占
   总成本 4-8%（对照表：总成本 ~112k¥/天，return=−cost/200）。

## Phase 2 矩阵（另文，此处只锚定与环境的接口）

24h 稀疏终端为主（headroom 136 分），算法：MAPPO / MATD3 / 均匀重分配
(IRCR, credit=R/T) / STAS-blend(conserved-causal, mix 早开低顶 0.03-0.05
可退火) / STAS-paper（消融）。新增门控：信用保真度探针（反事实真值
vs credit 秩相关）+ 影子基线退火。mix 调度依据 07-17 曲线诊断。

## 附录: 冒烟结果与门控修订 (2026-07-17 晚, 已获批准)

10k 冒烟 (env-v2-smoke-02, 校准后配置): MAPPO 最优验证 -490.6 (超过特权
base-stock -513); 反事实 delta: 订购 +282.3, 强制直达 +33.8, 打乱路由 -1.2。
门控判定: G1 ✓ / G2 ✓ / G3 ✓ / G4 ✓ (份额 23.1%)。

修订 1 (G1): 阈值只作用于 forced_direct。理由: 每对节点的两条绕行路线
ETA 近似对称, permutation 检验结构性无信息量; 学到的技能是"高峰避开
直达", 恰由 forced_direct 检出。permutation 保留为报告项。
修订 2 (G4): 带宽上限 12%->25%。理由: 原带宽是知道"氢负荷晚峰撞道路
晚峰"这一核心张力之前的先验; 实测该份额随策略能力单调下降
(idle 53% -> 5k MAPPO 40% -> 10k MAPPO 23%), 是能力度量而非固定税,
Phase 2 作为逐算法追踪指标。

校准三件套 (均已提交): h2_tank_init_ratio 0.4 (修复 reset 只读 min_ratio
的死键 bug), h2_action_order_max_peak_hours 2.0, h2_tank_power 按峰值
选型 [2000,2000,3400,4200]。Phase 1 至此收官, 环境参数冻结于
scripts/env_v2_overrides.py。
