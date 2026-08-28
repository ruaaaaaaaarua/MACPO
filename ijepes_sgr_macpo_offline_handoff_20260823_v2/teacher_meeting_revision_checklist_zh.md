# 老师会议修改清单（后续执行依据）

整理自本次会议录音转写。状态分为：已完成、待修改、暂缓。当前论文数值以
`ijepes_sgr_macpo_revised_en.tex` 为准；未经实验或代码证据，不为迎合表述而改动参数或结果。

## 本轮实施状态（2026-08-26）

- 已补强引言中“传统调度困难 → MARL → 安全 MARL”的方法选择逻辑。
- 已增加系统模型到 SGR–MACPO 的过渡，并把 GRU、集中式 critics 和约束更新分别对应到本文问题。
- 已从归档训练入口核实真实配置，并新增 Experimental setup 参数表。
- 已确认训练使用 `gamma=1.0`，论文保留真实值并说明其对应未折扣的 24 h 有限时域目标。
- 已明确一个 policy-training seed 与三个 evaluation-day seeds 的区别。
- 已将基线展示名统一为 MAPPO 和 fixed-penalty MAPPO；GRU 作为三种方法共享的受控架构设置。
- 环境示意图和算法示意图仍按用户安排暂缓，后续由用户绘制后再替换。

## 一、最高优先级：论文逻辑与算法动机

### 待修改

- 在引言以及“系统模型 → 算法”之间补清楚方法选择逻辑：
  1. 电—氢交易与调度具有非线性和随机性；
  2. 基于模型的优化依赖精确模型、场景生成且在线计算成本较高；
  3. 各微网执行时只有局部信息，因此需要多智能体、部分可观测建模；
  4. 普通 MARL 不能直接保证配电网电压安全，因此需要显式安全成本和约束策略更新。
- 让读者明确 SGR–MACPO 是针对本文问题设计的，而不是在模型后突然引入一个通用算法。
- 每个算法设计都要回答“为什么需要”：GRU 对应延迟和部分可观测性；集中式 reward/cost critic
  对应全局耦合；约束方向、recovery 和 backtracking 对应电压成本控制。
- 检查算法章节只讲问题建模、网络结构和更新流程；训练批量、学习率、迭代次数等实验设置移到
  Experimental setup。

### 已基本完成但需最终复核

- 已建立 constrained Markov game，并给出局部观测、动作、reward 和 voltage cost。
- 不需要堆砌通用的 Markov 状态转移公式；只保留本文真正使用的物理状态转移和约束。

## 二、参数表与实验口径

### 待修改

- 在 Experimental setup 增加完整参数表，至少包括：算法名称、是否使用 GRU、hidden size、
  discount factor、GAE、batch/rollout、learning rate、KL limit、CG iterations、damping、
  backtracking trials/ratio、penalty coefficient、voltage-cost budget、训练/测试 seeds。
- 复核基线名称。老师建议主图和表中简化为 MAPPO、Penalty MAPPO、SGR–MACPO，GRU 是否启用
  放入参数表，避免把“GRU-MAPPO”包装成独立创新。
- 老师建议 discount factor 使用 0.99，而当前稿件写的是实际设置 `gamma=1`。必须先以代码、配置和
  checkpoint 训练事实为准：若训练确实使用 1，不得只把论文改成 0.99；只能解释有限时域选择，
  或重新实验后再改。
- 统一 penalty 的命名。若系数确实固定为 1，可在实验设置中明确说明；图例是否保留
  “Fixed-penalty”需结合老师希望简化名称的意见最终决定。
- 始终准确写成“一个 training seed + 三个 deterministic evaluation-day seeds”。三个测试日不是
  三个独立训练随机种子，不得写成多训练种子统计显著性证据。

## 三、实验与 contribution 的对应关系

### 已完成

- 训练 reward 与 voltage-cost 曲线已拆成两张图，避免双子图信息拥挤；cost 使用对数坐标并标出
  0.02 阈值。
- 已删除 28 天平均、`a0–a7`、`[-1,1]` normalized action 图。实验图使用代表性测试日和物理单位。
- 原氢气服务柱状图不能证明结果由算法设计造成，已从主文移除，不再将相关性表述为因果性。
- 当前电压运行图使用代表日对比不同算法，标出 0.95 p.u. 下限、越限时段以及 hourly voltage
  cost，对应安全约束算法贡献。
- 当前氢气延迟反事实保持同一 checkpoint 和测试日，只改变 delivery timing，并展示电解槽功率、
  储氢和在途氢，对应延迟物流环境建模贡献。

### 待修改或按需要补强

- 每张实验图的正文解释都必须包含：比较对象、唯一变化因素、物理现象、与哪一项 contribution
  对应，以及结论边界。
- 电压图目前能支撑“整体安全机制有效”，但不能区分 cost critic、recovery、backtracking 中哪一项
  单独造成改善。若后续要对具体更新组件作因果声明，需要同环境、同设置的组件消融。
- 老师建议在解释电压变化时结合可再生能源出力。当前 PCC 功率已经提供直接电气联系；若后续数据
  足够且版面允许，可再加入 PV/WT 总出力作为辅助，而不是为了增加图而增加图。
- 不展示专门突出 seed-31 失败的附加图；正文继续诚实保留 2/3 安全天和“不是硬安全保证”的边界。
- 当前不制作补充材料。

## 四、环境示意图（用户之后自行绘制，当前暂缓）

- 不直接使用明显的 AI 整图；可生成单个 PNG 图标，再转 SVG 后手工排版和统一风格。
- 配电网在示意图中写成通用 `Distribution network`，无需把 `Swiss` 放在图内；数据来源在正文引用。
- 未实际展示或分析的 voltage profile、reactive power 不要硬塞进示意图；交易若只涉及有功，就突出
  active-power flow。
- 建议三层结构：
  1. 上层：electricity market 与 hydrogen market；
  2. 中层：多个结构相似的 microgrids，以及卡车运输和 delivery delay；
  3. 下层：distribution network，各微网通过 PCC 映射到不同节点。
- 微网不必画四个巨大的重复结构，可画三个加省略号表示一般的 (M) 个微网；内部 PV、WT、battery、
  electrolyzer、H2 tank、loads 等设备缩小但保留。
- 图底部单独设置 legend，清楚区分信息流、功率流和氢气物流；减少留白，整体紧凑。

## 五、算法示意图（用户之后自行绘制，当前暂缓）

- 图中要让读者直接看懂：observation 从哪里来、actor 如何产生 action、joint action 如何进入环境、
  环境如何返回 reward 和 voltage cost、两个 critic/advantage 如何形成、策略如何更新。
- 必须突出本文相对 MACPO 的实际适配或改进，而不是只画一个通用 actor–critic 框图。
- 区分“环境交互流”和“网络训练/更新流”，可使用两套颜色或线型，避免所有箭头混在一起。
- 图要有足够信息量，但公式细节仍在正文中完整给出；示意图负责结构和流程，不代替数学定义。

## 六、写作、图注与排版

### 已完成

- Swiss-PDGs 数据来源已在环境设置中给出参考文献。
- 当前多面板图均有 `(a)–(c)` 标记，caption 逐项说明含义。
- 结果分析已由单纯汇总数字扩展到具体运行时段和物理变量。

### 待最终检查

- 正文中的图引用统一按期刊风格写为 `Fig. X` / `Figs. X and Y`；caption 是否显示 `Figure X`
  由 Elsevier 模板控制。
- 算法配色只用于算法比较；其他动作或物理量图不要复用同一套颜色而造成“仍在比较算法”的误解。
- 所有量使用映射后的真实物理名称和单位，不出现 `a0`、`a1` 或 normalized action。
- 检查表格和图片浮动造成的空白，但不为排版强行改动实验内容。

## 七、后续修改顺序建议

1. 补强引言和模型到算法的过渡逻辑。
2. 复核真实训练参数，解决 `gamma`、基线命名和 seed 口径。
3. 新增 Experimental setup 参数表，并把实验参数从算法章节移过去。
4. 逐段检查算法公式、更新流程和贡献陈述是否一一对应。
5. 用户完成环境图与算法图后替换 Fig. 1–2，并统一图注、术语和引用。
6. 最后统一全文 Fig./Table 写法、单位、缩写、浮动位置并重新编译检查。
