# 多智能体电—氢微网论文服务器归档

归档日期：2026-08-26（Asia/Shanghai）

## 状态

- 已从服务器下载并在本地解压。
- 远端原始代码、实验和权重未删除、未改写。
- 远端仅新增了用于传输的压缩包和 Git bundle。
- 压缩包共 5,708 个文件；解压后约 733 MB。
- 共保留 53 个最终可恢复 checkpoint：Env-v6 Swiss 6 个，Env-v6.1 47 个。
- 大量中间 checkpoint、JAX 编译缓存、W&B 缓存、Python 缓存和完整 Conda 环境未下载。

## 完整性

压缩包：`paper_repro_archive_20260826.tar.gz`

SHA-256：

`8ec4c013db702b46255f8b8da47e216bbf14da4ce7f00d7cd8ebdee6ed69bf8b`

本地校验结果与服务器一致；压缩流检查和路径安全检查均已通过。

## 目录用途

### 当前论文对应代码

`内容/env-v6-handoff-wt/HyperMARL-main/`

这是 2026-08-23 更新后的 Env-v6 论文快照，对应 Git 分支
`codex/env-v6-handoff-20260726`，最新归档提交为 `3d652bf`。

关键文件：

- `baselines/MAPPO/safe_gru_trainer.py`：MAPPO、Penalty、Lagrangian、MACPO/SGR-MACPO 训练器。
- `baselines/MAPPO/shared_system_macpo.py`：共享系统约束更新。
- `envs/microgrid/microgrid_env.py`：电—氢微网环境。
- `scripts/run_env_v3_safe_matrix.py`：安全算法实验入口。
- `scripts/run_voltage_reference.py`：电压参考策略评估。
- `docs/ENV_V6_HANDOFF.md`：Env-v6 交接说明。

### 后续修正与消融代码

`内容/env_v6_1_dev/HyperMARL-main/`

对应 Git 分支 `codex/env-v6.1-improvements-20260726`，提交 `f4b8ac3`，包含：

- CPO 代理尺度与恢复分支修正；
- PCC 无功动作与电压安全带实验；
- MAPPO、Penalty MAPPO、Lagrangian、MACPO 对比；
- 按日枚举评估与不同种子复现；
- 软裕度、critic 拟合、无功容量、预算等消融配置。

### 当前论文原始训练与评估结果

`内容/env_v6_swiss_runs/`

包含 Env-v6 Swiss 的 smoke/long 训练曲线、配置、日志、rollout 和最终权重。当前稿件中的
MAPPO、fixed-penalty 与 MACPO 基础对比，以及部分氢交付统计可从这里追溯。

`内容/env_v6_swiss_reference_compact2/`

包含 2026-08-23 生成的 seed 31 电压参考 JSON 和逐时 CSV。当前稿件使用的
`0.685` 原始电压代价可在该参考结果中精确追溯。

### Env-v6.1 结果与后续消融素材

`内容/env_v6_1_runs/`

- `REPORT.md`：完整实验研究报告与结论修正记录。
- `PLAN.md`：逐轮实验计划和预注册判据。
- `eval/`：按日评估、rollout 和 HTML 曲线。
- `b1_*` 至 `b10_*`：MAPPO、不同 penalty、Lagrangian 基线与种子复现。
- `n*`：MACPO 安全预算、无功容量、soft margin、critic 等消融。
- `*_ext3000`：延长到 3000 updates 的收敛实验。
- `h2kg_3000`、`h2kwh_ctrl3000`：氢价单位修正对照。
- `renfc_3000`：可再生能源前瞻观测试验。
- `eval/n24_ext3000_rollout.json`：当前 28 天动作分析图的重要原始 rollout。

### 数据与诊断

- `内容/datasets/Swiss-PDGs/grids/`：Swiss-PDGs 网络数据，论文使用 MV `347_1`。
- `内容/env-v6-handoff-wt/HyperMARL-main/envs/Italian_data.csv`：时序数据。
- `内容/env-v6-handoff-wt/HyperMARL-main/envs/microgrid/italian_day_splits.json`：训练/测试/验证日划分。
- `内容/env_v6_diag_20260726/`：电压越限、无功动作和反事实诊断。
- `内容/env_v6_swiss_reference*/`：电压参考策略输出。

### Git 历史

`内容/hypermarl-microgrid/paper_code_history_20260826.bundle`

该 bundle 已验证为完整历史，包含 Env-v6、Env-v6.1 以及相关实验分支。需要恢复仓库时可在
Git 环境中从此 bundle 克隆；当前日常修改应优先使用上面的两个代码快照。

## 服务器运行环境摘要

- Ubuntu 22.04.3 LTS
- Python 3.10.8
- JAX / jaxlib 0.4.25
- NumPy 1.26.4
- SciPy 1.13.1
- Flax 0.8.4
- Optax 0.2.2
- Gymnasium 0.29.1
- Hydra 1.3.2
- OmegaConf 2.3.0
- NetworkX 3.2.1
- Matplotlib 3.9.0
- NVIDIA GeForce RTX 3090 24 GB
- NVIDIA driver 595.58.03

服务器 base 环境中未安装 pandas 和 pandapower；Swiss 网络由项目自身数据和代码路径读取。

## 后续本地工作原则

1. 论文正文与现有成图继续以工作区中的 `ijepes_sgr_macpo_offline_handoff_20260823_v2/` 为起点。
2. 重新制图时，从本归档的 JSON/JSONL/CSV/rollout 读取原始数据，不从 PNG 反推数值。
3. 普通论文修改、排版、统计和绘图均在本地完成。
4. 只有确实需要重新训练的消融才重新启用服务器；优先从最终 checkpoint 继续，而不是从零训练。
5. 任何论文数值在改写前，都要核对真实配置、训练种子与评估种子的区别。

