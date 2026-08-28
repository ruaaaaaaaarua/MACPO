# hypermarl-microgrid

本仓库用于在氢能微电网环境中训练 HyperMARL、IPPO、MAPPO，并保留 MATRPO 作为 baseline。

## 目录结构

```text
hypermarl-microgrid/
|-- README.md
|-- requirements-gpu.txt
|-- HyperMARL-main/
|   |-- envs/              # 微电网环境与 Italian 数据
|   |-- baselines/
|   |   `-- MATRPO/        # MATRPO baseline 算法
|   `-- scripts/run_ctde_40k.sh
`-- result/                # 已整理的实验结果
    `-- generated/         # 新生成结果，已被 Git 忽略
```

训练时只需要加入 `HyperMARL-main`：

```bash
export PYTHONPATH="${PWD}/HyperMARL-main:${PYTHONPATH}"
```

## 环境

建议使用 Linux + NVIDIA GPU + CUDA 12.x + Python 3.10+。

```bash
python -m venv .venv
source .venv/bin/activate

pip install "jax[cuda12_pip]==0.4.25" \
  --extra-index-url https://pypi.org/simple \
  -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

pip install -r requirements-gpu.txt
python -c "import jax; print(jax.devices())"
```

不要在 GPU 环境中直接安装 `jaxlib==0.4.25`，否则可能装成 CPU 版本。

## 训练

主实验入口：

```bash
bash HyperMARL-main/scripts/run_ctde_40k.sh
```

快速测试：

```bash
TOTAL_TIMESTEPS=4800 bash HyperMARL-main/scripts/run_ctde_40k.sh
```

默认 CTDE 设置：

- 环境：`FullCDA-ReserveDemand + Price30`
- episode：24 小时
- 数据划分：`italian_split=train`
- `TOTAL_TIMESTEPS=960000`
- `SEED=30`

## 算法入口

| 算法 | 位置 |
| --- | --- |
| HyperMARL-MLP / IPPO | `HyperMARL-main/baselines/IPPO/ippo_ff_shared_weights_mlp_hypernets.py` |
| HyperMARL-Linear / IPPO | `HyperMARL-main/baselines/IPPO/ippo_ff_shared_weights_linear_hypernets.py` |
| IPPO-FuPS | `HyperMARL-main/baselines/IPPO/ippo_ff_shared_weights.py` |
| IPPO-NoPS | `HyperMARL-main/baselines/IPPO/ippo_ff_nps.py` |
| MAPPO-IA-CTDE | `HyperMARL-main/baselines/MAPPO/mappo_ff_shared_weights.py`，配置 `mappo_ff_independent_actors_microgrid` |
| HyperMARL-MLP-CTDE / MAPPO | `HyperMARL-main/baselines/MAPPO/mappo_ff_shared_weights_mlp_hypernets_microgrid.py` |
| MATRPO baseline | `HyperMARL-main/baselines/MATRPO/` |

说明：`MAPPO-IA-CTDE` 使用集中式 critic，但每个 agent 拥有独立 actor；`HyperMARL-MLP-CTDE` 同样为每个 agent 生成不同 actor，不过所有 actor 参数由同一个 MLP hypernetwork 根据 agent id / embedding 生成；`MATRPO` baseline 也要求每个 agent 使用独立 actor，并共享一个 critic。

单独运行 Hydra 配置示例：

```bash
cd HyperMARL-main
python baselines/IPPO/ippo_ff_shared_weights_mlp_hypernets.py \
  --config-name=ippo_ff_shared_weights_mlp_hypernets_microgrid \
  ALG="HyperMARL-MLP" WANDB_MODE=disabled
```

## 输出

训练曲线：

```text
result/generated/returns/
```

运行日志：

```text
result/generated/runtime_logs/
```

自定义输出目录：

```bash
HYPERMARL_OUTPUT_DIR=/path/to/output bash HyperMARL-main/scripts/run_ctde_40k.sh
```

`result/generated/` 不进入 Git。历史图表、CSV 和报告可整理到 `result/figures/`、`result/tables/`、`result/reports/`。

## 说明

- 微电网环境在 `HyperMARL-main/envs/microgrid/`。
- Italian 数据在 `HyperMARL-main/envs/Italian_data.csv`。
- MATRPO baseline 算法在 `HyperMARL-main/baselines/MATRPO/`。
- 训练后的推理、绘图、诊断和报告脚本已从 `HyperMARL-main/` 根目录移除。
- 训练细节和排错记录见 `HyperMARL-main/AGENTS.md`。

## 微电网环境说明与逻辑审查

环境核心代码位于 `HyperMARL-main/envs/microgrid/`，训练脚本通过 `MicrogridContinuousEnv` 和 `baselines/utils/microgrid_vec_env.py` 调用。默认配置为 4 个 agent、24 小时 episode、5 维连续动作、17 维有效观测。

主要功能：构建含光伏、风电、电负荷、氢负荷、电池、制氢电解槽、储氢罐、外部电网、外部氢市场、内部电力/氢 CDA 市场的协同优化环境。代码中的历史变量名可能仍使用 `load_h` 或“热负荷”，但本文档统一表述为氢负荷；这只是负荷含义表述的转换，不改变算法输入输出。Italian 数据从 `HyperMARL-main/envs/Italian_data.csv` 读取，可按 `italian_day_splits.json` 做场景划分；氢负荷曲线可由电负荷派生。P2P 市场、气网和 LMP 电价逻辑保留为可选模块，默认主训练路径使用 CDA。

动作含义：每个 agent 输入 5 维连续动作，环境内部裁剪到 `[-1, 1]` 后解释为电解槽功率、电池充放电功率、电力报价、氢报价、储氢罐充放功率。当前配置中负荷侧氢能需求全部按氢负荷处理，不再作为独立热负荷控制问题；该表述调整不影响动作维度和训练算法。

观测内容：每个 agent 的观测向量按以下顺序拼接：

```text
0  光伏出力 / 光伏容量
1  风电出力 / 风电容量
2  电负荷 / 电负荷峰值
3  氢负荷 / 氢负荷峰值
4  电池 SOC
5  储氢罐库存比例
6  上一时刻电力市场清算价归一化
7  上一时刻氢市场清算价归一化
8  小时 sin 编码
9  小时 cos 编码
10 当前外部购电价格归一化
11 上一时刻外部电网交互功率归一化
12 上一时刻外部氢市场交互量归一化
13-16 未来 4 个 pending 氢交付桶
```

奖励目标：所有 agent 使用共享系统奖励 `-base_cost / reward_scale`，目标是最小化基础运行成本：

```text
base_cost = C_grid + C_h2
```

其中 `C_grid` 为外部电网购售电成本，`C_h2` 为外部氢市场购售氢成本。SOC/氢库存、终端库存、动作平滑等惩罚项以及 `market_bonus` 仍保留在 `info` 中用于诊断，但不再进入训练 reward；内部 CDA 交易主要用于系统内撮合，系统共享奖励中买卖转移不重复计入。

逻辑审查结果：
- `reset()` 会重置 SOC、储氢量、市场价格、外部交互量、pending 氢交付和每日数据；`step()` 会检查动作形状、裁剪动作、按物理约束更新电池和储氢罐，再进行电力/氢市场撮合与外部市场平衡。
- 电池 SOC、储氢库存、电解槽功率、储氢罐充放功率均有裁剪逻辑，未发现越界更新导致训练无法运行的明显问题。
- 氢交易支持延迟交付：内部氢成交后先记账，后续时段交付；当前时段缺口仍由外部氢市场平衡。该设计避免了把未到货氢直接用于当前氢负荷，但论文描述中需要明确说明。
- episode 结束时未到货 pending 氢不会再次结算，代码注释说明是为避免重复记账；这属于建模假设，建议在实验说明中保留。
- 训练接口与 IPPO/MAPPO/HyperMARL 匹配：向量化 wrapper 返回扁平观测、reward、termination、truncation 和 info，训练脚本会读取连续动作维度。

需要后续确认的风险点：
- `microgrid_env.py` 局部注释仍写有“6-d action / a5 热负荷氢比例”，但实际配置和代码均为 5 维动作；这是旧注释和当前氢负荷表述不一致，不影响当前运行。
- 默认 `agent_types` 为 `["mg", "mg", "mg", "mg"]`，不会触发 `producer/consumer` 标记；当前核心交易按净需求判断，若后续使用角色专属逻辑需重新检查。
- `MicrogridContinuousEnv.action_space` 标为无限 Box，而 `MicrogridVecEnv` 和环境内部实际按 `[-1, 1]` 裁剪；训练时应按 `[-1, 1]` 输出动作。
- P2P、气网、LMP 等可选分支默认不是主路径；若作为论文消融实验开启，建议单独做约束和收益核算检查。
- `data_generator_italian_tmp.py` 是 0 字节遗留文件，未被导入，可后续清理；未发现其影响训练。

结论：默认微电网训练路径未发现影响训练运行的明显逻辑错误。最小验证已通过：导入 `MicrogridContinuousEnv`、执行一次 `reset()` 和随机合法动作 `step()` 后，得到 `obs=(4, 17)`、`reward=(4, 1)`、`done=(4,)`。

## 许可和引用

`HyperMARL-main/` 遵循上游 HyperMARL 许可。使用本代码时，请引用相关论文和原始 HyperMARL / MATRPO 工作。
