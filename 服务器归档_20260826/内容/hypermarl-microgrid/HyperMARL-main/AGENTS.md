# HyperMARL 训练手册（给未来接手的 AI agent / 人类看）

> 本文件记录在 **AutoDL RTX 4090** 机器上跑 HyperMARL microgrid 实验的全部经验和踩过的坑。
> 新会话接手时**第一件事就是读这份**，避免重复踩坑。

---

## 0. 一眼速览（TL;DR）

- **项目路径**：`/root/autodl-tmp/HyperMARL-main`（**不是** `/root/HyperMARL-main`）
- **硬件**：RTX 4090 (24G) + 120G RAM + CUDA 12.6 驱动 + cuDNN 8.9
- **环境**：Python 3.10.8（`/root/miniconda3/`），JAX 0.4.25 + jaxlib 0.4.25+cuda12.cudnn89
- **训练一轮 4 算法（500k steps）并行**：约 **10 分钟**（首行输出要等 ~6-8 分钟 JIT 编译，别误以为挂了）
- **wandb 默认 disabled**，指标不会自动落盘，必须靠已打的补丁（见 §6）

---

## 1. 关键路径

| 用途 | 路径 |
|---|---|
| HyperMARL 主项目 | `/root/autodl-tmp/HyperMARL-main` |
| 微电网环境 | `/root/autodl-tmp/HyperMARL-main/envs` |
| 训练脚本 | `baselines/IPPO/ippo_ff_*.py` |
| Hydra config | `baselines/IPPO/config/*.yaml` |
| microgrid env wrapper | `baselines/utils/microgrid_vec_env.py` |
| 训练日志（自己建） | `logs/*.log` |
| Hydra 默认输出 | `outputs/YYYY-MM-DD/HH-MM-SS/` |

---

## 2. 环境：已经装好了，别再瞎装

### ✅ 已就绪
- JAX GPU 版：`jax==0.4.25`、`jaxlib==0.4.25+cuda12.cudnn89`
- `flax==0.8.4`、`chex==0.1.86`、`optax==0.2.2`、`distrax==0.1.5`、`orbax-checkpoint==0.5.15`
- `gymnasium==0.29.1`、`pettingzoo==1.24.3`、`SuperSuit==3.9.2`、`hydra-core==1.3.2`、`wandb==0.17.0` 等

### ❌ 永远不要做这些
1. **别装** CPU 版 `jaxlib==0.4.25`
   - 依赖已合并到根目录 `requirements-gpu.txt`；JAX GPU 需要按根目录 `README.md` 先单独安装
2. **别装** `jax[cuda12_pip]==0.4.14`
   - 这个版本**没有** `cuda12_pip` 这个 extra（只会装个纯 Python 的 jax 壳，jaxlib 反而被卸掉）
3. **别理** 这个 warning：
   ```
   The NVIDIA driver's CUDA version is 12.6 which is older than the ptxas CUDA version (12.9.86)
   ```
   驱动 12.6 向前兼容 ptxas 12.9，只是 XLA 编译不能并行（慢一点），**不影响运行**

### 🔧 如果真的需要重新装 JAX GPU（极端情况）
```bash
pip uninstall -y jax jaxlib
pip install "jax[cuda12_pip]==0.4.25" \
    --extra-index-url https://pypi.org/simple \
    -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

### 验证 GPU 可用
```bash
python -c "import jax; print(jax.devices())"
# 期望：[cuda(id=0)]
```

---

## 3. 四种算法、四份 config

| 算法 | 脚本 | Config 名 | 默认 `ALG` 字段 |
|---|---|---|---|
| HyperMARL (MLP hypernet) | `baselines/IPPO/ippo_ff_shared_weights_mlp_hypernets.py` | `ippo_ff_shared_weights_mlp_hypernets_microgrid` | `Hypermarl(IPPO)` ⚠️ |
| HyperMARL (Linear hypernet) | `baselines/IPPO/ippo_ff_shared_weights_linear_hypernets.py` | `ippo_ff_shared_weights_linear_hypernets_microgrid` | `Hypermarl(IPPO)` ⚠️ |
| IPPO-FuPS（共享权重） | `baselines/IPPO/ippo_ff_shared_weights.py` | `ippo_ff_shared_weights_microgrid` | `IPPO-FuPS` |
| IPPO-NoPS（独立参数） | `baselines/IPPO/ippo_ff_nps.py` | `ippo_ff_nps_microgrid` | `IPPO-NoPS` |

**⚠️ 坑 #1：两个 hypernet 的 `ALG` 字段冲突**
两个 hypernet config 都 inherits from `ippo_ff_shared_weights_hypernets_microgrid`，`ALG` 都是 `"Hypermarl(IPPO)"`。
这导致 metrics 保存文件名一样，**后写的覆盖先写的**。并行跑时必须用 hydra override 区分：
```
ALG="HyperMARL-MLP"     # MLP 跑加这个
ALG="HyperMARL-Linear"  # Linear 跑加这个
```

---

## 4. 启动训练：复制粘贴就能用

### 前置
```bash
cd /root/autodl-tmp/HyperMARL-main
mkdir -p logs
```

### 标准配方：4 算法并行，每个 500k steps，独立保存 npy/png
```bash
cd /root/autodl-tmp/HyperMARL-main && \
nohup bash -c '
XLA_PYTHON_CLIENT_MEM_FRACTION=0.22 python baselines/IPPO/ippo_ff_shared_weights_mlp_hypernets.py \
    --config-name=ippo_ff_shared_weights_mlp_hypernets_microgrid \
    TOTAL_TIMESTEPS=500000 ALG="HyperMARL-MLP" > logs/mlp.log 2>&1 &
sleep 10
XLA_PYTHON_CLIENT_MEM_FRACTION=0.22 python baselines/IPPO/ippo_ff_shared_weights_linear_hypernets.py \
    --config-name=ippo_ff_shared_weights_linear_hypernets_microgrid \
    TOTAL_TIMESTEPS=500000 ALG="HyperMARL-Linear" > logs/linear.log 2>&1 &
sleep 10
XLA_PYTHON_CLIENT_MEM_FRACTION=0.22 python baselines/IPPO/ippo_ff_shared_weights.py \
    --config-name=ippo_ff_shared_weights_microgrid \
    TOTAL_TIMESTEPS=500000 > logs/fups.log 2>&1 &
sleep 10
XLA_PYTHON_CLIENT_MEM_FRACTION=0.22 python baselines/IPPO/ippo_ff_nps.py \
    --config-name=ippo_ff_nps_microgrid \
    TOTAL_TIMESTEPS=500000 > logs/nops.log 2>&1 &
wait
echo "ALL DONE $(date)" > logs/all_done.txt
' > logs/launcher.log 2>&1 &
disown
```

### 配方解释
- `nohup ... & disown`：和 `screen` 作用一样，断开 SSH / Claude 会话关闭 shell 也不会停。Claude 没法 attach screen，所以统一用这个
- `sleep 10`：**错峰启动避免 4 个进程同时扎堆编译**。虽然 GPU 能扛住，CPU 编译错峰更稳
- `XLA_PYTHON_CLIENT_MEM_FRACTION=0.22`：每进程最多用 22% GPU 显存（24G × 22% ≈ 5.3G）。4 个 × 22% = 88%，留 12% buffer
- 训练产物（`returns_microgrid_<ALG>.npy/.png`）默认保存到 **仓库根目录的 `result/generated/returns/`**；可用 `HYPERMARL_OUTPUT_DIR=/path/to/output` 覆盖

### 多 seed 批量（想要 5 seeds）
```bash
for SEED in 1 2 3 4 5; do
    XLA_PYTHON_CLIENT_MEM_FRACTION=0.18 python baselines/IPPO/ippo_ff_shared_weights_mlp_hypernets.py \
        --config-name=ippo_ff_shared_weights_mlp_hypernets_microgrid \
        TOTAL_TIMESTEPS=500000 SEED=$SEED ALG="HyperMARL-MLP-s$SEED" \
        > logs/mlp_s$SEED.log 2>&1 &
    sleep 8
done
wait
```
5 个并行时 `MEM_FRACTION=0.18`（5×0.18=0.9），GPU 刚好够。

---

## 5. 监控与排查

### 看训练进度
```bash
tail -f logs/mlp.log
# 每 100 updates 打一行 `update: x/5208` + `SPS: avg inst`
```

### ⚠️ 坑 #2：首行输出要等 6-8 分钟别以为挂了
- smoke test (TOTAL_TIMESTEPS=960) 只要 ~30 秒出 `update: 0/10`
- **正式 500k** 第一次 JIT 编译涉及 hypernet 生成整个网络参数，编译图巨大，**单进程也要 ~8-12 分钟**才出首行 `update: 0/5208`
- 看到 GPU 利用率 0-5%、显存已占满 5G+、CPU 单核 100% = **在编译，等着**
- 看到 GPU 利用率 > 20% = **已经在训练了**

### ⚠️ 坑 #3：别用 `... | tail -N` 接训练
```bash
# 错误示范：stdout 被 tail 缓冲，训练跑 10 分钟你一个字都看不到
python train.py 2>&1 | tail -10

# 正确：直接重定向到文件，然后 tail -f 文件
python train.py > logs/run.log 2>&1 &
tail -f logs/run.log
```

### 检查是否真的跑完
```bash
cat logs/all_done.txt                              # 有内容 = 全部完成
grep "update: 5200" logs/*.log                     # 每个文件都有就是跑完了
ps -ef | grep -E "baselines/IPPO" | grep -v grep   # 空 = 全部退出
```

### 显存 / GPU
```bash
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader
watch -n 1 nvidia-smi
```

---

## 6. ⚠️ 必要的代码补丁（已打）

### 问题
`baselines/utils/utils.py:log_train_metrics` 原本只把 metrics 推 wandb，**不存本地**。
`WANDB_MODE=disabled` 时所有 reward/loss 曲线数据都**丢了**。
而且画的 PNG 文件名硬编码 `ippo_ff_{env}.png`，4 个算法并行跑会互相覆盖。

### 已做的修改（已 in place，不用再改）
`baselines/utils/utils.py` 中 `log_train_metrics` 的前几行，从
```python
if metrics.get("returned_episode_returns") is not None:
    plt.plot(metrics["returned_episode_returns"])
    plt.savefig(f"ippo_ff_{config['ENV_NAME']}.png")
    ...
```
改成按 `ALG` tag 分别保存 npy + png：
```python
alg_tag = str(config.get("ALG", "alg")).replace("/", "_").replace(" ", "_")
out_dir = _training_output_dir(config, "returns")
out_png = out_dir / f"returns_{config['ENV_NAME']}_{alg_tag}.png"
plt.savefig(out_png)
...
import numpy as _np
_np.save(out_dir / f"returns_{config['ENV_NAME']}_{alg_tag}.npy", _arr)
```
**文件保存位置**：默认 `result/generated/returns/`，不是 `HyperMARL-main/` 源码根目录。

### 如果想要 wandb 可视化
把任一 config override `WANDB_MODE=online`（已登录）或 `WANDB_MODE=offline`（本地，事后 `wandb sync`）。离线模式最省心。

---

## 7. 已知性能基线（seed=30）

### 500k 步（探索期快照）

末段 200 updates 的 stochastic 回报：

| 排名 | 算法 | Final Return | 状态 |
|:-:|---|---:|---|
| 🥇 | HyperMARL-MLP | -18.66 ± 6.84 | 基本收敛 |
| 🥈 | HyperMARL-Linear | -40.70 ± 7.63 | 还在上升 |
| 🥉 | IPPO-NoPS | -49.68 ± 7.34 | 斜率减缓 |
| 4 | IPPO-FuPS | -61.96 ± 8.00 | 仍在爬 |

### 1M 步（主实验，2026-04-18 夜间）

| 排名 | 算法 | Final Return (train) | 1-episode Deterministic Return | Δ vs 500k |
|:-:|---|---:|---:|---:|
| 🥇 | HyperMARL-MLP | **-13.72 ± 6.80** | **+0.12** ✅ | +4.94 |
| 🥈 | HyperMARL-Linear | -21.89 ± 7.00 | -4.26 | +18.81 |
| 🥉 | IPPO-FuPS | -32.25 ± 7.23 | -11.61 | +29.71 |
| 4 | IPPO-NoPS | **-57.23 ± 7.48** ⚠️ | -34.16 | **-7.55（退化）** |

**关键发现**：
- **排名稳定**：训练曲线和 1-episode deterministic 推理排名完全一致
- **IPPO-NoPS 退化**：500k → 1M 反而变差 7.55，符合独立参数在 MARL 中的数据效率瓶颈
- **HyperMARL-MLP 唯一达到正收益**：deterministic 推理 +0.12，学到合理的 TOU 响应调度

**单 seed 结论仅供参考**，多 seed 平均才有统计意义。

### 训练后分析脚本
`HyperMARL-main` 根目录中的推理、绘图、诊断、汇总脚本已移除；本仓库只保留训练代码和环境代码。

---

## 8. 其他需要知道的

- **性能换算**：稳态 SPS ~900/process，500k steps = 5208 updates ≈ 6 秒训练（JIT 之后）。4 个并行互相挤 CPU，实测 ~10 分钟整轮完成
- **`NUM_UPDATES = TOTAL_TIMESTEPS // NUM_STEPS // NUM_ENVS`** = `500000 // 24 // 4 = 5208`
- **首次编译慢**：hypernet 脚本编译 `_update_step` 要 ~5-8 分钟（CPU 单核编译，因为驱动/ptxas 版本不匹配无法并行编译）。**这是硬开销，加钱也没用**
- **`print` 每 100 updates 才打一行**，所以进度条不要指望高频更新
- **microgrid env 是纯 NumPy** 的（通过 `envs.microgrid.MicrogridContinuousEnv`），每个 step 都要 Python ↔ NumPy，不能纯 GPU 跑
- **AutoDL 关机不丢数据盘** `/root/autodl-tmp`，所以模型/日志放这里安全
- **长任务 → `wait && shutdown -h now`** 省钱（AutoDL 按时计费）

---

## 9. 快速调试清单（下次出问题先自查）

- [ ] `pwd` 是不是 `/root/autodl-tmp/HyperMARL-main`？
- [ ] `python -c "import jax; print(jax.devices())"` 输出是 `[cuda(id=0)]` 吗？不是的话**千万别再跑 requirements.txt**，要 §2 的重装命令
- [ ] `/root/autodl-tmp/HyperMARL-main/envs/microgrid` 还在吗？
- [ ] 等了多久？少于 10 分钟就别喊"挂了"，去看 `ps` 和 `nvidia-smi`
- [ ] `logs/*.log` tail 有没有 Traceback？
- [ ] hypernet 两个想并行跑？别忘加 `ALG="HyperMARL-MLP"` / `ALG="HyperMARL-Linear"` override
- [ ] 要保留 metrics？`WANDB_MODE=disabled` 的情况下靠 §6 的补丁存 npy，**补丁已在 `baselines/utils/utils.py`**
- [ ] 要推理 / 保留权重？必须 override `CHECKPOINT=True`（默认是 False），否则**一个权重文件都不会留**
- [ ] 自动关机脚本里 `shutdown` 前**必须加长 sleep**（见 §10）

---

## 10. ⚠️ 自动关机陷阱（2026-04-18 实战踩过）

### 现象
`wait && sleep 30 && shutdown -h +1` 这种写法在训练比预期快时是**灾难**：
- 原以为 1M 步 × 4 并行要 25 分钟
- 实际只跑了 **18.5 分钟**（错峰 `sleep 10` 让 4 个进程没扎堆抢 CPU，编译反而更快）
- 人还没回来看一眼，机器已经自己关了

### 教训
自动关机模式下，shutdown 命令前至少留 **10 分钟缓冲**，让人有反悔时间：
```bash
# 差：几乎零缓冲，关机前没机会干预
wait && sleep 30 && /sbin/shutdown -h +1

# 好：10 分钟窗口，足够 SSH 上来 shutdown -c 取消
wait
echo "ALL DONE $(date)" > logs/all_done.txt
sleep 600   # 十分钟反悔窗口
/sbin/shutdown -h +1 "reached shutdown timer"
```

或者干脆写成 cron-style："定时查到 `all_done.txt` 存在满 N 分钟才关机"。

### 补救
**即使被自动关机了，数据盘 `/root/autodl-tmp/` 数据完全不丢**。重启后：
```bash
cd /root/autodl-tmp/HyperMARL-main
cat logs/run_summary.txt   # 确认起止时间
tail -5 logs/*.log          # 确认 update 跑到最后
ls ../result/generated/returns/returns_microgrid_*.npy  # reward 曲线
find wandb -type d -name "*_steps_*"   # checkpoint 路径
```

---

## 11. Checkpoint 结构速查（用于推理）

训练时 `CHECKPOINT=True` 会把权重存到：
```
wandb/wandb/offline-run-<timestamp>-<id>/files/models/<RUN_NAME>/<STEP_TAG>.<AGENT_TAG>/
```

注意路径里有**两层 `wandb/wandb/`**（外层是 `WANDB_DIR`，内层是 wandb 自己新建的）。

### 每个算法 checkpoint 数量
| 算法 | # ckpt | AGENT_TAG 格式 |
|---|:-:|---|
| HyperMARL-MLP | 1 | `agent_30_seed` |
| HyperMARL-Linear | 1 | `agent_30_seed` |
| IPPO-FuPS | 1 | `agent_30_seed` |
| IPPO-NoPS | **4** | `agent_{0,1,2,3}_30_seed`（每个 agent 独立） |

### Orbax 格式
每个 ckpt 目录内部结构（orbax `PyTreeCheckpointHandler`）：
```
<ckpt_dir>/
  checkpoints           # aggregate 文件
  <numeric_subdirs>/    # 各层参数
```

### 加载示例
```python
from orbax.checkpoint import checkpointer
from orbax.checkpoint.pytree_checkpoint_handler import PyTreeCheckpointHandler

ckptr = checkpointer.Checkpointer(
    PyTreeCheckpointHandler(aggregate_filename="checkpoints")
)
# target 是与 train_state.params 同结构的 pytree
restored_params = ckptr.restore(str(ckpt_dir), item=target_pytree)
```

推理/rollout 脚本不再随训练源码保留。

---

*最后更新：2026-04-19 由 Claude 维护。下次改动请同步更新本文件，特别是"已知坑位"和"性能基线"两节。*
