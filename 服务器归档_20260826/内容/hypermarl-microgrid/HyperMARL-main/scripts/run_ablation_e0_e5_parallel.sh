#!/usr/bin/env bash
# E0-E5 消融实验 — 进程级并行,6 个后台进程错峰启动。
#
# 实验矩阵 (每项相对当前 override 的单变量改动,h2_price_unit=kg 为基线):
#   E0  复现 bug: h2_price_unit=kwh  → return 方差爆炸 (参照组)
#   E1  单位修正: h2_price_unit=kg  (default, R1 fix)
#   E2  供需可行: 降 consumer load_h_peak 到可被内部市场满足
#   E3  惩罚进 reward: penalty_in_reward_enable=true
#   E4  市场 bonus 进 reward: market_bonus_in_reward_enable=true
#   E5  PPO 稳定化: ANNEAL_LR + NUM_MINIBATCHES + VF_COEF
#
# 用法 (从 hypermarl-microgrid 根目录):
#   bash HyperMARL-main/scripts/run_ablation_e0_e5_parallel.sh [smoke|full|both]
#
# 快速 smoke (~30s/实验 × 6 并行 ≈ 1-2 min):
#   bash HyperMARL-main/scripts/run_ablation_e0_e5_parallel.sh smoke
#
# 完整短跑 (120k steps, ~10 min 整体墙钟):
#   bash HyperMARL-main/scripts/run_ablation_e0_e5_parallel.sh full
#
# 先 smoke 再 full:
#   bash HyperMARL-main/scripts/run_ablation_e0_e5_parallel.sh both

set -euo pipefail

MODE="${1:-both}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
HM_ROOT="${REPO_ROOT}/HyperMARL-main"
RUN_TS="$(date +%Y%m%d_%H%M%S)"
OUT_ROOT="${OUT_ROOT:-${REPO_ROOT}/result/ablation_e0_e5_${RUN_TS}}"
TRAIN_SCRIPT="${HM_ROOT}/baselines/MAPPO/mappo_ff_shared_weights_mlp_hypernets_microgrid.py"
CONFIG_NAME="mappo_ff_shared_weights_mlp_hypernets_microgrid"

export PYTHONPATH="${HM_ROOT}:${PYTHONPATH:-}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export SEED="${SEED:-30}"
export NUM_ENVS="${NUM_ENVS:-10}"
# 6 进程并行: 6 × 0.13 ≈ 0.78 < 1.0;留 22% buffer 给 JIT 峰值
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.13}"

SMOKE_STEPS=480
FULL_STEPS="${TOTAL_TIMESTEPS:-120000}"

# ─── 公共 FullCDA-ReserveDemand-Price30 overrides ──────────────────────────
# 完整沿用 microgrid_experiment_overrides.py 内容, 方便独立运行此脚本
COMMON="episode_length:24,multi_day_episode_enable:false,episode_days:1,day_boundary_interval:24,day_boundary_info_enable:true,daily_truncation_enable:false,italian_split_enable:true,italian_split_name:train,terminal_h2_shortfall_value_enable:false"
PRICE="lambda_h2_buy:30.0,lambda_h2_sell:3.0,pv_cap:[7500.0,1500.0,500.0,2000.0],wt_cap:[1500.0,6000.0,3000.0,500.0],load_h_peak:[750.0,600.0,2925.0,3656.25]"
FULLCDA="elec_internal_cda_enable:true,h2_internal_cda_enable:true,gas_network_enable:false,gas_price_dynamic_enable:false,gas_price_bidirectional_enable:false,gas_price_obs_enable:false,gas_pressure_obs_enable:false,h2_transport_loss:0.0,h2_market_schedule_enable:false,h2_market_lag_enable:true,h2_delivery_lag:4,h2_pending_obs_enable:true,h2_pending_obs_horizon:4,h2_pending_summary_obs_enable:true,h2_cap_aware_buy_enable:true,h2_delivery_reservation_enable:true,h2_delivery_reservation_horizon:4,h2_delivery_reservation_ratio:1.0,h2_buyer_reservation_demand_enable:true,h2_buyer_reservation_agent_indices:[2,3],h2_buyer_reservation_target_ratios:[0.0,0.0,0.35,0.45],h2_buyer_reservation_demand_gain:1.0,h2_buyer_reservation_max_order_fraction:0.25"

BASE_OVERRIDES="${COMMON},${PRICE},${FULLCDA}"

# ─── 各实验专属 overrides ───────────────────────────────────────────────────
# E0: 故意保留 h2_price_unit=kwh 的 bug 作为对照
E0_EXTRA="h2_price_unit:kwh"
E0_PPO=""

# E1: h2_price_unit=kg (config.py 已改为默认值, 此处无需额外覆盖)
E1_EXTRA=""
E1_PPO=""

# E2: 降低 consumer H2 负荷使供需可行
# load_h_peak A2: 2925→1500, A3: 3656.25→1800 (kW_thermal, ~48% 降幅)
E2_EXTRA="load_h_peak:[750.0,600.0,1500.0,1800.0]"
E2_PPO=""

# E3: 惩罚项进 reward (penalty_coef 用更保守的 200 避免压过 base_cost 梯度)
E3_EXTRA="penalty_in_reward_enable:true,soc_penalty_coef:200.0,h2_penalty_coef:200.0"
E3_PPO=""

# E4: 内部 H2 成交 bonus 进 reward (0.05 元/kWh_H2 ≈ 12% of base_cost)
E4_EXTRA="h2_internal_trade_bonus_enable:true,h2_internal_trade_bonus_coef:0.05,market_bonus_in_reward_enable:true"
E4_PPO=""

# E5: PPO 稳定化 (hydra 顶层参数, 不走 MICROGRID_CONFIG_OVERRIDES)
E5_EXTRA=""
E5_PPO="ANNEAL_LR=True NUM_MINIBATCHES=4 VF_COEF=0.5"

mkdir -p "${OUT_ROOT}/logs"
echo "${OUT_ROOT}" > "${REPO_ROOT}/result/.last_ablation_out" 2>/dev/null || true

# ─── 单个实验运行函数 ────────────────────────────────────────────────────────
run_exp() {
    local name="$1"       # E0 / E1 / ...
    local extra="$2"      # 追加到 BASE_OVERRIDES 的 microgrid 覆盖
    local ppo="$3"        # 追加到 python 命令的 hydra 顶层参数
    local steps="$4"      # TOTAL_TIMESTEPS
    local log="${OUT_ROOT}/logs/${name}_seed${SEED}_${steps}steps.log"

    local mg_overrides="${BASE_OVERRIDES}"
    if [[ -n "${extra}" ]]; then
        mg_overrides="${mg_overrides},${extra}"
    fi

    local exp_dir="${OUT_ROOT}/${name}_${steps}steps"
    mkdir -p "${exp_dir}/returns"
    export HYPERMARL_OUTPUT_DIR="${exp_dir}"
    export HYPERMARL_PROGRESS_LOG="${exp_dir}/progress.jsonl"

    echo "[${name}] start steps=${steps} log=${log}"

    (
        cd "${HM_ROOT}"
        # shellcheck disable=SC2086
        python "${TRAIN_SCRIPT}" \
            --config-name="${CONFIG_NAME}" \
            ALG="Ablation-${name}-s${SEED}" \
            SEED="${SEED}" \
            TOTAL_TIMESTEPS="${steps}" \
            NUM_ENVS="${NUM_ENVS}" \
            WANDB_MODE="${WANDB_MODE}" \
            CHECKPOINT=False \
            ${ppo} \
            "+MICROGRID_CONFIG_OVERRIDES={${mg_overrides}}"
    ) > "${log}" 2>&1

    local exit_code=$?
    if [[ ${exit_code} -ne 0 ]]; then
        echo "[${name}] FAILED (exit=${exit_code}) — see ${log}" >&2
    else
        echo "[${name}] done steps=${steps}"
    fi
    return ${exit_code}
}

# ─── smoke 模式: 顺序跑 6 个 smoke, 快速验证全部不崩 ───────────────────────
run_smoke_all() {
    echo "=== SMOKE (${SMOKE_STEPS} steps each) ==="
    local failed=0
    for exp_args in \
        "E0|${E0_EXTRA}|${E0_PPO}" \
        "E1|${E1_EXTRA}|${E1_PPO}" \
        "E2|${E2_EXTRA}|${E2_PPO}" \
        "E3|${E3_EXTRA}|${E3_PPO}" \
        "E4|${E4_EXTRA}|${E4_PPO}" \
        "E5|${E5_EXTRA}|${E5_PPO}"
    do
        IFS='|' read -r name extra ppo <<< "${exp_args}"
        run_exp "${name}_smoke" "${extra}" "${ppo}" "${SMOKE_STEPS}" || failed=$((failed+1))
    done
    if [[ ${failed} -gt 0 ]]; then
        echo "SMOKE FAILED for ${failed} experiments. Fix before running full." >&2
        return 1
    fi
    echo "=== ALL SMOKES PASSED ==="
}

# ─── full 模式: 6 个实验进程级并行,错峰启动 ────────────────────────────────
run_full_parallel() {
    echo "=== FULL PARALLEL (${FULL_STEPS} steps each, 6 processes) ==="
    echo "OUT_ROOT=${OUT_ROOT}"
    echo "SEED=${SEED} NUM_ENVS=${NUM_ENVS} XLA_MEM_FRAC=${XLA_PYTHON_CLIENT_MEM_FRACTION}"

    local pids=()

    for exp_args in \
        "E0|${E0_EXTRA}|${E0_PPO}" \
        "E1|${E1_EXTRA}|${E1_PPO}" \
        "E2|${E2_EXTRA}|${E2_PPO}" \
        "E3|${E3_EXTRA}|${E3_PPO}" \
        "E4|${E4_EXTRA}|${E4_PPO}" \
        "E5|${E5_EXTRA}|${E5_PPO}"
    do
        IFS='|' read -r name extra ppo <<< "${exp_args}"
        # 每个实验独立子进程, 不继承 set -e
        (
            set +e
            run_exp "${name}" "${extra}" "${ppo}" "${FULL_STEPS}"
        ) &
        pids+=($!)
        echo "[launcher] ${name} pid=$!"
        # 错峰 15s: 避免 6 个进程同时扎堆编译 CPU/GPU
        sleep 15
    done

    echo "[launcher] waiting for ${#pids[@]} processes..."
    local failed=0
    for pid in "${pids[@]}"; do
        wait "${pid}" || failed=$((failed+1))
    done

    echo "=== FULL PARALLEL DONE (failed=${failed}) $(date) ==="
    echo "DONE $(date)" > "${OUT_ROOT}/logs/all_done.txt"
    return ${failed}
}

# ─── 汇总统计: 打印各实验最终 return (从 progress.jsonl) ────────────────────
summarize_results() {
    echo ""
    echo "=== RESULTS SUMMARY ==="
    for name in E0 E1 E2 E3 E4 E5; do
        local prog="${OUT_ROOT}/${name}_${FULL_STEPS}steps/progress.jsonl"
        if [[ -f "${prog}" ]]; then
            # 最后一行的 episode_return 字段
            local last_ret
            last_ret=$(python3 -c "
import json, sys
lines = [l for l in open('${prog}') if l.strip()]
if lines:
    d = json.loads(lines[-1])
    v = d.get('episode_return', d.get('returned_episode_returns', 'N/A'))
    print(f'{v:.4f}' if isinstance(v, float) else str(v))
else:
    print('no data')
" 2>/dev/null || echo "parse_error")
            echo "  ${name}: last_return=${last_ret}"
        else
            echo "  ${name}: progress log not found"
        fi
    done
    echo ""
    echo "Logs dir: ${OUT_ROOT}/logs/"
    echo "To compare return curves manually:"
    echo "  ls ${OUT_ROOT}/*/returns/*.npy"
}

# ─── dispatch ────────────────────────────────────────────────────────────────
case "${MODE}" in
    smoke)
        run_smoke_all
        ;;
    full)
        run_full_parallel
        summarize_results
        ;;
    both)
        echo "=== Step 1/2: smoke tests ==="
        run_smoke_all
        echo "=== Step 2/2: full parallel runs ==="
        run_full_parallel
        summarize_results
        ;;
    *)
        echo "Usage: $0 [smoke|full|both]" >&2
        exit 1
        ;;
esac
