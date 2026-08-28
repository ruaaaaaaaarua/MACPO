#!/usr/bin/env bash
# Serial compare per prompt: MAPPO-IA-CTDE then HyperMARL-MAPPO.
#
# Budget: TOTAL_TIMESTEPS = 5000 * 24 = 120000 (not multiplied by NUM_ENVS)
# Parallel: NUM_ENVS=10, CPU threads=8
# Environment: FullCDA-ReserveDemand + Price30 (internal P2P via CDA)
#
# Usage (from repo root):
#   bash HyperMARL-main/scripts/run_compare_mappo_ia_hypermarl_serial.sh smoke
#   bash HyperMARL-main/scripts/run_compare_mappo_ia_hypermarl_serial.sh serial

set -euo pipefail

MODE="${1:-serial}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
HM_ROOT="${REPO_ROOT}/HyperMARL-main"
RUN_TS="$(date +%Y%m%d_%H%M%S)"
OUT_ROOT="${OUT_ROOT:-${REPO_ROOT}/result/compare_mappo_ia_hypermarl_10env_5kep_${RUN_TS}}"

export PYTHONPATH="${HM_ROOT}:${PYTHONPATH:-}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export SEED="${SEED:-30}"
export NUM_ENVS="${NUM_ENVS:-10}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-8}"
export XLA_FLAGS="${XLA_FLAGS:-} --xla_cpu_multi_thread_eigen=true"

case "${MODE}" in
  smoke) TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-480}" ;;
  *) TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-120000}" ;;
esac

OVERRIDE_ARGS="$(python3 - <<'PY'
from scripts.microgrid_experiment_overrides import HYDRA_OVERRIDE_ARGS
print(HYDRA_OVERRIDE_ARGS)
PY
)"
# Checkpoint every 100 updates for periodic action eval (24000 env steps @ NUM_ENVS=10).
CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-24000}"

mkdir -p "${OUT_ROOT}/logs" "${OUT_ROOT}/actors" "${OUT_ROOT}/action_analysis"
echo "${OUT_ROOT}" > "${REPO_ROOT}/result/.last_compare_mappo_ia_hypermarl_out"

run_mappo_ia() {
  local run_tag="mappo_ia_${SEED}_${RUN_TS}_$$"
  export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.70}"
  export HYPERMARL_OUTPUT_DIR="${OUT_ROOT}/mappo_independent"
  export HYPERMARL_PROGRESS_LOG="${OUT_ROOT}/logs/progress_mappo_ia.jsonl"
  export WANDB_DIR="${HYPERMARL_OUTPUT_DIR}/wandb"
  rm -f "${HYPERMARL_PROGRESS_LOG}"
  mkdir -p "${HYPERMARL_OUTPUT_DIR}/returns" "${WANDB_DIR}"
  local log="${OUT_ROOT}/logs/mappo_ia_seed${SEED}_${RUN_TS}.log"
  echo "[MAPPO-IA-CTDE] log=${log} TOTAL_TIMESTEPS=${TOTAL_TIMESTEPS} NUM_ENVS=${NUM_ENVS}"
  (
    cd "${HM_ROOT}"
    python baselines/MAPPO/mappo_ff_shared_weights.py \
      --config-name=mappo_ff_independent_actors_microgrid \
      ALG="MAPPO-IA-CTDE-5kEp" \
      EXP_NAME="mappo_ff_independent_actors_microgrid" \
      RUN_NAME="microgrid__MAPPO-IA__seed${SEED}__${run_tag}" \
      SEED="${SEED}" \
      TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS}" \
      NUM_ENVS="${NUM_ENVS}" \
      WANDB_MODE="${WANDB_MODE}" \
      CHECKPOINT=True \
      CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL}" \
      ANNEAL_LR=True \
      NUM_MINIBATCHES=4 \
      VF_COEF=0.5 \
      "${OVERRIDE_ARGS}"
  ) 2>&1 | tee "${log}"
}

run_hypermarl_mappo() {
  local run_tag="hypermarl_${SEED}_${RUN_TS}_$$"
  export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.70}"
  export HYPERMARL_OUTPUT_DIR="${OUT_ROOT}/hypermarl_mappo"
  export HYPERMARL_PROGRESS_LOG="${OUT_ROOT}/logs/progress_hypermarl_mappo.jsonl"
  export WANDB_DIR="${HYPERMARL_OUTPUT_DIR}/wandb"
  rm -f "${HYPERMARL_PROGRESS_LOG}"
  mkdir -p "${HYPERMARL_OUTPUT_DIR}/returns" "${WANDB_DIR}"
  local log="${OUT_ROOT}/logs/hypermarl_mappo_seed${SEED}_${RUN_TS}.log"
  echo "[HyperMARL-MAPPO] log=${log} TOTAL_TIMESTEPS=${TOTAL_TIMESTEPS} NUM_ENVS=${NUM_ENVS}"
  (
    cd "${HM_ROOT}"
    python baselines/MAPPO/mappo_ff_shared_weights_mlp_hypernets_microgrid.py \
      --config-name=mappo_ff_shared_weights_mlp_hypernets_microgrid \
      ALG="HyperMARL-MAPPO-5kEp" \
      EXP_NAME="mappo_ff_shared_weights_mlp_hypernets_microgrid" \
      RUN_NAME="microgrid__HyperMARL-MAPPO__seed${SEED}__${run_tag}" \
      SEED="${SEED}" \
      TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS}" \
      NUM_ENVS="${NUM_ENVS}" \
      WANDB_MODE="${WANDB_MODE}" \
      CHECKPOINT=True \
      CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL}" \
      ANNEAL_LR=True \
      NUM_MINIBATCHES=4 \
      VF_COEF=0.5 \
      "${OVERRIDE_ARGS}"
  ) 2>&1 | tee "${log}"
}

find_latest_checkpoint() {
  local search_root="$1"
  find "${search_root}" -type d -name "*_steps_*_updates.*" 2>/dev/null | sort | tail -1
}

export_actors() {
  local ckpt_ia
  local ckpt_hyper
  ckpt_ia="$(find_latest_checkpoint "${OUT_ROOT}/mappo_independent")"
  ckpt_hyper="$(find_latest_checkpoint "${OUT_ROOT}/hypermarl_mappo")"
  if [[ -n "${ckpt_ia}" ]]; then
    python "${HM_ROOT}/scripts/export_actor_weights.py" \
      --checkpoint-dir "${ckpt_ia}" \
      --out-npz "${OUT_ROOT}/actors/mappo_ia_actor_seed${SEED}.npz" \
      --algorithm "MAPPO-IA-CTDE-5kEp" \
      --seed "${SEED}"
  else
    echo "[warn] MAPPO-IA checkpoint not found under ${OUT_ROOT}/mappo_independent" >&2
  fi
  if [[ -n "${ckpt_hyper}" ]]; then
    python "${HM_ROOT}/scripts/export_actor_weights.py" \
      --checkpoint-dir "${ckpt_hyper}" \
      --out-npz "${OUT_ROOT}/actors/hypermarl_mappo_actor_seed${SEED}.npz" \
      --algorithm "HyperMARL-MAPPO-5kEp" \
      --seed "${SEED}"
  else
    echo "[warn] HyperMARL-MAPPO checkpoint not found under ${OUT_ROOT}/hypermarl_mappo" >&2
  fi
}

postprocess() {
  python "${HM_ROOT}/scripts/plot_compare_mappo_ia_hypermarl.py" \
    --out-root "${OUT_ROOT}" \
    --num-envs "${NUM_ENVS}"
  export_actors
  python "${HM_ROOT}/scripts/analyze_microgrid_policy_actions.py" \
    --out-root "${OUT_ROOT}" \
    --seed "${SEED}" \
    --all-checkpoints
}

case "${MODE}" in
  mappo)
    run_mappo_ia
    ;;
  hypermarl)
    run_hypermarl_mappo
    ;;
  smoke|serial)
    echo "=== serial MAPPO-IA-CTDE -> HyperMARL-MAPPO (${MODE}) ==="
    echo "OUT_ROOT=${OUT_ROOT}"
    echo "SEED=${SEED} TOTAL_TIMESTEPS=${TOTAL_TIMESTEPS} NUM_ENVS=${NUM_ENVS}"
    echo "Expected NUM_UPDATES=$((TOTAL_TIMESTEPS / 24 / NUM_ENVS))"
    run_mappo_ia
    run_hypermarl_mappo
    postprocess
    echo "ALL DONE $(date)" | tee "${OUT_ROOT}/logs/all_done.txt"
    ;;
  parallel)
    # 两个算法并行启动,各占 40% GPU 显存,完成后统一后处理
    echo "=== parallel MAPPO-IA-CTDE || HyperMARL-MAPPO ==="
    echo "OUT_ROOT=${OUT_ROOT}"
    echo "SEED=${SEED} TOTAL_TIMESTEPS=${TOTAL_TIMESTEPS} NUM_ENVS=${NUM_ENVS}"
    echo "Expected NUM_UPDATES=$((TOTAL_TIMESTEPS / 24 / NUM_ENVS))"
    export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.40}"
    run_mappo_ia &
    PID_IA=$!
    sleep 20   # 错峰 JIT 编译
    run_hypermarl_mappo &
    PID_HM=$!
    echo "[launcher] MAPPO-IA pid=${PID_IA}  HyperMARL pid=${PID_HM}"
    wait "${PID_IA}" && echo "[done] MAPPO-IA finished"
    wait "${PID_HM}" && echo "[done] HyperMARL finished"
    postprocess
    echo "ALL DONE $(date)" | tee "${OUT_ROOT}/logs/all_done.txt"
    ;;
  postprocess)
    postprocess
    ;;
  *)
    echo "Unknown mode: ${MODE}" >&2
    exit 1
    ;;
esac
