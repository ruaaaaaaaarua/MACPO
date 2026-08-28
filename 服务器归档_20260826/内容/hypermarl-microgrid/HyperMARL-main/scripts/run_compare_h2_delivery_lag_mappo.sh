#!/usr/bin/env bash
# Serial MAPPO-IA-CTDE: instant H2 delivery vs fixed 4h delayed delivery.
#
# Budget: TOTAL_TIMESTEPS = 5000 * 24 = 120000 (agent-env transitions, not × NUM_ENVS)
# Parallel: NUM_ENVS=6, CPU threads=3
#
# Usage (from repo root):
#   bash HyperMARL-main/scripts/run_compare_h2_delivery_lag_mappo.sh smoke
#   bash HyperMARL-main/scripts/run_compare_h2_delivery_lag_mappo.sh
#   bash HyperMARL-main/scripts/run_compare_h2_delivery_lag_mappo.sh postprocess

set -euo pipefail

MODE="${1:-serial}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
HM_ROOT="${REPO_ROOT}/HyperMARL-main"
RUN_TS="$(date +%Y%m%d_%H%M%S)"
OUT_ROOT="${OUT_ROOT:-${REPO_ROOT}/result/compare_h2_delivery_lag_mappo/${RUN_TS}}"

export PYTHONPATH="${HM_ROOT}:${PYTHONPATH:-}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export SEED="${SEED:-30}"
export NUM_ENVS="${NUM_ENVS:-6}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-3}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-3}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-3}"
export XLA_FLAGS="${XLA_FLAGS:-} --xla_cpu_multi_thread_eigen=true"

case "${MODE}" in
  smoke) TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-480}" ;;
  *) TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-120000}" ;;
esac

CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-12000}"
NUM_UPDATES=$((TOTAL_TIMESTEPS / 24 / NUM_ENVS))

mkdir -p "${OUT_ROOT}/env_configs" "${OUT_ROOT}/logs" "${OUT_ROOT}/figures" "${OUT_ROOT}/rollouts"
python3 "${HM_ROOT}/scripts/microgrid_experiment_overrides_h2_delivery_lag.py" "${OUT_ROOT}/env_configs"
echo "${OUT_ROOT}" > "${REPO_ROOT}/result/.last_compare_h2_delivery_lag_out"

INSTANT_OVERRIDE_ARGS="$(python3 - <<'PY'
from scripts.microgrid_experiment_overrides_h2_delivery_lag import INSTANT_HYDRA_OVERRIDE_ARGS
print(INSTANT_HYDRA_OVERRIDE_ARGS)
PY
)"
LAG4H_OVERRIDE_ARGS="$(python3 - <<'PY'
from scripts.microgrid_experiment_overrides_h2_delivery_lag import LAG4H_HYDRA_OVERRIDE_ARGS
print(LAG4H_HYDRA_OVERRIDE_ARGS)
PY
)"

run_mappo_instant() {
  local run_tag="instant_${SEED}_${RUN_TS}_$$"
  export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.70}"
  export HYPERMARL_OUTPUT_DIR="${OUT_ROOT}/instant"
  export HYPERMARL_PROGRESS_LOG="${OUT_ROOT}/logs/progress_instant.jsonl"
  export WANDB_DIR="${HYPERMARL_OUTPUT_DIR}/wandb"
  rm -f "${HYPERMARL_PROGRESS_LOG}"
  mkdir -p "${HYPERMARL_OUTPUT_DIR}/returns" "${WANDB_DIR}"
  local log="${OUT_ROOT}/logs/train_instant.log"
  echo "[EnvA-InstantH2] log=${log} TOTAL_TIMESTEPS=${TOTAL_TIMESTEPS} NUM_ENVS=${NUM_ENVS}"
  (
    cd "${HM_ROOT}"
    python baselines/MAPPO/mappo_ff_shared_weights.py \
      --config-name=mappo_ff_independent_actors_microgrid \
      ALG="MAPPO-IA-InstantH2-5kEp" \
      EXP_NAME="mappo_ff_independent_actors_microgrid" \
      RUN_NAME="microgrid__MAPPO-IA-InstantH2__seed${SEED}__${run_tag}" \
      SEED="${SEED}" \
      TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS}" \
      NUM_ENVS="${NUM_ENVS}" \
      WANDB_MODE="${WANDB_MODE}" \
      CHECKPOINT=True \
      CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL}" \
      ANNEAL_LR=True \
      NUM_MINIBATCHES=4 \
      VF_COEF=0.5 \
      ${INSTANT_OVERRIDE_ARGS}
  ) 2>&1 | tee "${log}"
}

run_mappo_lag4h() {
  local run_tag="lag4h_${SEED}_${RUN_TS}_$$"
  export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.70}"
  export HYPERMARL_OUTPUT_DIR="${OUT_ROOT}/lag4h"
  export HYPERMARL_PROGRESS_LOG="${OUT_ROOT}/logs/progress_lag4h.jsonl"
  export WANDB_DIR="${HYPERMARL_OUTPUT_DIR}/wandb"
  rm -f "${HYPERMARL_PROGRESS_LOG}"
  mkdir -p "${HYPERMARL_OUTPUT_DIR}/returns" "${WANDB_DIR}"
  local log="${OUT_ROOT}/logs/train_lag4h.log"
  echo "[EnvB-Lag4hH2] log=${log} TOTAL_TIMESTEPS=${TOTAL_TIMESTEPS} NUM_ENVS=${NUM_ENVS}"
  (
    cd "${HM_ROOT}"
    python baselines/MAPPO/mappo_ff_shared_weights.py \
      --config-name=mappo_ff_independent_actors_microgrid \
      ALG="MAPPO-IA-Lag4hH2-5kEp" \
      EXP_NAME="mappo_ff_independent_actors_microgrid" \
      RUN_NAME="microgrid__MAPPO-IA-Lag4hH2__seed${SEED}__${run_tag}" \
      SEED="${SEED}" \
      TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS}" \
      NUM_ENVS="${NUM_ENVS}" \
      WANDB_MODE="${WANDB_MODE}" \
      CHECKPOINT=True \
      CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL}" \
      ANNEAL_LR=True \
      NUM_MINIBATCHES=4 \
      VF_COEF=0.5 \
      ${LAG4H_OVERRIDE_ARGS}
  ) 2>&1 | tee "${log}"
}

postprocess() {
  python "${HM_ROOT}/scripts/plot_compare_h2_delivery_lag_mappo.py" \
    --out-root "${OUT_ROOT}" \
    --num-envs "${NUM_ENVS}" \
    --total-timesteps "${TOTAL_TIMESTEPS}"
  python "${HM_ROOT}/scripts/analyze_h2_delivery_lag_rollout.py" \
    --out-root "${OUT_ROOT}" \
    --seed "${SEED}"
}

case "${MODE}" in
  instant)
    run_mappo_instant
    ;;
  lag4h)
    run_mappo_lag4h
    ;;
  smoke|serial)
    echo "=== serial H2 delivery lag MAPPO (${MODE}) ==="
    echo "OUT_ROOT=${OUT_ROOT}"
    echo "SEED=${SEED} TOTAL_TIMESTEPS=${TOTAL_TIMESTEPS} NUM_ENVS=${NUM_ENVS}"
    echo "Expected NUM_UPDATES=${NUM_UPDATES}"
    echo "Actual env steps ≈ $((NUM_UPDATES * 24 * NUM_ENVS))"
    run_mappo_instant
    run_mappo_lag4h
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
