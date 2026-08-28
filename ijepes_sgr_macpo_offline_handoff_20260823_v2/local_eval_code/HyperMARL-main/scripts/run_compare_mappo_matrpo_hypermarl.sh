#!/usr/bin/env bash
# Compare MAPPO-IA / MATRPO / HyperMARL-MAPPO on microgrid (5000 episodes).
#
# Budget: TOTAL_TIMESTEPS = 5000 * 24 = 120000 (teacher definition)
#
# Usage:
#   bash HyperMARL-main/scripts/run_compare_mappo_matrpo_hypermarl.sh smoke
#   bash HyperMARL-main/scripts/run_compare_mappo_matrpo_hypermarl.sh parallel

set -euo pipefail

MODE="${1:-parallel}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
HM_ROOT="${REPO_ROOT}/HyperMARL-main"
OUT_ROOT="${REPO_ROOT}/result/compare_mappo_matrpo_hypermarl"

export PYTHONPATH="${HM_ROOT}:${PYTHONPATH:-}"
export WANDB_MODE="${WANDB_MODE:-disabled}"
export SEED="${SEED:-30}"

case "${MODE}" in
  smoke) TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-480}" ;;
  *) TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-120000}" ;;
esac

OVERRIDE_ARGS="$(python3 - <<'PY'
from scripts.microgrid_experiment_overrides import HYDRA_OVERRIDE_ARGS
print(HYDRA_OVERRIDE_ARGS)
PY
)"

mkdir -p "${OUT_ROOT}/logs"

run_mappo_ia() {
  local mem_frac="${1:-0.28}"
  local run_tag="mappo_ia_${SEED}_$(date +%Y%m%d_%H%M%S)_$$"
  export XLA_PYTHON_CLIENT_MEM_FRACTION="${mem_frac}"
  export HYPERMARL_OUTPUT_DIR="${OUT_ROOT}/mappo_independent"
  export HYPERMARL_PROGRESS_LOG="${OUT_ROOT}/logs/progress_mappo_ia.jsonl"
  rm -f "${HYPERMARL_PROGRESS_LOG}"
  mkdir -p "${HYPERMARL_OUTPUT_DIR}/returns" "${HYPERMARL_OUTPUT_DIR}/logs"
  local log="${OUT_ROOT}/logs/mappo_ia_seed${SEED}_$(date +%Y%m%d_%H%M%S).log"
  echo "[MAPPO-IA] log=${log} TOTAL_TIMESTEPS=${TOTAL_TIMESTEPS}"
  (
    cd "${HM_ROOT}"
    python baselines/MAPPO/mappo_ff_shared_weights.py \
      --config-name=mappo_ff_independent_actors_microgrid \
      ALG="MAPPO-IA-CTDE-5kEp" \
      EXP_NAME="mappo_ff_independent_actors_microgrid" \
      RUN_NAME="microgrid__MAPPO-IA__seed${SEED}__${run_tag}" \
      SEED="${SEED}" \
      TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS}" \
      WANDB_MODE="${WANDB_MODE}" \
      CHECKPOINT=True \
      "${OVERRIDE_ARGS}"
  ) 2>&1 | tee "${log}"
}

run_hypermarl_mappo() {
  local mem_frac="${1:-0.28}"
  local run_tag="hypermarl_${SEED}_$(date +%Y%m%d_%H%M%S)_$$"
  export XLA_PYTHON_CLIENT_MEM_FRACTION="${mem_frac}"
  export HYPERMARL_OUTPUT_DIR="${OUT_ROOT}/hypermarl_mappo"
  export HYPERMARL_PROGRESS_LOG="${OUT_ROOT}/logs/progress_hypermarl_mappo.jsonl"
  rm -f "${HYPERMARL_PROGRESS_LOG}"
  mkdir -p "${HYPERMARL_OUTPUT_DIR}/returns" "${HYPERMARL_OUTPUT_DIR}/logs"
  local log="${OUT_ROOT}/logs/hypermarl_mappo_seed${SEED}_$(date +%Y%m%d_%H%M%S).log"
  echo "[HyperMARL-MAPPO] log=${log} TOTAL_TIMESTEPS=${TOTAL_TIMESTEPS}"
  (
    cd "${HM_ROOT}"
    python baselines/MAPPO/mappo_ff_shared_weights_mlp_hypernets_microgrid.py \
      --config-name=mappo_ff_shared_weights_mlp_hypernets_microgrid \
      ALG="HyperMARL-MAPPO-5kEp" \
      EXP_NAME="mappo_ff_shared_weights_mlp_hypernets_microgrid" \
      RUN_NAME="microgrid__HyperMARL-MAPPO__seed${SEED}__${run_tag}" \
      SEED="${SEED}" \
      TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS}" \
      WANDB_MODE="${WANDB_MODE}" \
      CHECKPOINT=True \
      "${OVERRIDE_ARGS}"
  ) 2>&1 | tee "${log}"
}

run_matrpo() {
  export HYPERMARL_OUTPUT_DIR="${OUT_ROOT}/matrpo"
  mkdir -p "${HYPERMARL_OUTPUT_DIR}/returns" "${HYPERMARL_OUTPUT_DIR}/logs"
  local log="${OUT_ROOT}/logs/matrpo_seed${SEED}_$(date +%Y%m%d_%H%M%S).log"
  echo "[MATRPO] log=${log} TOTAL_TIMESTEPS=${TOTAL_TIMESTEPS}"
  (
    cd "${HM_ROOT}"
    python baselines/MATRPO/train_matrpo_microgrid.py \
      --seed "${SEED}" \
      --total-timesteps "${TOTAL_TIMESTEPS}" \
      --episode-length 24 \
      --alg "MATRPO-24h-FullCDA-ReserveDemand-5kEp" \
      --log-dir "${OUT_ROOT}/logs"
  ) 2>&1 | tee "${log}"
}

case "${MODE}" in
  mappo) run_mappo_ia 0.70 ;;
  hypermarl) run_hypermarl_mappo 0.70 ;;
  matrpo) run_matrpo ;;
  smoke|parallel)
    echo "=== compare MAPPO-IA / MATRPO / HyperMARL-MAPPO (${MODE}) ==="
    echo "OUT_ROOT=${OUT_ROOT} SEED=${SEED} TOTAL_TIMESTEPS=${TOTAL_TIMESTEPS}"
    run_mappo_ia 0.28 &
    pid1=$!
    run_hypermarl_mappo 0.28 &
    pid2=$!
    run_matrpo &
    pid3=$!
    wait "${pid1}"
    wait "${pid2}"
    wait "${pid3}"
    python "${HM_ROOT}/scripts/plot_compare_mappo_matrpo_hypermarl.py" --out-root "${OUT_ROOT}"
    ;;
  *)
    echo "Unknown mode: ${MODE}" >&2
    exit 1
    ;;
esac
