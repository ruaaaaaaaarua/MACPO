#!/usr/bin/env bash
# H2/kg temp experiment: 4 parallel envs, 40k episodes (teacher CTDE budget).
#
# Budget: TOTAL_TIMESTEPS = 40000 * 24 = 960000 (not multiplied by NUM_ENVS)
# Parallel: NUM_ENVS=4  =>  NUM_UPDATES = 960000 // 24 // 4 = 10000
# Environment: FullCDA-ReserveDemand + Price30/kg (h2_price_unit=kg)
#
# Usage (from repo root):
#   bash HyperMARL-main/scripts/run_compare_mappo_ia_hypermarl_serial_h2_kg_4env_40k.sh smoke
#   bash HyperMARL-main/scripts/run_compare_mappo_ia_hypermarl_serial_h2_kg_4env_40k.sh serial

set -euo pipefail

MODE="${1:-serial}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
HM_ROOT="${REPO_ROOT}/HyperMARL-main"
RUN_TS="$(date +%Y%m%d_%H%M%S)"
OUT_ROOT="${OUT_ROOT:-${REPO_ROOT}/result/compare_mappo_ia_hypermarl_h2kg_4env_40kep_${RUN_TS}}"

export PYTHONPATH="${HM_ROOT}:${PYTHONPATH:-}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export SEED="${SEED:-30}"
export NUM_ENVS="${NUM_ENVS:-4}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-8}"
export XLA_FLAGS="${XLA_FLAGS:-} --xla_cpu_multi_thread_eigen=true"

IA_ALG_TAG="MAPPO-IA-CTDE-40kEp-h2kg-4env"
HYPER_ALG_TAG="HyperMARL-MAPPO-40kEp-h2kg-4env"
EPISODE_LENGTH=24

case "${MODE}" in
  smoke) TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-480}" ;;
  *) TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-960000}" ;;
esac

NUM_EPISODES=$((TOTAL_TIMESTEPS / EPISODE_LENGTH))
NUM_UPDATES=$((TOTAL_TIMESTEPS / EPISODE_LENGTH / NUM_ENVS))

OVERRIDE_ARGS="$(python3 - <<'PY'
from scripts.microgrid_experiment_overrides_h2_kg_tmp import HYDRA_OVERRIDE_ARGS
print(HYDRA_OVERRIDE_ARGS)
PY
)"
# Checkpoint every 100 updates @ NUM_ENVS=4: 100 * 4 * 24 = 9600 env steps.
CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-9600}"

mkdir -p "${OUT_ROOT}/logs" "${OUT_ROOT}/actors" "${OUT_ROOT}/action_analysis"
echo "${OUT_ROOT}" > "${REPO_ROOT}/result/.last_compare_mappo_ia_hypermarl_h2kg_4env_out"

run_mappo_ia() {
  local run_tag="mappo_ia_h2kg_4env_${SEED}_${RUN_TS}_$$"
  export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.70}"
  export HYPERMARL_OUTPUT_DIR="${OUT_ROOT}/mappo_independent"
  export HYPERMARL_PROGRESS_LOG="${OUT_ROOT}/logs/progress_mappo_ia.jsonl"
  export WANDB_DIR="${HYPERMARL_OUTPUT_DIR}/wandb"
  rm -f "${HYPERMARL_PROGRESS_LOG}"
  mkdir -p "${HYPERMARL_OUTPUT_DIR}/returns" "${WANDB_DIR}"
  local log="${OUT_ROOT}/logs/mappo_ia_seed${SEED}_${RUN_TS}.log"
  echo "[MAPPO-IA-CTDE h2/kg 4env] log=${log} TOTAL_TIMESTEPS=${TOTAL_TIMESTEPS} NUM_ENVS=${NUM_ENVS}"
  (
    cd "${HM_ROOT}"
    python baselines/MAPPO/mappo_ff_shared_weights.py \
      --config-name=mappo_ff_independent_actors_microgrid \
      ALG="${IA_ALG_TAG}" \
      EXP_NAME="mappo_ff_independent_actors_microgrid" \
      RUN_NAME="microgrid__MAPPO-IA-h2kg-4env__seed${SEED}__${run_tag}" \
      SEED="${SEED}" \
      TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS}" \
      NUM_ENVS="${NUM_ENVS}" \
      WANDB_MODE="${WANDB_MODE}" \
      CHECKPOINT=True \
      CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL}" \
      "${OVERRIDE_ARGS}"
  ) 2>&1 | tee "${log}"
}

run_hypermarl_mappo() {
  local run_tag="hypermarl_h2kg_4env_${SEED}_${RUN_TS}_$$"
  export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.70}"
  export HYPERMARL_OUTPUT_DIR="${OUT_ROOT}/hypermarl_mappo"
  export HYPERMARL_PROGRESS_LOG="${OUT_ROOT}/logs/progress_hypermarl_mappo.jsonl"
  export WANDB_DIR="${HYPERMARL_OUTPUT_DIR}/wandb"
  rm -f "${HYPERMARL_PROGRESS_LOG}"
  mkdir -p "${HYPERMARL_OUTPUT_DIR}/returns" "${WANDB_DIR}"
  local log="${OUT_ROOT}/logs/hypermarl_mappo_seed${SEED}_${RUN_TS}.log"
  echo "[HyperMARL-MAPPO h2/kg 4env] log=${log} TOTAL_TIMESTEPS=${TOTAL_TIMESTEPS} NUM_ENVS=${NUM_ENVS}"
  (
    cd "${HM_ROOT}"
    python baselines/MAPPO/mappo_ff_shared_weights_mlp_hypernets_microgrid.py \
      --config-name=mappo_ff_shared_weights_mlp_hypernets_microgrid \
      ALG="${HYPER_ALG_TAG}" \
      EXP_NAME="mappo_ff_shared_weights_mlp_hypernets_microgrid" \
      RUN_NAME="microgrid__HyperMARL-MAPPO-h2kg-4env__seed${SEED}__${run_tag}" \
      SEED="${SEED}" \
      TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS}" \
      NUM_ENVS="${NUM_ENVS}" \
      WANDB_MODE="${WANDB_MODE}" \
      CHECKPOINT=True \
      CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL}" \
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
      --algorithm "${IA_ALG_TAG}" \
      --seed "${SEED}"
  else
    echo "[warn] MAPPO-IA checkpoint not found under ${OUT_ROOT}/mappo_independent" >&2
  fi
  if [[ -n "${ckpt_hyper}" ]]; then
    python "${HM_ROOT}/scripts/export_actor_weights.py" \
      --checkpoint-dir "${ckpt_hyper}" \
      --out-npz "${OUT_ROOT}/actors/hypermarl_mappo_actor_seed${SEED}.npz" \
      --algorithm "${HYPER_ALG_TAG}" \
      --seed "${SEED}"
  else
    echo "[warn] HyperMARL-MAPPO checkpoint not found under ${OUT_ROOT}/hypermarl_mappo" >&2
  fi
}

postprocess() {
  python "${HM_ROOT}/scripts/plot_compare_mappo_ia_hypermarl.py" \
    --out-root "${OUT_ROOT}" \
    --num-envs "${NUM_ENVS}" \
    --total-timesteps "${TOTAL_TIMESTEPS}" \
    --episode-length "${EPISODE_LENGTH}" \
    --ia-tag "${IA_ALG_TAG}" \
    --hyper-tag "${HYPER_ALG_TAG}" \
    --env-note "FullCDA-ReserveDemand + Price30/kg（h2_price_unit=kg）；4 并行 env；内部 P2P 互济经 CDA 撮合"
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
    echo "=== serial MAPPO-IA-CTDE -> HyperMARL-MAPPO h2/kg 4env (${MODE}) ==="
    echo "OUT_ROOT=${OUT_ROOT}"
    echo "SEED=${SEED} TOTAL_TIMESTEPS=${TOTAL_TIMESTEPS} NUM_ENVS=${NUM_ENVS}"
    echo "Expected episodes=${NUM_EPISODES} NUM_UPDATES=${NUM_UPDATES}"
    run_mappo_ia
    run_hypermarl_mappo
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
