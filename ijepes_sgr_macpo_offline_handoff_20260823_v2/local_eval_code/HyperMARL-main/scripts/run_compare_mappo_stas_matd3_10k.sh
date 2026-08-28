#!/usr/bin/env bash
# Sequential 10k-episode comparison for MAPPO-IA, STAS-MAPPO, and MATD3.

set -euo pipefail

MODE="${1:-sequential}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
HM_ROOT="${REPO_ROOT}/HyperMARL-main"
OUT_ROOT="${OUT_ROOT:-${REPO_ROOT}/result/compare_mappo_stas_matd3_10k}"
SEED="${SEED:-30}"
TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-240000}"
WANDB_MODE="${WANDB_MODE:-disabled}"

if [[ "${MODE}" == "smoke" ]]; then
  TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-480}"
  MATD3_EPISODES="${MATD3_EPISODES:-2}"
  MATD3_CHECKPOINT_INTERVAL="${MATD3_CHECKPOINT_INTERVAL:-1}"
else
  MATD3_EPISODES="${MATD3_EPISODES:-10000}"
  MATD3_CHECKPOINT_INTERVAL="${MATD3_CHECKPOINT_INTERVAL:-1000}"
fi

export PYTHONPATH="${HM_ROOT}:${PYTHONPATH:-}"
export WANDB_MODE

OVERRIDE_ARG="$(cd "${HM_ROOT}" && python - <<'PY'
from scripts.microgrid_experiment_overrides import HYDRA_OVERRIDE_ARGS
print(HYDRA_OVERRIDE_ARGS)
PY
)"

mkdir -p "${OUT_ROOT}/logs"
MANIFEST="${OUT_ROOT}/experiment_manifest.jsonl"
: > "${MANIFEST}"

record_manifest() {
  local name="$1"
  local command="$2"
  python - "$MANIFEST" "$name" "$command" "$SEED" "$TOTAL_TIMESTEPS" "$MATD3_EPISODES" <<'PY'
import json
import sys
from datetime import datetime

path, name, command, seed, total_timesteps, matd3_episodes = sys.argv[1:]
row = {
    "time": datetime.utcnow().isoformat() + "Z",
    "name": name,
    "command": command,
    "seed": int(seed),
    "total_timesteps": int(total_timesteps),
    "matd3_episodes": int(matd3_episodes),
}
with open(path, "a", encoding="utf-8") as f:
    f.write(json.dumps(row, ensure_ascii=False) + "\n")
PY
}

run_mappo_ia() {
  local run_tag="mappo_ia_${SEED}_$(date +%Y%m%d_%H%M%S)"
  export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_MEM_FRACTION:-0.70}"
  export HYPERMARL_OUTPUT_DIR="${OUT_ROOT}/mappo_ia"
  export HYPERMARL_PROGRESS_LOG="${OUT_ROOT}/logs/progress_mappo_ia.jsonl"
  mkdir -p "${HYPERMARL_OUTPUT_DIR}/returns" "${HYPERMARL_OUTPUT_DIR}/logs"
  local log="${OUT_ROOT}/logs/mappo_ia_seed${SEED}_$(date +%Y%m%d_%H%M%S).log"
  local cmd="python baselines/MAPPO/mappo_ff_shared_weights.py --config-name=mappo_ff_independent_actors_microgrid ALG=MAPPO-IA-CTDE-10kEp EXP_NAME=mappo_ff_independent_actors_microgrid RUN_NAME=microgrid__MAPPO-IA__seed${SEED}__${run_tag} SEED=${SEED} TOTAL_TIMESTEPS=${TOTAL_TIMESTEPS} WANDB_MODE=${WANDB_MODE} CHECKPOINT=True EVAL_INTERVAL=100000000 CAPTURE_VIDEO_INTERVAL=null ${OVERRIDE_ARG}"
  echo "[MAPPO-IA] ${cmd}"
  record_manifest "MAPPO-IA" "${cmd}"
  (
    cd "${HM_ROOT}"
    python baselines/MAPPO/mappo_ff_shared_weights.py \
      --config-name=mappo_ff_independent_actors_microgrid \
      ALG="MAPPO-IA-CTDE-10kEp" \
      EXP_NAME="mappo_ff_independent_actors_microgrid" \
      RUN_NAME="microgrid__MAPPO-IA__seed${SEED}__${run_tag}" \
      SEED="${SEED}" \
      TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS}" \
      WANDB_MODE="${WANDB_MODE}" \
      CHECKPOINT=True \
      EVAL_INTERVAL=100000000 \
      CAPTURE_VIDEO_INTERVAL=null \
      "${OVERRIDE_ARG}"
  ) 2>&1 | tee "${log}"
}

run_stas_mappo() {
  local coef="$1"
  local coef_tag="${coef/./p}"
  local run_tag="stas_mix${coef_tag}_${SEED}_$(date +%Y%m%d_%H%M%S)"
  export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_MEM_FRACTION:-0.70}"
  export HYPERMARL_OUTPUT_DIR="${OUT_ROOT}/stas_mix${coef_tag}"
  export HYPERMARL_PROGRESS_LOG="${OUT_ROOT}/logs/progress_stas_mix${coef_tag}.jsonl"
  mkdir -p "${HYPERMARL_OUTPUT_DIR}/returns" "${HYPERMARL_OUTPUT_DIR}/logs"
  local log="${OUT_ROOT}/logs/stas_mix${coef_tag}_seed${SEED}_$(date +%Y%m%d_%H%M%S).log"
  local cmd="python baselines/STAS-MAPPO/mappo_stas.py --config-name=stas_mappo_microgrid ALG=STAS-MAPPO-mix${coef_tag}-10kEp EXP_NAME=stas_mappo_microgrid RUN_NAME=microgrid__STAS-MAPPO-mix${coef_tag}__seed${SEED}__${run_tag} SEED=${SEED} TOTAL_TIMESTEPS=${TOTAL_TIMESTEPS} STAS.MIX_COEF=${coef} WANDB_MODE=${WANDB_MODE} CHECKPOINT=True EVAL_INTERVAL=100000000 CAPTURE_VIDEO_INTERVAL=null ${OVERRIDE_ARG}"
  echo "[STAS-MAPPO ${coef}] ${cmd}"
  record_manifest "STAS-MAPPO-${coef}" "${cmd}"
  (
    cd "${HM_ROOT}"
    python baselines/STAS-MAPPO/mappo_stas.py \
      --config-name=stas_mappo_microgrid \
      ALG="STAS-MAPPO-mix${coef_tag}-10kEp" \
      EXP_NAME="stas_mappo_microgrid" \
      RUN_NAME="microgrid__STAS-MAPPO-mix${coef_tag}__seed${SEED}__${run_tag}" \
      SEED="${SEED}" \
      TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS}" \
      STAS.MIX_COEF="${coef}" \
      WANDB_MODE="${WANDB_MODE}" \
      CHECKPOINT=True \
      EVAL_INTERVAL=100000000 \
      CAPTURE_VIDEO_INTERVAL=null \
      "${OVERRIDE_ARG}"
  ) 2>&1 | tee "${log}"
}

run_matd3() {
  export HYPERMARL_OUTPUT_DIR="${OUT_ROOT}/matd3"
  mkdir -p "${HYPERMARL_OUTPUT_DIR}/returns" "${HYPERMARL_OUTPUT_DIR}/logs"
  local log="${OUT_ROOT}/logs/matd3_seed${SEED}_$(date +%Y%m%d_%H%M%S).log"
  local cmd="python baselines/MATD3/train_matd3_microgrid.py --seed ${SEED} --episodes ${MATD3_EPISODES} --episode-length 24 --alg MATD3-10kEp --checkpoint-interval ${MATD3_CHECKPOINT_INTERVAL}"
  if [[ "${MODE}" == "smoke" ]]; then
    cmd="${cmd} --batch-size 16 --start-steps 0 --update-after 0 --hidden-dim 64 --log-interval 1"
  fi
  echo "[MATD3] ${cmd}"
  record_manifest "MATD3" "${cmd}"
  (cd "${HM_ROOT}" && eval "${cmd}") 2>&1 | tee "${log}"
}

case "${MODE}" in
  smoke)
    echo "=== smoke: STAS smoke + MATD3 tiny run ==="
    (cd "${HM_ROOT}" && python baselines/STAS-MAPPO/smoke_test.py) 2>&1 | tee "${OUT_ROOT}/logs/stas_smoke.log"
    run_matd3
    ;;
  mappo) run_mappo_ia ;;
  stas02) run_stas_mappo 0.2 ;;
  stas05) run_stas_mappo 0.5 ;;
  stas08) run_stas_mappo 0.8 ;;
  matd3) run_matd3 ;;
  sequential)
    echo "=== sequential 10k comparison ==="
    echo "OUT_ROOT=${OUT_ROOT} SEED=${SEED} TOTAL_TIMESTEPS=${TOTAL_TIMESTEPS}"
    run_mappo_ia
    run_stas_mappo 0.2
    run_stas_mappo 0.5
    run_stas_mappo 0.8
    run_matd3
    ;;
  *)
    echo "Unknown mode: ${MODE}" >&2
    exit 1
    ;;
esac
