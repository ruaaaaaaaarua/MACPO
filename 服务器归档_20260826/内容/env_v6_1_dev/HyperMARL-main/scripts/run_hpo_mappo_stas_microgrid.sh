#!/usr/bin/env bash
# Staged HPO launcher for MAPPO-IA and STAS-MAPPO on the microgrid task.

set -euo pipefail

MODE="${1:-dry-run}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
HM_ROOT="${REPO_ROOT}/HyperMARL-main"
OUT_ROOT="${OUT_ROOT:-${REPO_ROOT}/result/hpo_mappo_stas_20260708}"

export PYTHONPATH="${HM_ROOT}:${PYTHONPATH:-}"
export WANDB_MODE="${WANDB_MODE:-disabled}"

mkdir -p "${OUT_ROOT}/logs"

"${PYTHON:-python}" "${HM_ROOT}/scripts/hpo_mappo_stas_microgrid.py" "${MODE}" --root "${OUT_ROOT}"
