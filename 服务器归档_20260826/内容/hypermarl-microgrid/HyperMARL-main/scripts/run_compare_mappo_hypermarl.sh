#!/usr/bin/env bash
# Compare plain MAPPO vs HyperMARL-MAPPO on the same microgrid env and budget.
#
# Usage (from repo root):
#   bash HyperMARL-main/scripts/run_compare_mappo_hypermarl.sh smoke
#   bash HyperMARL-main/scripts/run_compare_mappo_hypermarl.sh parallel
#   bash HyperMARL-main/scripts/run_compare_mappo_hypermarl.sh mappo
#   bash HyperMARL-main/scripts/run_compare_mappo_hypermarl.sh hypermarl
#
# Environment overrides are copied unchanged from run_ctde_40k.sh (teacher params).

set -euo pipefail

MODE="${1:-parallel}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
HM_ROOT="${REPO_ROOT}/HyperMARL-main"
OUT_ROOT="${REPO_ROOT}/result/compare_mappo_hypermarl"

export PYTHONPATH="${HM_ROOT}:${PYTHONPATH:-}"
export WANDB_MODE="${WANDB_MODE:-disabled}"
export HYPERMARL_OUTPUT_DIR="${HYPERMARL_OUTPUT_DIR:-${OUT_ROOT}}"
export SEED="${SEED:-30}"

case "${MODE}" in
  smoke) TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-4800}" ;;
  *) TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-480000}" ;;
esac

COMMON_24H_OVERRIDES="episode_length:24,multi_day_episode_enable:false,episode_days:1,day_boundary_interval:24,day_boundary_info_enable:true,daily_truncation_enable:false,italian_split_enable:true,italian_split_name:train,terminal_h2_shortfall_value_enable:false"
PRICE30_SUPPLY_DEMAND_OVERRIDES="lambda_h2_buy:30.0,lambda_h2_sell:3.0,pv_cap:[7500.0,1500.0,500.0,2000.0],wt_cap:[1500.0,6000.0,3000.0,500.0],load_h_peak:[750.0,600.0,2925.0,3656.25]"
FULLCDA_OVERRIDES="elec_internal_cda_enable:true,h2_internal_cda_enable:true,gas_network_enable:false,gas_price_dynamic_enable:false,gas_price_bidirectional_enable:false,gas_price_obs_enable:false,gas_pressure_obs_enable:false,h2_transport_loss:0.0,h2_market_schedule_enable:false,h2_market_lag_enable:true,h2_delivery_lag:4,h2_pending_obs_enable:true,h2_pending_obs_horizon:4,h2_pending_summary_obs_enable:true,h2_cap_aware_buy_enable:true,h2_delivery_reservation_enable:true,h2_delivery_reservation_horizon:4,h2_delivery_reservation_ratio:1.0,h2_buyer_reservation_demand_enable:true,h2_buyer_reservation_agent_indices:[2,3],h2_buyer_reservation_target_ratios:[0.0,0.0,0.35,0.45],h2_buyer_reservation_demand_gain:1.0,h2_buyer_reservation_max_order_fraction:0.25"
OVERRIDE_ARGS="+MICROGRID_CONFIG_OVERRIDES={${COMMON_24H_OVERRIDES},${PRICE30_SUPPLY_DEMAND_OVERRIDES},${FULLCDA_OVERRIDES}}"

mkdir -p "${OUT_ROOT}/returns" "${OUT_ROOT}/logs" "${OUT_ROOT}/runtime_logs"

run_mappo_baseline() {
  local mem_frac="${1:-0.45}"
  local run_tag="mappo_${SEED}_$(date +%Y%m%d_%H%M%S)_$$"
  export XLA_PYTHON_CLIENT_MEM_FRACTION="${mem_frac}"
  export HYPERMARL_OUTPUT_DIR="${OUT_ROOT}/mappo_baseline"
  mkdir -p "${HYPERMARL_OUTPUT_DIR}/returns" "${HYPERMARL_OUTPUT_DIR}/runtime_logs"
  local log="${OUT_ROOT}/logs/mappo_baseline_seed${SEED}_$(date +%Y%m%d_%H%M%S).log"
  echo "[MAPPO-baseline] log=${log} TOTAL_TIMESTEPS=${TOTAL_TIMESTEPS}"
  (
    cd "${HM_ROOT}"
    python baselines/MAPPO/mappo_ff_shared_weights_microgrid.py \
      --config-name=mappo_ff_shared_weights_microgrid \
      ALG="MAPPO-baseline-24h-FullCDA-ReserveDemand-5kEp" \
      EXP_NAME="mappo_ff_shared_weights_microgrid" \
      RUN_NAME="microgrid__MAPPO-baseline__seed${SEED}__${run_tag}" \
      SEED="${SEED}" \
      TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS}" \
      WANDB_MODE="${WANDB_MODE}" \
      CHECKPOINT=True \
      "${OVERRIDE_ARGS}"
  ) 2>&1 | tee "${log}"
}

run_hypermarl_mappo() {
  local mem_frac="${1:-0.45}"
  local run_tag="hypermarl_${SEED}_$(date +%Y%m%d_%H%M%S)_$$"
  export XLA_PYTHON_CLIENT_MEM_FRACTION="${mem_frac}"
  export HYPERMARL_OUTPUT_DIR="${OUT_ROOT}/hypermarl_mappo"
  mkdir -p "${HYPERMARL_OUTPUT_DIR}/returns" "${HYPERMARL_OUTPUT_DIR}/runtime_logs"
  local log="${OUT_ROOT}/logs/hypermarl_mappo_seed${SEED}_$(date +%Y%m%d_%H%M%S).log"
  echo "[HyperMARL-MAPPO] log=${log} TOTAL_TIMESTEPS=${TOTAL_TIMESTEPS}"
  (
    cd "${HM_ROOT}"
    python baselines/MAPPO/mappo_ff_shared_weights_mlp_hypernets_microgrid.py \
      --config-name=mappo_ff_shared_weights_mlp_hypernets_microgrid \
      ALG="HyperMARL-MAPPO-24h-FullCDA-ReserveDemand-5kEp" \
      EXP_NAME="mappo_ff_shared_weights_mlp_hypernets_microgrid" \
      RUN_NAME="microgrid__HyperMARL-MAPPO__seed${SEED}__${run_tag}" \
      SEED="${SEED}" \
      TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS}" \
      WANDB_MODE="${WANDB_MODE}" \
      CHECKPOINT=True \
      "${OVERRIDE_ARGS}"
  ) 2>&1 | tee "${log}"
}

case "${MODE}" in
  mappo)
    run_mappo_baseline "${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.70}"
    ;;
  hypermarl)
    run_hypermarl_mappo "${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.70}"
    ;;
  smoke|parallel)
    echo "=== compare MAPPO vs HyperMARL-MAPPO (${MODE}) ==="
    echo "REPO_ROOT=${REPO_ROOT}"
    echo "OUT_ROOT=${OUT_ROOT}"
    echo "SEED=${SEED} TOTAL_TIMESTEPS=${TOTAL_TIMESTEPS}"
    run_mappo_baseline 0.45 &
    pid_mappo=$!
    run_hypermarl_mappo 0.45 &
    pid_hyper=$!
    wait "${pid_mappo}"
    wait "${pid_hyper}"
    python "${HM_ROOT}/scripts/plot_compare_mappo_hypermarl.py" --out-root "${OUT_ROOT}"
    ;;
  *)
    echo "Unknown mode: ${MODE}" >&2
    exit 1
    ;;
esac
