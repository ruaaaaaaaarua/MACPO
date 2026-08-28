#!/usr/bin/env bash
# CTDE (MAPPO + MLP hypernets) baseline — FullCDA-ReserveDemand + Price30, 40k episodes.
#
# Environment: 24h episode, italian_split=train, seed=30
# Budget: TOTAL_TIMESTEPS=960000 (= 40k episodes × 24 steps × 4 parallel envs)
#
# Usage (from repo root):
#   export PYTHONPATH="${PWD}/HyperMARL-main:${PYTHONPATH}"
#   bash HyperMARL-main/scripts/run_ctde_40k.sh
#
# Short smoke test (~2 min after JIT):
#   TOTAL_TIMESTEPS=4800 bash HyperMARL-main/scripts/run_ctde_40k.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
HM_ROOT="${REPO_ROOT}/HyperMARL-main"

export PYTHONPATH="${HM_ROOT}:${PYTHONPATH:-}"
export WANDB_MODE="${WANDB_MODE:-disabled}"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.70}"
export HYPERMARL_OUTPUT_DIR="${HYPERMARL_OUTPUT_DIR:-${REPO_ROOT}/result/generated}"

CONFIG_NAME="mappo_ff_shared_weights_mlp_hypernets_microgrid"
TRAIN_SCRIPT="${HM_ROOT}/baselines/MAPPO/mappo_ff_shared_weights_mlp_hypernets_microgrid.py"
ALG="${ALG:-HyperMARL-MLP-CTDE-24h-FullCDA-ReserveDemand-40kEp}"
SEED="${SEED:-30}"
TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-960000}"
RUN_LOG_DIR="${RUN_LOG_DIR:-${HYPERMARL_OUTPUT_DIR}/runtime_logs}"
mkdir -p "${HYPERMARL_OUTPUT_DIR}/returns" "${RUN_LOG_DIR}"

if [[ "${HYPERMARL_CAPTURE_LOG:-1}" != "0" ]]; then
  SAFE_ALG="$(printf '%s' "${ALG}" | tr '/ ' '__')"
  RUN_LOG="${RUN_LOG:-${RUN_LOG_DIR}/run_ctde_40k_${SAFE_ALG}_seed${SEED}_$(date +%Y%m%d_%H%M%S).log}"
  exec > >(tee -a "${RUN_LOG}") 2>&1
fi

# FullCDA-RD + Price30Sell3 + Peak50 supply/demand (matches paper main result env)
COMMON_24H_OVERRIDES="episode_length:24,multi_day_episode_enable:false,episode_days:1,day_boundary_interval:24,day_boundary_info_enable:true,daily_truncation_enable:false,italian_split_enable:true,italian_split_name:train,terminal_h2_shortfall_value_enable:false"
PRICE30_SUPPLY_DEMAND_OVERRIDES="lambda_h2_buy:30.0,lambda_h2_sell:3.0,pv_cap:[7500.0,1500.0,500.0,2000.0],wt_cap:[1500.0,6000.0,3000.0,500.0],load_h_peak:[750.0,600.0,2925.0,3656.25]"
FULLCDA_OVERRIDES="elec_internal_cda_enable:true,h2_internal_cda_enable:true,gas_network_enable:false,gas_price_dynamic_enable:false,gas_price_bidirectional_enable:false,gas_price_obs_enable:false,gas_pressure_obs_enable:false,h2_transport_loss:0.0,h2_market_schedule_enable:false,h2_market_lag_enable:true,h2_delivery_lag:4,h2_pending_obs_enable:true,h2_pending_obs_horizon:4,h2_pending_summary_obs_enable:true,h2_cap_aware_buy_enable:true,h2_delivery_reservation_enable:true,h2_delivery_reservation_horizon:4,h2_delivery_reservation_ratio:1.0,h2_buyer_reservation_demand_enable:true,h2_buyer_reservation_agent_indices:[2,3],h2_buyer_reservation_target_ratios:[0.0,0.0,0.35,0.45],h2_buyer_reservation_demand_gain:1.0,h2_buyer_reservation_max_order_fraction:0.25"
MICROGRID_OVERRIDES="{${COMMON_24H_OVERRIDES},${PRICE30_SUPPLY_DEMAND_OVERRIDES},${FULLCDA_OVERRIDES}}"

echo "=== CTDE 40k training ==="
echo "REPO_ROOT=${REPO_ROOT}"
echo "ALG=${ALG}"
echo "SEED=${SEED}"
echo "TOTAL_TIMESTEPS=${TOTAL_TIMESTEPS}"
echo "WANDB_MODE=${WANDB_MODE}"
echo "HYPERMARL_OUTPUT_DIR=${HYPERMARL_OUTPUT_DIR}"
echo "RUN_LOG=${RUN_LOG:-disabled}"
echo "PYTHONPATH=${PYTHONPATH}"
echo "MICROGRID_CONFIG_OVERRIDES=${MICROGRID_OVERRIDES}"
echo

cd "${HM_ROOT}"

python "${TRAIN_SCRIPT}" \
  --config-name="${CONFIG_NAME}" \
  ALG="${ALG}" \
  SEED="${SEED}" \
  TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS}" \
  WANDB_MODE="${WANDB_MODE}" \
  CHECKPOINT=True \
  "+MICROGRID_CONFIG_OVERRIDES={${COMMON_24H_OVERRIDES},${PRICE30_SUPPLY_DEMAND_OVERRIDES},${FULLCDA_OVERRIDES}}"
