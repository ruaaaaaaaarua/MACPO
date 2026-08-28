"""Single-seed runner definitions for the exploratory env-v3-safe comparison."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from baselines.MAPPO.safe_gru_trainer import SafeGRUMAPPOTrainer
from scripts.env_v3_safe_overrides import env_v3_safe_overrides, hydra_override_arg


# Derived from the final twenty stochastic daily costs of the env-v3 balanced
# run (median / deterministic cost = 0.495, clipped to rho=1.0) and its
# empirical balanced reference budget.  Keeping both values explicit makes
# the v4 course reproducible without reading historical run files at launch.
V4_CURRICULUM_RHO = 1.0
V4_EVAL_BALANCED_BUDGET = 0.4662967833789416
V4_CURRICULUM_TARGET = V4_CURRICULUM_RHO * V4_EVAL_BALANCED_BUDGET


EXPERIMENTS = {
    "dense_ff_mappo_anneal": {
        "family": "ff",
        "algorithm": "mappo",
        "seed": 30,
        "exploratory": True,
    },
    "dense_gru_mappo_anneal": {
        "family": "gru",
        "algorithm": "mappo",
        "seed": 30,
        "exploratory": True,
    },
    "dense_gru_mappo_lagrangian": {
        "family": "gru",
        "algorithm": "lagrangian",
        "seed": 30,
        "exploratory": True,
    },
    "dense_gru_macpo": {
        "family": "gru",
        "algorithm": "macpo",
        "seed": 30,
        "exploratory": True,
    },
    "dense_gru_macpo_history_communication": {
        "family": "gru",
        "algorithm": "macpo",
        "seed": 30,
        "exploratory": True,
        "include_previous_action": True,
        "include_transaction_message": True,
    },
    "dense_gru_macpo_two_stage_intent_d_strict": {
        "family": "gru",
        "algorithm": "macpo",
        "seed": 30,
        "exploratory": True,
        "include_previous_action": True,
        "include_transaction_message": True,
        "two_stage_intent": True,
        "intent_dim": 3,
        "intent_broadcast_mode": "full",
        # Replaced by the empirical safety reference before a formal run.
        "cost_budget": 0.0,
        "budget_mode": "strict",
    },
    "dense_gru_macpo_two_stage_intent_d_balanced": {
        "family": "gru",
        "algorithm": "macpo",
        "seed": 30,
        "exploratory": True,
        "include_previous_action": True,
        "include_transaction_message": True,
        "two_stage_intent": True,
        "intent_dim": 3,
        "intent_broadcast_mode": "full",
        "cost_budget": 0.3,
        "budget_mode": "balanced",
    },
    "dense_gru_macpo_two_stage_intent_no_broadcast_d_balanced": {
        "family": "gru",
        "algorithm": "macpo",
        "seed": 30,
        "exploratory": True,
        "include_previous_action": True,
        "include_transaction_message": True,
        "two_stage_intent": True,
        "intent_dim": 3,
        "intent_broadcast_mode": "other_zero",
        "cost_budget": 0.3,
        "budget_mode": "balanced",
    },
    "v4_full_intent_curriculum": {
        "family": "gru",
        "algorithm": "macpo",
        "seed": 30,
        "exploratory": True,
        "include_previous_action": True,
        "include_transaction_message": True,
        "two_stage_intent": True,
        "intent_dim": 3,
        "intent_broadcast_mode": "full",
        "intent_residual_limit": 0.25,
        "intent_residual_coef": 0.01,
        "cost_budget": V4_CURRICULUM_TARGET,
        "curriculum_d_start": 3.5,
        "curriculum_d_target": V4_CURRICULUM_TARGET,
        "curriculum_updates": 200,
        "curriculum_log_std_start": -1.0,
        "curriculum_log_std_end": -2.3,
        "env_overrides": {
            "h2_day_ahead_forecast_enable": True,
            "h2_day_ahead_forecast_horizons": [4, 6, 10],
        },
    },
    "v4_no_broadcast_curriculum": {
        "family": "gru",
        "algorithm": "macpo",
        "seed": 30,
        "exploratory": True,
        "include_previous_action": True,
        "include_transaction_message": True,
        "two_stage_intent": True,
        "intent_dim": 3,
        "intent_broadcast_mode": "other_zero",
        "intent_residual_limit": 0.25,
        "intent_residual_coef": 0.01,
        "cost_budget": V4_CURRICULUM_TARGET,
        "curriculum_d_start": 3.5,
        "curriculum_d_target": V4_CURRICULUM_TARGET,
        "curriculum_updates": 200,
        "curriculum_log_std_start": -1.0,
        "curriculum_log_std_end": -2.3,
        "env_overrides": {
            "h2_day_ahead_forecast_enable": True,
            "h2_day_ahead_forecast_horizons": [4, 6, 10],
        },
    },
    "v5_supply_intent_full": {
        "family": "gru", "algorithm": "macpo", "seed": 30, "exploratory": True,
        "include_previous_action": True, "include_transaction_message": True,
        "two_stage_intent": True, "intent_dim": 3, "intent_broadcast_mode": "full",
        "intent_residual_limit": 0.25, "intent_residual_coef": 0.01,
        "h2_supply_intent_message_enable": True, "cost_budget": 0.0,
        "env_overrides": {"h2_supply_intent_message_enable": True},
    },
    "v5_supply_intent_no_broadcast": {
        "family": "gru", "algorithm": "macpo", "seed": 30, "exploratory": True,
        "include_previous_action": True, "include_transaction_message": True,
        "two_stage_intent": True, "intent_dim": 3, "intent_broadcast_mode": "other_zero",
        "intent_residual_limit": 0.25, "intent_residual_coef": 0.01,
        "h2_supply_intent_message_enable": True, "cost_budget": 0.0,
        "env_overrides": {"h2_supply_intent_message_enable": True},
    },
    "v52_full_gru_mappo": {
        "family": "gru", "algorithm": "mappo", "seed": 30, "exploratory": True,
        "include_previous_action": True, "include_transaction_message": True,
        "two_stage_intent": True, "intent_dim": 3,
        "intent_broadcast_mode": "full", "communication_scope": "full",
        "intent_residual_limit": 0.25, "intent_residual_coef": 0.01,
        "h2_supply_intent_message_enable": True,
        "curriculum_updates": 300,
        "curriculum_log_std_start": -1.0, "curriculum_log_std_end": -2.3,
        "env_overrides": {"h2_supply_intent_message_enable": True, "soc_init": 0.5},
    },
    "v52_full_gru_macpo": {
        "family": "gru", "algorithm": "macpo", "seed": 30, "exploratory": True,
        "include_previous_action": True, "include_transaction_message": True,
        "two_stage_intent": True, "intent_dim": 3,
        "intent_broadcast_mode": "full", "communication_scope": "full",
        "intent_residual_limit": 0.25, "intent_residual_coef": 0.01,
        "h2_supply_intent_message_enable": True,
        "cost_budget": 0.0, "budget_mode": "v52_nominal_curriculum",
        "curriculum_d_start": 0.0, "curriculum_d_target": 0.0,
        "curriculum_updates": 300,
        "curriculum_log_std_start": -1.0, "curriculum_log_std_end": -2.3,
        "env_overrides": {"h2_supply_intent_message_enable": True, "soc_init": 0.5},
    },
    "v52_self_only_gru_macpo": {
        "family": "gru", "algorithm": "macpo", "seed": 30, "exploratory": True,
        "include_previous_action": True, "include_transaction_message": True,
        "two_stage_intent": True, "intent_dim": 3,
        "intent_broadcast_mode": "self_only", "communication_scope": "self_only",
        "intent_residual_limit": 0.25, "intent_residual_coef": 0.01,
        "h2_supply_intent_message_enable": True,
        "cost_budget": 0.0, "budget_mode": "v52_nominal_curriculum",
        "curriculum_d_start": 0.0, "curriculum_d_target": 0.0,
        "curriculum_updates": 300,
        "curriculum_log_std_start": -1.0, "curriculum_log_std_end": -2.3,
        "env_overrides": {"h2_supply_intent_message_enable": True, "soc_init": 0.5},
    },
    "v6_nocomm_gru_mappo": {
        "family": "gru", "algorithm": "mappo", "seed": 30, "exploratory": True,
        "num_envs": 2, "env_parallel_backend": "process", "fused_rollout_kernel": True,
        "include_previous_action": True, "include_transaction_message": False,
        "two_stage_intent": False, "h2_supply_intent_message_enable": False,
        "cost_budget": 1.0, "voltage_cost_scale": 0.02,
        "curriculum_updates": 300,
        "curriculum_log_std_start": -1.0, "curriculum_log_std_end": -2.3,
        "env_overrides": {"h2_local_supply_facts_enable": True, "soc_init": 0.5},
    },
    "v6_nocomm_gru_mappo_penalty": {
        "family": "gru", "algorithm": "mappo_penalty", "seed": 30, "exploratory": True,
        "num_envs": 2, "env_parallel_backend": "process", "fused_rollout_kernel": True,
        "include_previous_action": True, "include_transaction_message": False,
        "two_stage_intent": False, "h2_supply_intent_message_enable": False,
        "cost_budget": 1.0, "voltage_cost_scale": 0.02,
        "fixed_cost_penalty_coef": 1.0,
        "curriculum_updates": 300,
        "curriculum_log_std_start": -1.0, "curriculum_log_std_end": -2.3,
        "env_overrides": {"h2_local_supply_facts_enable": True, "soc_init": 0.5},
    },
    "v6_nocomm_gru_macpo": {
        "family": "gru", "algorithm": "macpo", "seed": 30, "exploratory": True,
        "num_envs": 2, "env_parallel_backend": "process", "fused_rollout_kernel": True,
        "include_previous_action": True, "include_transaction_message": False,
        "two_stage_intent": False, "h2_supply_intent_message_enable": False,
        "cost_budget": 1.0, "voltage_cost_scale": 0.02,
        "curriculum_updates": 300,
        "curriculum_log_std_start": -1.0, "curriculum_log_std_end": -2.3,
        "env_overrides": {"h2_local_supply_facts_enable": True, "soc_init": 0.5},
    },
    # Env-v6.1 diagnosis follow-up (2026-07-26). The 1000-update MACPO run
    # plateaued from update ~400 with the constraint violated 18x on average:
    # 48 transitions/update is far too small for a trust-region cost critic, and
    # the KL cap pinned at 0.01 with accept rate 1.0 shows the step size, not the
    # direction, was binding. bigbatch raises the on-policy batch 8x and doubles
    # the trust region; nothing else moves so the comparison stays clean.
    "v6_nocomm_gru_macpo_bigbatch": {
        "family": "gru", "algorithm": "macpo", "seed": 30, "exploratory": True,
        "num_envs": 16, "env_parallel_backend": "process", "fused_rollout_kernel": True,
        "macpo_max_kl": 0.02,
        "include_previous_action": True, "include_transaction_message": False,
        "two_stage_intent": False, "h2_supply_intent_message_enable": False,
        "cost_budget": 1.0, "voltage_cost_scale": 0.02,
        "curriculum_updates": 300,
        "curriculum_log_std_start": -1.0, "curriculum_log_std_end": -2.3,
        "env_overrides": {"h2_local_supply_facts_enable": True, "soc_init": 0.5},
    },
    # bigbatch plus the optional PCC reactive-power action (action_dim 7 -> 8).
    # Day-clamp counterfactuals showed ~6/20 train day types cannot reach the
    # voltage floor with active power alone (background load pushes vmin below
    # 0.95 even with the electrolyser off and the battery fully rationed), so
    # hard safety on those days requires capacitive support. Checkpoints from
    # the 7-dim variants do not load into this one.
    "v6q_nocomm_gru_macpo_bigbatch": {
        "family": "gru", "algorithm": "macpo", "seed": 30, "exploratory": True,
        "num_envs": 16, "env_parallel_backend": "process", "fused_rollout_kernel": True,
        "macpo_max_kl": 0.02,
        "include_previous_action": True, "include_transaction_message": False,
        "two_stage_intent": False, "h2_supply_intent_message_enable": False,
        "cost_budget": 1.0, "voltage_cost_scale": 0.02,
        "curriculum_updates": 300,
        "curriculum_log_std_start": -1.0, "curriculum_log_std_end": -2.3,
        "env_overrides": {
            "h2_local_supply_facts_enable": True,
            "soc_init": 0.5,
            "pcc_q_action_enable": True,
            # ~cancel each microgrid's own peak load Q (~600 kvar) with margin
            # for net capacitive support; the derived default of
            # 1.2*(pv_cap+bat_power) is unrealistically large for agent 0.
            "pcc_q_apparent_cap_kva": 1200.0,
        },
    },
    # bigbatch with the training budget tightened from raw 0.02 to 0.01, while
    # evaluation still judges safety at raw 0.02.  The retry1 run (conservative
    # cost branch + episode-sum surrogate) left 8 of its 13 unsafe day types at
    # raw vcost 0.016-0.050, i.e. within ~2.5x of the budget it was trained
    # against: the recovery line search only has to reach the budget, so days
    # that land just outside it are never pushed further.  Halving the training
    # budget gives the recovery branch margin without touching the reported
    # criterion.  ``apply_env_v6_calibration`` overwrites both
    # ``voltage_cost_scale`` and ``cost_budget`` from the calibration file, so
    # the tightening has to ride on ``cost_budget_scale``, which is applied
    # *after* that overwrite.  Effective raw budget = cost_budget *
    # voltage_cost_scale = 0.5 * 0.02 = 0.01.
    "v6_nocomm_gru_macpo_tightbudget": {
        "family": "gru", "algorithm": "macpo", "seed": 30, "exploratory": True,
        "num_envs": 16, "env_parallel_backend": "process", "fused_rollout_kernel": True,
        "macpo_max_kl": 0.02,
        "include_previous_action": True, "include_transaction_message": False,
        "two_stage_intent": False, "h2_supply_intent_message_enable": False,
        "cost_budget": 1.0, "voltage_cost_scale": 0.02, "cost_budget_scale": 0.5,
        "curriculum_updates": 300,
        "curriculum_log_std_start": -1.0, "curriculum_log_std_end": -2.3,
        "env_overrides": {"h2_local_supply_facts_enable": True, "soc_init": 0.5},
    },
    # Round 3 / N4: the two round-2 interventions combined.  N1 (reactive power)
    # and N3 (tightened training budget) each lifted the safe-day count from
    # 15/28 to 18/28, but they made *different* days safe -- N1 gained days
    # 0/23/25, N3 gained days 0/14/20 -- because they attack different failure
    # modes: N1 removes a physical reachability limit on the hard day types,
    # N3 stops the recovery line search from halting exactly on the budget line
    # for the borderline days.  The mechanisms are orthogonal, so combining them
    # should be roughly additive.  Effective raw budget = 0.5 * 0.02 = 0.01;
    # evaluation still judges at raw 0.02.
    "v6q_nocomm_gru_macpo_tightbudget": {
        "family": "gru", "algorithm": "macpo", "seed": 30, "exploratory": True,
        "num_envs": 16, "env_parallel_backend": "process", "fused_rollout_kernel": True,
        "macpo_max_kl": 0.02,
        "include_previous_action": True, "include_transaction_message": False,
        "two_stage_intent": False, "h2_supply_intent_message_enable": False,
        "cost_budget": 1.0, "voltage_cost_scale": 0.02, "cost_budget_scale": 0.5,
        "curriculum_updates": 300,
        "curriculum_log_std_start": -1.0, "curriculum_log_std_end": -2.3,
        "env_overrides": {
            "h2_local_supply_facts_enable": True,
            "soc_init": 0.5,
            "pcc_q_action_enable": True,
            "pcc_q_apparent_cap_kva": 1200.0,
        },
    },
    # Round 3 / N5: N1 with the per-agent apparent-power capacity doubled from
    # 1200 to 2400 kVA, everything else identical.  Under N1 the reactive action
    # cut the six hard day types by 1.25x-4373x but only one of them crossed the
    # budget, which is consistent with the *direction* being right and the
    # *amount* being insufficient.  Available Q is sqrt(S^2 - (pv + |p_bat|)^2);
    # at the evening peak pv is zero, so this is close to a pure capacity sweep
    # rather than a test of the apparent-power sharing rule.  If day 24 still
    # does not respond at 2400 kVA, its deficit is not a reactive-support
    # problem and it should be reported as a physical boundary.
    # Round 5 / N7: N5's capacity + N4's budget pressure.  Round 3 measured the two
    # mechanisms as additive (N1 + N3 -> N4 landed exactly on the predicted 20/28),
    # and N4/N5 fixed disjoint marginal days (14 vs 21), so the union is the
    # prediction here: 21-22/28.  The other half of the motivation is the margin
    # finding: with the pure-hinge cost the policy has no incentive to keep any
    # headroom, and days 14/20/1/18 fail only because their mean sits at 1-1.6x the
    # budget with no slack against the environment's within-day realisation noise.
    # A tighter training budget is the blunt way to manufacture that margin, and it
    # is already known to work (day 14: 0.0197 unsafe under N1 -> 0.0037 safe under
    # N4).  Cost: N3/N4 both paid ~30% more economically, so N7 is expected to give
    # back some of N5's exceptional 1.293e6.
    "v6q_nocomm_gru_macpo_qcap2400_tight": {
        "family": "gru", "algorithm": "macpo", "seed": 30, "exploratory": True,
        "num_envs": 16, "env_parallel_backend": "process", "fused_rollout_kernel": True,
        "macpo_max_kl": 0.02,
        "include_previous_action": True, "include_transaction_message": False,
        "two_stage_intent": False, "h2_supply_intent_message_enable": False,
        "cost_budget": 1.0, "voltage_cost_scale": 0.02,
        "cost_budget_scale": 0.5,
        "curriculum_updates": 300,
        "curriculum_log_std_start": -1.0, "curriculum_log_std_end": -2.3,
        "env_overrides": {
            "h2_local_supply_facts_enable": True,
            "soc_init": 0.5,
            "pcc_q_action_enable": True,
            "pcc_q_apparent_cap_kva": 2400.0,
        },
    },
    # Round 5 / N9: third point of the reactive-capacity sweep, for the day-24
    # boundary claim only.  Everything except pcc_q_apparent_cap_kva is identical to
    # N5, so 0 / 1200 / 2400 / 4800 form a clean one-variable sweep.
    "v6q_nocomm_gru_macpo_qcap4800": {
        "family": "gru", "algorithm": "macpo", "seed": 30, "exploratory": True,
        "num_envs": 16, "env_parallel_backend": "process", "fused_rollout_kernel": True,
        "macpo_max_kl": 0.02,
        "include_previous_action": True, "include_transaction_message": False,
        "two_stage_intent": False, "h2_supply_intent_message_enable": False,
        "cost_budget": 1.0, "voltage_cost_scale": 0.02,
        "curriculum_updates": 300,
        "curriculum_log_std_start": -1.0, "curriculum_log_std_end": -2.3,
        "env_overrides": {
            "h2_local_supply_facts_enable": True,
            "soc_init": 0.5,
            "pcc_q_action_enable": True,
            "pcc_q_apparent_cap_kva": 4800.0,
        },
    },
    "v6q_nocomm_gru_macpo_allthree": {
        "family": "gru", "algorithm": "macpo", "seed": 30, "exploratory": True,
        "num_envs": 16, "env_parallel_backend": "process", "fused_rollout_kernel": True,
        "macpo_max_kl": 0.02,
        "include_previous_action": True, "include_transaction_message": False,
        "two_stage_intent": False, "h2_supply_intent_message_enable": False,
        "cost_budget": 1.0, "voltage_cost_scale": 0.02,
        "cost_budget_scale": 0.5,
        "critic_epochs": 8,
        "curriculum_updates": 300,
        "curriculum_log_std_start": -1.0, "curriculum_log_std_end": -2.3,
        "env_overrides": {
            "h2_local_supply_facts_enable": True,
            "soc_init": 0.5,
            "pcc_q_action_enable": True,
            "pcc_q_apparent_cap_kva": 2400.0,
        },
    },
    "v6q_nocomm_gru_macpo_qcap9600": {
        "family": "gru", "algorithm": "macpo", "seed": 30, "exploratory": True,
        "num_envs": 16, "env_parallel_backend": "process", "fused_rollout_kernel": True,
        "macpo_max_kl": 0.02,
        "include_previous_action": True, "include_transaction_message": False,
        "two_stage_intent": False, "h2_supply_intent_message_enable": False,
        "cost_budget": 1.0, "voltage_cost_scale": 0.02,
        "curriculum_updates": 300,
        "curriculum_log_std_start": -1.0, "curriculum_log_std_end": -2.3,
        "env_overrides": {
            "h2_local_supply_facts_enable": True,
            "soc_init": 0.5,
            "pcc_q_action_enable": True,
            "pcc_q_apparent_cap_kva": 9600.0,
        },
    },
    "v6q_nocomm_gru_macpo_qcap4800_tight": {
        "family": "gru", "algorithm": "macpo", "seed": 30, "exploratory": True,
        "num_envs": 16, "env_parallel_backend": "process", "fused_rollout_kernel": True,
        "macpo_max_kl": 0.02,
        "include_previous_action": True, "include_transaction_message": False,
        "two_stage_intent": False, "h2_supply_intent_message_enable": False,
        "cost_budget": 1.0, "voltage_cost_scale": 0.02,
        "cost_budget_scale": 0.5,
        "curriculum_updates": 300,
        "curriculum_log_std_start": -1.0, "curriculum_log_std_end": -2.3,
        "env_overrides": {
            "h2_local_supply_facts_enable": True,
            "soc_init": 0.5,
            "pcc_q_action_enable": True,
            "pcc_q_apparent_cap_kva": 4800.0,
        },
    },
    "v6q_nocomm_gru_macpo_qcap4800_allthree": {
        "family": "gru", "algorithm": "macpo", "seed": 30, "exploratory": True,
        "num_envs": 16, "env_parallel_backend": "process", "fused_rollout_kernel": True,
        "macpo_max_kl": 0.02,
        "include_previous_action": True, "include_transaction_message": False,
        "two_stage_intent": False, "h2_supply_intent_message_enable": False,
        "cost_budget": 1.0, "voltage_cost_scale": 0.02,
        "cost_budget_scale": 0.5,
        "critic_epochs": 8,
        "curriculum_updates": 300,
        "curriculum_log_std_start": -1.0, "curriculum_log_std_end": -2.3,
        "env_overrides": {
            "h2_local_supply_facts_enable": True,
            "soc_init": 0.5,
            "pcc_q_action_enable": True,
            "pcc_q_apparent_cap_kva": 4800.0,
        },
    },
    "v6q_nocomm_gru_macpo_qcap9600_tight": {
        "family": "gru", "algorithm": "macpo", "seed": 30, "exploratory": True,
        "num_envs": 16, "env_parallel_backend": "process", "fused_rollout_kernel": True,
        "macpo_max_kl": 0.02,
        "include_previous_action": True, "include_transaction_message": False,
        "two_stage_intent": False, "h2_supply_intent_message_enable": False,
        "cost_budget": 1.0, "voltage_cost_scale": 0.02,
        "curriculum_updates": 300,
        "curriculum_log_std_start": -1.0, "curriculum_log_std_end": -2.3,
        "cost_budget_scale": 0.5,
        "env_overrides": {
            "h2_local_supply_facts_enable": True,
            "soc_init": 0.5,
            "pcc_q_action_enable": True,
            "pcc_q_apparent_cap_kva": 9600.0,
        },
    },
    "v6q_nocomm_gru_macpo_qcap4800_budget025": {
        "family": "gru", "algorithm": "macpo", "seed": 30, "exploratory": True,
        "num_envs": 16, "env_parallel_backend": "process", "fused_rollout_kernel": True,
        "macpo_max_kl": 0.02,
        "include_previous_action": True, "include_transaction_message": False,
        "two_stage_intent": False, "h2_supply_intent_message_enable": False,
        "cost_budget": 1.0, "voltage_cost_scale": 0.02,
        "curriculum_updates": 300,
        "curriculum_log_std_start": -1.0, "curriculum_log_std_end": -2.3,
        "cost_budget_scale": 0.25,
        "env_overrides": {
            "h2_local_supply_facts_enable": True,
            "soc_init": 0.5,
            "pcc_q_action_enable": True,
            "pcc_q_apparent_cap_kva": 4800.0,
        },
    },
    "v6q_nocomm_gru_macpo_softc8_knee955": {
        "seed": 30,
        "critic_epochs": 8,
        "family": "gru", "algorithm": "macpo", "exploratory": True,
        "num_envs": 16, "env_parallel_backend": "process", "fused_rollout_kernel": True,
        "macpo_max_kl": 0.02,
        "include_previous_action": True, "include_transaction_message": False,
        "two_stage_intent": False, "h2_supply_intent_message_enable": False,
        "cost_budget": 1.0, "voltage_cost_scale": 0.02,
        "curriculum_updates": 300,
        "curriculum_log_std_start": -1.0, "curriculum_log_std_end": -2.3,
        "env_overrides": {
            "h2_local_supply_facts_enable": True,
            "soc_init": 0.5,
            "pcc_q_action_enable": True,
            "pcc_q_apparent_cap_kva": 4800.0,
            "power_flow_vmin_pu": 0.955,
            "power_flow_vmax_pu": 1.045,
        },
    },
    "v6q_nocomm_gru_macpo_softc8_knee97_s31": {
        "seed": 31,
        "critic_epochs": 8,
        "family": "gru", "algorithm": "macpo", "exploratory": True,
        "num_envs": 16, "env_parallel_backend": "process", "fused_rollout_kernel": True,
        "macpo_max_kl": 0.02,
        "include_previous_action": True, "include_transaction_message": False,
        "two_stage_intent": False, "h2_supply_intent_message_enable": False,
        "cost_budget": 1.0, "voltage_cost_scale": 0.02,
        "curriculum_updates": 300,
        "curriculum_log_std_start": -1.0, "curriculum_log_std_end": -2.3,
        "env_overrides": {
            "h2_local_supply_facts_enable": True,
            "soc_init": 0.5,
            "pcc_q_action_enable": True,
            "pcc_q_apparent_cap_kva": 4800.0,
            "power_flow_vmin_pu": 0.97,
            "power_flow_vmax_pu": 1.03,
        },
    },
    "v6q_nocomm_gru_macpo_softc8_knee97_s32": {
        "seed": 32,
        "critic_epochs": 8,
        "family": "gru", "algorithm": "macpo", "exploratory": True,
        "num_envs": 16, "env_parallel_backend": "process", "fused_rollout_kernel": True,
        "macpo_max_kl": 0.02,
        "include_previous_action": True, "include_transaction_message": False,
        "two_stage_intent": False, "h2_supply_intent_message_enable": False,
        "cost_budget": 1.0, "voltage_cost_scale": 0.02,
        "curriculum_updates": 300,
        "curriculum_log_std_start": -1.0, "curriculum_log_std_end": -2.3,
        "env_overrides": {
            "h2_local_supply_facts_enable": True,
            "soc_init": 0.5,
            "pcc_q_action_enable": True,
            "pcc_q_apparent_cap_kva": 4800.0,
            "power_flow_vmin_pu": 0.97,
            "power_flow_vmax_pu": 1.03,
        },
    },
    "v6q_nocomm_gru_macpo_softc8_knee97_q2400": {
        "seed": 30,
        "critic_epochs": 8,
        "family": "gru", "algorithm": "macpo", "exploratory": True,
        "num_envs": 16, "env_parallel_backend": "process", "fused_rollout_kernel": True,
        "macpo_max_kl": 0.02,
        "include_previous_action": True, "include_transaction_message": False,
        "two_stage_intent": False, "h2_supply_intent_message_enable": False,
        "cost_budget": 1.0, "voltage_cost_scale": 0.02,
        "curriculum_updates": 300,
        "curriculum_log_std_start": -1.0, "curriculum_log_std_end": -2.3,
        "env_overrides": {
            "h2_local_supply_facts_enable": True,
            "soc_init": 0.5,
            "pcc_q_action_enable": True,
            "pcc_q_apparent_cap_kva": 2400.0,
            "power_flow_vmin_pu": 0.97,
            "power_flow_vmax_pu": 1.03,
        },
    },
    "v6q_nocomm_gru_mappo_knee97": {
        "seed": 30,
        "critic_epochs": 8,
        "family": "gru", "algorithm": "mappo", "exploratory": True,
        "num_envs": 16, "env_parallel_backend": "process", "fused_rollout_kernel": True,
        "macpo_max_kl": 0.02,
        "include_previous_action": True, "include_transaction_message": False,
        "two_stage_intent": False, "h2_supply_intent_message_enable": False,
        "cost_budget": 1.0, "voltage_cost_scale": 0.02,
        "curriculum_updates": 300,
        "curriculum_log_std_start": -1.0, "curriculum_log_std_end": -2.3,
        "env_overrides": {
            "h2_local_supply_facts_enable": True,
            "soc_init": 0.5,
            "pcc_q_action_enable": True,
            "pcc_q_apparent_cap_kva": 4800.0,
            "power_flow_vmin_pu": 0.97,
            "power_flow_vmax_pu": 1.03,
        },
    },
    "v6q_nocomm_gru_mappopen01_knee97": {
        "seed": 30,
        "critic_epochs": 8,
        "family": "gru", "algorithm": "mappo_penalty", "exploratory": True,
        "num_envs": 16, "env_parallel_backend": "process", "fused_rollout_kernel": True,
        "macpo_max_kl": 0.02,
        "fixed_cost_penalty_coef": 0.1,
        "include_previous_action": True, "include_transaction_message": False,
        "two_stage_intent": False, "h2_supply_intent_message_enable": False,
        "cost_budget": 1.0, "voltage_cost_scale": 0.02,
        "curriculum_updates": 300,
        "curriculum_log_std_start": -1.0, "curriculum_log_std_end": -2.3,
        "env_overrides": {
            "h2_local_supply_facts_enable": True,
            "soc_init": 0.5,
            "pcc_q_action_enable": True,
            "pcc_q_apparent_cap_kva": 4800.0,
            "power_flow_vmin_pu": 0.97,
            "power_flow_vmax_pu": 1.03,
        },
    },
    "v6q_nocomm_gru_mappopen03_knee97": {
        "seed": 30,
        "critic_epochs": 8,
        "family": "gru", "algorithm": "mappo_penalty", "exploratory": True,
        "num_envs": 16, "env_parallel_backend": "process", "fused_rollout_kernel": True,
        "macpo_max_kl": 0.02,
        "fixed_cost_penalty_coef": 0.3,
        "include_previous_action": True, "include_transaction_message": False,
        "two_stage_intent": False, "h2_supply_intent_message_enable": False,
        "cost_budget": 1.0, "voltage_cost_scale": 0.02,
        "curriculum_updates": 300,
        "curriculum_log_std_start": -1.0, "curriculum_log_std_end": -2.3,
        "env_overrides": {
            "h2_local_supply_facts_enable": True,
            "soc_init": 0.5,
            "pcc_q_action_enable": True,
            "pcc_q_apparent_cap_kva": 4800.0,
            "power_flow_vmin_pu": 0.97,
            "power_flow_vmax_pu": 1.03,
        },
    },
    "v6q_nocomm_gru_mappopen003_knee97": {
        "seed": 30,
        "critic_epochs": 8,
        "family": "gru", "algorithm": "mappo_penalty", "exploratory": True,
        "num_envs": 16, "env_parallel_backend": "process", "fused_rollout_kernel": True,
        "macpo_max_kl": 0.02,
        "fixed_cost_penalty_coef": 0.03,
        "include_previous_action": True, "include_transaction_message": False,
        "two_stage_intent": False, "h2_supply_intent_message_enable": False,
        "cost_budget": 1.0, "voltage_cost_scale": 0.02,
        "curriculum_updates": 300,
        "curriculum_log_std_start": -1.0, "curriculum_log_std_end": -2.3,
        "env_overrides": {
            "h2_local_supply_facts_enable": True,
            "soc_init": 0.5,
            "pcc_q_action_enable": True,
            "pcc_q_apparent_cap_kva": 4800.0,
            "power_flow_vmin_pu": 0.97,
            "power_flow_vmax_pu": 1.03,
        },
    },
    "v6q_nocomm_gru_mappopen001_knee97_s31": {
        "seed": 31,
        "critic_epochs": 8,
        "family": "gru", "algorithm": "mappo_penalty", "exploratory": True,
        "num_envs": 16, "env_parallel_backend": "process", "fused_rollout_kernel": True,
        "macpo_max_kl": 0.02,
        "fixed_cost_penalty_coef": 0.01,
        "include_previous_action": True, "include_transaction_message": False,
        "two_stage_intent": False, "h2_supply_intent_message_enable": False,
        "cost_budget": 1.0, "voltage_cost_scale": 0.02,
        "curriculum_updates": 300,
        "curriculum_log_std_start": -1.0, "curriculum_log_std_end": -2.3,
        "env_overrides": {
            "h2_local_supply_facts_enable": True,
            "soc_init": 0.5,
            "pcc_q_action_enable": True,
            "pcc_q_apparent_cap_kva": 4800.0,
            "power_flow_vmin_pu": 0.97,
            "power_flow_vmax_pu": 1.03,
        },
    },
    "v6q_nocomm_gru_mappopen001_knee97_s32": {
        "seed": 32,
        "critic_epochs": 8,
        "family": "gru", "algorithm": "mappo_penalty", "exploratory": True,
        "num_envs": 16, "env_parallel_backend": "process", "fused_rollout_kernel": True,
        "macpo_max_kl": 0.02,
        "fixed_cost_penalty_coef": 0.01,
        "include_previous_action": True, "include_transaction_message": False,
        "two_stage_intent": False, "h2_supply_intent_message_enable": False,
        "cost_budget": 1.0, "voltage_cost_scale": 0.02,
        "curriculum_updates": 300,
        "curriculum_log_std_start": -1.0, "curriculum_log_std_end": -2.3,
        "env_overrides": {
            "h2_local_supply_facts_enable": True,
            "soc_init": 0.5,
            "pcc_q_action_enable": True,
            "pcc_q_apparent_cap_kva": 4800.0,
            "power_flow_vmin_pu": 0.97,
            "power_flow_vmax_pu": 1.03,
        },
    },
    "v6q_nocomm_gru_mappopen001_knee97": {
        "seed": 30,
        "critic_epochs": 8,
        "family": "gru", "algorithm": "mappo_penalty", "exploratory": True,
        "num_envs": 16, "env_parallel_backend": "process", "fused_rollout_kernel": True,
        "macpo_max_kl": 0.02,
        "fixed_cost_penalty_coef": 0.01,
        "include_previous_action": True, "include_transaction_message": False,
        "two_stage_intent": False, "h2_supply_intent_message_enable": False,
        "cost_budget": 1.0, "voltage_cost_scale": 0.02,
        "curriculum_updates": 300,
        "curriculum_log_std_start": -1.0, "curriculum_log_std_end": -2.3,
        "env_overrides": {
            "h2_local_supply_facts_enable": True,
            "soc_init": 0.5,
            "pcc_q_action_enable": True,
            "pcc_q_apparent_cap_kva": 4800.0,
            "power_flow_vmin_pu": 0.97,
            "power_flow_vmax_pu": 1.03,
        },
    },
    "v6q_nocomm_gru_mappopen1_knee97": {
        "seed": 30,
        "critic_epochs": 8,
        "family": "gru", "algorithm": "mappo_penalty", "exploratory": True,
        "num_envs": 16, "env_parallel_backend": "process", "fused_rollout_kernel": True,
        "macpo_max_kl": 0.02,
        "fixed_cost_penalty_coef": 1.0,
        "include_previous_action": True, "include_transaction_message": False,
        "two_stage_intent": False, "h2_supply_intent_message_enable": False,
        "cost_budget": 1.0, "voltage_cost_scale": 0.02,
        "curriculum_updates": 300,
        "curriculum_log_std_start": -1.0, "curriculum_log_std_end": -2.3,
        "env_overrides": {
            "h2_local_supply_facts_enable": True,
            "soc_init": 0.5,
            "pcc_q_action_enable": True,
            "pcc_q_apparent_cap_kva": 4800.0,
            "power_flow_vmin_pu": 0.97,
            "power_flow_vmax_pu": 1.03,
        },
    },
    "v6q_nocomm_gru_mappopen10_knee97": {
        "seed": 30,
        "critic_epochs": 8,
        "family": "gru", "algorithm": "mappo_penalty", "exploratory": True,
        "num_envs": 16, "env_parallel_backend": "process", "fused_rollout_kernel": True,
        "macpo_max_kl": 0.02,
        "fixed_cost_penalty_coef": 10.0,
        "include_previous_action": True, "include_transaction_message": False,
        "two_stage_intent": False, "h2_supply_intent_message_enable": False,
        "cost_budget": 1.0, "voltage_cost_scale": 0.02,
        "curriculum_updates": 300,
        "curriculum_log_std_start": -1.0, "curriculum_log_std_end": -2.3,
        "env_overrides": {
            "h2_local_supply_facts_enable": True,
            "soc_init": 0.5,
            "pcc_q_action_enable": True,
            "pcc_q_apparent_cap_kva": 4800.0,
            "power_flow_vmin_pu": 0.97,
            "power_flow_vmax_pu": 1.03,
        },
    },
    "v6q_nocomm_gru_lagr_knee97": {
        "seed": 30,
        "critic_epochs": 8,
        "family": "gru", "algorithm": "lagrangian", "exploratory": True,
        "num_envs": 16, "env_parallel_backend": "process", "fused_rollout_kernel": True,
        "macpo_max_kl": 0.02,
        "include_previous_action": True, "include_transaction_message": False,
        "two_stage_intent": False, "h2_supply_intent_message_enable": False,
        "cost_budget": 1.0, "voltage_cost_scale": 0.02,
        "curriculum_updates": 300,
        "curriculum_log_std_start": -1.0, "curriculum_log_std_end": -2.3,
        "env_overrides": {
            "h2_local_supply_facts_enable": True,
            "soc_init": 0.5,
            "pcc_q_action_enable": True,
            "pcc_q_apparent_cap_kva": 4800.0,
            "power_flow_vmin_pu": 0.97,
            "power_flow_vmax_pu": 1.03,
        },
    },
    "v6q_nocomm_gru_macpo_softc8_knee97": {
        "seed": 30,
        "critic_epochs": 8,
        "family": "gru", "algorithm": "macpo", "exploratory": True,
        "num_envs": 16, "env_parallel_backend": "process", "fused_rollout_kernel": True,
        "macpo_max_kl": 0.02,
        "include_previous_action": True, "include_transaction_message": False,
        "two_stage_intent": False, "h2_supply_intent_message_enable": False,
        "cost_budget": 1.0, "voltage_cost_scale": 0.02,
        "curriculum_updates": 300,
        "curriculum_log_std_start": -1.0, "curriculum_log_std_end": -2.3,
        "env_overrides": {
            "h2_local_supply_facts_enable": True,
            "soc_init": 0.5,
            "pcc_q_action_enable": True,
            "pcc_q_apparent_cap_kva": 4800.0,
            "power_flow_vmin_pu": 0.97,
            "power_flow_vmax_pu": 1.03,
        },
    },
    # H2 price rebased from yuan/kWh_H2 to yuan/kg (LHV 33.33 kWh/kg), i.e. every
    # hydrogen price divided by 33.33: buy 1000 -> 30 yuan/kg, sell 100 -> 3 yuan/kg.
    # Identical to v6q_nocomm_gru_macpo_softc8_knee97 in every other key.
    # NOT comparable to any earlier run: the objective itself changed.
    "v6q_macpo_knee97_h2kg": {
        "seed": 30,
        "critic_epochs": 8,
        "family": "gru", "algorithm": "macpo", "exploratory": True,
        "num_envs": 16, "env_parallel_backend": "process", "fused_rollout_kernel": True,
        "macpo_max_kl": 0.02,
        "include_previous_action": True, "include_transaction_message": False,
        "two_stage_intent": False, "h2_supply_intent_message_enable": False,
        "cost_budget": 1.0, "voltage_cost_scale": 0.02,
        "curriculum_updates": 300,
        "curriculum_log_std_start": -1.0, "curriculum_log_std_end": -2.3,
        "env_overrides": {
            "h2_local_supply_facts_enable": True,
            "soc_init": 0.5,
            "pcc_q_action_enable": True,
            "pcc_q_apparent_cap_kva": 4800.0,
            "power_flow_vmin_pu": 0.97,
            "power_flow_vmax_pu": 1.03,
            "lambda_h2": 0.495,
            "lambda_h2_buy": 0.90,
            "lambda_h2_sell": 0.09,
            "h2_price_min": 0.09,
            "h2_price_max": 0.90,
            "h2_price_init": 0.495,
        },
    },
    # Identical to v6q_macpo_knee97_h2kg except for three renewable-forecast
    # keys.  Tests whether A0's anti-phase electrolyzer (it produces at night
    # on bought power and idles through the PV peak) is caused by the absence
    # of any PV/WT look-ahead in the observation.  The forecast is noisy on
    # purpose: multiplicative error on generation, std 0.15*sqrt(H).
    "v6q_macpo_knee97_h2kg_renfc": {
        "seed": 30,
        "critic_epochs": 8,
        "family": "gru", "algorithm": "macpo", "exploratory": True,
        "num_envs": 16, "env_parallel_backend": "process", "fused_rollout_kernel": True,
        "macpo_max_kl": 0.02,
        "include_previous_action": True, "include_transaction_message": False,
        "two_stage_intent": False, "h2_supply_intent_message_enable": False,
        "cost_budget": 1.0, "voltage_cost_scale": 0.02,
        "curriculum_updates": 300,
        "curriculum_log_std_start": -1.0, "curriculum_log_std_end": -2.3,
        "env_overrides": {
            "h2_local_supply_facts_enable": True,
            "soc_init": 0.5,
            "pcc_q_action_enable": True,
            "pcc_q_apparent_cap_kva": 4800.0,
            "power_flow_vmin_pu": 0.97,
            "power_flow_vmax_pu": 1.03,
            "lambda_h2": 0.495,
            "lambda_h2_buy": 0.90,
            "lambda_h2_sell": 0.09,
            "h2_price_min": 0.09,
            "h2_price_max": 0.90,
            "h2_price_init": 0.495,
            "renewable_forecast_enable": True,
            "renewable_forecast_horizons": [2, 4, 6],
            "renewable_forecast_noise_std": 0.15,
        },
    },
    "v6q_nocomm_gru_macpo_softc8_knee98": {
        "seed": 30,
        "critic_epochs": 8,
        "family": "gru", "algorithm": "macpo", "exploratory": True,
        "num_envs": 16, "env_parallel_backend": "process", "fused_rollout_kernel": True,
        "macpo_max_kl": 0.02,
        "include_previous_action": True, "include_transaction_message": False,
        "two_stage_intent": False, "h2_supply_intent_message_enable": False,
        "cost_budget": 1.0, "voltage_cost_scale": 0.02,
        "curriculum_updates": 300,
        "curriculum_log_std_start": -1.0, "curriculum_log_std_end": -2.3,
        "env_overrides": {
            "h2_local_supply_facts_enable": True,
            "soc_init": 0.5,
            "pcc_q_action_enable": True,
            "pcc_q_apparent_cap_kva": 4800.0,
            "power_flow_vmin_pu": 0.98,
            "power_flow_vmax_pu": 1.02,
        },
    },
    "v6q_nocomm_gru_macpo_softmargin_criticfit_seed31": {
        "seed": 31,
        "critic_epochs": 8,
        "family": "gru", "algorithm": "macpo", "exploratory": True,
        "num_envs": 16, "env_parallel_backend": "process", "fused_rollout_kernel": True,
        "macpo_max_kl": 0.02,
        "include_previous_action": True, "include_transaction_message": False,
        "two_stage_intent": False, "h2_supply_intent_message_enable": False,
        "cost_budget": 1.0, "voltage_cost_scale": 0.02,
        "curriculum_updates": 300,
        "curriculum_log_std_start": -1.0, "curriculum_log_std_end": -2.3,
        "env_overrides": {
            "h2_local_supply_facts_enable": True,
            "soc_init": 0.5,
            "pcc_q_action_enable": True,
            "pcc_q_apparent_cap_kva": 4800.0,
            "power_flow_vmin_pu": 0.96,
            "power_flow_vmax_pu": 1.04,
        },
    },
    "v6q_nocomm_gru_macpo_softmargin_criticfit_seed32": {
        "seed": 32,
        "critic_epochs": 8,
        "family": "gru", "algorithm": "macpo", "exploratory": True,
        "num_envs": 16, "env_parallel_backend": "process", "fused_rollout_kernel": True,
        "macpo_max_kl": 0.02,
        "include_previous_action": True, "include_transaction_message": False,
        "two_stage_intent": False, "h2_supply_intent_message_enable": False,
        "cost_budget": 1.0, "voltage_cost_scale": 0.02,
        "curriculum_updates": 300,
        "curriculum_log_std_start": -1.0, "curriculum_log_std_end": -2.3,
        "env_overrides": {
            "h2_local_supply_facts_enable": True,
            "soc_init": 0.5,
            "pcc_q_action_enable": True,
            "pcc_q_apparent_cap_kva": 4800.0,
            "power_flow_vmin_pu": 0.96,
            "power_flow_vmax_pu": 1.04,
        },
    },
    "v6q_nocomm_gru_macpo_qcap2400_softmargin_seed31": {
        "seed": 31,
        "family": "gru", "algorithm": "macpo", "exploratory": True,
        "num_envs": 16, "env_parallel_backend": "process", "fused_rollout_kernel": True,
        "macpo_max_kl": 0.02,
        "include_previous_action": True, "include_transaction_message": False,
        "two_stage_intent": False, "h2_supply_intent_message_enable": False,
        "cost_budget": 1.0, "voltage_cost_scale": 0.02,
        "curriculum_updates": 300,
        "curriculum_log_std_start": -1.0, "curriculum_log_std_end": -2.3,
        "env_overrides": {
            "h2_local_supply_facts_enable": True,
            "soc_init": 0.5,
            "pcc_q_action_enable": True,
            "pcc_q_apparent_cap_kva": 2400.0,
            "power_flow_vmin_pu": 0.96,
            "power_flow_vmax_pu": 1.04,
        },
    },
    "v6q_nocomm_gru_macpo_qcap4800_softmargin_seed31": {
        "seed": 31,
        "family": "gru", "algorithm": "macpo", "exploratory": True,
        "num_envs": 16, "env_parallel_backend": "process", "fused_rollout_kernel": True,
        "macpo_max_kl": 0.02,
        "include_previous_action": True, "include_transaction_message": False,
        "two_stage_intent": False, "h2_supply_intent_message_enable": False,
        "cost_budget": 1.0, "voltage_cost_scale": 0.02,
        "curriculum_updates": 300,
        "curriculum_log_std_start": -1.0, "curriculum_log_std_end": -2.3,
        "env_overrides": {
            "h2_local_supply_facts_enable": True,
            "soc_init": 0.5,
            "pcc_q_action_enable": True,
            "pcc_q_apparent_cap_kva": 4800.0,
            "power_flow_vmin_pu": 0.96,
            "power_flow_vmax_pu": 1.04,
        },
    },
    "v6q_nocomm_gru_macpo_qcap2400_softmargin": {
        "seed": 30,
        "family": "gru", "algorithm": "macpo", "exploratory": True,
        "num_envs": 16, "env_parallel_backend": "process", "fused_rollout_kernel": True,
        "macpo_max_kl": 0.02,
        "include_previous_action": True, "include_transaction_message": False,
        "two_stage_intent": False, "h2_supply_intent_message_enable": False,
        "cost_budget": 1.0, "voltage_cost_scale": 0.02,
        "curriculum_updates": 300,
        "curriculum_log_std_start": -1.0, "curriculum_log_std_end": -2.3,
        "env_overrides": {
            "h2_local_supply_facts_enable": True,
            "soc_init": 0.5,
            "pcc_q_action_enable": True,
            "pcc_q_apparent_cap_kva": 2400.0,
            "power_flow_vmin_pu": 0.96,
            "power_flow_vmax_pu": 1.04,
        },
    },
    "v6q_nocomm_gru_macpo_qcap4800_softmargin_criticfit": {
        "seed": 30,
        "critic_epochs": 8,
        "family": "gru", "algorithm": "macpo", "exploratory": True,
        "num_envs": 16, "env_parallel_backend": "process", "fused_rollout_kernel": True,
        "macpo_max_kl": 0.02,
        "include_previous_action": True, "include_transaction_message": False,
        "two_stage_intent": False, "h2_supply_intent_message_enable": False,
        "cost_budget": 1.0, "voltage_cost_scale": 0.02,
        "curriculum_updates": 300,
        "curriculum_log_std_start": -1.0, "curriculum_log_std_end": -2.3,
        "env_overrides": {
            "h2_local_supply_facts_enable": True,
            "soc_init": 0.5,
            "pcc_q_action_enable": True,
            "pcc_q_apparent_cap_kva": 4800.0,
            "power_flow_vmin_pu": 0.96,
            "power_flow_vmax_pu": 1.04,
        },
    },
    "v6q_nocomm_gru_macpo_qcap4800_softmargin": {
        "family": "gru", "algorithm": "macpo", "seed": 30, "exploratory": True,
        "num_envs": 16, "env_parallel_backend": "process", "fused_rollout_kernel": True,
        "macpo_max_kl": 0.02,
        "include_previous_action": True, "include_transaction_message": False,
        "two_stage_intent": False, "h2_supply_intent_message_enable": False,
        "cost_budget": 1.0, "voltage_cost_scale": 0.02,
        "curriculum_updates": 300,
        "curriculum_log_std_start": -1.0, "curriculum_log_std_end": -2.3,
        "env_overrides": {
            "h2_local_supply_facts_enable": True,
            "soc_init": 0.5,
            "pcc_q_action_enable": True,
            "pcc_q_apparent_cap_kva": 4800.0,
            "power_flow_vmin_pu": 0.96,
            "power_flow_vmax_pu": 1.04,
        },
    },
    "v6q_nocomm_gru_macpo_qcap2400": {
        "family": "gru", "algorithm": "macpo", "seed": 30, "exploratory": True,
        "num_envs": 16, "env_parallel_backend": "process", "fused_rollout_kernel": True,
        "macpo_max_kl": 0.02,
        "include_previous_action": True, "include_transaction_message": False,
        "two_stage_intent": False, "h2_supply_intent_message_enable": False,
        "cost_budget": 1.0, "voltage_cost_scale": 0.02,
        "curriculum_updates": 300,
        "curriculum_log_std_start": -1.0, "curriculum_log_std_end": -2.3,
        "env_overrides": {
            "h2_local_supply_facts_enable": True,
            "soc_init": 0.5,
            "pcc_q_action_enable": True,
            "pcc_q_apparent_cap_kva": 2400.0,
        },
    },
    # Round 4 / N6: N1 with the critics given 8 gradient steps per update instead
    # of 1.  Motivated by the 2026-07-30 offline diagnosis rather than by a
    # hyperparameter hunch: on every run so far -- including the successful ones --
    # the measured trust-region ratio rho = (actual cost drop) / (surrogate's
    # predicted cost drop) sits at ~0 and does NOT degrade with step size, i.e.
    # the surrogate's magnitude is uninformative everywhere, which is why the
    # textbook feasibility branch could not be rescued by any trust-region tuning.
    # The reason turned out to be upstream: the cost critic's out-of-sample RMSE
    # equals the cost value itself (retry1 4.46 vs 4.02, N1 1.63 vs 1.65), because
    # it took exactly one Adam step per update.  If 8 steps shrink that RMSE and
    # safety improves, the cost signal was simply under-fitted; if the RMSE shrinks
    # in-sample but out-of-sample error and safety do not move, the residual is
    # irreducible given the observation, which is direct evidence for the next
    # intervention (electrical-load / PV lookahead features -- v6 agents currently
    # have none).  Either outcome is informative.
    "v6q_nocomm_gru_macpo_criticfit": {
        "family": "gru", "algorithm": "macpo", "seed": 30, "exploratory": True,
        "num_envs": 16, "env_parallel_backend": "process", "fused_rollout_kernel": True,
        "macpo_max_kl": 0.02,
        "include_previous_action": True, "include_transaction_message": False,
        "two_stage_intent": False, "h2_supply_intent_message_enable": False,
        "cost_budget": 1.0, "voltage_cost_scale": 0.02,
        "critic_epochs": 8,
        "curriculum_updates": 300,
        "curriculum_log_std_start": -1.0, "curriculum_log_std_end": -2.3,
        "env_overrides": {
            "h2_local_supply_facts_enable": True,
            "soc_init": 0.5,
            "pcc_q_action_enable": True,
            "pcc_q_apparent_cap_kva": 1200.0,
        },
    },
}


def build_gru_config(name: str, *, updates: int) -> dict[str, Any]:
    """Build the explicit config consumed by the independent GRU trainer."""
    spec = EXPERIMENTS[name]
    if spec["family"] != "gru":
        raise ValueError(f"{name} is an FF baseline, not a GRU variant")
    if updates < 1:
        raise ValueError("updates must be positive")
    env_overrides = env_v3_safe_overrides()
    env_overrides.update(dict(spec.get("env_overrides", {})))
    config: dict[str, Any] = {
        "seed": spec["seed"],
        "num_envs": int(spec.get("num_envs", 4)),
        "num_steps": 24,
        "total_updates": updates,
        "anneal_lr": True,
        "hidden_size": 128,
        "lr": 3e-4,
        "gamma": 1.0,
        "gae_lambda": 0.95,
        "clip_eps": 0.2,
        "entropy_coef": 0.01,
        "lagrange_lr": 0.05,
        "cost_budget": float(spec.get("cost_budget", 0.0)),
        "voltage_cost_scale": float(spec.get("voltage_cost_scale", 1.0)),
        "fixed_cost_penalty_coef": float(spec.get("fixed_cost_penalty_coef", 1.0)),
        "fused_rollout_kernel": bool(spec.get("fused_rollout_kernel", False)),
        "safety_budget_mode": spec.get("budget_mode", "fixed"),
        "macpo_max_kl": float(spec.get("macpo_max_kl", 0.01)),
        "macpo_cg_iterations": 10,
        "macpo_damping": 1e-2,
        "include_previous_action": bool(spec.get("include_previous_action", False)),
        "include_transaction_message": bool(
            spec.get("include_transaction_message", False)
        ),
        "two_stage_intent": bool(spec.get("two_stage_intent", False)),
        "intent_dim": int(spec.get("intent_dim", 3)),
        "intent_broadcast_mode": str(spec.get("intent_broadcast_mode", "full")),
        "communication_scope": spec.get("communication_scope"),
        "intent_residual_limit": float(spec.get("intent_residual_limit", 0.25)),
        "intent_residual_coef": float(spec.get("intent_residual_coef", 0.0)),
        "h2_supply_intent_message_enable": bool(
            spec.get("h2_supply_intent_message_enable", False)
        ),
        "curriculum_d_start": spec.get("curriculum_d_start"),
        "curriculum_d_target": spec.get("curriculum_d_target"),
        "curriculum_updates": int(spec.get("curriculum_updates", 0)),
        "curriculum_log_std_start": spec.get("curriculum_log_std_start"),
        "curriculum_log_std_end": spec.get("curriculum_log_std_end"),
        "env_parallel_backend": str(spec.get("env_parallel_backend", "serial")),
        "env_overrides": env_overrides,
    }
    # Optional keys are attached only when the spec sets them, so a variant that
    # does not opt in produces a config dict identical to the one its existing
    # checkpoints were fingerprinted against.
    if "critic_epochs" in spec:
        config["critic_epochs"] = int(spec["critic_epochs"])
    return config


def apply_env_v6_calibration(
    config: dict[str, Any],
    calibration: dict[str, Any],
    *,
    budget_scale: float = 1.0,
) -> dict[str, Any]:
    """Apply one passing native Swiss MV calibration to a v6 trainer config.

    ``budget_scale`` lets a variant train against a tighter cost budget than the
    calibration's reporting budget (raw 0.02).  It is applied after the
    calibration overwrite, and is a keyword with default 1.0 so that every
    existing variant's config -- and therefore its checkpoint fingerprint --
    is bit-identical to before.
    """
    if calibration.get("environment") != "env-v6-swiss":
        raise ValueError("Env-v6 training requires an env-v6-swiss calibration")
    selection = calibration.get("selection")
    if not calibration.get("feasible") or not isinstance(selection, dict):
        raise ValueError("Env-v6 training requires a passing physical gate")
    for key in ("pcc_injection_scale", "background_load_scale"):
        if not np.isclose(float(calibration.get(key, np.nan)), 1.0):
            raise ValueError(f"Env-v6 calibration {key} must be 1.0")
    economic_scale = float(calibration.get("economic_reward_scale_yuan", 0.0))
    cost_scale = float(calibration.get("training_cost_scale", 0.0))
    cost_budget = float(calibration.get("training_cost_budget", 0.0))
    if economic_scale <= 0.0 or cost_scale <= 0.0 or cost_budget <= 0.0:
        raise ValueError("Env-v6 calibration normalization scales must be positive")
    overrides = config["env_overrides"]
    overrides.update(
        {
            "power_flow_model": "swiss_mv",
            "power_flow_case_dir": str(selection["case_dir"]),
            "power_flow_pcc_bus_ids": [int(bus) for bus in selection["pcc_bus_ids"]],
            "power_flow_background_load_scale": 1.0,
            "power_flow_pcc_injection_scale": 1.0,
            "reward_scale": economic_scale,
        }
    )
    config["voltage_cost_scale"] = cost_scale
    budget_scale = float(budget_scale)
    if not np.isfinite(budget_scale) or budget_scale <= 0.0:
        raise ValueError("budget_scale must be finite and positive")
    config["cost_budget"] = cost_budget * budget_scale
    config["curriculum_d_start"] = None
    config["curriculum_d_target"] = None
    return config


def calibrated_safety_budgets(reference: dict[str, Any]) -> dict[str, float]:
    """Turn an empirical safe-reference report into daily MACPO budgets."""
    c_ref = float(reference["c_ref"])
    c_idle = float(reference["c_idle"])
    c_best = min(c_ref, c_idle)
    return {
        "strict": c_best + 0.10 * (c_idle - c_best),
        "balanced": c_best + 0.50 * (c_idle - c_best),
    }


def apply_safety_reference(
    config: dict[str, Any], reference: dict[str, Any]
) -> dict[str, Any]:
    """Apply legacy budgets or the Env-v5.2 nominal-to-zero curriculum."""
    mode = str(config["safety_budget_mode"])
    if mode in {"strict", "balanced"}:
        config["cost_budget"] = calibrated_safety_budgets(reference)[mode]
    elif mode == "v52_nominal_curriculum":
        selection = reference.get("selection")
        if not reference.get("feasible") or selection is None:
            raise ValueError("Env-v5.2 training requires a passing calibration")
        config["curriculum_d_start"] = float(
            selection["nominal_max_daily_cost"]
        )
        config["curriculum_d_target"] = 0.0
        config["cost_budget"] = 0.0
    return config


def reconcile_metrics_for_resume(
    metrics_path: str | Path,
    *,
    checkpoint_update: int,
) -> int:
    """Atomically discard metric rows newer than the restored checkpoint."""
    path = Path(metrics_path)
    if not path.exists():
        return 0
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line]
    rows = [json.loads(line) for line in lines]
    kept = [row for row in rows if int(row["update"]) <= int(checkpoint_update)]
    if checkpoint_update > 0 and not any(
        int(row["update"]) == int(checkpoint_update) for row in kept
    ):
        raise ValueError(
            f"metrics do not contain restored checkpoint update {checkpoint_update}"
        )
    removed = len(rows) - len(kept)
    if removed:
        temporary = path.with_name(path.name + ".resume-tmp")
        temporary.write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in kept),
            encoding="utf-8",
        )
        temporary.replace(path)
    return removed


def build_ff_command(name: str, *, updates: int) -> list[str]:
    """Return, without executing, the existing FF-MAPPO annealing command."""
    spec = EXPERIMENTS[name]
    if spec["family"] != "ff":
        raise ValueError(f"{name} is a GRU variant, not an FF baseline")
    root = Path(__file__).resolve().parents[1]
    total_timesteps = updates * 24 * 4
    return [
        sys.executable,
        "-m",
        "baselines.MAPPO.mappo_ff_shared_weights",
        "--config-name",
        "mappo_ff_independent_actors_microgrid",
        f"SEED={spec['seed']}",
        "NUM_SEEDS=1",
        f"TOTAL_TIMESTEPS={total_timesteps}",
        "ANNEAL_LR=true",
        "GAMMA=1.0",
        hydra_override_arg(env_v3_safe_overrides()),
    ]


def run(
    name: str,
    *,
    updates: int,
    dry_run: bool = False,
    run_dir: str | Path | None = None,
    checkpoint_interval: int = 25,
    validation_interval: int = 100,
    resume: str | Path | None = None,
    safety_reference: str | Path | None = None,
    env_v6_calibration: str | Path | None = None,
    env_parallel_backend: str | None = None,
    background_load_scale: float | None = None,
    pcc_injection_scale: float | None = None,
) -> dict[str, Any]:
    """Run one exploratory single-seed variant or return its launch description."""
    if name not in EXPERIMENTS:
        raise ValueError(f"Unknown experiment {name!r}")
    spec = EXPERIMENTS[name]
    output_root = Path(run_dir) if run_dir is not None else None
    if spec["family"] == "ff":
        if resume is not None:
            raise ValueError("FF resume is managed by its existing trainer, not this GRU runner")
        command = build_ff_command(name, updates=updates)
        if output_root is not None:
            checkpoint_root = output_root / "checkpoints"
            checkpoint_root.mkdir(parents=True, exist_ok=True)
            command.extend(
                [
                    f"CHECKPOINT_INTERVAL={checkpoint_interval * 24 * 4}",
                    f"+TRAINING_CHECKPOINT_PATH={checkpoint_root / (name + '.msgpack')}",
                ]
            )
        if not dry_run:
            subprocess.run(command, check=True)
        return {"variant": name, "exploratory": True, "command": command}

    if safety_reference is not None and env_v6_calibration is not None:
        raise ValueError("only one calibration interface may be provided")
    config = build_gru_config(name, updates=updates)
    if background_load_scale is not None:
        config["env_overrides"]["power_flow_background_load_scale"] = float(
            background_load_scale
        )
    if pcc_injection_scale is not None:
        config["env_overrides"]["power_flow_pcc_injection_scale"] = float(
            pcc_injection_scale
        )
    if safety_reference is not None:
        reference = json.loads(Path(safety_reference).read_text(encoding="utf-8"))
        apply_safety_reference(config, reference)
    if env_v6_calibration is not None:
        calibration = json.loads(
            Path(env_v6_calibration).read_text(encoding="utf-8")
        )
        apply_env_v6_calibration(
            config,
            calibration,
            budget_scale=float(EXPERIMENTS[name].get("cost_budget_scale", 1.0)),
        )
    if env_parallel_backend is not None:
        config["env_parallel_backend"] = str(env_parallel_backend)
    checkpoint_dir = (
        output_root / "checkpoints" / name if output_root is not None else None
    )
    metrics_path = (
        output_root / f"{name}.metrics.jsonl" if output_root is not None else None
    )
    if dry_run:
        return {
            "variant": name,
            "exploratory": True,
            "config": config,
            "target_updates": updates,
            "checkpoint_interval": checkpoint_interval,
            "validation_interval": validation_interval,
            "checkpoint_dir": str(checkpoint_dir) if checkpoint_dir is not None else None,
            "metrics_path": str(metrics_path) if metrics_path is not None else None,
            "resume": str(resume) if resume is not None else None,
        }
    trainer = SafeGRUMAPPOTrainer(config)
    try:
        start_update = 0
        if resume is not None:
            start_update = trainer.load_checkpoint(resume, algorithm=spec["algorithm"])
        if start_update > updates:
            raise ValueError("checkpoint update exceeds requested total updates")
        if resume is not None and metrics_path is not None:
            reconcile_metrics_for_resume(
                metrics_path,
                checkpoint_update=start_update,
            )
        rollout_root = output_root / "rollouts" / name if output_root is not None else None

        def save_validation(update: int) -> None:
            if rollout_root is None:
                return
            rollout_root.mkdir(parents=True, exist_ok=True)
            (rollout_root / f"update_{update:06d}.json").write_text(
                json.dumps(trainer.deterministic_rollout(seed=spec["seed"]), indent=2),
                encoding="utf-8",
            )

        metrics = trainer.train(
            updates - start_update,
            algorithm=spec["algorithm"],
            start_update=start_update,
            checkpoint_dir=checkpoint_dir,
            checkpoint_interval=checkpoint_interval,
            metrics_path=metrics_path,
            validation_interval=validation_interval if output_root is not None else 0,
            validation_callback=save_validation if output_root is not None else None,
        )
        report = trainer.deterministic_rollout(seed=spec["seed"])
    finally:
        trainer.close()
    result = {
        "variant": name,
        "exploratory": True,
        "algorithm": spec["algorithm"],
        # The resolved config, so a run directory is self-describing.  This is the config the
        # trainer actually received (post-calibration), and it is the only artifact that records
        # the voltage band the run TRAINED against -- soft-margin runs train on a tightened band
        # and are evaluated on the true one, and that distinction was previously recoverable only
        # from the launcher spec at the right commit.  Written to the summary rather than into
        # the trainer config on purpose: the checkpoint fingerprint is a sha256 of the trainer
        # config, so adding a key there would invalidate every existing checkpoint.
        "config": config,
        "target_updates": updates,
        "resumed_from_update": start_update,
        "dimensions": {
            "num_agents": trainer.num_agents,
            "action_dim": trainer.action_dim,
            "base_obs_dim": trainer.base_obs_dim,
            "actor_obs_dim": trainer.obs_dim,
        },
        "metrics": metrics,
        "deterministic_rollout": report,
    }
    if output_root is not None:
        output_root.mkdir(parents=True, exist_ok=True)
        (output_root / f"{name}.json").write_text(
            json.dumps(result, indent=2, default=str), encoding="utf-8"
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("variant", choices=sorted(EXPERIMENTS))
    parser.add_argument("--updates", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--checkpoint-interval", type=int, default=25)
    parser.add_argument("--validation-interval", type=int, default=100)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--safety-reference", type=Path)
    parser.add_argument("--env-v6-calibration", type=Path)
    parser.add_argument("--env-parallel-backend", choices=("serial", "process"))
    parser.add_argument("--background-load-scale", type=float)
    parser.add_argument("--pcc-injection-scale", type=float)
    args = parser.parse_args()
    print(
        json.dumps(
            run(
                args.variant,
                updates=args.updates,
                dry_run=args.dry_run,
                run_dir=args.run_dir,
                checkpoint_interval=args.checkpoint_interval,
                validation_interval=args.validation_interval,
                resume=args.resume,
                safety_reference=args.safety_reference,
                env_v6_calibration=args.env_v6_calibration,
                env_parallel_backend=args.env_parallel_backend,
                background_load_scale=args.background_load_scale,
                pcc_injection_scale=args.pcc_injection_scale,
            ),
            indent=2,
            default=str,
        )
    )


if __name__ == "__main__":
    main()
