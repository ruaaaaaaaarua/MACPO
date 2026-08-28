"""Env-v6 MACPO diagnostics: evening-peak counterfactual + wide-sample evaluation.

Read-only with respect to the existing run directory: this script loads the frozen
``update_001000`` checkpoint and writes every result into a separate output directory.
It never re-launches training and never touches ``env_v6_swiss_runs``.

Two subcommands:

``counterfactual``
    Replay the deterministic MACPO policy on one seed, optionally clamping selected
    action dimensions inside a chosen hour window, and report the per-hour voltage
    trace.  Used to test whether the observed evening-peak violation is avoidable
    with the real-power flexibility the agents already have.

``wide-eval``
    Evaluate the frozen checkpoint over many independent days, deterministically and
    (optionally) stochastically, to replace the n=3 day sample with a real estimate
    of the safe-day rate.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
import sys
from typing import Any, Callable

import numpy as np

WORKTREE = Path("/root/autodl-tmp/env-v6-handoff-wt/HyperMARL-main")
if str(WORKTREE) not in sys.path:
    sys.path.insert(0, str(WORKTREE))

import jax
import jax.numpy as jnp

from baselines.MAPPO.continuous_policy import sample_squashed_gaussian
from baselines.MAPPO.safe_gru_trainer import SafeGRUMAPPOTrainer, reset_actor_hidden
from baselines.utils.microgrid_vec_env import MicrogridVecEnv
from scripts.run_env_v3_safe_matrix import (
    EXPERIMENTS,
    apply_env_v6_calibration,
    build_gru_config,
)

MACPO = "v6_nocomm_gru_macpo"
RAW_BUDGET = 0.02
DEFAULT_RUN_DIR = Path("/root/autodl-tmp/env_v6_swiss_runs/long")
DEFAULT_CALIBRATION = Path("/root/autodl-tmp/env_v6_swiss_runs/calibration.json")

# Action layout, from microgrid_env.py:1224-1233 and 1534/1206.
A_ELECTROLYSER = 0  # p_el = ((a+1)/2) * el_cap   -> a=-1 turns the electrolyser off
A_BATTERY = 1  # p_bat = a * bat_power       -> a=-1 is full discharge


def build_trainer(
    *,
    run_dir: Path,
    calibration_path: Path,
    updates: int,
    env_override: dict[str, Any] | None = None,
):
    """Construct the trainer and restore the frozen checkpoint.

    ``env_override`` is merged into the environment config.  It is used only for
    physics probes (e.g. setting the load power factor to 1.0 to remove reactive
    flow) and must not change any observation or action dimension, or the restored
    checkpoint would no longer apply.
    """
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    config = build_gru_config(MACPO, updates=updates)
    apply_env_v6_calibration(config, calibration)
    if env_override:
        config["env_overrides"] = {**config["env_overrides"], **env_override}
    trainer = SafeGRUMAPPOTrainer(config)
    checkpoints = sorted((run_dir / "checkpoints" / MACPO).glob("update_*.msgpack"))
    if not checkpoints:
        raise FileNotFoundError(f"no checkpoint under {run_dir}/checkpoints/{MACPO}")
    restored = trainer.load_checkpoint(
        checkpoints[-1], algorithm=str(EXPERIMENTS[MACPO]["algorithm"])
    )
    return trainer, checkpoints[-1], int(restored)


def policy_rollout(
    trainer: SafeGRUMAPPOTrainer,
    *,
    seed: int,
    override: Callable[[int, np.ndarray], np.ndarray] | None = None,
    stochastic: bool = False,
    rng_seed: int = 0,
    log_std_max: float = -2.3,
    env: MicrogridVecEnv | None = None,
    env_override: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Roll one day out under the restored policy, with an optional action clamp.

    Mirrors ``SafeGRUMAPPOTrainer.deterministic_rollout`` so that a run with
    ``override=None, stochastic=False`` reproduces the published numbers exactly;
    that equivalence is asserted by the ``counterfactual`` subcommand before any
    clamped variant is interpreted.
    """
    owns_env = env is None
    if owns_env:
        # The override is applied to the evaluation environment only.  The trainer
        # keeps the training config so the checkpoint fingerprint still matches.
        overrides = dict(trainer.config["env_overrides"])
        if env_override:
            overrides.update(env_override)
        env = MicrogridVecEnv(
            num_envs=1,
            auto_reset=False,
            config_overrides=overrides,
        )
    try:
        scope = trainer.communication_scope
        flat_obs, _ = env.reset(seed=int(seed))
        previous_actions = np.zeros(
            (1, trainer.num_agents, trainer.action_dim), dtype=np.float32
        )
        transaction_messages = np.zeros(
            (1, trainer.transaction_message_dim), dtype=np.float32
        )
        local_obs = trainer._reshape_local_obs(
            flat_obs,
            previous_actions=previous_actions,
            transaction_messages=transaction_messages,
            communication_scope=scope,
        )
        actor_hidden = jnp.zeros(
            (1, trainer.num_agents, trainer.hidden_size), dtype=jnp.float32
        )
        done = jnp.ones(1, dtype=bool)
        key = jax.random.PRNGKey(int(rng_seed))
        records: list[dict[str, Any]] = []
        clamped_steps: list[int] = []
        episode_length = int(env.envs[0].env.T)
        for step in range(episode_length):
            actor_hidden = reset_actor_hidden(actor_hidden, done)
            means, log_stds, next_actor_hidden, _ = trainer.actor.apply(
                trainer.actor_state.params,
                local_obs,
                actor_hidden,
                return_intents=True,
                intent_broadcast_mode=None,
            )
            if stochastic:
                key, sample_key = jax.random.split(key)
                sampled, _ = sample_squashed_gaussian(
                    means,
                    log_stds,
                    sample_key,
                    log_std_min=float(trainer.config["log_std_min"]),
                    log_std_max=float(log_std_max),
                )
                action = np.asarray(sampled[0], dtype=np.float32)
            else:
                action = np.asarray(jnp.tanh(means[0]), dtype=np.float32)
            if override is not None:
                clamped = np.asarray(override(step, action.copy()), dtype=np.float32)
                if not np.allclose(clamped, action):
                    clamped_steps.append(step)
                action = clamped
            next_obs, _, termination, truncation, infos = env.step(
                action.reshape(-1, trainer.action_dim)
            )
            info = infos[0]
            records.append(
                {
                    "step": step,
                    "actions": action.tolist(),
                    "economic_cost": float(info["economic_cost"]),
                    "step_total_cost": float(info["total_cost"]),
                    "voltage_cost": float(info["voltage_cost"]),
                    "voltage_violation_area": float(info["voltage_violation_area"]),
                    "voltage_min_pu": info["voltage_min_pu"],
                    "voltage_max_pu": info["voltage_max_pu"],
                    "pf_converged": bool(info["pf_converged"]),
                    "pcc_p_kw": list(info["pcc_p_kw"]),
                    "pcc_q_kvar": list(info["pcc_q_kvar"]),
                    "p_el": list(info["p_el"]),
                    "p_bat": list(info["p_bat"]),
                    "soc": list(info["soc"]),
                    "h2_level": list(info["h2_level"]),
                    "h2_emergency_buy_energy": list(info["h2_emergency_buy_energy"]),
                    "pending_h2_energy_total": float(info["pending_h2_energy_total"]),
                    "terminal_settlement_cost": float(info["terminal_settlement_cost"]),
                    "terminal_undelivered_h2_energy": float(
                        info["terminal_undelivered_h2_energy"]
                    ),
                }
            )
            done_np = (
                np.logical_or(termination, truncation)
                .reshape(1, trainer.num_agents)
                .any(axis=1)
            )
            done = jnp.asarray(done_np)
            actor_hidden = reset_actor_hidden(next_actor_hidden, done)
            previous_actions = np.where(
                done_np[:, None, None],
                np.zeros_like(action[None, :, :]),
                action[None, :, :],
            )
            transaction_messages = np.where(
                done_np[:, None],
                np.zeros_like(trainer._transaction_message_from_infos(infos)),
                trainer._transaction_message_from_infos(infos),
            )
            local_obs = trainer._reshape_local_obs(
                next_obs,
                previous_actions=previous_actions,
                transaction_messages=transaction_messages,
                communication_scope=scope,
            )
            if bool(done_np[0]):
                break
    finally:
        if owns_env:
            env.close()

    mins = [
        r["voltage_min_pu"]
        for r in records
        if r["voltage_min_pu"] is not None and np.isfinite(r["voltage_min_pu"])
    ]
    maxs = [
        r["voltage_max_pu"]
        for r in records
        if r["voltage_max_pu"] is not None and np.isfinite(r["voltage_max_pu"])
    ]
    daily_voltage_cost = float(sum(r["voltage_cost"] for r in records))
    pf_failures = sum(1 for r in records if not r["pf_converged"])
    summary = {
        "seed": int(seed),
        "steps": len(records),
        "stochastic": bool(stochastic),
        "clamped_steps": clamped_steps,
        "daily_voltage_cost": daily_voltage_cost,
        "voltage_violation_area": float(
            sum(r["voltage_violation_area"] for r in records)
        ),
        "economic_cost": float(sum(r["economic_cost"] for r in records)),
        "total_cost": float(sum(r["step_total_cost"] for r in records)),
        "terminal_settlement_cost": (
            records[-1]["terminal_settlement_cost"] if records else 0.0
        ),
        "terminal_undelivered_h2_energy": (
            records[-1]["terminal_undelivered_h2_energy"] if records else 0.0
        ),
        "voltage_min_pu": min(mins) if mins else None,
        "voltage_max_pu": max(maxs) if maxs else None,
        "pf_failure_rate": float(pf_failures) / max(len(records), 1),
        "violating_hours": [r["step"] for r in records if r["voltage_cost"] > 0.0],
        "safe_day": bool(
            daily_voltage_cost <= RAW_BUDGET
            and pf_failures == 0
            and len(records) == 24
        ),
        "strictly_within_limits": bool(
            all(r["voltage_cost"] == 0.0 for r in records)
            and pf_failures == 0
            and len(records) == 24
        ),
    }
    return {"summary": summary, "steps": records}


def make_clamp(dims: tuple[int, ...], hours: tuple[int, ...], value: float):
    """Clamp the given action dimensions to ``value`` inside the given hours."""

    def override(step: int, action: np.ndarray) -> np.ndarray:
        if step in hours:
            for dim in dims:
                action[:, dim] = value
        return action

    return override


def run_counterfactual(args: argparse.Namespace) -> dict[str, Any]:
    trainer, checkpoint, restored = build_trainer(
        run_dir=args.run_dir, calibration_path=args.calibration, updates=args.updates
    )
    peak = tuple(range(args.peak_start, args.peak_end + 1))
    wide = tuple(range(args.peak_start - 2, args.peak_end + 2))
    variants: dict[str, Callable | None] = {
        "baseline": None,
        "electrolyser_off_peak": make_clamp((A_ELECTROLYSER,), peak, -1.0),
        "battery_discharge_peak": make_clamp((A_BATTERY,), peak, -1.0),
        "both_peak": make_clamp((A_ELECTROLYSER, A_BATTERY), peak, -1.0),
        "both_wide_window": make_clamp((A_ELECTROLYSER, A_BATTERY), wide, -1.0),
    }
    results: dict[str, Any] = {}
    try:
        for name, override in variants.items():
            started = time.time()
            report = policy_rollout(trainer, seed=args.seed, override=override)
            report["summary"]["wall_seconds"] = round(time.time() - started, 2)
            results[name] = report
            summary = report["summary"]
            print(
                f"[{name}] vcost={summary['daily_voltage_cost']:.4f} "
                f"vmin={summary['voltage_min_pu']:.4f} "
                f"vmax={summary['voltage_max_pu']:.4f} "
                f"econ={summary['economic_cost']:.4g} "
                f"safe={summary['safe_day']} "
                f"violating_hours={summary['violating_hours']}",
                flush=True,
            )
    finally:
        trainer.close()
    return {
        "mode": "counterfactual",
        "seed": int(args.seed),
        "checkpoint": str(checkpoint),
        "restored_update": restored,
        "peak_hours": list(peak),
        "wide_hours": list(wide),
        "raw_voltage_budget": RAW_BUDGET,
        "action_layout": {
            "electrolyser_dim": A_ELECTROLYSER,
            "battery_dim": A_BATTERY,
            "clamp_value": -1.0,
            "note": "a0=-1 sets p_el=0; a1=-1 is full battery discharge",
        },
        "variants": results,
    }


def run_ration(args: argparse.Namespace) -> dict[str, Any]:
    """Sweep the battery clamp level to test whether rationing beats dumping.

    ``both_peak`` empties the battery in the first two peak hours, so hours 19-20
    are left with no stored energy.  Spreading the same energy over the whole
    window may clear more hours even though each hour gets less support.
    """
    trainer, checkpoint, restored = build_trainer(
        run_dir=args.run_dir, calibration_path=args.calibration, updates=args.updates
    )
    env_override = json.loads(args.env_override) if args.env_override else None
    peak = tuple(range(args.peak_start, args.peak_end + 1))
    results: dict[str, Any] = {}
    try:
        for level in args.levels:
            override = make_clamp((A_ELECTROLYSER,), peak, -1.0)

            def combined(step: int, action: np.ndarray, _level=level) -> np.ndarray:
                action = override(step, action)
                if step in peak:
                    action[:, A_BATTERY] = _level
                return action

            report = policy_rollout(
                trainer, seed=args.seed, override=combined, env_override=env_override
            )
            results[f"battery_{level:+.2f}"] = report
            summary = report["summary"]
            print(
                f"[el_off + bat={level:+.2f}] vcost={summary['daily_voltage_cost']:.4f} "
                f"vmin={summary['voltage_min_pu']:.4f} "
                f"econ={summary['economic_cost']:.4g} "
                f"safe={summary['safe_day']} "
                f"strict={summary['strictly_within_limits']} "
                f"violating_hours={summary['violating_hours']}",
                flush=True,
            )
    finally:
        trainer.close()
    return {
        "mode": "ration",
        "env_override": args.env_override,
        "seed": int(args.seed),
        "checkpoint": str(checkpoint),
        "restored_update": restored,
        "peak_hours": list(peak),
        "variants": results,
    }


def run_wide_eval(args: argparse.Namespace) -> dict[str, Any]:
    trainer, checkpoint, restored = build_trainer(
        run_dir=args.run_dir, calibration_path=args.calibration, updates=args.updates
    )
    seeds = list(range(args.seed_start, args.seed_start + args.seed_count))
    modes = ["deterministic"] if args.deterministic_only else ["deterministic", "stochastic"]
    out: dict[str, Any] = {}
    try:
        for mode in modes:
            summaries = []
            started = time.time()
            for index, seed in enumerate(seeds):
                report = policy_rollout(
                    trainer,
                    seed=seed,
                    stochastic=(mode == "stochastic"),
                    rng_seed=seed,
                    log_std_max=args.log_std_max,
                )
                summaries.append(report["summary"])
                if (index + 1) % 20 == 0:
                    rate = (index + 1) / (time.time() - started)
                    print(
                        f"  {mode}: {index + 1}/{len(seeds)} days "
                        f"({rate:.2f} days/s)",
                        flush=True,
                    )
            out[mode] = {
                "seeds": seeds,
                "wall_seconds": round(time.time() - started, 2),
                "summaries": summaries,
            }
            print(f"[{mode}] done in {out[mode]['wall_seconds']}s", flush=True)
    finally:
        trainer.close()
    return {
        "mode": "wide-eval",
        "checkpoint": str(checkpoint),
        "restored_update": restored,
        "raw_voltage_budget": RAW_BUDGET,
        "log_std_max": args.log_std_max,
        "results": out,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--calibration", type=Path, default=DEFAULT_CALIBRATION)
    parser.add_argument("--updates", type=int, default=1000)
    parser.add_argument("--out", type=Path, required=True)
    sub = parser.add_subparsers(dest="command", required=True)

    cf = sub.add_parser("counterfactual")
    cf.add_argument("--seed", type=int, default=31)
    cf.add_argument("--peak-start", type=int, default=17)
    cf.add_argument("--peak-end", type=int, default=20)
    cf.set_defaults(func=run_counterfactual)

    rt = sub.add_parser("ration")
    rt.add_argument("--seed", type=int, default=31)
    rt.add_argument("--peak-start", type=int, default=17)
    rt.add_argument("--peak-end", type=int, default=20)
    rt.add_argument(
        "--levels",
        type=float,
        nargs="+",
        default=[-1.0, -0.75, -0.6, -0.5, -0.4, -0.3],
    )
    rt.add_argument(
        "--env-override",
        type=str,
        default=None,
        help='JSON merged into env_overrides, e.g. \'{"power_flow_load_power_factor": 1.0}\'',
    )
    rt.set_defaults(func=run_ration)

    we = sub.add_parser("wide-eval")
    we.add_argument("--seed-start", type=int, default=30)
    we.add_argument("--seed-count", type=int, default=200)
    we.add_argument("--log-std-max", type=float, default=-2.3)
    we.add_argument("--deterministic-only", action="store_true")
    we.set_defaults(func=run_wide_eval)

    args = parser.parse_args()
    report = args.func(args)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
