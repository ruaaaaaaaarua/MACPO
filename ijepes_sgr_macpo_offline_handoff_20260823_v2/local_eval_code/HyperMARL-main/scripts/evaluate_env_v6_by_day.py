"""Day-enumerated evaluation for Env-v6 checkpoints.

Motivation
----------
The Italian profile source contains only 28 complete days (20 train / 4 test /
4 validation per ``envs/microgrid/italian_day_splits.json``).  Seed-sampled
evaluation draws days *with replacement* from the train pool, so "n seeds" vastly
overstates the effective sample size: within one underlying day the daily voltage
cost varies by at most a few percent, while across days it spans 0 to ~1.9.  This
script therefore enumerates each day exactly once (via ``italian_day_indices``)
and treats the day, not the seed, as the unit of analysis.  Residual stochasticity
(H2 orders, traffic, synthetic sub-profiles) is averaged over a small number of
seeds per day.

The held-out days (test + validation) are evaluated with the same machinery,
giving the first generalization measurement for the frozen checkpoint.

Usage
-----
    python scripts/evaluate_env_v6_by_day.py \
        --run-dir /root/autodl-tmp/env_v6_swiss_runs/long \
        --calibration /root/autodl-tmp/env_v6_swiss_runs/calibration.json \
        --out <new-output-dir>/by_day_eval.json

Day overrides are applied to the evaluation environment only; the trainer keeps
the training config so the checkpoint fingerprint check still passes.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
import sys
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import jax.numpy as jnp

from baselines.MAPPO.safe_gru_trainer import SafeGRUMAPPOTrainer, reset_actor_hidden
from baselines.utils.microgrid_vec_env import MicrogridVecEnv
from scripts.run_env_v3_safe_matrix import (
    EXPERIMENTS,
    apply_env_v6_calibration,
    build_gru_config,
)

RAW_BUDGET = 0.02
SPLIT_MANIFEST = REPO_ROOT / "envs" / "microgrid" / "italian_day_splits.json"


def load_day_splits() -> dict[int, str]:
    manifest = json.loads(SPLIT_MANIFEST.read_text(encoding="utf-8"))
    return {int(day): str(entry["split"]) for day, entry in manifest["days"].items()}


def build_trainer(*, variant: str, run_dir: Path, calibration_path: Path, updates: int):
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    config = build_gru_config(variant, updates=updates)
    apply_env_v6_calibration(
        config,
        calibration,
        budget_scale=float(EXPERIMENTS[variant].get("cost_budget_scale", 1.0)),
    )
    trainer = SafeGRUMAPPOTrainer(config)
    checkpoints = sorted((run_dir / "checkpoints" / variant).glob("update_*.msgpack"))
    if not checkpoints:
        raise FileNotFoundError(f"no checkpoint under {run_dir}/checkpoints/{variant}")
    trainer.load_checkpoint(
        checkpoints[-1], algorithm=str(EXPERIMENTS[variant]["algorithm"])
    )
    return trainer, checkpoints[-1]


def parse_env_override(item: str) -> tuple[str, Any]:
    """Parse a KEY=VALUE pair, decoding VALUE as JSON when possible."""
    if "=" not in item:
        raise argparse.ArgumentTypeError(f"expected KEY=VALUE, got {item!r}")
    key, raw = item.split("=", 1)
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        value = raw
    return key.strip(), value


def deterministic_day_rollout(
    trainer: SafeGRUMAPPOTrainer,
    *,
    day: int,
    seed: int,
    extra_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """One deterministic day with the profile day pinned via italian_day_indices.

    ``extra_overrides`` is applied last, so an evaluation can score a policy under a
    different env config than it trained on -- needed to evaluate a soft-margin
    variant (trained at 0.96/1.04) at the real voltage band 0.95/1.05.
    """
    overrides = dict(trainer.config["env_overrides"])
    overrides["italian_day_indices"] = [int(day)]
    overrides.update(extra_overrides or {})
    env = MicrogridVecEnv(num_envs=1, auto_reset=False, config_overrides=overrides)
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
        records = []
        for step in range(int(env.envs[0].env.T)):
            actor_hidden = reset_actor_hidden(actor_hidden, done)
            means, _, next_actor_hidden, _ = trainer.actor.apply(
                trainer.actor_state.params,
                local_obs,
                actor_hidden,
                return_intents=True,
                intent_broadcast_mode=None,
            )
            action = np.asarray(jnp.tanh(means[0]), dtype=np.float32)
            next_obs, _, termination, truncation, infos = env.step(
                action.reshape(-1, trainer.action_dim)
            )
            info = infos[0]
            records.append(
                {
                    "step": step,
                    "economic_cost": float(info["economic_cost"]),
                    "voltage_cost": float(info["voltage_cost"]),
                    "voltage_min_pu": info["voltage_min_pu"],
                    "voltage_max_pu": info["voltage_max_pu"],
                    "pf_converged": bool(info["pf_converged"]),
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
    vcost = float(sum(r["voltage_cost"] for r in records))
    pf_failures = sum(1 for r in records if not r["pf_converged"])
    return {
        "day": int(day),
        "seed": int(seed),
        "steps": len(records),
        "daily_voltage_cost": vcost,
        "economic_cost": float(sum(r["economic_cost"] for r in records)),
        "voltage_min_pu": min(mins) if mins else None,
        "voltage_max_pu": max(maxs) if maxs else None,
        "pf_failure_rate": float(pf_failures) / max(len(records), 1),
        "violating_hours": [r["step"] for r in records if r["voltage_cost"] > 0.0],
        "safe_day": bool(vcost <= RAW_BUDGET and pf_failures == 0 and len(records) == 24),
    }


def aggregate_split(day_rows: list[dict[str, Any]], split: str) -> dict[str, Any]:
    rows = [r for r in day_rows if r["split"] == split or split == "all"]
    if not rows:
        return {}
    vcost = np.array([r["vcost_mean"] for r in rows])
    safe = sum(1 for r in rows if r["safe_all_seeds"])
    return {
        "days": len(rows),
        "safe_days_all_seeds": safe,
        "safe_day_rate": safe / len(rows),
        "vcost_mean_of_day_means": float(vcost.mean()),
        "vcost_max_day_mean": float(vcost.max()),
        "vmin_worst": float(min(r["vmin_worst"] for r in rows)),
        "vmax_worst": float(max(r["vmax_worst"] for r in rows)),
        "econ_mean_of_day_means": float(np.mean([r["econ_mean"] for r in rows])),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", default="v6_nocomm_gru_macpo")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--updates", type=int, default=1000)
    parser.add_argument("--seeds-per-day", type=int, default=3)
    parser.add_argument("--seed-base", type=int, default=1000)
    parser.add_argument(
        "--days",
        type=int,
        nargs="*",
        default=None,
        help="Day indices to evaluate (default: all days in the split manifest)",
    )
    parser.add_argument(
        "--env-override",
        type=parse_env_override,
        action="append",
        default=None,
        metavar="KEY=VALUE",
        help=(
            "Override an env config key for evaluation only, on top of the run's own "
            "env_overrides (repeatable). Use this to score a variant trained on a "
            "shifted voltage band at the real band, e.g. "
            "--env-override power_flow_vmin_pu=0.95 --env-override power_flow_vmax_pu=1.05"
        ),
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    extra_overrides = dict(args.env_override or [])
    if extra_overrides:
        print(f"eval env overrides: {extra_overrides}", flush=True)

    splits = load_day_splits()
    days = args.days if args.days else sorted(splits)
    trainer, checkpoint = build_trainer(
        variant=args.variant,
        run_dir=args.run_dir,
        calibration_path=args.calibration,
        updates=args.updates,
    )
    day_rows: list[dict[str, Any]] = []
    try:
        for day in days:
            started = time.time()
            runs = [
                deterministic_day_rollout(
                    trainer,
                    day=day,
                    seed=args.seed_base + day * 100 + k,
                    extra_overrides=extra_overrides,
                )
                for k in range(args.seeds_per_day)
            ]
            vcosts = [r["daily_voltage_cost"] for r in runs]
            row = {
                "day": int(day),
                "split": splits.get(int(day), "unknown"),
                "vcost_mean": float(np.mean(vcosts)),
                "vcost_min": float(np.min(vcosts)),
                "vcost_max": float(np.max(vcosts)),
                "vmin_worst": float(min(r["voltage_min_pu"] for r in runs)),
                "vmax_worst": float(max(r["voltage_max_pu"] for r in runs)),
                "econ_mean": float(np.mean([r["economic_cost"] for r in runs])),
                "safe_all_seeds": bool(all(r["safe_day"] for r in runs)),
                "pf_failures": int(sum(r["pf_failure_rate"] > 0 for r in runs)),
                "violating_hours_union": sorted(
                    {h for r in runs for h in r["violating_hours"]}
                ),
                "runs": runs,
                "wall_seconds": round(time.time() - started, 2),
            }
            day_rows.append(row)
            print(
                f"[day {day:2d} {row['split']:>10}] vcost={row['vcost_mean']:.4f} "
                f"[{row['vcost_min']:.4f},{row['vcost_max']:.4f}] "
                f"vmin={row['vmin_worst']:.4f} vmax={row['vmax_worst']:.4f} "
                f"safe={row['safe_all_seeds']} hours={row['violating_hours_union']}",
                flush=True,
            )
    finally:
        trainer.close()

    report = {
        "variant": args.variant,
        "checkpoint": str(checkpoint),
        "seeds_per_day": args.seeds_per_day,
        "raw_voltage_budget": RAW_BUDGET,
        "eval_env_overrides": extra_overrides,
        "by_split": {
            split: aggregate_split(day_rows, split)
            for split in ("train", "test", "validation", "all")
        },
        "days": day_rows,
    }
    for split, agg in report["by_split"].items():
        if agg:
            print(
                f"== {split:>10}: {agg['safe_days_all_seeds']}/{agg['days']} safe days "
                f"({agg['safe_day_rate']:.0%}), vcost mean {agg['vcost_mean_of_day_means']:.4f}, "
                f"worst vmin {agg['vmin_worst']:.4f}, worst vmax {agg['vmax_worst']:.4f}"
            )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
