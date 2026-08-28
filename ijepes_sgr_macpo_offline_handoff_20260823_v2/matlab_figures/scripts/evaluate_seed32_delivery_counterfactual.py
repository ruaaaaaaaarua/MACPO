"""Evaluate one SGR-MACPO checkpoint with delayed and instant H2 delivery.

This script is intended to run in the canonical server environment.  It loads
the existing update-1000 checkpoint and performs two deterministic seed-32
rollouts without training.  Checkpoint-compatible observation dimensions are
preserved by keeping traffic observations enabled and changing only the
physical delivery lag after the checkpoint has been restored.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from baselines.MAPPO.safe_gru_trainer import SafeGRUMAPPOTrainer
from scripts.run_env_v3_safe_matrix import (
    EXPERIMENTS,
    apply_env_v6_calibration,
    build_gru_config,
)


VARIANT = "v6_nocomm_gru_macpo"
RUN_DIR = Path("/root/autodl-tmp/env_v6_swiss_runs/long")
CALIBRATION_PATH = Path("/root/autodl-tmp/env_v6_swiss_runs/calibration.json")
CHECKPOINT = RUN_DIR / "checkpoints" / VARIANT / "update_001000.msgpack"
OUTPUT = RUN_DIR / "seed32_delivery_counterfactual.json"

EXPECTED_DELAYED = {
    "economic_cost": 3817816.7437592787,
    "daily_voltage_cost": 0.0,
    "voltage_min_pu": 0.9572422358159942,
    "planned_h2_order": 43434.296241760254,
    "emergency_h2_buy": 43162.64447402954,
    "late_h2_order": 13377.458387374878,
    "pending_h2_energy": 8380.8515625,
}


def assert_reference_match(summary: dict[str, object]) -> None:
    mismatches = {}
    for key, expected in EXPECTED_DELAYED.items():
        actual = float(summary[key])
        tolerance = max(1.0e-7, abs(expected) * 1.0e-9)
        if abs(actual - expected) > tolerance:
            mismatches[key] = {"expected": expected, "actual": actual}
    if mismatches:
        raise RuntimeError(
            "Canonical delayed rollout does not match the archived paper source: "
            + json.dumps(mismatches, sort_keys=True)
        )


def main() -> None:
    calibration = json.loads(CALIBRATION_PATH.read_text(encoding="utf-8"))
    config = build_gru_config(VARIANT, updates=1000)
    apply_env_v6_calibration(config, calibration)
    trainer = SafeGRUMAPPOTrainer(config)
    try:
        restored = trainer.load_checkpoint(
            CHECKPOINT, algorithm=str(EXPERIMENTS[VARIANT]["algorithm"])
        )
        delayed = trainer.deterministic_rollout(seed=32)
        assert_reference_match(delayed["summary"])

        original_overrides = dict(trainer.config["env_overrides"])
        instant_overrides = dict(original_overrides)
        instant_overrides["h2_instant_delivery_counterfactual"] = True
        trainer.config["env_overrides"] = instant_overrides
        instant = trainer.deterministic_rollout(seed=32)
        trainer.config["env_overrides"] = original_overrides
    finally:
        trainer.close()

    report = {
        "variant": VARIANT,
        "checkpoint": str(CHECKPOINT),
        "restored_update": int(restored),
        "seed": 32,
        "comparison": "same checkpoint and seed; physical H2 delivery lag only",
        "delayed_delivery_lag_hours": int(original_overrides["h2_delivery_lag"]),
        "instant_delivery_lag_hours": 0,
        "instant_delivery_implementation": (
            "checkpoint-compatible evaluation switch; internal and planned-external "
            "purchases are delivered in the clearing step"
        ),
        "delayed": delayed,
        "instant": instant,
    }
    OUTPUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(OUTPUT),
                "delayed": delayed["summary"],
                "instant": instant["summary"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
