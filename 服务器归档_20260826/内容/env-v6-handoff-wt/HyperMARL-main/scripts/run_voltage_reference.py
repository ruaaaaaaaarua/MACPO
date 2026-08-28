"""Run a non-training voltage-coordination reference rollout."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

# Allow direct execution from the repository root (``python scripts/...py``)
# without requiring callers to set PYTHONPATH manually.
import sys
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from baselines.MAPPO.safe_gru_trainer import SafeGRUMAPPOTrainer
from scripts.analyze_env_v6_swiss import _latest_checkpoint
from scripts.run_env_v3_safe_matrix import EXPERIMENTS, apply_env_v6_calibration, build_gru_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--variant", default="v6_nocomm_gru_macpo")
    parser.add_argument("--seed", type=int, default=31)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    calibration = json.loads(args.calibration.read_text(encoding="utf-8"))
    config = build_gru_config(args.variant, updates=1000)
    apply_env_v6_calibration(config, calibration)
    trainer = SafeGRUMAPPOTrainer(config)
    try:
        checkpoint = _latest_checkpoint(args.run_dir, args.variant)
        restored = trainer.load_checkpoint(
            checkpoint, algorithm=str(EXPERIMENTS[args.variant]["algorithm"])
        )
        baseline = trainer.deterministic_rollout(seed=args.seed)
        reference = trainer.deterministic_rollout(
            seed=args.seed, voltage_reference=True
        )
    finally:
        trainer.close()

    report = {
        "environment": "env-v6-swiss",
        "variant": args.variant,
        "seed": args.seed,
        "checkpoint": checkpoint.name,
        "restored_update": restored,
        "policy_type": "rule_based_voltage_coordination_reference",
        "baseline_summary": baseline["summary"],
        "reference_summary": reference["summary"],
        "reference_steps": reference["steps"],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / f"env_v6_voltage_reference_seed{args.seed}.json"
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(output), "baseline": baseline["summary"], "reference": reference["summary"]}, indent=2))


if __name__ == "__main__":
    main()
