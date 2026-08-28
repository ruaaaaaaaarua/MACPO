#!/usr/bin/env python3
"""Evaluate deterministic rule baselines on locked validation and test days."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from baselines.utils.fixed_scenario_eval import (  # noqa: E402
    DEFAULT_NOISE_SEEDS,
    TEST_DAYS,
    TEST_NOISE_SEEDS,
    VALIDATION_DAYS,
    append_evaluation_record,
    build_scenarios,
    evaluate_contextual_policy,
)
from baselines.utils.rule_baselines import RULE_BASELINES  # noqa: E402
from scripts.run_abc_multialg_parallel import group_abc_spec  # noqa: E402


def run_rule_baselines(output_root, base_overrides=None):
    output_root = Path(output_root)
    if output_root.exists():
        raise FileExistsError(f"output root must be fresh: {output_root}")
    output_root.mkdir(parents=True)
    records_path = output_root / "rule_baselines.jsonl"
    overrides = dict(
        group_abc_spec().env_overrides
        if base_overrides is None
        else base_overrides
    )
    overrides["italian_split_strategy"] = "manifest"

    split_specs = {
        "validation": build_scenarios(VALIDATION_DAYS, DEFAULT_NOISE_SEEDS),
        "test": build_scenarios(TEST_DAYS, TEST_NOISE_SEEDS),
    }
    compact = {"results": {}}
    for split_name, scenarios in split_specs.items():
        compact["results"][split_name] = {}
        for algorithm, policy in RULE_BASELINES.items():
            result = evaluate_contextual_policy(
                policy,
                overrides,
                scenarios,
                algorithm=algorithm,
                split_name=split_name,
            )
            append_evaluation_record(
                records_path,
                result,
                training_episode=0,
            )
            compact["results"][split_name][algorithm] = {
                "privileged_diagnostic": result["privileged_diagnostic"],
                **result["summary"],
            }

    summary_path = output_root / "rule_baselines_summary.json"
    summary_path.write_text(
        json.dumps(
            compact,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    return {
        "records_path": str(records_path),
        "summary_path": str(summary_path),
        "summary": compact,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args()
    result = run_rule_baselines(args.output_root)
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
