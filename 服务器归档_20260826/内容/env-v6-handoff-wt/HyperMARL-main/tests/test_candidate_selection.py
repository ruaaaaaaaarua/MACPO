import json
import tempfile
import unittest
from pathlib import Path

from scripts.run_fair_pipeline_continuation import (
    _stas_checkpoint_eligible,
    select_mappo_name,
)


def _write_eval(path: Path, values):
    with path.open("w", encoding="utf-8") as handle:
        for episode, value in zip((9000, 9500, 10000), values):
            handle.write(
                json.dumps(
                    {
                        "training_episode": episode,
                        "summary": {"return_mean": value},
                    }
                )
                + "\n"
            )


class CandidateSelectionTest(unittest.TestCase):
    def test_stas_checkpoint_requires_active_credit_mixing(self):
        healthy = {
            "last_explained_variance": 0.3,
            "last_conservation_error": 1e-6,
            "last_mix_coef": 0.05,
            "gate": {"disabled": False},
        }
        self.assertTrue(_stas_checkpoint_eligible(healthy))
        self.assertFalse(
            _stas_checkpoint_eligible({**healthy, "last_mix_coef": 0.0})
        )
        self.assertFalse(
            _stas_checkpoint_eligible({**healthy, "gate": {"disabled": True}})
        )

    def test_tie_within_two_prefers_smaller_mappo(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            small = root / "small.jsonl"
            large = root / "large.jsonl"
            _write_eval(small, (-300.0, -298.0, -297.0))
            _write_eval(large, (-299.0, -297.0, -296.0))
            name, scores = select_mappo_name(small, large)
        self.assertEqual(name, "stable_mappo_128")
        self.assertLessEqual(abs(scores["stable_mappo_256"] - scores["stable_mappo_128"]), 2)

    def test_clear_higher_score_selects_256(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            small = root / "small.jsonl"
            large = root / "large.jsonl"
            _write_eval(small, (-310.0, -309.0, -308.0))
            _write_eval(large, (-300.0, -299.0, -298.0))
            name, _ = select_mappo_name(small, large)
        self.assertEqual(name, "stable_mappo_256")


if __name__ == "__main__":
    unittest.main()
