import unittest
from pathlib import Path

import numpy as np

try:
    from baselines.utils.environment_stream import reset_environment_stream
except ImportError:
    reset_environment_stream = None


ROOT = Path(__file__).resolve().parents[1]


class _ProfileStream:
    """Minimal reset stream with independently observable day/profile draws."""

    def __init__(self):
        self._rng = np.random.RandomState()
        self.history = []

    def reset(self, seed=None):
        if seed is not None:
            self._rng = np.random.RandomState(int(seed))
        observation = {
            "day": int(self._rng.choice([0, 2, 3, 4, 5, 6])),
            "profile": self._rng.normal(size=(4, 24)).astype(np.float32),
        }
        self.history.append(observation)
        return observation, {}


class STASResumeEnvironmentStreamTest(unittest.TestCase):
    def test_resumed_day_and_profile_equal_uninterrupted_stream(self):
        self.assertIsNotNone(
            reset_environment_stream,
            "environment stream resume helper has not been implemented",
        )
        completed_updates = 7

        uninterrupted = _ProfileStream()
        expected, _ = uninterrupted.reset(seed=30)
        for _ in range(completed_updates):
            expected, _ = uninterrupted.reset()

        resumed = _ProfileStream()
        actual, _ = reset_environment_stream(
            resumed,
            seed=30,
            completed_resets=completed_updates,
        )

        self.assertEqual(actual["day"], expected["day"])
        np.testing.assert_array_equal(actual["profile"], expected["profile"])
        self.assertEqual(len(resumed.history), completed_updates + 1)

    def test_stas_training_entry_uses_seed_first_stream_restore(self):
        source = (
            ROOT / "baselines" / "STAS-MAPPO" / "mappo_stas.py"
        ).read_text(encoding="utf-8")
        self.assertIn("reset_environment_stream(", source)
        self.assertNotIn(
            "for _ in range(start_update):\n            raw_obsv, _ = env.reset()\n"
            "        raw_obsv, _ = env.reset(seed=int_seed)",
            source,
        )


if __name__ == "__main__":
    unittest.main()
