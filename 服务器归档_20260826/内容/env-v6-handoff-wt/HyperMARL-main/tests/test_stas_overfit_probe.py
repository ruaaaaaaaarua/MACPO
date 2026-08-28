import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
STAS_ROOT = ROOT / "baselines" / "STAS-MAPPO"
for candidate in (ROOT, STAS_ROOT):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

try:
    from scripts import run_stas_overfit_probe as probe
except ImportError:
    probe = None

from stas_mappo.conserved_credit import ConservedSTASCreditAssigner
from stas_mappo.credit_conservation import discounted_team_return


class OverfitProbeScriptPresenceTest(unittest.TestCase):
    def test_probe_script_exists(self):
        self.assertIsNotNone(probe, "run_stas_overfit_probe.py is missing")


@unittest.skipIf(probe is None, "probe script has not been implemented")
class OverfitProbeBehaviorTest(unittest.TestCase):
    def _dataset(self, samples=32, steps=3):
        rng = np.random.default_rng(30)
        return probe.ProbeDataset(
            obs=rng.normal(size=(samples, 2, steps, 4)).astype(np.float32),
            actions=rng.normal(size=(samples, 2, steps, 1)).astype(np.float32),
            rewards=rng.normal(size=(samples, 2, steps)).astype(np.float32),
            dones=np.zeros((samples, 2, steps), dtype=np.float32),
        )

    def _run_stubbed_main(self, output, *, passed):
        dataset = self._dataset(samples=32, steps=24)
        dimensions = {"obs_dim": 4, "action_dim": 1, "n_agents": 2}
        metrics = {
            "normalized_mse": 5e-4 if passed else 2e-3,
            "raw_mse": 1.0,
            "ev_values": [0.96, 0.96, 0.96],
        }
        assertions = {
            "sample_count_exactly_32": True,
            "numeric_diagnostics_finite": True,
            "normalized_mse_within_threshold": passed,
            "all_final_ev_within_threshold": True,
            "deterministic_ev_range": True,
            "target_consistency": True,
        }
        gate = {"assertions": assertions, "ev_range": 0.0, "passed": passed}
        fake_assigner = SimpleNamespace(buffer=object())
        argv = [
            "--seed", "30",
            "--samples", "32",
            "--max-updates", "5000",
            "--mse-threshold", "0.001",
            "--ev-threshold", "0.95",
            "--output", str(output),
        ]
        with (
            mock.patch.object(probe, "seed_everything"),
            mock.patch.object(
                probe,
                "collect_probe_dataset",
                return_value=(dataset, dimensions, {"italian_split_name": "train"}),
            ) as collect,
            mock.patch.object(
                probe, "ConservedSTASCreditAssigner", return_value=fake_assigner
            ),
            mock.patch.object(probe, "route_memorization_dataset", return_value=0.0),
            mock.patch.object(probe, "credit_buffer_sha256", return_value="a" * 64),
            mock.patch.object(
                probe, "train_probe", return_value=(25, 0.01, metrics, gate)
            ) as train,
            mock.patch.object(probe, "_source_commit", return_value="abc123"),
        ):
            exit_code = probe.main(argv)

        collect.assert_called_once_with(seed=30, samples=32, sequence_length=24)
        self.assertEqual(train.call_args.kwargs["max_updates"], 5000)
        self.assertEqual(len(metrics["ev_values"]), 3)
        return exit_code, json.loads(output.read_text(encoding="utf-8"))

    def test_locked_credit_config_uses_real_causal_stas_capacity(self):
        config = probe.build_credit_config(
            obs_dim=17,
            action_dim=6,
            n_agents=4,
            seq_length=24,
            device="cpu",
        )

        self.assertEqual(config.obs_dim, 17)
        self.assertEqual(config.action_dim, 6)
        self.assertEqual(config.n_agents, 4)
        self.assertEqual(config.seq_length, 24)
        self.assertEqual(config.emb_dim, 128)
        self.assertEqual(config.n_heads, 4)
        self.assertEqual(config.n_layers, 2)
        self.assertEqual(config.batch_size, 32)
        self.assertEqual(config.buffer_size, 32)
        self.assertEqual(config.dropout, 0.0)
        self.assertTrue(config.causal)
        self.assertTrue(config.conserve_discounted)

    def test_all_samples_follow_production_target_path_then_buffers_are_isolated(self):
        dataset = self._dataset()
        config = probe.build_credit_config(
            obs_dim=4,
            action_dim=1,
            n_agents=2,
            seq_length=3,
            device="cpu",
        )
        assigner = ConservedSTASCreditAssigner(config)

        target_error = probe.route_memorization_dataset(assigner, dataset)

        self.assertEqual(assigner.normalizer.count, 32)
        self.assertEqual(assigner.episodes_seen, 32)
        self.assertEqual(assigner.rollouts_seen, 1)
        self.assertEqual(len(assigner.buffer), 32)
        self.assertEqual(len(assigner.holdout_buffer), 32)
        self.assertLessEqual(target_error, 1e-6)
        expected = discounted_team_return(dataset.rewards, config.gamma)
        np.testing.assert_allclose(
            assigner.buffer.get_all()[-1], expected, rtol=0.0, atol=1e-6
        )

        self.assertFalse(
            np.shares_memory(
                assigner.buffer.storage[0][0],
                assigner.holdout_buffer.storage[0][0],
            )
        )
        untouched = assigner.holdout_buffer.storage[0][0].copy()
        assigner.buffer.storage[0][0].fill(999.0)
        np.testing.assert_array_equal(assigner.holdout_buffer.storage[0][0], untouched)

    def test_dataset_hash_is_stable_and_sensitive_to_stored_data(self):
        dataset = self._dataset()
        config = probe.build_credit_config(
            obs_dim=4,
            action_dim=1,
            n_agents=2,
            seq_length=3,
            device="cpu",
        )
        first = ConservedSTASCreditAssigner(config)
        second = ConservedSTASCreditAssigner(config)
        probe.route_memorization_dataset(first, dataset)
        probe.route_memorization_dataset(second, dataset)

        digest = probe.credit_buffer_sha256(first.buffer)
        self.assertEqual(len(digest), 64)
        self.assertEqual(digest, probe.credit_buffer_sha256(second.buffer))
        second.buffer.storage[0][2][0, 0] += 1.0
        self.assertNotEqual(digest, probe.credit_buffer_sha256(second.buffer))

    def test_gate_rejects_nonfinite_or_threshold_violation(self):
        passing = probe.evaluate_probe_gate(
            sample_count=32,
            normalized_mse=1e-3,
            raw_mse=2.0,
            ev_values=[0.95, 0.95, 0.95],
            target_error=1e-6,
            mse_threshold=1e-3,
            ev_threshold=0.95,
        )
        self.assertTrue(passing["passed"])
        self.assertTrue(all(passing["assertions"].values()))

        invalid = probe.evaluate_probe_gate(
            sample_count=32,
            normalized_mse=float("nan"),
            raw_mse=2.0,
            ev_values=[0.95, 0.95, 0.95],
            target_error=1e-6,
            mse_threshold=1e-3,
            ev_threshold=0.95,
        )
        self.assertFalse(invalid["passed"])
        self.assertFalse(invalid["assertions"]["numeric_diagnostics_finite"])

    def test_gate_requires_exactly_three_final_ev_measurements(self):
        for ev_values in ([0.95, 0.95], [0.95, 0.95, 0.95, 0.95]):
            with self.subTest(ev_count=len(ev_values)):
                result = probe.evaluate_probe_gate(
                    sample_count=32,
                    normalized_mse=1e-3,
                    raw_mse=2.0,
                    ev_values=ev_values,
                    target_error=1e-6,
                    mse_threshold=1e-3,
                    ev_threshold=0.95,
                )

                self.assertFalse(result["passed"])
                self.assertFalse(
                    result["assertions"]["all_final_ev_within_threshold"]
                )

    def test_measure_probe_evaluates_exactly_three_consecutive_ev_values(self):
        count = 32
        zeros = np.zeros((count, 2, 24), dtype=np.float32)
        assigner = SimpleNamespace(
            model=mock.Mock(),
            holdout_buffer=SimpleNamespace(
                get_all=mock.Mock(
                    return_value=(
                        zeros[..., None],
                        zeros[..., None],
                        zeros,
                        zeros,
                        np.zeros(count, dtype=np.float32),
                    )
                )
            ),
            normalizer=SimpleNamespace(
                normalize=lambda values: np.asarray(values, dtype=np.float32),
                std=1.0,
                mean=0.0,
            ),
            _predict_normalized_returns=mock.Mock(
                return_value=(None, probe.torch.zeros(count))
            ),
            holdout_explained_variance=mock.Mock(side_effect=[0.96, 0.96, 0.96]),
        )

        metrics = probe.measure_probe(assigner)

        self.assertEqual(metrics["ev_values"], [0.96, 0.96, 0.96])
        self.assertEqual(assigner.holdout_explained_variance.call_count, 3)

    def test_main_requests_locked_probe_budget_and_writes_success_or_gate_failure(self):
        for passed, expected_exit in ((True, 0), (False, 1)):
            with self.subTest(passed=passed), tempfile.TemporaryDirectory() as tmpdir:
                output = Path(tmpdir) / "probe.json"
                exit_code, payload = self._run_stubbed_main(output, passed=passed)

                self.assertEqual(exit_code, expected_exit)
                self.assertEqual(payload["passed"], passed)
                self.assertEqual(payload["sample_count"], 32)
                self.assertEqual(payload["resolved_config"]["sequence_length"], 24)
                self.assertEqual(payload["resolved_config"]["max_updates"], 5000)
                self.assertEqual(len(payload["ev_values"]), 3)

    def test_main_rejects_dataset_not_exactly_32_by_24_before_model_use(self):
        invalid_datasets = (
            self._dataset(samples=31, steps=24),
            self._dataset(samples=32, steps=23),
        )
        dimensions = {"obs_dim": 4, "action_dim": 1, "n_agents": 2}
        for dataset in invalid_datasets:
            with (
                self.subTest(shape=dataset.obs.shape),
                tempfile.TemporaryDirectory() as tmpdir,
            ):
                output = Path(tmpdir) / "probe.json"
                with (
                    mock.patch.object(probe, "seed_everything"),
                    mock.patch.object(
                        probe,
                        "collect_probe_dataset",
                        return_value=(dataset, dimensions, {}),
                    ),
                    mock.patch.object(
                        probe, "ConservedSTASCreditAssigner"
                    ) as assigner_type,
                    mock.patch.object(probe, "_source_commit", return_value="abc123"),
                ):
                    exit_code = probe.main(["--output", str(output)])

                payload = json.loads(output.read_text(encoding="utf-8"))
                self.assertEqual(exit_code, 1)
                self.assertFalse(payload["passed"])
                self.assertIn("32 samples of length 24", payload["error"])
                assigner_type.assert_not_called()


if __name__ == "__main__":
    unittest.main()
