import importlib
import inspect
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import yaml

STAS_ROOT = Path(__file__).resolve().parents[1] / "baselines" / "STAS-MAPPO"
if str(STAS_ROOT) not in sys.path:
    sys.path.insert(0, str(STAS_ROOT))

from stas_mappo.conserved_credit import ConservedSTASCreditAssigner  # noqa: E402
from stas_mappo.credit import EpisodeCreditBuffer, STASCreditConfig  # noqa: E402

diagnostics = importlib.import_module("stas_mappo.diagnostics")
mappo_stas = importlib.import_module("mappo_stas")


class RolloutDiagnosticsModuleTest(unittest.TestCase):
    def test_diagnostics_module_is_available(self):
        try:
            module = importlib.import_module("stas_mappo.diagnostics")
        except ModuleNotFoundError:
            module = None

        self.assertIsNotNone(
            module,
            "rollout diagnostics require a focused stas_mappo.diagnostics module",
        )


class BufferTargetErrorTest(unittest.TestCase):
    def test_all_buffers_are_consistent_and_corruption_is_detected(self):
        compute_target_error = getattr(diagnostics, "compute_target_error", None)
        self.assertTrue(
            callable(compute_target_error),
            "diagnostics must compute target error from current buffer contents",
        )
        gamma = 0.97
        train = EpisodeCreditBuffer(capacity=2)
        holdout = EpisodeCreditBuffer(capacity=2)
        for index, destination in enumerate((train, holdout), start=1):
            rewards = np.array(
                [[index, 0.5, -0.25], [0.1, -0.2, 0.3]], dtype=np.float32
            )
            target = diagnostics.discounted_team_return(
                rewards[None, ...], gamma
            )[0]
            destination.add(
                np.zeros((2, 3, 1), dtype=np.float32),
                np.zeros((2, 3, 1), dtype=np.float32),
                rewards,
                np.zeros((2, 3), dtype=np.float32),
                target,
            )

        healthy = compute_target_error((train, holdout), gamma)
        self.assertLessEqual(healthy, 1e-4)
        self.assertEqual(compute_target_error((), gamma), 0.0)

        stored = holdout.storage[0]
        corrupted_rewards = stored[2].copy()
        corrupted_rewards[0, 0] += 0.5
        holdout.storage[0] = (*stored[:2], corrupted_rewards, *stored[3:])

        corrupted = compute_target_error((train, holdout), gamma)
        self.assertGreater(corrupted, 0.49)

    def test_real_assigner_tracks_rollout_and_diagnostic_target_error(self):
        assigner = ConservedSTASCreditAssigner(_small_conserved_config())
        self.assertEqual(assigner.last_target_error, 0.0)

        assigner.add_rollout(*_small_rollout())
        self.assertLessEqual(assigner.last_target_error, 1e-6)

        stored = assigner.buffer.storage[0]
        corrupted_rewards = stored[2].copy()
        corrupted_rewards[0, 0] += 0.5
        assigner.buffer.storage[0] = (
            *stored[:2],
            corrupted_rewards,
            *stored[3:],
        )
        record = diagnostics.build_rollout_record(
            assigner,
            update=1,
            episode=1,
            global_step=3,
        )

        self.assertGreater(record["target_error"], 0.49)
        self.assertEqual(assigner.last_target_error, record["target_error"])


class GateStateTest(unittest.TestCase):
    def test_gate_phases_and_reasons_are_distinct(self):
        classify_gate_state = getattr(diagnostics, "classify_gate_state", None)
        self.assertTrue(
            callable(classify_gate_state),
            "diagnostics must classify each quality-gate decision",
        )
        common = {
            "minimum_training_buffer": 2,
            "warmup_episodes": 10,
            "explained_variance_threshold": 0.2,
            "negative_streak": 3,
            "disabled": False,
        }

        insufficient = classify_gate_state(
            training_buffer_size=1,
            episodes_seen=100,
            explained_variance=1.0,
            mix_coef=0.1,
            **common,
        )
        warmup = classify_gate_state(
            training_buffer_size=2,
            episodes_seen=9,
            explained_variance=1.0,
            mix_coef=0.0,
            **common,
        )
        ev_blocked = classify_gate_state(
            training_buffer_size=2,
            episodes_seen=10,
            explained_variance=0.1,
            mix_coef=0.0,
            **common,
        )
        active = classify_gate_state(
            training_buffer_size=2,
            episodes_seen=11,
            explained_variance=0.3,
            mix_coef=0.05,
            **common,
        )

        self.assertEqual(
            (insufficient["phase"], insufficient["reason"]),
            ("insufficient_training_buffer", "train_buffer_below_minimum"),
        )
        self.assertEqual(
            (warmup["phase"], warmup["reason"]),
            ("warmup", "episode_warmup"),
        )
        self.assertEqual(
            (ev_blocked["phase"], ev_blocked["reason"]),
            ("ev_blocked", "explained_variance_below_threshold"),
        )
        self.assertEqual((active["phase"], active["reason"]), ("active", "credit_mix_active"))
        self.assertTrue(active["active"])
        self.assertEqual(active["negative_streak"], 3)
        self.assertFalse(active["disabled"])

    def test_nonfinite_ev_reports_the_actual_positive_mix_as_active(self):
        assigner = _fake_assigner(last_loss=0.75)
        assigner.last_explained_variance = np.nan
        assigner.last_mix_coef = 0.05
        record = diagnostics.build_rollout_record(
            assigner,
            update=1,
            episode=4,
            global_step=96,
        )
        state = record["gate"]

        self.assertIsNone(record["explained_variance"])
        self.assertEqual(record["mix_coef"], 0.05)
        self.assertEqual(state["phase"], "active")
        self.assertTrue(state["active"])
        self.assertEqual(state["reason"], "credit_mix_active_with_invalid_ev")

    def test_nonfinite_ev_with_zero_or_nonfinite_mix_is_explicitly_inactive(self):
        for mix_coef in (0.0, np.inf):
            with self.subTest(mix_coef=mix_coef):
                assigner = _fake_assigner(last_loss=0.75)
                assigner.last_explained_variance = np.nan
                assigner.last_mix_coef = mix_coef
                record = diagnostics.build_rollout_record(
                    assigner,
                    update=1,
                    episode=4,
                    global_step=96,
                )
                state = record["gate"]
                self.assertIsNone(record["explained_variance"])
                if np.isfinite(mix_coef):
                    self.assertEqual(record["mix_coef"], 0.0)
                else:
                    self.assertIsNone(record["mix_coef"])
                self.assertEqual(state["phase"], "invalid_ev")
                self.assertFalse(state["active"])
                self.assertEqual(state["reason"], "nonfinite_explained_variance")

    def test_finite_ev_at_threshold_keeps_existing_ramp_boundary(self):
        state = diagnostics.classify_gate_state(
            training_buffer_size=2,
            minimum_training_buffer=2,
            episodes_seen=10,
            warmup_episodes=10,
            explained_variance=0.2,
            explained_variance_threshold=0.2,
            mix_coef=0.0,
        )

        self.assertEqual((state["phase"], state["reason"]), ("ramp", "ramp_mix_zero"))
        self.assertFalse(state["active"])


def _fake_assigner(last_loss=np.nan):
    gamma = 0.99
    buffer = EpisodeCreditBuffer(capacity=8)
    holdout_buffer = EpisodeCreditBuffer(capacity=8)
    rewards = np.array([[1.0, -0.5], [0.25, 0.75]], dtype=np.float32)
    target = diagnostics.discounted_team_return(rewards[None, ...], gamma)[0]
    buffer.add(
        np.zeros((2, 2, 1), dtype=np.float32),
        np.zeros((2, 2, 1), dtype=np.float32),
        rewards,
        np.zeros((2, 2), dtype=np.float32),
        target,
    )
    return SimpleNamespace(
        config=SimpleNamespace(
            gamma=gamma,
            batch_size=1,
            buffer_size=8,
            warmup_episodes=0,
            explained_variance_threshold=0.2,
            mix_coef=0.1,
        ),
        buffer=buffer,
        holdout_buffer=holdout_buffer,
        episodes_seen=4,
        last_loss=last_loss,
        last_explained_variance=0.4,
        last_mix_coef=0.1,
        last_conservation_error=1e-6,
        gate=SimpleNamespace(negative_streak=0, disabled=False),
    )


class JsonlDiagnosticsTest(unittest.TestCase):
    def test_two_completed_rollouts_append_two_monotonic_records(self):
        write_diagnostic = getattr(diagnostics, "write_rollout_diagnostic", None)
        self.assertTrue(
            callable(write_diagnostic), "training diagnostics need a JSONL writer"
        )

        required = {
            "schema_version",
            "update",
            "episode",
            "global_step",
            "reward_model_loss",
            "explained_variance",
            "mix_coef",
            "target_error",
            "conservation_error",
            "gate",
        }
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "nested" / "rollouts.jsonl"
            assigner = _fake_assigner(last_loss=0.75)
            for update in (1, 2):
                write_diagnostic(
                    path,
                    assigner,
                    update=update,
                    episode=update * 4,
                    global_step=update * 96,
                )

            lines = path.read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(lines), 2)
            records = [json.loads(line) for line in lines]
            self.assertTrue(all(required <= set(record) for record in records))
            self.assertEqual([record["update"] for record in records], [1, 2])
            self.assertEqual([record["episode"] for record in records], [4, 8])
            self.assertEqual([record["global_step"] for record in records], [96, 192])
            for record in records:
                self.assertTrue(
                    {"phase", "active", "reason", "negative_streak", "disabled"}
                    <= set(record["gate"])
                )

    def test_null_path_is_a_no_op(self):
        append_record = getattr(diagnostics, "append_rollout_record", None)
        self.assertTrue(callable(append_record), "JSON diagnostics need a JSONL writer")
        with tempfile.TemporaryDirectory() as temporary_directory:
            result = append_record(None, {"schema_version": 1})
            self.assertFalse(result)
            self.assertEqual(list(Path(temporary_directory).iterdir()), [])

    def test_null_path_skips_record_construction(self):
        write_diagnostic = getattr(diagnostics, "write_rollout_diagnostic", None)
        self.assertTrue(
            callable(write_diagnostic),
            "null diagnostics must bypass record construction entirely",
        )
        self.assertFalse(
            write_diagnostic(
                None,
                object(),
                update=1,
                episode=4,
                global_step=96,
            )
        )

    def test_warmup_loss_is_json_null(self):
        build_record = getattr(diagnostics, "build_rollout_record", None)
        self.assertTrue(callable(build_record), "JSON diagnostics need a record builder")
        record = build_record(
            _fake_assigner(last_loss=np.nan),
            update=1,
            episode=4,
            global_step=96,
        )
        self.assertIsNone(record["reward_model_loss"])

    def test_nonfinite_diagnostic_scalars_are_json_null(self):
        cases = (
            ("last_loss", np.inf, "reward_model_loss"),
            ("last_explained_variance", np.nan, "explained_variance"),
            ("last_mix_coef", np.inf, "mix_coef"),
            ("last_conservation_error", -np.inf, "conservation_error"),
        )

        for attribute, value, field in cases:
            with self.subTest(field=field):
                assigner = _fake_assigner(last_loss=0.75)
                setattr(assigner, attribute, value)
                record = diagnostics.build_rollout_record(
                    assigner,
                    update=1,
                    episode=4,
                    global_step=96,
                )
                self.assertIsNone(record[field])

    def test_nan_buffer_reward_makes_target_error_json_null(self):
        assigner = _fake_assigner(last_loss=0.75)
        stored = assigner.buffer.storage[0]
        corrupted_rewards = stored[2].copy()
        corrupted_rewards[0, 0] = np.nan
        assigner.buffer.storage[0] = (
            *stored[:2],
            corrupted_rewards,
            *stored[3:],
        )

        record = diagnostics.build_rollout_record(
            assigner,
            update=1,
            episode=4,
            global_step=96,
        )

        self.assertIsNone(record["target_error"])

    def test_infinite_buffer_target_makes_target_error_json_null(self):
        assigner = _fake_assigner(last_loss=0.75)
        stored = assigner.buffer.storage[0]
        assigner.buffer.storage[0] = (*stored[:4], np.inf)

        record = diagnostics.build_rollout_record(
            assigner,
            update=1,
            episode=4,
            global_step=96,
        )

        self.assertIsNone(record["target_error"])

    def test_nonfinite_record_is_rejected_before_open_without_partial_line(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "rollouts.jsonl"
            original = b'{"schema_version": 1}\n'
            path.write_bytes(original)

            with mock.patch.object(
                Path,
                "open",
                side_effect=AssertionError("invalid JSON must be rejected before open"),
            ) as open_mock:
                with self.assertRaises(ValueError):
                    diagnostics.append_rollout_record(path, {"bad": np.inf})

            open_mock.assert_not_called()
            self.assertEqual(path.read_bytes(), original)


class ConservationErrorHelperTest(unittest.TestCase):
    def test_reports_actual_discounted_return_difference(self):
        compute_conservation_error = getattr(
            diagnostics, "compute_conservation_error", None
        )
        self.assertTrue(
            callable(compute_conservation_error),
            "diagnostics must compare actual MAPPO rewards with original rewards",
        )
        original = np.array(
            [[[1.0, -0.5, 0.25], [0.1, 0.2, -0.3]]], dtype=np.float32
        )
        self.assertEqual(compute_conservation_error(original, original, 0.99), 0.0)
        changed = original.copy()
        changed[0, 0, 0] += 2.0
        self.assertGreater(compute_conservation_error(changed, original, 0.99), 1.99)


def _small_conserved_config(**overrides):
    values = {
        "obs_dim": 1,
        "action_dim": 1,
        "n_agents": 2,
        "seq_length": 3,
        "gamma": 0.99,
        "emb_dim": 4,
        "n_heads": 1,
        "n_layers": 1,
        "sample_num": 1,
        "dropout": 0.0,
        "buffer_size": 4,
        "batch_size": 1,
        "warmup_rollouts": 999,
        "conserve_discounted": True,
        "warmup_episodes": 0,
        "ramp_episodes": 1,
        "max_mix_coef": 0.3,
        "explained_variance_threshold": 0.2,
        "device": "cpu",
    }
    values.update(overrides)
    return STASCreditConfig(**values)


def _small_rollout():
    rng = np.random.default_rng(17)
    return (
        rng.normal(size=(1, 2, 3, 1)).astype(np.float32),
        rng.normal(size=(1, 2, 3, 1)).astype(np.float32),
        rng.normal(size=(1, 2, 3)).astype(np.float32),
        np.zeros((1, 2, 3), dtype=np.float32),
    )


class CurrentRolloutConservationTest(unittest.TestCase):
    def test_mix_zero_recomputes_instead_of_retaining_stale_error(self):
        assigner = ConservedSTASCreditAssigner(_small_conserved_config())
        assigner.last_conservation_error = 123.0
        obs, actions, rewards, dones = _small_rollout()

        training_rewards, _ = assigner.process_rollout(
            obs, actions, rewards, dones
        )

        self.assertEqual(assigner.last_mix_coef, 0.0)
        np.testing.assert_array_equal(training_rewards, rewards)
        self.assertEqual(assigner.last_conservation_error, 0.0)

    def test_positive_mix_reports_actual_blended_reward_conservation(self):
        assigner = ConservedSTASCreditAssigner(
            _small_conserved_config(explained_variance_threshold=-1.0)
        )
        obs, actions, rewards, dones = _small_rollout()

        training_rewards, _ = assigner.process_rollout(
            obs, actions, rewards, dones
        )

        actual_error = diagnostics.compute_conservation_error(
            training_rewards, rewards, assigner.config.gamma
        )
        self.assertGreater(assigner.last_mix_coef, 0.0)
        self.assertLessEqual(actual_error, 1e-4)
        self.assertAlmostEqual(
            assigner.last_conservation_error, actual_error, delta=1e-12
        )


class TrainingEntryDiagnosticsWiringTest(unittest.TestCase):
    def test_config_defaults_diagnostics_path_to_null(self):
        config_path = STAS_ROOT / "config" / "stas_mappo_microgrid.yaml"
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        self.assertIn(
            "DIAGNOSTICS_PATH",
            config["STAS"],
            "STAS config must expose an opt-in diagnostics path",
        )
        self.assertIsNone(config["STAS"]["DIAGNOSTICS_PATH"])

    def test_training_loop_writes_once_after_global_step_with_one_based_update(self):
        source = inspect.getsource(mappo_stas.make_train)
        call = "write_rollout_diagnostic("
        self.assertEqual(
            source.count(call),
            1,
            "the training loop must append exactly one diagnostic per update",
        )
        writer_position = source.index(call)
        self.assertGreater(writer_position, source.index("global_step +="))
        call_source = source[writer_position : writer_position + 500]
        self.assertIn("update=update + 1", call_source)
        self.assertIn(
            'episode=global_step // config["NUM_STEPS"]',
            call_source,
        )
        self.assertIn("global_step=global_step", call_source)


if __name__ == "__main__":
    unittest.main()
