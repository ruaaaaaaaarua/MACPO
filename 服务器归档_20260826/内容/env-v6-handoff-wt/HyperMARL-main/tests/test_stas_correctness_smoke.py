import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from scripts import run_stas_correctness_smoke as smoke
except ImportError:
    smoke = None


class CorrectnessSmokeScriptPresenceTest(unittest.TestCase):
    def test_smoke_script_exists(self):
        self.assertIsNotNone(smoke, "run_stas_correctness_smoke.py is missing")


@unittest.skipIf(smoke is None, "smoke script has not been implemented")
class CorrectnessSmokeBehaviorTest(unittest.TestCase):
    def test_two_phase_commands_lock_settings_and_restore_both_states(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            specs = smoke.build_phase_specs(Path(tmpdir))

        self.assertEqual(set(specs), {"phase1", "phase2"})
        phase1 = " ".join(specs["phase1"].command)
        phase2 = " ".join(specs["phase2"].command)
        for command in (phase1, phase2):
            self.assertIn("SEED=30", command)
            self.assertIn("NUM_ENVS=4", command)
            self.assertIn("NUM_STEPS=24", command)
            self.assertIn("ACTOR_LAYERS=[256,256]", command)
            self.assertIn("CRITIC_LAYERS=[256,256]", command)
            self.assertIn("ACTIVATION=relu", command)
            self.assertIn("+POLICY_MODE=squashed_gaussian", command)
            self.assertIn("+LOG_STD_MIN=-2.5", command)
            self.assertIn("+LOG_STD_MAX=-0.5", command)
            self.assertIn("LOG_STD_INIT=-1.0", command)
            self.assertIn("ENT_COEF=0", command)
            self.assertIn("+STAS.CONSERVE_DISCOUNTED=true", command)
            self.assertIn("+STAS.BIDIRECTIONAL=false", command)
            self.assertIn("+STAS.WARMUP_EPISODES=200", command)
            self.assertIn("+STAS.RAMP_EPISODES=800", command)
            self.assertIn("+STAS.MAX_MIX_COEF=0.1", command)
            self.assertIn("+STAS.EXPLAINED_VARIANCE_THRESHOLD=0.2", command)
            self.assertIn("external_h2_dependency_penalty_enable:true", command)
            self.assertIn("h2_learnable_rolling_order_enable:true", command)
            self.assertIn("italian_split_name:train", command)
            self.assertNotIn("italian_split_name:test", command)
            self.assertNotIn("italian_day_indices", command)
        self.assertIn("TOTAL_TIMESTEPS=1440", phase1)
        self.assertIn("TOTAL_TIMESTEPS=2400", phase2)
        self.assertIn("+TRAINING_CHECKPOINT_LOAD_PATH=", phase2)
        self.assertIn("+STAS.CHECKPOINT_LOAD_PATH=", phase2)
        self.assertNotEqual(specs["phase1"].diagnostics, specs["phase2"].diagnostics)

    def _diagnostic(self, update):
        return {
            "schema_version": 1,
            "update": update,
            "episode": update * 4,
            "global_step": update * 96,
            "reward_model_loss": None if update < 5 else 0.125,
            "explained_variance": 0.0,
            "mix_coef": 0.0,
            "target_error": 0.0,
            "conservation_error": 0.0,
            "gate": {"phase": "warmup", "active": False, "reason": "episode_warmup"},
        }

    def _write_jsonl(self, path, records):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            "".join(json.dumps(record, allow_nan=False) + "\n" for record in records),
            encoding="utf-8",
        )

    def _materialize_valid_artifacts(self, specs):
        phase1 = specs["phase1"]
        phase2 = specs["phase2"]
        for spec in specs.values():
            spec.output_dir.mkdir(parents=True, exist_ok=True)
            spec.log.parent.mkdir(parents=True, exist_ok=True)
            spec.jax_checkpoint.parent.mkdir(parents=True, exist_ok=True)
            spec.log.write_text("phase complete\n", encoding="utf-8")
            spec.jax_checkpoint.write_bytes(b"jax-state")

        phase2.log.write_text(
            "Loaded full training checkpoint from phase1 (episode=60, update=15)\n"
            "Loaded full STAS state from phase1 (global_step=1440, rollouts_seen=15)\n",
            encoding="utf-8",
        )
        self._write_jsonl(
            phase1.diagnostics, [self._diagnostic(update) for update in range(1, 16)]
        )
        self._write_jsonl(
            phase2.diagnostics, [self._diagnostic(update) for update in range(16, 26)]
        )

        rewards = np.arange(6, dtype=np.float32).reshape(2, 3) / 10.0
        weights = np.power(0.99, np.arange(3, dtype=np.float64))
        target = float(np.sum(rewards * weights[None, :]))
        item = (
            np.zeros((2, 3, 4), dtype=np.float32),
            np.zeros((2, 3, 1), dtype=np.float32),
            rewards,
            np.zeros((2, 3), dtype=np.float32),
            target,
        )
        phase1_state = {
            "update": 15,
            "episode": 60,
            "global_step": 1440,
            "rollouts_seen": 15,
            "episodes_seen": 60,
            "normalizer": {"count": 60, "mean": 0.0, "m2": 1.0},
            "config": {"gamma": 0.99},
            "buffer_storage": [item],
            "holdout_buffer_storage": [],
            "last_conservation_error": 0.0,
        }
        phase2_state = {
            **phase1_state,
            "update": 25,
            "episode": 100,
            "global_step": 2400,
            "rollouts_seen": 25,
            "episodes_seen": 100,
            "normalizer": {"count": 100, "mean": 0.0, "m2": 1.0},
        }
        torch.save(phase1_state, phase1.stas_checkpoint)
        torch.save(phase2_state, phase2.stas_checkpoint)

        episodes = []
        for day in (8, 17, 21, 23):
            for seed in (4200,):
                episodes.append(
                    {
                        "day": day,
                        "seed": seed,
                        "steps": 24,
                        "return": -1.0,
                        "base_cost": 100.0,
                    }
                )
        self._write_jsonl(
            phase2.validation,
            [{"schema_version": 1, "training_episode": 100, "episodes": episodes}],
        )

    def test_artifact_audit_accepts_exact_resume_sequences_and_validation(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            specs = smoke.build_phase_specs(Path(tmpdir))
            self._materialize_valid_artifacts(specs)

            result = smoke.audit_smoke_artifacts(
                specs,
                exit_codes={"phase1": 0, "phase2": 0},
                source_commit="abc123",
                invocation=["python", "scripts/run_stas_correctness_smoke.py"],
            )

        self.assertTrue(result["passed"])
        self.assertTrue(all(result["assertions"].values()))
        self.assertEqual(result["schema_version"], 1)
        self.assertEqual(result["source_commit"], "abc123")
        self.assertEqual(result["diagnostic_counts"], {"phase1": 15, "phase2": 10})
        self.assertIn(
            "final_validation_contains_no_test_scenarios", result["assertions"]
        )
        self.assertNotIn("no_test_scenario_accessed", result["assertions"])
        self.assertGreaterEqual(len(result["artifacts"]), 9)
        self.assertTrue(
            all(len(metadata["sha256"]) == 64 for metadata in result["artifacts"].values())
        )

    def test_artifact_audit_rejects_active_mix_during_warmup(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            specs = smoke.build_phase_specs(Path(tmpdir))
            self._materialize_valid_artifacts(specs)
            records = [self._diagnostic(update) for update in range(16, 26)]
            records[-1]["mix_coef"] = 0.01
            self._write_jsonl(specs["phase2"].diagnostics, records)

            result = smoke.audit_smoke_artifacts(
                specs,
                exit_codes={"phase1": 0, "phase2": 0},
                source_commit="abc123",
                invocation=["python", "scripts/run_stas_correctness_smoke.py"],
            )

        self.assertFalse(result["passed"])
        self.assertFalse(result["assertions"]["warmup_mix_zero_and_gate_inactive"])

    def test_artifact_audit_rejects_vacuous_target_check_without_buffers(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            specs = smoke.build_phase_specs(Path(tmpdir))
            self._materialize_valid_artifacts(specs)
            state = torch.load(
                specs["phase2"].stas_checkpoint,
                map_location="cpu",
                weights_only=False,
            )
            state["buffer_storage"] = []
            state["holdout_buffer_storage"] = []
            torch.save(state, specs["phase2"].stas_checkpoint)

            result = smoke.audit_smoke_artifacts(
                specs,
                exit_codes={"phase1": 0, "phase2": 0},
                source_commit="abc123",
                invocation=["python", "scripts/run_stas_correctness_smoke.py"],
            )

        self.assertFalse(result["passed"])
        self.assertFalse(result["assertions"].get("checkpoint_buffers_nonempty", True))

    def _audit_payload(self, exit_codes):
        return {
            "schema_version": 1,
            "source_commit": "abc123",
            "command": ["python", "scripts/run_stas_correctness_smoke.py"],
            "commands": {},
            "resolved_config": {},
            "exit_codes": dict(exit_codes),
            "artifacts": {},
            "assertions": {"subprocesses_exit_zero": all(
                code == 0 for code in exit_codes.values()
            )},
            "errors": [],
            "passed": all(code == 0 for code in exit_codes.values()),
        }

    def test_main_runs_phase1_then_phase2_and_writes_gate_results(self):
        order = []

        def fake_run_phase(spec):
            order.append(spec.name)
            return 0

        def fake_audit(_specs, *, exit_codes, **_kwargs):
            return self._audit_payload(exit_codes)

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "smoke"
            with (
                mock.patch.object(smoke, "_run_phase", side_effect=fake_run_phase),
                mock.patch.object(
                    smoke, "audit_smoke_artifacts", side_effect=fake_audit
                ),
                mock.patch.object(smoke, "_source_commit", return_value="abc123"),
            ):
                exit_code = smoke.main(["--root", str(root)])
            payload = json.loads(
                (root / "gate_results.json").read_text(encoding="utf-8")
            )

        self.assertEqual(exit_code, 0)
        self.assertEqual(order, ["phase1", "phase2"])
        self.assertEqual(payload["exit_codes"], {"phase1": 0, "phase2": 0})
        self.assertTrue(payload["passed"])

    def test_main_skips_phase2_after_phase1_failure_and_records_exit_codes(self):
        order = []

        def fake_run_phase(spec):
            order.append(spec.name)
            return 7

        def fake_audit(_specs, *, exit_codes, **_kwargs):
            return self._audit_payload(exit_codes)

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "smoke"
            with (
                mock.patch.object(smoke, "_run_phase", side_effect=fake_run_phase),
                mock.patch.object(
                    smoke, "audit_smoke_artifacts", side_effect=fake_audit
                ),
                mock.patch.object(smoke, "_source_commit", return_value="abc123"),
            ):
                exit_code = smoke.main(["--root", str(root)])
            payload = json.loads(
                (root / "gate_results.json").read_text(encoding="utf-8")
            )
            skip_log = (
                root / "phase2_060_100" / "train.log"
            ).read_text(encoding="utf-8")

        self.assertEqual(exit_code, 1)
        self.assertEqual(order, ["phase1"])
        self.assertEqual(payload["exit_codes"], {"phase1": 7, "phase2": -1})
        self.assertIn("phase2 skipped because phase1 failed", skip_log)

    def test_existing_root_rejection_preserves_every_byte_and_writes_nothing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "smoke"
            root.mkdir()
            existing = root / "prior-artifact.bin"
            existing.write_bytes(b"prior-checkpoint-evidence")
            gate = root / "gate_results.json"
            gate.write_bytes(b'{"passed": true, "sentinel": "keep-me"}\n')
            before = {
                str(path.relative_to(root)): path.read_bytes()
                for path in root.rglob("*")
                if path.is_file()
            }

            with mock.patch.object(smoke, "_source_commit", return_value="abc123"):
                exit_code = smoke.main(["--root", str(root)])

            after = {
                str(path.relative_to(root)): path.read_bytes()
                for path in root.rglob("*")
                if path.is_file()
            }

        self.assertEqual(exit_code, 1)
        self.assertEqual(after, before)

    def test_invalid_locked_args_cannot_overwrite_an_existing_root(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "smoke"
            root.mkdir()
            sentinel = root / "prior-artifact.bin"
            sentinel.write_bytes(b"prior-checkpoint-evidence")
            gate = root / "gate_results.json"
            gate.write_bytes(b'{"passed": true, "sentinel": "keep-me"}\n')
            before = {
                str(path.relative_to(root)): path.read_bytes()
                for path in root.rglob("*")
                if path.is_file()
            }

            with mock.patch.object(smoke, "_source_commit", return_value="abc123"):
                exit_code = smoke.main(
                    ["--root", str(root), "--seed", "31"]
                )

            after = {
                str(path.relative_to(root)): path.read_bytes()
                for path in root.rglob("*")
                if path.is_file()
            }

        self.assertEqual(exit_code, 1)
        self.assertEqual(after, before)


if __name__ == "__main__":
    unittest.main()
