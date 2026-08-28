import json
import tempfile
import unittest
from pathlib import Path

import jax
import numpy as np

from baselines.MAPPO.safe_gru_trainer import SafeGRUMAPPOTrainer


class SafeTrainingPersistenceTest(unittest.TestCase):
    @staticmethod
    def _config():
        return {
            "seed": 30,
            "num_envs": 1,
            "num_steps": 4,
            "hidden_size": 16,
            "total_updates": 4,
            "env_overrides": {
                "profile_source": "synthetic",
                "italian_split_enable": False,
                "episode_length": 4,
                "reward_emission_mode": "dense",
                "power_flow_enable": True,
                "terminal_economic_settlement_enable": False,
                "penalty_enable": False,
                "low_inventory_penalty_enable": False,
                "terminal_h2_floor_penalty_enable": False,
                "terminal_h2_shortfall_value_enable": False,
                "terminal_h2_settlement_in_reward_enable": False,
                "terminal_soc_floor_penalty_enable": False,
                "terminal_battery_salvage_enable": False,
                "stepwise_h2_floor_penalty_enable": False,
                "action_reg_enable": False,
            },
        }

    def test_checkpoint_restores_states_rng_and_lagrange_at_episode_boundary(self):
        trainer = SafeGRUMAPPOTrainer(self._config())
        try:
            trainer.update(trainer.collect_rollout(), algorithm="lagrangian")
            with tempfile.TemporaryDirectory() as temp_dir:
                checkpoint = Path(temp_dir) / "update_000001.msgpack"
                trainer.save_checkpoint(checkpoint, update=1, algorithm="lagrangian")
                restored = SafeGRUMAPPOTrainer(self._config())
                try:
                    self.assertEqual(
                        restored.load_checkpoint(checkpoint, algorithm="lagrangian"), 1
                    )
                    np.testing.assert_equal(
                        np.asarray(trainer._rng), np.asarray(restored._rng)
                    )
                    self.assertAlmostEqual(
                        trainer.lagrange_multiplier, restored.lagrange_multiplier
                    )
                    for left, right in zip(
                        jax.tree_util.tree_leaves(trainer.actor_state.params),
                        jax.tree_util.tree_leaves(restored.actor_state.params),
                    ):
                        np.testing.assert_allclose(left, right)
                finally:
                    restored.close()
        finally:
            trainer.close()

    def test_checkpoint_rejects_a_different_config_fingerprint(self):
        trainer = SafeGRUMAPPOTrainer(self._config())
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                checkpoint = Path(temp_dir) / "update_000000.msgpack"
                trainer.save_checkpoint(checkpoint, update=0, algorithm="mappo")
                incompatible = self._config()
                incompatible["lr"] = 1e-4
                restored = SafeGRUMAPPOTrainer(incompatible)
                try:
                    with self.assertRaisesRegex(ValueError, "configuration fingerprint"):
                        restored.load_checkpoint(checkpoint, algorithm="mappo")
                finally:
                    restored.close()
        finally:
            trainer.close()

    def test_train_appends_metrics_and_periodic_checkpoints(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            trainer = SafeGRUMAPPOTrainer(self._config())
            try:
                rows = trainer.train(
                    2,
                    algorithm="mappo",
                    checkpoint_dir=Path(temp_dir) / "checkpoints",
                    checkpoint_interval=1,
                    metrics_path=Path(temp_dir) / "metrics.jsonl",
                )
            finally:
                trainer.close()

            self.assertEqual([row["update"] for row in rows], [1, 2])
            metrics_rows = [
                json.loads(line)
                for line in (Path(temp_dir) / "metrics.jsonl").read_text().splitlines()
            ]
            self.assertEqual([row["update"] for row in metrics_rows], [1, 2])
            self.assertTrue((Path(temp_dir) / "checkpoints" / "update_000001.msgpack").is_file())
            self.assertTrue((Path(temp_dir) / "checkpoints" / "update_000002.msgpack").is_file())


if __name__ == "__main__":
    unittest.main()
