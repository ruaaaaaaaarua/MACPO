import json
import tempfile
import unittest
from pathlib import Path

from scripts import run_env_v3_safe_matrix as runner
from scripts.run_env_v3_safe_matrix import run


class EnvV3SafeMatrixResumeTest(unittest.TestCase):
    def test_resume_truncates_metrics_after_checkpoint_before_appending(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "metrics.jsonl"
            path.write_text(
                "".join(
                    json.dumps({"update": update, "value": update * 10}) + "\n"
                    for update in (1, 2, 3)
                ),
                encoding="utf-8",
            )

            removed = runner.reconcile_metrics_for_resume(path, checkpoint_update=2)

            rows = [json.loads(line) for line in path.read_text().splitlines()]
            self.assertEqual([row["update"] for row in rows], [1, 2])
            self.assertEqual(removed, 1)

    def test_gru_dry_run_describes_checkpoint_and_metrics_locations(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            result = run(
                "dense_gru_mappo_anneal",
                updates=500,
                dry_run=True,
                run_dir=Path(temp_dir),
                checkpoint_interval=25,
            )

        self.assertEqual(result["checkpoint_interval"], 25)
        self.assertEqual(result["checkpoint_dir"], str(Path(temp_dir) / "checkpoints" / "dense_gru_mappo_anneal"))
        self.assertEqual(result["metrics_path"], str(Path(temp_dir) / "dense_gru_mappo_anneal.metrics.jsonl"))
        self.assertEqual(result["target_updates"], 500)
        self.assertEqual(result["validation_interval"], 100)


if __name__ == "__main__":
    unittest.main()
