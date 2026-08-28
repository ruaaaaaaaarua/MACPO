import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
STAS_DIR = ROOT / "baselines" / "STAS-MAPPO"
for path in (str(ROOT), str(STAS_DIR)):
    if path not in sys.path:
        sys.path.insert(0, path)

from stas_mappo.credit import STASCreditConfig  # noqa: E402
from stas_mappo.conserved_credit import UniformCreditAssigner  # noqa: E402
from stas_mappo.credit_conservation import discounted_team_return  # noqa: E402


def _config(**overrides):
    kwargs = {
        "obs_dim": 4,
        "action_dim": 2,
        "n_agents": 3,
        "seq_length": 8,
        "gamma": 1.0,
        "mix_coef": 0.0,
        "device": "cpu",
        "buffer_size": 4,
        "batch_size": 2,
        "conserve_discounted": True,
        "quality_gate_enable": True,
        "warmup_episodes": 0,
        "ramp_episodes": 1,
        "max_mix_coef": 0.5,
        "mode": "uniform",
    }
    kwargs.update(overrides)
    return STASCreditConfig(**kwargs)


class UniformCreditAssignerTest(unittest.TestCase):
    """设计规格 Phase 2: IRCR 照妖镜基线的守恒性与平坦性。"""

    def test_credit_is_flat_and_conserves_team_return(self):
        assigner = UniformCreditAssigner(_config())
        rng = np.random.RandomState(0)
        rewards = np.zeros((2, 3, 8), dtype=np.float32)
        rewards[:, :, -1] = rng.uniform(-5.0, -1.0, size=(2, 3))  # 终端稀疏
        obs = rng.randn(2, 3, 8, 4).astype(np.float32)
        actions = rng.randn(2, 3, 8, 2).astype(np.float32)
        dones = np.zeros((2, 3, 8), dtype=np.float32)
        credit = assigner.credit_rewards(obs, actions, rewards, dones)
        self.assertEqual(credit.shape, rewards.shape)
        # 每个 env 内 credit 平坦 (gamma=1)。
        for env_id in range(2):
            values = credit[env_id].reshape(-1)
            self.assertAlmostEqual(float(values.std()), 0.0, places=6)
        # 守恒: 折扣团队和 == 原始折扣团队回报。
        targets = discounted_team_return(rewards.astype(np.float64), 1.0)
        redistributed = discounted_team_return(credit, 1.0)
        np.testing.assert_allclose(redistributed, targets, rtol=0, atol=1e-6)
        self.assertLess(assigner.last_conservation_error, 1e-6)

    def test_ev_reports_one_and_no_model_training(self):
        assigner = UniformCreditAssigner(_config())
        self.assertEqual(assigner.holdout_explained_variance(), 1.0)
        self.assertEqual(assigner.train_if_ready(), 0.0)


if __name__ == "__main__":
    unittest.main()
