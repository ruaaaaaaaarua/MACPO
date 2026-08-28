"""Episode-level training progress logging for comparison experiments."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


@dataclass
class EpisodeProgressLogger:
    """Log milestone summaries every `report_every` completed episodes."""

    log_path: Path
    report_every: int = 500
    algorithm: str = "unknown"
    episode_returns: List[float] = field(default_factory=list)
    best_mean_reward: float = float("-inf")

    def __post_init__(self) -> None:
        self.log_path = Path(self.log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def record_episode(self, episode_idx: int, episode_return: float) -> None:
        self.episode_returns.append(float(episode_return))
        if episode_idx % self.report_every != 0:
            return
        self._write_report(episode_idx)

    def record_update(
        self,
        update_idx: int,
        update_return: float,
        *,
        episodes_per_update: int,
    ) -> None:
        """For vectorized trainers that report one scalar per PPO update."""
        start_ep = update_idx * episodes_per_update
        for offset in range(episodes_per_update):
            ep = start_ep + offset + 1
            self.episode_returns.append(float(update_return))
            if ep % self.report_every == 0:
                self._write_report(ep)

    def finalize(self) -> None:
        if self.episode_returns:
            self._write_report(len(self.episode_returns))

    def _write_report(self, episode_idx: int) -> None:
        recent = self.episode_returns[-self.report_every :]
        recent_mean = sum(recent) / len(recent)
        cumulative_mean = sum(self.episode_returns) / len(self.episode_returns)
        self.best_mean_reward = max(self.best_mean_reward, recent_mean)
        recent_std = (
            (sum((x - recent_mean) ** 2 for x in recent) / len(recent)) ** 0.5
            if len(recent) > 1
            else 0.0
        )
        oscillation = "noticeable" if recent_std > abs(recent_mean) * 0.15 else "stable"
        payload = {
            "algorithm": self.algorithm,
            "episode": int(episode_idx),
            "recent_window": len(recent),
            "recent_mean_reward": recent_mean,
            "cumulative_mean_reward": cumulative_mean,
            "best_recent_mean_reward": self.best_mean_reward,
            "recent_std": recent_std,
            "stability_note": oscillation,
        }
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
        print(
            f"[progress] {self.algorithm} ep={episode_idx} "
            f"recent{len(recent)}={recent_mean:.2f} "
            f"cumulative={cumulative_mean:.2f} best_recent={self.best_mean_reward:.2f} "
            f"({oscillation})"
        )


def load_progress_jsonl(path: Path) -> List[dict]:
    if not path.exists():
        return []
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


_PROGRESS_LOGGER: Optional[EpisodeProgressLogger] = None


def get_progress_logger(algorithm: str) -> Optional[EpisodeProgressLogger]:
    import os

    global _PROGRESS_LOGGER
    path = os.environ.get("HYPERMARL_PROGRESS_LOG")
    if not path:
        return None
    if _PROGRESS_LOGGER is None:
        _PROGRESS_LOGGER = EpisodeProgressLogger(
            log_path=Path(path),
            report_every=500,
            algorithm=algorithm,
        )
    return _PROGRESS_LOGGER
