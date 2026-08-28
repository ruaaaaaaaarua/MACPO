"""Selection and convergence metrics for fixed-scenario evaluation curves."""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np


def compute_curve_metrics(
    records: Iterable[dict[str, Any]], final_episode: int = 30000
) -> dict[str, Any]:
    by_episode: dict[int, float] = {}
    for record in records:
        episode = int(record["training_episode"])
        if episode <= final_episode:
            by_episode[episode] = float(record["summary"]["return_mean"])
    final_points = [final_episode - 1000, final_episode - 500, final_episode]
    missing = [episode for episode in final_points if episode not in by_episode]
    if missing:
        raise ValueError(f"missing final evaluation points: {missing}")
    episodes = np.asarray(sorted(by_episode), dtype=np.float64)
    values = np.asarray([by_episode[int(episode)] for episode in episodes], dtype=np.float64)
    if episodes.size == 0 or not np.isfinite(values).all():
        raise ValueError("evaluation curve is empty or non-finite")
    if episodes[0] > 0:
        episodes = np.concatenate(([0.0], episodes))
        values = np.concatenate(([values[0]], values))
    if episodes[-1] < final_episode:
        episodes = np.concatenate((episodes, [float(final_episode)]))
        values = np.concatenate((values, [values[-1]]))
    auc = float(np.trapz(values, episodes))
    return {
        "final_points": final_points,
        "final_score": float(np.mean([by_episode[episode] for episode in final_points])),
        "auc": auc,
        "normalized_auc": auc / float(final_episode),
        "num_eval_points": int(len(by_episode)),
    }
