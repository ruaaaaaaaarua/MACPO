"""Small, bounded cross-agent safety reference for Env-v6 rollouts.

This module is deliberately outside the training update.  It changes only the
action sent to the environment during a diagnostic rollout.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def adjust_action(
    action: np.ndarray,
    diagnostics: dict[str, Any] | None = None,
    *,
    soc: np.ndarray | None = None,
    pcc_power: np.ndarray | None = None,
    bat_power: np.ndarray | None = None,
    el_cap: np.ndarray | None = None,
    h2_level: np.ndarray | None = None,
    pending_h2: np.ndarray | None = None,
) -> np.ndarray:
    """Apply a bounded safety coordination adjustment to normalized actions.

    Action semantics are Env-v6's normalized vector: ``a0`` electrolyzer,
    ``a1`` battery (positive charge, negative discharge), and ``a5`` planned
    H2 order.  The reference never writes inventory or pending state directly.
    """
    adjusted = np.asarray(action, dtype=np.float32).copy()
    if adjusted.ndim != 2 or adjusted.shape[1] < 6:
        raise ValueError(f"expected (agents, >=6) action array, got {adjusted.shape}")
    diagnostics = diagnostics or {}
    vmin = diagnostics.get("voltage_min_pu", 1.0)
    vmin = 1.0 if vmin is None or not np.isfinite(vmin) else float(vmin)
    agents = adjusted.shape[0]
    soc_arr = np.full(agents, 0.5, dtype=np.float32) if soc is None else np.asarray(soc, dtype=np.float32)
    pcc_arr = np.zeros(agents, dtype=np.float32) if pcc_power is None else np.asarray(pcc_power, dtype=np.float32)
    bat_scale = np.ones(agents, dtype=np.float32) if bat_power is None else np.maximum(np.asarray(bat_power, dtype=np.float32), 1e-6)
    el_scale = np.ones(agents, dtype=np.float32) if el_cap is None else np.maximum(np.asarray(el_cap, dtype=np.float32), 1e-6)
    h2_arr = np.full(agents, 200.0, dtype=np.float32) if h2_level is None else np.asarray(h2_level, dtype=np.float32)
    pending_arr = np.zeros(agents, dtype=np.float32) if pending_h2 is None else np.asarray(pending_h2, dtype=np.float32)
    for name, value in (("soc", soc_arr), ("pcc_power", pcc_arr), ("bat_power", bat_scale), ("el_cap", el_scale), ("h2_level", h2_arr), ("pending_h2", pending_arr)):
        if value.shape != (agents,):
            raise ValueError(f"{name} must have shape {(agents,)}, got {value.shape}")

    if vmin >= 0.97:
        return np.clip(adjusted, -1.0, 1.0)

    # The risk ramps from zero at 0.97 to one at 0.95.  This keeps the
    # pre-violation reference close to the learned action.
    risk = float(np.clip((0.97 - vmin) / 0.02, 0.0, 1.0))
    imports = np.maximum(pcc_arr, 0.0)
    import_scale = max(float(np.max(imports)), 1.0)
    import_ratio = imports / import_scale

    # First remove charging and shave electrolyzer demand on high-import PCCs.
    charging = np.maximum(adjusted[:, 1], 0.0)
    adjusted[:, 1] = adjusted[:, 1] - charging * (0.55 * risk)
    adjusted[:, 0] -= (0.12 * risk * import_ratio).astype(np.float32)

    if vmin < 0.95:
        # Once outside the limit, cancel all charging and select high-import,
        # high-SOC agents for a bounded discharge.  Low-SOC agents are spared.
        adjusted[:, 1] = np.minimum(adjusted[:, 1], 0.0)
        available_soc = np.clip((soc_arr - 0.15) / 0.70, 0.0, 1.0)
        score = (imports / import_scale) * available_soc
        if float(np.max(score)) <= 1e-6:
            score = available_soc
        score_scale = max(float(np.max(score)), 1e-6)
        requested_discharge = -np.clip(0.10 + 0.28 * risk * score / score_scale, 0.0, 0.38)
        adjusted[:, 1] = np.minimum(adjusted[:, 1], requested_discharge).astype(np.float32)
        adjusted[:, 0] -= (0.10 * risk * import_ratio).astype(np.float32)

    # This probe isolates electrical coordination. Keep H2 order actions
    # unchanged so any safety improvement is attributable to PCC/battery/
    # electrolyzer coordination rather than a changed transport schedule.
    return np.clip(adjusted, -1.0, 1.0).astype(np.float32)
