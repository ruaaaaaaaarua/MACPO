"""Short-horizon smoke of the corrected MACPO updater. Not a training claim.

Runs N updates from scratch with the v6 MACPO config (serial envs so it stays
light), printing per-update mode / cost / geometry so the branch behaviour of the
fixed CPO logic is visible. Also runs 2 MAPPO updates to confirm the untouched
branch still works.
"""

from __future__ import annotations

import json
import sys

sys.path.insert(0, "/root/autodl-tmp/env_v6_1_dev/HyperMARL-main")

from baselines.MAPPO.safe_gru_trainer import SafeGRUMAPPOTrainer
from scripts.run_env_v3_safe_matrix import apply_env_v6_calibration, build_gru_config

UPDATES = int(sys.argv[1]) if len(sys.argv) > 1 else 30

cal = json.load(open("/root/autodl-tmp/env_v6_swiss_runs/calibration.json"))

config = build_gru_config("v6_nocomm_gru_macpo", updates=1000)
apply_env_v6_calibration(config, cal)
config["num_envs"] = 4
config["env_parallel_backend"] = "serial"
config["fused_rollout_kernel"] = True
trainer = SafeGRUMAPPOTrainer(config)
modes = {}
try:
    for update in range(1, UPDATES + 1):
        rollout = trainer.collect_rollout(update_index=update)
        metrics = trainer.update(rollout, algorithm="macpo", update_index=update)
        mode = str(metrics.get("mode"))
        modes[mode] = modes.get(mode, 0) + 1
        print(
            f"u{update:03d} mode={mode:<18} accepted={metrics.get('accepted')} "
            f"vcost={metrics['daily_voltage_cost_normalized']:.2f} "
            f"gap={metrics.get('cost_gap', float('nan')):.2f} "
            f"maxdrop={metrics.get('max_linear_cost_drop', float('nan')):.2f} "
            f"kl={metrics.get('kl_after', float('nan')):.4f} "
            f"econ={metrics['daily_economic_cost_raw_yuan']:.3g}",
            flush=True,
        )
finally:
    trainer.close()
print("mode counts:", modes)

config2 = build_gru_config("v6_nocomm_gru_mappo", updates=1000)
apply_env_v6_calibration(config2, cal)
config2["num_envs"] = 2
config2["env_parallel_backend"] = "serial"
config2["fused_rollout_kernel"] = False
trainer2 = SafeGRUMAPPOTrainer(config2)
try:
    for update in range(1, 3):
        rollout = trainer2.collect_rollout(update_index=update)
        metrics = trainer2.update(rollout, algorithm="mappo", update_index=update)
        print(f"mappo u{update} actor_loss={metrics['actor_loss']:.4f} ok")
finally:
    trainer2.close()
print("SMOKE DONE")
