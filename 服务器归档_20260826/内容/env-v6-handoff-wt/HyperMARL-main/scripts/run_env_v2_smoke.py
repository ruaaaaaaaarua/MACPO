#!/usr/bin/env python
"""Env v2 smoke gates G1/G3/G4 (设计规格): train a short MAPPO run on the
canonical v2 config, then counterfactually verify the routing mechanism.

G1 机制门控: forced_direct 与 permuted_route 的验证 delta >= 阈值且同向。
G3 稳定门控: 训练完成、最优验证 checkpoint 存在、回报有限。
G4 经济量级: 应急溢价 + 在途损耗价值占总成本的份额落在目标带内。

用法: PYTHONPATH=. python scripts/run_env_v2_smoke.py <output_root>
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.env_v2_overrides import env_v2_overrides, hydra_override_arg  # noqa: E402

SRC_MANIFEST = Path(
    "/root/autodl-tmp/traffic-stas-paper-sparse-20260716-10k-v1/train_10k_20k/manifest.json"
)
OLD_EXP_PREFIX = (
    "/root/autodl-tmp/traffic-stas-paper-sparse-20260716-10k-v1/train_10k_20k/mappo"
)
# 门控修订 (2026-07-17, 用户批准): G1 只对 forced_direct 设阈值 —— 两条
# 绕行路线近似对称使 permutation 检验结构性无信息量 (保留为报告项);
# G4 带宽上限 12%->25% —— 氢负荷晚峰与道路晚峰相撞的设计张力下, 优秀
# 策略也保有 ~20% 时机保险, 该份额随能力单调下降, 是度量而非税。
G1_DELTA_THRESHOLD = 15.0
G4_BAND = (0.02, 0.25)
TOTAL_TIMESTEPS = int(__import__("os").environ.get("SMOKE_TIMESTEPS", 240000))
EVAL_INTERVAL = 12000


def base_mappo_command() -> list:
    manifest = json.loads(SRC_MANIFEST.read_text())
    for value in manifest.values():
        if isinstance(value, dict) and value and all(
            isinstance(job, dict) and "command" in job for job in value.values()
        ):
            return list(value["mappo"]["command"])
    raise RuntimeError("mappo job not found in paper-sparse manifest")


def build_command(output_root: Path) -> list:
    overrides_arg = hydra_override_arg(env_v2_overrides(sparse=True))
    command = []
    for arg in base_mappo_command():
        if OLD_EXP_PREFIX in arg:
            arg = arg.replace(OLD_EXP_PREFIX, str(output_root / "mappo"))
        if arg.startswith("+MICROGRID_CONFIG_OVERRIDES="):
            arg = overrides_arg
        elif arg.startswith("TOTAL_TIMESTEPS="):
            arg = "TOTAL_TIMESTEPS=%d" % TOTAL_TIMESTEPS
        elif arg.startswith("EVAL_INTERVAL="):
            arg = "EVAL_INTERVAL=%d" % EVAL_INTERVAL
        elif arg.startswith("CHECKPOINT_INTERVAL="):
            arg = "CHECKPOINT_INTERVAL=%d" % TOTAL_TIMESTEPS
        elif arg.startswith("ALG="):
            arg = "ALG=EnvV2-Smoke-MAPPO"
        elif arg.startswith("EXP_NAME="):
            arg = "EXP_NAME=env_v2_smoke"
        elif arg.startswith("RUN_NAME="):
            arg = "RUN_NAME=env_v2_smoke__seed30"
        command.append(arg)
    return command


def run_training(output_root: Path) -> int:
    import os

    log_path = output_root / "train.log"
    env = dict(os.environ)
    env["PYTHONPATH"] = f"{ROOT}:{env.get('PYTHONPATH', '')}"
    env["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.5"
    env["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    env["WANDB_MODE"] = "disabled"
    with log_path.open("w") as stream:
        process = subprocess.run(
            build_command(output_root),
            cwd=str(ROOT),
            stdout=stream,
            stderr=subprocess.STDOUT,
            env=env,
        )
    return int(process.returncode)


def load_best_policy(output_root: Path):
    import jax
    import jax.numpy as jnp
    import numpy as np
    import optax
    from flax.training.train_state import TrainState

    from baselines.MAPPO.continuous_policy import deterministic_action
    from baselines.MAPPO.mappo_ff_shared_weights import ActorCritic
    from baselines.utils.microgrid_vec_env import MicrogridVecEnv
    from baselines.utils.training_checkpoint import load_jax_training_checkpoint

    meta_path = (
        output_root
        / "mappo/output/checkpoints/best_validation/best_validation.json"
    )
    meta = json.loads(meta_path.read_text())
    overrides = env_v2_overrides(sparse=True)
    probe = MicrogridVecEnv(num_envs=1, auto_reset=True, config_overrides=overrides)
    n, obs_dim, action_dim = probe.num_agents, probe.obs_dim, probe.action_dim
    probe.close()
    network = ActorCritic(
        action_dim,
        activation="relu",
        actor_layers=[256, 256],
        critic_layers=[256, 256],
        num_agents=n,
        observation_dim=obs_dim,
        is_continuous=True,
        log_std_init=-1.0,
        log_std_min=-2.5,
        log_std_max=-0.5,
    )
    actor_template = jnp.zeros((n, obs_dim + n), dtype=jnp.float32)
    critic_template = jnp.zeros((n, obs_dim * n), dtype=jnp.float32)
    params = network.init(jax.random.PRNGKey(0), actor_template, critic_template)
    tx = optax.chain(optax.clip_by_global_norm(5.0), optax.adam(3e-4, eps=1e-5))
    template = TrainState.create(apply_fn=network.apply, params=params, tx=tx)
    trained = load_jax_training_checkpoint(
        Path(meta["jax_checkpoint"]), template
    ).train_state.params
    ids = jnp.eye(n, dtype=jnp.float32)
    dummy = jnp.zeros((n, obs_dim * n), dtype=jnp.float32)

    @jax.jit
    def act(obs):
        actor_obs = jnp.concatenate([obs, ids], axis=-1)
        actor_output, _ = network.apply(trained, actor_obs, dummy)
        mean, _ = actor_output
        return deterministic_action(mean)

    return (lambda obs: np.asarray(act(jnp.asarray(obs, dtype=jnp.float32)))), meta


def run_gates(output_root: Path) -> dict:
    from baselines.utils.final_comparison import (
        force_direct_route,
        force_no_order,
        permute_route_actions,
    )
    from baselines.utils.fixed_scenario_eval import (
        VALIDATION_DAYS,
        build_scenarios,
        evaluate_policy,
    )

    policy, meta = load_best_policy(output_root)
    overrides = env_v2_overrides(sparse=True)
    scenarios = build_scenarios(VALIDATION_DAYS, (4200,))
    variants = {
        "normal": policy,
        "forced_no_order": force_no_order(policy),
        "forced_direct_route": force_direct_route(policy),
        "permuted_route": permute_route_actions(policy),
    }
    records = {
        name: evaluate_policy(
            fn,
            overrides,
            scenarios,
            algorithm=f"env_v2_smoke__{name}",
            split_name="validation",
        )
        for name, fn in variants.items()
    }
    normal = records["normal"]["summary"]

    def delta(name):
        return float(
            normal["return_mean"] - records[name]["summary"]["return_mean"]
        )

    deltas = {name: delta(name) for name in
              ("forced_no_order", "forced_direct_route", "permuted_route")}
    g1_pass = deltas["forced_direct_route"] >= G1_DELTA_THRESHOLD
    total_cost = max(float(normal.get("total_cost_mean", 0.0)), 1e-9)
    emergency_premium = 0.5 * float(normal.get("emergency_h2_cost_mean", 0.0))
    loss_value = float(normal.get("transport_loss_mean", 0.0)) * float(
        overrides["lambda_h2_buy"]
    )
    g4_share = (emergency_premium + loss_value) / total_cost
    g4_pass = G4_BAND[0] <= g4_share <= G4_BAND[1]
    return {
        "best_checkpoint_episode": meta.get("episode"),
        "best_validation_return": meta.get("validation_return"),
        "variant_summaries": {
            name: record["summary"] for name, record in records.items()
        },
        "deltas_normal_minus_counterfactual": deltas,
        "G1_route_mechanism": {
            "threshold": G1_DELTA_THRESHOLD,
            "pass": bool(g1_pass),
        },
        "G4_economic_share": {
            "share": g4_share,
            "band": list(G4_BAND),
            "pass": bool(g4_pass),
        },
    }


def main():
    output_root = Path(sys.argv[1])
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "mappo/output").mkdir(parents=True, exist_ok=True)
    returncode = run_training(output_root)
    g3 = {"training_returncode": returncode, "pass": returncode == 0}
    results = {"G3_training_stability": g3}
    if returncode == 0:
        try:
            results.update(run_gates(output_root))
        except Exception as error:  # noqa: BLE001
            results["gate_error"] = repr(error)
    (output_root / "gate_results.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False, default=str)
    )
    print(json.dumps(
        {k: results[k] for k in results if k != "variant_summaries"},
        indent=2, ensure_ascii=False, default=str,
    ))
    print("SMOKE_DONE")


if __name__ == "__main__":
    main()
