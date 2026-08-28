#!/usr/bin/env python
"""STAS 信用保真度探针 (Phase 2 病因诊断).

核心问题: STAS 的注意力 credit 到底有没有识别真实的时序信用, 还是那 15 分
(blend vs uniform) 只是方差整形?

方法: 拿训好的 blend checkpoint, 在固定验证场景上采 on-policy 轨迹, 对每个
(步 t, agent i) 做单步反事实消融 —— 把该体该步动作换成 idle 基线、之后继续
on-policy 跑到底, 量终端团队回报变化 = 真实因果重要度。再算 STAS credit 与
该因果重要度的 Spearman 秩相关。

判读: 相关高 => 注意力学对了, STAS 输在注入方式 (救援有望, 试 advantage 塑形);
相关低/零 => 注意力没识别信用, 15 分是方差整形 (转诊断性负结果)。

用法: PYTHONPATH=. python scripts/stas_credit_fidelity_probe.py <blend_output_root> [n_episodes]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
STAS_DIR = ROOT / "baselines" / "STAS-MAPPO"
for path in (str(ROOT), str(STAS_DIR)):
    if path not in sys.path:
        sys.path.insert(0, path)

IDLE_ACTION_VALUE = -1.0  # 物理基线: 全 idle, 与 force_no_order 约定一致
GAMMA = 1.0


def _load_policy(meta, overrides):
    import jax
    import jax.numpy as jnp
    import optax
    from flax.training.train_state import TrainState

    from baselines.MAPPO.continuous_policy import deterministic_action
    from baselines.MAPPO.mappo_ff_shared_weights import ActorCritic
    from baselines.utils.microgrid_vec_env import MicrogridVecEnv
    from baselines.utils.training_checkpoint import load_jax_training_checkpoint

    probe = MicrogridVecEnv(num_envs=1, auto_reset=True, config_overrides=overrides)
    n, obs_dim, action_dim = probe.num_agents, probe.obs_dim, probe.action_dim
    probe.close()
    network = ActorCritic(
        action_dim, activation="relu", actor_layers=[256, 256],
        critic_layers=[256, 256], num_agents=n, observation_dim=obs_dim,
        is_continuous=True, log_std_init=-1.0, log_std_min=-2.5, log_std_max=-0.5,
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
        # squashed_gaussian 训练下 env 动作 = 均值 (deterministic_action 恒等)
        return deterministic_action(mean)

    def policy(obs):
        return np.asarray(act(jnp.asarray(obs, dtype=jnp.float32)), dtype=np.float32)

    return policy, n, obs_dim, action_dim


def _load_credit_assigner(overrides, obs_dim, action_dim, n, meta):
    from stas_mappo.credit import STASCreditConfig
    from stas_mappo.conserved_credit import ConservedSTASCreditAssigner
    from stas_mappo.checkpoint import load_credit_assigner_checkpoint

    config = STASCreditConfig(
        obs_dim=obs_dim, action_dim=action_dim, n_agents=n, seq_length=24,
        gamma=GAMMA, emb_dim=128, n_heads=4, n_layers=3, sample_num=5,
        lr=5e-4, weight_decay=1e-5, device="cuda", conserve_discounted=True,
        quality_gate_enable=True, warmup_episodes=500, ramp_episodes=2000,
        max_mix_coef=0.05, mode="conserved",
    )
    assigner = ConservedSTASCreditAssigner(config)
    load_credit_assigner_checkpoint(Path(meta["stas_checkpoint"]), assigner)
    return assigner


def _rankdata(values):
    """Average-rank (ties averaged), no scipy dependency."""
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    ranks[order] = np.arange(1, len(values) + 1, dtype=np.float64)
    # average ties
    sorted_vals = values[order]
    i = 0
    while i < len(values):
        j = i
        while j + 1 < len(values) and sorted_vals[j + 1] == sorted_vals[i]:
            j += 1
        if j > i:
            avg = np.mean(ranks[order[i:j + 1]])
            ranks[order[i:j + 1]] = avg
        i = j + 1
    return ranks


def _pearson(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.std() < 1e-12 or b.std() < 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _spearman(a, b):
    return _pearson(_rankdata(a), _rankdata(b))


def _run_episode(env, n, obs_dim, seed, policy, ablate=None):
    """跑一个确定性 episode。ablate=(t,i) 时在第 t 步把 agent i 动作设为 idle。

    返回 (obs_seq[T,n,obs_dim], act_seq[T,n,adim], rew_seq[T,n], team_return)。
    """
    obs, _ = env.reset(seed=seed)
    obs = np.asarray(obs, dtype=np.float32).reshape(n, obs_dim)
    obs_seq, act_seq, rew_seq = [], [], []
    t = 0
    while True:
        actions = policy(obs)  # [n, adim], 确定性
        if ablate is not None and t == ablate[0]:
            actions = actions.copy()
            actions[ablate[1], :] = IDLE_ACTION_VALUE
        obs_seq.append(obs.copy())
        act_seq.append(actions.copy())
        next_obs, rewards, terms, truncs, _ = env.step(actions)
        rewards = np.asarray(rewards, dtype=np.float32).reshape(n)
        rew_seq.append(rewards.copy())
        t += 1
        if bool(np.any(terms)) or bool(np.any(truncs)):
            break
        obs = np.asarray(next_obs, dtype=np.float32).reshape(n, obs_dim)
    team_return = float(np.sum(np.asarray(rew_seq, dtype=np.float64)))
    return (
        np.asarray(obs_seq, dtype=np.float32),
        np.asarray(act_seq, dtype=np.float32),
        np.asarray(rew_seq, dtype=np.float32),
        team_return,
    )


def _causal_importance(env, n, obs_dim, seed, policy, base_return, horizon):
    """逐 (t,i) 单步消融, 返回 [T,n]: c_{t,i} = G_base - G_ablate(t,i)。"""
    importance = np.zeros((horizon, n), dtype=np.float64)
    for t in range(horizon):
        for i in range(n):
            _, _, _, g = _run_episode(env, n, obs_dim, seed, policy, ablate=(t, i))
            importance[t, i] = base_return - g
    return importance


def _to_eat(arr_tn_last):
    """[T,n,...] -> [env=1, agent, time, ...] 供 credit_rewards 消费。"""
    arr = np.asarray(arr_tn_last)
    if arr.ndim == 2:  # [T,n]
        return arr.transpose(1, 0)[None, ...]
    return arr.transpose(1, 0, 2)[None, ...]  # [T,n,d]->[1,n,T,d]


def main():
    blend_root = Path(sys.argv[1])
    n_episodes = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    from scripts.env_v2_overrides import env_v2_overrides
    from baselines.utils.microgrid_vec_env import MicrogridVecEnv

    from baselines.utils.fixed_scenario_eval import VALIDATION_DAYS

    base_overrides = env_v2_overrides(sparse=True)
    meta = json.loads(
        (blend_root / "output/checkpoints/best_validation/best_validation.json").read_text()
    )
    policy, n, obs_dim, action_dim = _load_policy(meta, base_overrides)
    assigner = _load_credit_assigner(base_overrides, obs_dim, action_dim, n, meta)

    # 必须精确复刻固定验证协议 (钉天 + 锁定种子), 否则 rollout 走的是
    # 策略从未经历过的随机日, base_return 与 checkpoint 的 -547 严重不符,
    # credit 相关性无意义。VALIDATION_DAYS=(8,17,21,23), 锁定种子 4200。
    LOCKED_SEED = 4200
    days = list(VALIDATION_DAYS)[:n_episodes] if n_episodes <= len(VALIDATION_DAYS) \
        else list(VALIDATION_DAYS)

    def pin(day):
        ov = dict(base_overrides)
        ov.update({
            "italian_split_enable": True,
            "italian_split_strategy": "manifest",
            "italian_split_name": "validation",
            "italian_day_indices": [int(day)],
        })
        return ov

    # 确定性自检: 同天同种子跑两遍团队回报必须逐位相等。
    env0 = MicrogridVecEnv(num_envs=1, auto_reset=False, config_overrides=pin(days[0]))
    _, _, _, g1 = _run_episode(env0, n, obs_dim, LOCKED_SEED, policy)
    _, _, _, g2 = _run_episode(env0, n, obs_dim, LOCKED_SEED, policy)
    env0.close()
    assert abs(g1 - g2) < 1e-4, f"env not deterministic: {g1} vs {g2}"

    per_episode = []
    pooled_causal, pooled_credit = [], []
    for k, day in enumerate(days):
        seed = LOCKED_SEED
        env = MicrogridVecEnv(num_envs=1, auto_reset=False, config_overrides=pin(day))
        obs_seq, act_seq, rew_seq, base_return = _run_episode(env, n, obs_dim, seed, policy)
        horizon = obs_seq.shape[0]
        credit = assigner.credit_rewards(
            _to_eat(obs_seq), _to_eat(act_seq), _to_eat(rew_seq),
            np.zeros((1, n, horizon), dtype=np.float32),
        )[0]  # [n, T]
        importance = _causal_importance(
            env, n, obs_dim, seed, policy, base_return, horizon
        )  # [T, n]
        credit_flat = credit.transpose(1, 0).reshape(-1)   # (T*n,) 对齐 [t,i]
        causal_flat = importance.reshape(-1)
        sp = _spearman(causal_flat, credit_flat)
        pe = _pearson(causal_flat, credit_flat)
        # z-score 后入池 (跨 episode 量级不同)
        cz = (causal_flat - causal_flat.mean()) / (causal_flat.std() + 1e-12)
        rz = (credit_flat - credit_flat.mean()) / (credit_flat.std() + 1e-12)
        pooled_causal.extend(cz.tolist())
        pooled_credit.extend(rz.tolist())
        # top-5 因果步是否落在 credit 前 25%?
        k_top = max(1, len(causal_flat) // 20)
        top_causal = set(np.argsort(causal_flat)[-k_top:].tolist())
        thresh = np.quantile(credit_flat, 0.75)
        hit = np.mean([credit_flat[idx] >= thresh for idx in top_causal])
        env.close()
        per_episode.append({
            "day": int(day), "seed": seed, "base_return": base_return,
            "spearman": sp, "pearson": pe,
            "causal_std": float(causal_flat.std()),
            "credit_std": float(credit_flat.std()),
            "top_causal_in_credit_top25pct": float(hit),
        })
        print(f"[ep {k} day={day} seed={seed}] base={base_return:.1f} "
              f"spearman={sp:.3f} pearson={pe:.3f} topk_hit={hit:.2f}", flush=True)

    sp_vals = [e["spearman"] for e in per_episode if e["spearman"] == e["spearman"]]
    pooled_sp = _spearman(pooled_causal, pooled_credit)
    summary = {
        "checkpoint_episode": meta.get("episode"),
        "validation_return": meta.get("validation_return"),
        "n_episodes": n_episodes,
        "n_agents": n,
        "idle_baseline_value": IDLE_ACTION_VALUE,
        "spearman_mean": float(np.mean(sp_vals)) if sp_vals else float("nan"),
        "spearman_std": float(np.std(sp_vals)) if sp_vals else float("nan"),
        "spearman_pooled_zscored": pooled_sp,
        "topk_hit_mean": float(np.mean([e["top_causal_in_credit_top25pct"] for e in per_episode])),
        "per_episode": per_episode,
    }
    out = blend_root / "credit_fidelity_probe.json"
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print("\n=== CREDIT FIDELITY VERDICT ===")
    print(json.dumps({k: v for k, v in summary.items() if k != "per_episode"},
                     indent=2, ensure_ascii=False))
    print("PROBE_DONE")


if __name__ == "__main__":
    main()
