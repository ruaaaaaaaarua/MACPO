"""MATRPO 算法实现，适配微电网异构多智能体场景。

训练结构采用 CTDE：
- 执行端：每个智能体有一个独立 Actor，只看本地观测。
- 训练端：所有智能体共享一个 Critic，输入为全局状态。
- 更新端：Critic 用 MSE 拟合 GAE return；每个 Actor 分别执行一轮 TRPO 更新。

与原始代码的关键区别：
  1. 支持 N 个独立 Actor（separated policy）
  2. 每个 Actor 独立做 CG + line search
  3. Critic 使用 value normalization
  4. 加入 entropy 监控（不作为 loss，TRPO 靠 KL 约束控制探索）
"""

import torch
import torch.nn as nn
from torch.nn.utils.convert_parameters import parameters_to_vector, vector_to_parameters


class RunningMeanStd:
    """在线维护标量/张量的均值和方差，用于 value normalization。"""

    def __init__(self, shape=()):
        self.mean = torch.zeros(shape)
        self.var = torch.ones(shape)
        self.count = 1e-4

    def update(self, x):
        """用当前 batch 更新运行均值和方差。"""
        device = self.mean.device
        x = x.to(device)
        batch_mean = x.mean(dim=0)
        batch_var = x.var(dim=0)
        batch_count = x.shape[0]

        delta = batch_mean - self.mean
        total_count = self.count + batch_count

        self.mean = self.mean + delta * batch_count / total_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + delta.pow(2) * self.count * batch_count / total_count
        self.var = m2 / total_count
        self.count = total_count

    def normalize(self, x):
        """将原始值标准化，作为 Critic 的训练目标。"""
        return (x - self.mean) / (self.var.sqrt() + 1e-8)

    def denormalize(self, x):
        """将 Critic 输出还原到原始 reward/return 尺度。"""
        return x * (self.var.sqrt() + 1e-8) + self.mean


class MATRPO:
    """Multi-Agent Trust Region Policy Optimization。

    Args:
        actors: list of Actor (每个 agent 一个)
        critic: Critic (共享)
        各种 TRPO 超参数
    """

    def __init__(
        self,
        actors,
        critic,
        critic_lr=1e-3,
        max_kl=0.01,
        cg_iters=10,
        cg_damping=0.1,
        line_search_steps=10,
        critic_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        use_valuenorm=True,
        max_grad_norm=5.0,
    ):
        if len({id(actor) for actor in actors}) != len(actors):
            raise ValueError("MATRPO requires one independent Actor instance per agent.")
        actor_param_ids = [
            id(param)
            for actor in actors
            for param in actor.parameters()
        ]
        if len(set(actor_param_ids)) != len(actor_param_ids):
            raise ValueError("MATRPO actors must not share parameter tensors.")

        self.actors = actors
        self.critic = critic
        self.num_agents = len(actors)

        self.critic_optimizer = torch.optim.Adam(critic.parameters(), lr=critic_lr)

        self.max_kl = max_kl
        self.cg_iters = cg_iters
        self.cg_damping = cg_damping
        self.line_search_steps = line_search_steps
        self.critic_epochs = critic_epochs
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.max_grad_norm = max_grad_norm

        self.use_valuenorm = use_valuenorm
        if use_valuenorm:
            self.value_rms = RunningMeanStd(shape=())
            device = next(critic.parameters()).device
            self.value_rms.mean = self.value_rms.mean.to(device)
            self.value_rms.var = self.value_rms.var.to(device)

    # ============================================================
    # TRPO 核心方法（针对单个 Actor）
    # ============================================================

    def _get_flat_params(self, model):
        """把模型参数展平成一个向量，便于 TRPO 按向量做线搜索。"""
        return parameters_to_vector(model.parameters())

    def _set_flat_params(self, model, flat_params):
        """将展平向量写回模型参数。"""
        vector_to_parameters(flat_params, model.parameters())

    def _surrogate_loss(self, actor, obs, old_log_probs, actions, advantages):
        """TRPO 的替代目标 E[ratio * advantage]。"""
        dist = actor(obs)
        log_probs = dist.log_prob(actions).sum(dim=-1)
        ratio = torch.exp(log_probs - old_log_probs)
        return (ratio * advantages).mean()

    def _compute_kl(self, actor, obs, old_mean, old_std):
        """计算新旧高斯策略之间的平均 KL 散度。"""
        dist = actor(obs)
        kl = (
            torch.log(dist.stddev / old_std)
            + (old_std.pow(2) + (old_mean - dist.mean).pow(2))
            / (2.0 * dist.stddev.pow(2))
            - 0.5
        )
        return kl.sum(dim=-1).mean()

    def _fvp(self, actor, obs, p, old_mean, old_std):
        """Fisher-Vector Product，近似计算 KL Hessian 与向量 p 的乘积。"""
        # TRPO 不显式构造 Hessian，而是通过二阶自动微分得到 H*p。
        kl = self._compute_kl(actor, obs, old_mean, old_std)
        grads = torch.autograd.grad(kl, actor.parameters(), create_graph=True)
        flat_grad = torch.cat([g.view(-1) for g in grads])
        kl_v = (flat_grad * p).sum()
        grads2 = torch.autograd.grad(kl_v, actor.parameters())
        flat_grad2 = torch.cat([g.contiguous().view(-1) for g in grads2])
        # damping 用于数值稳定，避免 Fisher 矩阵病态导致共轭梯度发散。
        return flat_grad2 + p * self.cg_damping

    def _conjugate_gradient(self, actor, obs, b, old_mean, old_std):
        """用共轭梯度近似求解 Hx=b，其中 H 是 Fisher/KL Hessian。"""
        x = torch.zeros_like(b)
        r = b.clone()
        p = b.clone()
        rsold = torch.dot(r, r)

        for _ in range(self.cg_iters):
            # Ap 即 H*p，通过 Fisher-vector product 计算。
            Ap = self._fvp(actor, obs, p, old_mean, old_std)
            alpha = rsold / (torch.dot(p, Ap) + 1e-8)
            x += alpha * p
            r -= alpha * Ap
            rsnew = torch.dot(r, r)
            if rsnew < 1e-10:
                break
            p = r + (rsnew / rsold) * p
            rsold = rsnew
        return x

    def _trpo_step(self, actor, obs, actions, old_log_probs, advantages):
        """对单个 Actor 执行一步 TRPO 更新。

        Returns:
            dict: 包含 kl, surrogate_loss, entropy, update_success 等信息
        """
        # 保存旧策略分布参数，后续 KL 约束和重要性采样都以它为基准。
        with torch.no_grad():
            old_dist = actor(obs)
            old_mean = old_dist.mean.clone()
            old_std = old_dist.stddev.clone()
            entropy = old_dist.entropy().sum(dim=-1).mean().item()

        # 对替代目标求一阶梯度，作为 trust-region 更新的目标方向。
        surrogate_loss = self._surrogate_loss(actor, obs, old_log_probs, actions, advantages)
        grads = torch.autograd.grad(surrogate_loss, actor.parameters())
        loss_grad = torch.cat([g.view(-1) for g in grads]).detach()

        # 梯度接近 0 时不更新，避免后续除零或无意义线搜索。
        if torch.norm(loss_grad) < 1e-8:
            return {
                "kl": 0.0,
                "surrogate_loss": surrogate_loss.item(),
                "entropy": entropy,
                "success": False,
            }

        # 用共轭梯度求解自然梯度方向 H^{-1}g。
        step_dir = self._conjugate_gradient(actor, obs, loss_grad, old_mean, old_std)

        # 根据 max_kl 缩放完整步长，保证理论二次近似下满足 KL 约束。
        shs = 0.5 * torch.dot(
            step_dir,
            self._fvp(actor, obs, step_dir, old_mean, old_std),
        )
        lm = torch.sqrt(torch.clamp(shs / self.max_kl, min=1e-8))
        fullstep = step_dir / lm

        # 线搜索：逐步缩小步长，直到 KL 不超限且替代目标提升。
        old_params = self._get_flat_params(actor)
        success = False
        final_kl = 0.0

        for frac in [0.5**i for i in range(self.line_search_steps)]:
            new_params = old_params + frac * fullstep
            self._set_flat_params(actor, new_params)

            with torch.no_grad():
                new_loss = self._surrogate_loss(
                    actor, obs, old_log_probs, actions, advantages
                )
                kl = self._compute_kl(actor, obs, old_mean, old_std)

            if kl < self.max_kl and new_loss > surrogate_loss:
                success = True
                final_kl = kl.item()
                break

        if not success:
            # 若所有候选步长都失败，回滚到更新前参数。
            self._set_flat_params(actor, old_params)
            final_kl = 0.0

        return {
            "kl": final_kl,
            "surrogate_loss": surrogate_loss.item(),
            "entropy": entropy,
            "success": success,
        }

    # ============================================================
    # GAE 计算
    # ============================================================

    def compute_gae(self, rewards, values, next_values, dones):
        """计算 GAE(lambda) returns 和 advantages。"""
        T = len(rewards)
        advantages = torch.zeros(T, device=values.device)
        returns = torch.zeros(T, device=values.device)
        gae = 0.0

        for t in reversed(range(T)):
            # delta 是一步 TD 误差；done 会截断 episode 边界后的 bootstrap。
            delta = (
                rewards[t]
                + self.gamma * next_values[t] * (1 - dones[t])
                - values[t]
            )
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages[t] = gae
            returns[t] = gae + values[t]

        return returns, advantages

    # ============================================================
    # 主更新接口
    # ============================================================

    def update(self, rollout_data):
        """执行一次完整的 MATRPO 更新。

        Args:
            rollout_data: dict，包含:
                obs:       list of [T, obs_dim] per agent
                actions:   list of [T, act_dim] per agent
                log_probs: list of [T] per agent
                states:    [T, state_dim]  全局状态
                rewards:   [T]  共享奖励
                dones:     [T]
                next_states: [T, state_dim]

        Returns:
            info: dict，训练指标
        """
        states = rollout_data["states"]
        rewards = rollout_data["rewards"]
        dones = rollout_data["dones"]
        next_states = rollout_data["next_states"]

        # ---- Critic: 先用当前 Critic 估计 V(s) 和 V(s')，再计算 GAE ----
        with torch.no_grad():
            values = self.critic(states).squeeze(-1)
            next_values = self.critic(next_states).squeeze(-1)

            # Critic 若输出的是标准化 value，需要先反标准化再参与 TD/GAE。
            if self.use_valuenorm:
                self.value_rms.update(values)
                values_denorm = self.value_rms.denormalize(values)
                next_values_denorm = self.value_rms.denormalize(next_values)
            else:
                values_denorm = values
                next_values_denorm = next_values

        returns, advantages = self.compute_gae(
            rewards, values_denorm, next_values_denorm, dones
        )
        # Actor 更新使用标准化 advantage，降低不同 episode 成本尺度带来的训练不稳定。
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Critic 的监督目标与其输出尺度保持一致：启用 valuenorm 时训练标准化 return。
        if self.use_valuenorm:
            self.value_rms.update(returns)
            targets = self.value_rms.normalize(returns)
        else:
            targets = returns

        # ---- Critic 更新：共享价值网络拟合全局状态的 return ----
        critic_losses = []
        for _ in range(self.critic_epochs):
            v = self.critic(states).squeeze(-1)
            critic_loss = nn.MSELoss()(v, targets.detach())
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
            self.critic_optimizer.step()
            critic_losses.append(critic_loss.item())

        # ---- Actor 更新：每个智能体独立执行 TRPO，但共享同一条系统级 advantage ----
        actor_infos = []
        for i in range(self.num_agents):
            info_i = self._trpo_step(
                self.actors[i],
                rollout_data["obs"][i],
                rollout_data["actions"][i],
                rollout_data["log_probs"][i],
                advantages,
            )
            actor_infos.append(info_i)

        return {
            "critic_loss": sum(critic_losses) / len(critic_losses),
            "actor_infos": actor_infos,
            "returns_mean": returns.mean().item(),
        }
