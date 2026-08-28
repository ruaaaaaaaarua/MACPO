"""MATRPO 使用的神经网络结构。

本项目采用 separated policy + shared critic：
- 每个智能体拥有一个独立 Actor，输入本地观测，输出连续动作的高斯分布。
- 所有智能体共享一个 Critic，输入拼接后的全局状态，输出状态价值。
"""

import torch
import torch.nn as nn
from torch.distributions import Normal


def init_weights(m, gain=0.01):
    """对线性层使用正交初始化，并将 bias 初始化为 0。"""
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight, gain=gain)
        nn.init.constant_(m.bias, 0)


class Actor(nn.Module):
    """单个智能体的策略网络：局部观测 -> 连续动作高斯分布。"""

    def __init__(self, obs_dim, action_dim, hidden_dim=128):
        super().__init__()
        # 两层 MLP 输出动作均值；动作标准差不依赖状态，由可学习参数给出。
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )
        # 可学习的 log 标准差，shape=(1, action_dim)，forward 时扩展到 batch 维度。
        self.action_log_std = nn.Parameter(torch.zeros(1, action_dim))
        init_weights(self.net[0], gain=nn.init.calculate_gain("relu"))
        init_weights(self.net[2], gain=nn.init.calculate_gain("relu"))
        init_weights(self.net[4], gain=0.01)

    def forward(self, obs):
        """返回 Normal 分布对象，训练脚本从中 sample 或取 mean 作为动作。"""
        mean = self.net(obs)
        log_std = self.action_log_std.expand_as(mean)
        std = torch.exp(log_std)
        return Normal(mean, std)


class Critic(nn.Module):
    """共享价值网络：全局状态 -> 状态价值 V(s)。"""

    def __init__(self, state_dim, hidden_dim=128):
        super().__init__()
        # Critic 输入是所有智能体局部观测拼接后的全局状态。
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.apply(lambda m: init_weights(m, gain=1.0))

    def forward(self, state):
        """返回状态价值，shape 通常为 [batch, 1]。"""
        return self.net(state)
