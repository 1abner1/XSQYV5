import torch
import torch.nn as nn


class ActorCritic(nn.Module):
    def __init__(self, num_inputs, num_actions, num_agents):
        super(ActorCritic, self).__init__()

        # 定义卷积层
        self.conv_layers = nn.Sequential(
            nn.Conv2d(num_inputs, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=3, stride=1),
            nn.ReLU()
        )

        # 定义全连接层
        self.fc_layers = nn.Sequential(
            nn.Linear(32 * 7 * 7, 256),
            nn.ReLU()
        )

        # 定义策略网络（Actor）
        self.actors = nn.ModuleList([nn.Sequential(
            nn.Linear(256, num_actions),
            nn.Softmax(dim=-1)
        ) for _ in range(num_agents)])

        # 定义价值网络（Critic）
        self.critic = nn.Linear(256, 1)

    def forward(self, x, agent_idx=None):
        x = x / 255.0  # 将像素值范围缩放到[0,1]
        x = self.conv_layers(x)  # 卷积层处理
        x = x.view(-1, 32 * 7 * 7)  # 展平后送入全连接层
        x = self.fc_layers(x)  # 全连接层处理

        if agent_idx is not None:
            actor_out = self.actors[agent_idx](x)  # 策略网络的输出
        else:
            # 如果agent_idx为None，表示需要计算所有Agent的策略概率
            actor_out = torch.stack([actor_net(x) for actor_net in self.actors], dim=1)

        critic_out = self.critic(x)  # 价值网络的输出
        return actor_out, critic_out
