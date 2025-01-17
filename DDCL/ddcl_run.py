#基于双重决策和先验知识的虚实迁移持续学习方法
# 首先创建代码总体框架
import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
#感知阶段
def extract_fused_features(real_image, fake_image, input_channels=3, feature_dim=64):
    """
    提取融合后的特征向量。

    :param real_image: 真实场景图像，形状为 [batch_size, 3, 32, 32]
    :param fake_image: 虚拟场景图像，形状为 [batch_size, 3, 32, 32]
    :param input_channels: 图像通道数，默认为3（RGB图像）
    :param feature_dim: 最终输出的特征向量维度，默认为64
    :return: 融合后的特征向量，形状为 [batch_size, feature_dim]
    """

    # 定义卷积神经网络部分


class FeatureFusionNetwork(nn.Module):
    def __init__(self, input_channels=3, feature_dim=64):
        """初始化 FeatureFusionNetwork 类

        参数：
        - input_channels (int): 输入图像的通道数（默认是3，即RGB图像）
        - feature_dim (int): 输出特征向量的维度（默认是64）
        """
        super(FeatureFusionNetwork, self).__init__()

        # 定义卷积神经网络结构
        # 第一层卷积，将输入的通道数从 input_channels（默认为3）转换为 32 个输出通道
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, stride=1, padding=1)
        # 第二层卷积，将通道数从 32 转换为 64
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        # 第三层卷积，将通道数从 64 转换为 128
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)

        # 定义全连接层，假设输入图像大小为 32x32，经过卷积后输出的特征图大小为 128x32x32
        # 首先展平，输入到一个 512 维的全连接层
        self.fc1 = nn.Linear(128 * 32 * 32, 512)
        # 第二个全连接层，将 512 维的向量映射到 feature_dim（默认为64）维
        self.fc2 = nn.Linear(512, feature_dim)

        # 融合部分：将两个特征向量进行融合，拼接成一个更大的向量后，再映射到 feature_dim 维
        self.fc_fusion = nn.Linear(2 * feature_dim, feature_dim)

    def forward(self, real_image, fake_image):
        """前向传播，计算真实图像和虚拟图像的融合特征

        参数：
        - real_image (Tensor): 真实场景的输入图像
        - fake_image (Tensor): 虚拟场景的输入图像

        返回：
        - fused_features (Tensor): 融合后的特征向量
        """
        # 处理真实场景图像
        x_real = F.relu(self.conv1(real_image))  # 使用ReLU激活函数进行激活
        x_real = F.relu(self.conv2(x_real))
        x_real = F.relu(self.conv3(x_real))
        x_real = x_real.view(x_real.size(0), -1)  # 展平特征图为一维向量
        x_real = F.relu(self.fc1(x_real))  # 全连接层1
        real_features = F.relu(self.fc2(x_real))  # 全连接层2，得到真实图像的特征

        # 处理虚拟场景图像
        x_fake = F.relu(self.conv1(fake_image))  # 使用ReLU激活函数进行激活
        x_fake = F.relu(self.conv2(x_fake))
        x_fake = F.relu(self.conv3(x_fake))
        x_fake = x_fake.view(x_fake.size(0), -1)  # 展平特征图为一维向量
        x_fake = F.relu(self.fc1(x_fake))  # 全连接层1
        fake_features = F.relu(self.fc2(x_fake))  # 全连接层2，得到虚拟图像的特征

        # 融合两个特征向量
        combined_features = torch.cat((real_features, fake_features), dim=1)  # 在通道维度（dim=1）拼接两个特征向量
        fused_features = F.relu(self.fc_fusion(combined_features))  # 经过全连接层融合后的特征

        return fused_features  # 返回融合后的特征

    # 初始化模型
    model = FeatureFusionNetwork(input_channels=input_channels, feature_dim=feature_dim)

    # 前向传播，得到融合后的特征向量
    fused_features = model(real_image, fake_image)

    return fused_features


# 示例
# if __name__ == "__main__":
#     # 假设输入图像大小为 [batch_size, 3, 32, 32]，例如 batch_size=8
#     real_image = torch.randn(8, 3, 32, 32)  # 8张真实图像
#     fake_image = torch.randn(8, 3, 32, 32)  # 8张虚拟图像
#
#     # 提取融合后的特征
#     fused_features = extract_fused_features(real_image, fake_image)
#     print(fused_features.shape)  # 输出形状应该是 [8, 64]，即8个样本，64维特征向量

#决策阶段
class Policy_Learning(nn.Module):
    def __init__(self,state_dim,action_dim):
        super(Policy_Learning, self).__init__()

    def actor(self,state_dim,action_dim):
        nn.Sequential(
                nn.Linear(state_dim, 64),
                nn.Tanh(),
                nn.Linear(64, 64),
                nn.Tanh(),
                nn.Linear(64, action_dim),
                nn.Tanh()
        )
    def critic(self,state_dim):
         nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )
    def update(self):
        pass
    def Get_action(self,state):
        action_mean = actor(state)
        cov_mat = torch.diag(self.action_var).unsqueeze(dim=0)
        dist = MultivariateNormal(action_mean, cov_mat)
        action = dist.sample()  # 动作采用多元高斯分布采样
        action_logprob = dist.log_prob(action)  # 这个动作概率就相当于优先经验回放的is_weight

    return action.detach(), action_logprob.detach()

#语义特征知识库，存储文本知识
def store_text_data(text, file_name):
    """
    将文本数据存储到指定的文件中。

    参数:
    - text: 要存储的文本数据（字符串）。
    - file_name: 存储文本的文件名（字符串）。
    """
    try:
        # 以写入模式打开文件，如果文件不存在则会自动创建
        with open(file_name, 'w', encoding='utf-8') as file:
            file.write(text)  # 写入文本数据
        print(f"数据已成功存储到文件 {file_name}")
    except Exception as e:
        print(f"存储文本数据时出错: {e}")


def select_best_strategy(strategies, error_weight=0.4, risk_weight=0.3, reward_weight=0.3):
    """
    基于错误率、风险率、奖励值等评估指标的判别器函数，选择最佳决策策略。

    参数:
    - strategies: 一个包含多个策略对象的列表，每个策略包含错误率、风险率和奖励值等信息。
    - error_weight: 错误率的权重，默认为0.4。
    - risk_weight: 风险率的权重，默认为0.3。
    - reward_weight: 奖励值的权重，默认为0.3。

    返回:
    - 最优策略对象。
    """

    class Strategy:
        def __init__(self, name, error_rate, risk_rate, reward_value):
            """
            初始化策略对象

            :param name: 策略名称
            :param error_rate: 错误率
            :param risk_rate: 风险率
            :param reward_value: 奖励值
            """
            self.name = name
            self.error_rate = error_rate
            self.risk_rate = risk_rate
            self.reward_value = reward_value

        def __repr__(self):
            return f"Strategy(name={self.name}, error_rate={self.error_rate}, risk_rate={self.risk_rate}, reward_value={self.reward_value})"

    def evaluate(strategy):
        """
        评估策略的综合得分，返回一个得分

        :param strategy: 策略对象
        :return: 综合得分
        """
        # 错误率、风险率和奖励值的评分，我们假设越低的错误率和风险率越好，奖励值越高越好
        score = (1 - strategy.error_rate) * error_weight + \
                (1 - strategy.risk_rate) * risk_weight + \
                strategy.reward_value * reward_weight
        return score

    # 寻找最优策略
    best_score = -np.inf
    best_strategy = None

    for strategy in strategies:
        score = evaluate(strategy)
        print(f"Strategy {strategy.name}: Score = {score}")
        if score > best_score:
            best_score = score
            best_strategy = strategy

    return best_strategy


# 示例策略
strategies = [
    Strategy("Strategy A", error_rate=0.1, risk_rate=0.2, reward_value=0.8),
    Strategy("Strategy B", error_rate=0.05, risk_rate=0.15, reward_value=0.9),
    Strategy("Strategy C", error_rate=0.15, risk_rate=0.25, reward_value=0.7),
]

# 调用函数选择最佳策略
best_strategy = select_best_strategy(strategies)

print(f"最佳策略是: {best_strategy.name}")


#知识体控制
def knowledge_based_control(frame, error_threshold=3000, stop_threshold=5000):
    """
    根据输入的图像数据决定无人车的移动。

    参数:
    - frame: 来自摄像头的图像数据（一个BGR图像帧）。
    - error_threshold: 每个区域内黑色像素数量的阈值，用于判断障碍物的存在。
    - stop_threshold: 如果前方区域有障碍物，控制车停止的阈值。

    返回:
    - steering_angle: 转向角度，负值表示左转，正值表示右转，0表示直行。
    - speed: 车速，0表示停止，正值表示前进。
    """

    # 转换为灰度图
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 应用阈值处理，假设障碍物是黑色的区域
    _, thresholded = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)

    # 获取图像的中间三部分：左、中、右
    height, width = thresholded.shape
    left_region = thresholded[:, 0:width // 3]
    center_region = thresholded[:, width // 3:2 * width // 3]
    right_region = thresholded[:, 2 * width // 3:]

    # 计算每个区域内黑色区域的数量，黑色区域可能是障碍物
    left_obstacle = np.sum(left_region == 0)
    center_obstacle = np.sum(center_region == 0)
    right_obstacle = np.sum(right_region == 0)

    # 根据障碍物的位置做决策
    if center_obstacle > stop_threshold:  # 前方有障碍物，停止
        steering_angle = 0
        speed = 0
        print("前方有障碍物，停止！")
    elif left_obstacle > error_threshold:  # 左边有障碍物，向右转
        steering_angle = 15  # 向右转
        speed = 50
        print("左侧有障碍物，右转！")
    elif right_obstacle > error_threshold:  # 右边有障碍物，向左转
        steering_angle = -15  # 向左转
        speed = 50
        print("右侧有障碍物，左转！")
    else:  # 没有障碍物，继续前进
        steering_angle = 0
        speed = 50
        print("前方没有障碍物，直行！")

    return steering_angle, speed


# 使用摄像头获取图像并做决策
def run_car():
    # 初始化摄像头
    camera = cv2.VideoCapture(0)

    while True:
        ret, frame = camera.read()
        if not ret:
            print("无法读取图像，退出！")
            break

        # 调用函数来做出决策
        steering_angle, speed = knowledge_based_control(frame)

        # 显示决策后的图像
        cv2.putText(frame, f"Steering: {steering_angle} Speed: {speed}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Car View", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    # 释放资源
    camera.release()
    cv2.destroyAllWindows()

#actor-critic
# 定义Actor-Critic网络
class ActorCritic(nn.Module):
    def __init__(self, input_size, action_space):
        super(ActorCritic, self).__init__()

        # Actor网络：用于输出动作的概率分布
        self.actor = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, action_space),  # 输出每个动作的概率
            nn.Softmax(dim=-1)  # 使用Softmax使得输出是一个概率分布
        )

        # Critic网络：用于输出状态的价值
        self.critic = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 1)  # 输出状态的价值
        )

    def forward(self, state):
        """
        输入状态，分别输出动作的概率和状态的价值。
        """
        action_probs = self.actor(state)
        state_value = self.critic(state)
        return action_probs, state_value


# 定义Actor-Critic的训练过程
def actor_critic(state, action, reward, next_state, done, model, optimizer, gamma=0.99):
    """
    Actor-Critic算法的核心训练函数。

    参数:
    - state: 当前状态
    - action: 当前动作
    - reward: 当前奖励
    - next_state: 下一状态
    - done: 是否结束
    - model: Actor-Critic模型
    - optimizer: 优化器
    - gamma: 折扣因子，默认0.99

    返回:
    - actor_loss: Actor的损失
    - critic_loss: Critic的损失
    """
    state = torch.tensor(state, dtype=torch.float32)
    next_state = torch.tensor(next_state, dtype=torch.float32)
    reward = torch.tensor(reward, dtype=torch.float32)
    done = torch.tensor(done, dtype=torch.float32)

    # 前向传播，获取当前状态的动作概率和状态价值
    action_probs, state_value = model(state)

    # 计算Critic损失：均方误差（MSE）
    with torch.no_grad():
        _, next_state_value = model(next_state)
        target_value = reward + (1 - done) * gamma * next_state_value.squeeze()  # 计算目标值
    critic_loss = (state_value.squeeze() - target_value).pow(2).mean()  # 均方误差

    # 计算Actor损失：策略梯度
    action_log_probs = torch.log(action_probs.squeeze(0)[action])  # 当前动作的对数概率
    advantage = target_value - state_value.squeeze()  # 优势函数（目标值 - 当前价值）
    actor_loss = -(action_log_probs * advantage).mean()  # 策略梯度的损失

    # 总损失：Actor损失 + Critic损失
    total_loss = actor_loss + critic_loss

    # 优化器更新
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    return actor_loss.item(), critic_loss.item()

#多演员多评论家网络
def multi_actor_critic_training(state_dim, action_dim, n_actors=3, n_critics=3, gamma=0.99, lr=1e-3, episodes=1000,
                                batch_size=32, memory_capacity=10000):
    """
    基于多演员（Actor）和多评论家（Critic）的强化学习训练函数。
    参数:
    - state_dim (int): 状态空间维度
    - action_dim (int): 动作空间维度
    - n_actors (int): 演员数量（默认3个）
    - n_critics (int): 评论家数量（默认3个）
    - gamma (float): 折扣因子（默认0.99）
    - lr (float): 学习率（默认1e-3）
    - episodes (int): 训练回合数（默认1000）
    - batch_size (int): 每次训练的样本大小（默认32）
    - memory_capacity (int): 经验池容量（默认10000）
    """

    # 定义神经网络模型
    class Actor(nn.Module):
        def __init__(self, state_dim, action_dim):
            super(Actor, self).__init__()
            self.fc1 = nn.Linear(state_dim, 128)
            self.fc2 = nn.Linear(128, 64)
            self.fc3 = nn.Linear(64, action_dim)
            self.softmax = nn.Softmax(dim=-1)  # 用于生成概率分布

        def forward(self, state):
            x = torch.relu(self.fc1(state))
            x = torch.relu(self.fc2(x))
            action_probs = self.softmax(self.fc3(x))
            return action_probs

    class Critic(nn.Module):
        def __init__(self, state_dim):
            super(Critic, self).__init__()  # 调用父类的初始化方法

            # 定义神经网络的三层全连接层（fc1, fc2, fc3）
            # fc1: 输入维度是state_dim，输出维度是128
            self.fc1 = nn.Linear(state_dim, 128)

            # fc2: 输入维度是128，输出维度是64
            self.fc2 = nn.Linear(128, 64)

            # fc3: 输入维度是64，输出维度是1（因为我们要预测状态的价值）
            self.fc3 = nn.Linear(64, 1)  # 输出一个值，评估状态的价值

        def forward(self, state):
            # 前向传播过程：将输入状态通过三层全连接层进行处理

            # 使用ReLU激活函数对第一层的输出进行非线性变换
            x = torch.relu(self.fc1(state))

            # 使用ReLU激活函数对第二层的输出进行非线性变换
            x = torch.relu(self.fc2(x))

            # 最后一层的输出是状态的价值，不使用激活函数（线性输出）
            value = self.fc3(x)

            # 返回状态的价值
            return value

class ExperienceReplay:
    def __init__(self, capacity=10000):
        # 初始化经验回放缓冲区，设置最大容量为capacity
        self.capacity = capacity
        # 使用deque作为存储结构，deque在插入和删除元素时效率较高
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        # 将一个新的经验（状态，动作，奖励，下一状态，是否结束）添加到回放缓冲区
        # 这个经验会作为一个元组存储
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        # 从回放缓冲区随机采样一个batch的经验
        # batch_size决定了采样的数量
        return random.sample(self.buffer, batch_size)

    def size(self):
        # 返回当前回放缓冲区中存储的经验数量
        return len(self.buffer)

    # 初始化多个Actor（智能体的策略网络），每个Actor用于处理不同的环境实例或不同的策略
    # n_actors表示有多少个Actor，state_dim是状态维度，action_dim是动作维度
    actors = [Actor(state_dim, action_dim) for _ in range(n_actors)]

    # 初始化多个Critic（状态价值估计网络），每个Critic用于估计不同的状态值
    # n_critics表示有多少个Critic，state_dim是状态维度
    critics = [Critic(state_dim) for _ in range(n_critics)]

    # 初始化一个主Critic，用于估计主策略的状态值
    main_critic = Critic(state_dim)

    # 初始化每个Actor的优化器，这里使用Adam优化器，学习率为lr
    actor_optimizers = [optim.Adam(actor.parameters(), lr=lr) for actor in actors]

    # 初始化每个Critic的优化器，使用Adam优化器，学习率为lr
    critic_optimizers = [optim.Adam(critic.parameters(), lr=lr) for critic in critics]

    # 初始化主Critic的优化器，使用Adam优化器，学习率为lr
    main_critic_optimizer = optim.Adam(main_critic.parameters(), lr=lr)

    # 初始化经验回放缓冲区，指定容量为memory_capacity
    # ExperienceReplay是用于存储智能体与环境交互经验的缓冲区
    memory = ExperienceReplay(capacity=memory_capacity)

def normalize(values, min_val=0.0, max_val=1.0):
    """将指标值归一化到 [min_val, max_val] 范围内"""
    return (values - min_val) / (max_val - min_val)

def select_action(state, actor_idx):
    """根据演员选择策略"""
    state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
    action_probs = actors[actor_idx](state)
    action = torch.multinomial(action_probs, 1).item()  # 根据概率分布选择动作
    return action

def evaluate_action(state, action, critic_idx):
    """根据评论家评估动作的价值"""
    state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
    value = critics[critic_idx](state)
    return value

def update_main_critic():
    """更新主评论家网络"""
    if len(memory) > 0:  # 检查经验回放是否为空
        # 从经验回放中采样一批数据
        batch = memory.sample(batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)  # 解包批次数据

        # 将状态、下一个状态、奖励、done标志转换为tensor
        states = torch.tensor(states, dtype=torch.float32)  # 当前状态
        next_states = torch.tensor(next_states, dtype=torch.float32)  # 下一个状态
        rewards = torch.tensor(rewards, dtype=torch.float32)  # 奖励
        dones = torch.tensor(dones, dtype=torch.float32)  # 是否终止的标志

        # 计算当前状态的价值和下一个状态的价值
        values = main_critic(states)  # 当前状态的价值
        next_values = main_critic(next_states)  # 下一个状态的价值

        # 计算目标价值（使用贝尔曼方程）
        target_values = rewards + (1 - dones) * gamma * next_values.squeeze()  # 目标价值

        # 计算评论家网络的损失
        critic_loss = nn.MSELoss()(values.squeeze(), target_values)  # 使用均方误差（MSE）作为损失函数

        # 清空梯度
        main_critic_optimizer.zero_grad()

        # 反向传播计算梯度
        critic_loss.backward()

        # 更新主评论家网络参数
        main_critic_optimizer.step()


def update_actors(state, action, actor_idx):
    """根据评论家的反馈更新演员策略"""
    # 将状态和动作转换为Tensor，并增加一个批次维度（unsqueeze(0)）
    state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
    action = torch.tensor(action, dtype=torch.long).unsqueeze(0)

    # 使用演员网络计算给定状态下的动作概率分布
    action_probs = actors[actor_idx](state)

    # 计算所选动作的对数概率
    log_prob = torch.log(action_probs[0, action])  # 选择的动作的对数概率

    # 使用主评论家网络计算给定状态的价值估计
    value = main_critic(state)  # 主评论家的价值估计

    # 计算优势函数：优势 = 价值 - 价值（这里的价值已经通过主评论家得到）
    advantage = value - main_critic(state)

    # 计算演员的损失：策略梯度方法，损失与对数概率和优势函数的乘积成反比
    actor_loss = -log_prob * advantage.detach()  # 使用策略梯度方法进行优化

    # 清空当前演员网络的梯度
    actor_optimizers[actor_idx].zero_grad()

    # 反向传播计算梯度
    actor_loss.backward()

    # 更新演员网络参数
    actor_optimizers[actor_idx].step()


def update_critics(state, reward, next_state, done, critic_idx):
    """根据奖励和下一个状态更新评论家网络"""

    # 将状态、下一个状态、奖励和结束标志转换为PyTorch张量，并增加一个批次维度
    state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
    next_state = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)
    reward = torch.tensor(reward, dtype=torch.float32)
    done = torch.tensor(done, dtype=torch.float32)

    # 使用评论家网络计算当前状态的价值估计
    value = critics[critic_idx](state)

    # 使用评论家网络计算下一个状态的价值估计
    next_value = critics[critic_idx](next_state)

    # 计算目标值（TD目标）：目标值 = 奖励 + (1 - done) * γ * 下一个状态的价值
    target_value = reward + (1 - done) * gamma * next_value

    # 计算评论家的损失（均方误差损失），用于度量当前状态价值估计与目标值之间的误差
    critic_loss = nn.MSELoss()(value.squeeze(), target_value)

    # 清空当前评论家网络的梯度
    critic_optimizers[critic_idx].zero_grad()

    # 反向传播计算评论家网络的梯度
    critic_loss.backward()

    # 使用优化器更新评论家网络的参数
    critic_optimizers[critic_idx].step()


# 训练过程
for episode in range(episodes):
    state = np.random.rand(state_dim)  # 随机初始化状态
    done = False
    total_reward = 0

    while not done:
        # 每个演员选择动作
        actions = [select_action(state, actor_idx) for actor_idx in range(n_actors)]

        # 每个演员执行并得到奖励
        action = random.choice(actions)  # 随机选择一个演员的动作
        reward = np.random.rand()  # 假设奖励是随机的
        next_state = np.random.rand(state_dim)  # 假设下一个状态是随机的
        done = random.choice([True, False])  # 假设是否完成是随机的

        # 存储经验
        memory.push(state, action, reward, next_state, done)

        # 更新评论家和演员网络
        for actor_idx in range(n_actors):
            update_actors(state, action, actor_idx)

        for critic_idx in range(n_critics):
            update_critics(state, reward, next_state, done, critic_idx)

        # 更新主评论家
        update_main_critic()

        total_reward += reward
        state = next_state  # 更新状态

    print(f"Episode {episode + 1}/{episodes}, Total Reward: {total_reward}")
#判别器一，决策模式选择
def decision_maker(error_rate, risk_value, reward_value):
    """
    根据错误率、风险值和奖励值来选择决策模式。
    如果综合值低于0.5，则选择知识体决策，否则返回动态决策。

    参数:
    - error_rate (float): 错误率 (0到1之间)
    - risk_value (float): 风险值 (0到1之间)
    - reward_value (float): 奖励值 (0到1之间)

    返回:
    - decision (dict or str): 返回决策模式
    """

    # 知识体决策
    knowledge_base_decision = {
        "obstacle": "turn_right_30_degrees_and_move_5m",
        "target": "move_straight_at_average_speed",
        "dynamic_obstacle": "intercept_continuously"
    }

    # 计算判别器的综合值
    weight_error = 0.3
    weight_risk = 0.4
    weight_reward = 0.3

    discriminative_value = (1 - error_rate) * weight_error + (1 - risk_value) * weight_risk + reward_value * weight_reward

    # 如果判别器的综合值低于0.5，选择知识体决策
    if discriminative_value < 0.5:
        print(f"Decision Mode: Knowledge-Based")
        return knowledge_base_decision
    else:
        print(f"Decision Mode: Dynamic")
        # 假设动态决策模式是由其他策略或算法实现的
        return "dynamic_decision_algorithm"

# 初始化模型并进行训练
def train_actor_critic():
    # 环境参数
    input_size = 4  # 假设状态空间维度为4
    action_space = 2  # 假设有2个动作（例如：向左，向右）

    # 创建Actor-Critic模型
    model = ActorCritic(input_size, action_space)

    # 优化器
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 假设训练过程中使用简单的状态和动作
    state = np.random.rand(input_size)  # 当前状态
    action = np.random.randint(action_space)  # 当前动作
    reward = np.random.rand()  # 当前奖励
    next_state = np.random.rand(input_size)  # 下一状态
    done = np.random.randint(2)  # 是否结束（1表示结束，0表示继续）

    # 训练过程
    actor_loss, critic_loss = actor_critic(state, action, reward, next_state, done, model, optimizer)

    print(f"Actor Loss: {actor_loss}, Critic Loss: {critic_loss}")

