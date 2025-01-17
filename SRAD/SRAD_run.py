import torch
import torch.nn as nn

class CriticNetwork(nn.Module):
    def __init__(self, input_size):
        super(CriticNetwork, self).__init__()
        # 这里定义一个简单的全连接网络作为Critic
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)  # 输出一个状态值

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        value = self.fc3(x)
        return value


class SafetyModule:
    def __init__(self, input_size):
        # 初始化两个Critic网络，分别处理不同的风险等级
        self.critic_low_risk = CriticNetwork(input_size)
        self.critic_high_risk = CriticNetwork(input_size)

        # 使用相同的优化器更新两个Critic
        self.optimizer = optim.Adam(list(self.critic_low_risk.parameters()) +
                                    list(self.critic_high_risk.parameters()), lr=0.001)

    def select_critic(self, state, risk_level):
        """
        根据风险等级选择合适的Critic网络
        """
        if risk_level == "high":
            return self.critic_high_risk(state)
        else:
            return self.critic_low_risk(state)

    def evaluate_risk(self, state, external_factors):
        """
        根据外部因素和当前状态评估风险等级
        通过一些简单的阈值来评估风险等级
        """
        if external_factors['obstacle_proximity'] < 0.5:  # 障碍物接近，判定为高风险
            risk_level = "high"
        else:
            risk_level = "low"
        return risk_level

    def train(self, state, reward, next_state, done, external_factors):
        """
        根据风险等级选择对应的Critic，计算损失并更新模型
        """
        # 评估风险等级
        risk_level = self.evaluate_risk(state, external_factors)

        # 获取当前状态对应的Critic评估值
        critic_value = self.select_critic(state, risk_level)

        # 下一状态的价值评估
        next_state_value = self.select_critic(next_state, risk_level).detach()

        # 计算目标值
        target_value = reward + (1 - done) * 0.99 * next_state_value

        # 计算损失
        critic_loss = (critic_value - target_value).pow(2).mean()

        # 反向传播并优化
        self.optimizer.zero_grad()
        critic_loss.backward()
        self.optimizer.step()

        return critic_loss.item()


# 示例：训练过程
def train_safety_module():
    input_size = 4  # 状态空间维度
    safety_module = SafetyModule(input_size)

    # 假设训练时给定的状态和外部因素
    state = np.random.rand(input_size)  # 当前状态
    next_state = np.random.rand(input_size)  # 下一状态
    reward = np.random.rand()  # 当前奖励
    done = np.random.randint(2)  # 是否结束（1表示结束，0表示继续）
    external_factors = {'obstacle_proximity': np.random.rand()}  # 外部因素（障碍物距离等）

    # 转换为Tensor
    state_tensor = torch.tensor(state, dtype=torch.float32)
    next_state_tensor = torch.tensor(next_state, dtype=torch.float32)

    # 训练一个步骤
    loss = safety_module.train(state_tensor, reward, next_state_tensor, done, external_factors)
    print(f"Critic Loss: {loss}")


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

    # 自恢复模块
    def self_recovery_module(image, error_threshold=3000):
        """
        根据四旋翼无人机的环境图像判断飞行方向，避免碰撞。

        参数：
        - image: 当前图像数据，形状为 (84, 84, 3)，表示环境图像。
        - error_threshold: 用于检测障碍物的阈值，黑色像素数量超过此阈值时认为该区域有障碍物。

        返回：
        - direction: 返回四旋翼飞行的控制指令。
          控制指令包括：'move_left', 'move_right', 'move_up', 'move_down', 'move_backward', 'move_forward'
        """
        # 转换为灰度图像
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # 反转图像颜色，假设障碍物是黑色区域
        _, thresholded = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)

        # 获取图像的中间五个部分：左、中、右、上、下
        height, width = thresholded.shape
        left_region = thresholded[:, 0:width // 3]  # 左区域
        center_region = thresholded[:, width // 3:2 * width // 3]  # 中心区域
        right_region = thresholded[:, 2 * width // 3:]  # 右区域
        top_region = thresholded[0:height // 3, :]  # 上区域
        bottom_region = thresholded[height // 3:, :]  # 下区域

        # 计算每个区域内黑色像素的数量
        left_obstacle = np.sum(left_region == 0)
        right_obstacle = np.sum(right_region == 0)
        top_obstacle = np.sum(top_region == 0)
        bottom_obstacle = np.sum(bottom_region == 0)
        center_obstacle = np.sum(center_region == 0)

        # 判断每个区域的障碍物是否超过阈值
        if left_obstacle > error_threshold:
            direction = "move_right"  # 左边有障碍物，向右转
        elif right_obstacle > error_threshold:
            direction = "move_left"  # 右边有障碍物，向左转
        elif top_obstacle > error_threshold:
            direction = "move_down"  # 上方有障碍物，向下飞
        elif bottom_obstacle > error_threshold:
            direction = "move_up"  # 下方有障碍物，向上飞
        elif center_obstacle > error_threshold:
            direction = "move_backward"  # 前方有障碍物，向后退
        else:
            direction = "move_forward"  # 无障碍物，向前进

        return direction

    # 使用摄像头获取图像并做出决策
    def run_drone():
        # 初始化摄像头
        camera = cv2.VideoCapture(0)

        while True:
            ret, frame = camera.read()
            if not ret:
                print("无法读取图像，退出！")
                break

            # 调用自恢复模块函数来做出决策
            direction = self_recovery_module(frame)
            print(f"飞行指令: {direction}")

            # 显示图像
            cv2.putText(frame, f"Direction: {direction}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow("Drone View", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # 释放资源
        camera.release()
        cv2.destroyAllWindows()
#样本标签任务池
def find_similar_images_in_pool(sample_labels_pool, input_image_path, top_n=5, model='resnet50'):
    """
    查找与输入图片最相似的图片及其标签。

    参数:
    sample_labels_pool (dict): 样本标签池，每个元素包含图片路径和标签。
    input_image_path (str): 输入图片路径。
    top_n (int): 返回最相似的前n张图片，默认为5。
    model (str): 预训练模型名称，默认为 'resnet50'。

    返回:
    list: 包含最相似图片的标签和路径的元组，按相似度降序排列。
    """

    # 使用预训练的ResNet模型提取图片特征
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if model == 'resnet50':
        model = models.resnet50(pretrained=True).to(device)
    else:
        raise ValueError("Unsupported model: Only 'resnet50' is supported.")

    model.eval()

    # 图像预处理
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 提取特征的函数
    def extract_features(image_path):
        image = Image.open(image_path).convert("RGB")
        image = preprocess(image).unsqueeze(0).to(device)
        with torch.no_grad():
            features = model(image)
        return features.cpu().numpy().flatten()

    # 创建一个字典来存储池中图片的特征
    image_features = {}
    for idx, sample in sample_labels_pool.items():
        image_path = sample["image"]
        image_features[idx] = extract_features(image_path)

    # 提取输入图片的特征
    input_features = extract_features(input_image_path)

    similarities = []

    # 计算输入图片与标签池中其他图片的相似度
    for idx, features in image_features.items():
        similarity = cosine_similarity([input_features], [features])[0][0]
        similarities.append((similarity, idx))

    # 按照相似度降序排列
    similarities.sort(reverse=True, key=lambda x: x[0])

    # 输出最相似的图片和标签
    result = []
    for sim, idx in similarities[:top_n]:
        result.append((sample_labels_pool[idx]["label"], sample_labels_pool[idx]["image"], sim))

    return result


# 示例：样本标签池
sample_labels_pool = {
    0: {"image": "image1.jpg", "label": "向前移动"},
    1: {"image": "image2.jpg", "label": "向后移动"},
    2: {"image": "image3.jpg", "label": "向左移动"},
    3: {"image": "image4.jpg", "label": "向右移动"},
    4: {"image": "image5.jpg", "label": "向上移动"},
    5: {"image": "image6.jpg", "label": "向下移动"},
}

# 使用该函数进行查找
input_image_path = "test_image.jpg"  # 输入图片路径
top_n = 5  # 返回最相似的前5张图片
result = find_similar_images_in_pool(sample_labels_pool, input_image_path, top_n)

# 打印结果
for sim, label, image_path in result:
    print(f"相似度: {sim:.4f}， 标签: {label}， 图片: {image_path}")

#风险安全策略
def risk_based_strategy(success_rate, safety_distance, collision_rate, reward_value, smoothness, max_distance,
                        completion_time, weights):
    """
    根据评估指标计算综合风险值并选择安全策略。

    参数:
    success_rate (float): 成功率
    safety_distance (float): 安全距离
    collision_rate (float): 碰撞率
    reward_value (float): 奖励值
    smoothness (float): 平滑度
    max_distance (float): 最远距离
    completion_time (float): 完成时间
    weights (dict): 各指标的权重

    返回:
    tuple: (综合风险值, 选择的安全策略)
    """

    def normalize(values, min_val=0.0, max_val=1.0):
        """将指标值归一化到 [min_val, max_val] 范围内"""
        return (values - min_val) / (max_val - min_val)

    # 计算综合风险值
    def calculate_risk(success_rate, safety_distance, collision_rate, reward_value, smoothness, max_distance,
                       completion_time, weights):
        # 对各个指标进行归一化
        success_rate = normalize(success_rate)
        safety_distance = normalize(safety_distance)
        collision_rate = normalize(collision_rate)
        reward_value = normalize(reward_value)
        smoothness = normalize(smoothness)
        max_distance = normalize(max_distance)
        completion_time = normalize(completion_time)

        # 计算综合风险值（加权和）
        risk_value = (
                success_rate * weights['success_rate'] +
                safety_distance * weights['safety_distance'] +
                (1 - collision_rate) * weights['collision_rate'] +  # 碰撞率越低越好
                reward_value * weights['reward_value'] +
                smoothness * weights['smoothness'] +
                max_distance * weights['max_distance'] +
                (1 - completion_time) * weights['completion_time']  # 完成时间越短越好
        )

        return risk_value

    # 选择策略
    def choose_strategy(risk_value):
        """根据综合风险值选择安全策略"""
        if risk_value < 0.2:
            return "策略1：最安全策略"
        elif risk_value < 0.4:
            return "策略2：安全策略"
        elif risk_value < 0.6:
            return "策略3：中等安全策略"
        elif risk_value < 1.0:
            return "策略4：高风险策略"
        else:
            return "策略5：危险策略"

    # 计算综合风险值
    risk_value = calculate_risk(success_rate, safety_distance, collision_rate, reward_value, smoothness, max_distance,
                                completion_time, weights)

    # 根据风险值选择策略
    strategy = choose_strategy(risk_value)

    return risk_value, strategy


# 示例权重
weights = {
    'success_rate': 0.2,  # 成功率的权重
    'safety_distance': 0.3,  # 安全距离的权重
    'collision_rate': 0.2,  # 碰撞率的权重
    'reward_value': 0.1,  # 奖励值的权重
    'smoothness': 0.05,  # 平滑度的权重
    'max_distance': 0.1,  # 最远距离的权重
    'completion_time': 0.05  # 完成时间的权重
}

# 示例输入：评估指标值
success_rate = 0.9  # 假设成功率是0.9
safety_distance = 0.8  # 假设安全距离是0.8
collision_rate = 0.1  # 假设碰撞率是0.1
reward_value = 0.85  # 假设奖励值是0.85
smoothness = 0.75  # 假设平滑度是0.75
max_distance = 1.2  # 假设最远距离是1.2
completion_time = 0.5  # 假设完成时间是0.5

# 调用函数
risk_value, strategy = risk_based_strategy(success_rate, safety_distance, collision_rate, reward_value, smoothness,
                                           max_distance, completion_time, weights)

# 输出结果
print(f"综合风险值: {risk_value:.4f}")
print(f"选择的安全策略: {strategy}")

def train_actor_critic(env_name='CartPole-v1',
                       n_episodes=1000,
                       max_timesteps=200,
                       lr=0.001,
                       gamma=0.99,
                       hidden_size=64):
    """
    训练一个基于 Actor-Critic 算法的强化学习模型。

    参数:
    env_name (str): 使用的环境名称（默认为 'CartPole-v1'）。
    n_episodes (int): 训练的回合数。
    max_timesteps (int): 每一回合的最大步数。
    lr (float): 学习率。
    gamma (float): 折扣因子。
    hidden_size (int): 隐藏层大小。

    返回:
    model (ActorCritic): 训练好的模型。
    """

    # 创建环境
    env = gym.make(env_name)

    # 环境维度
    input_dim = env.observation_space.shape[0]
    output_dim = env.action_space.n

    # 设定设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 定义 Actor-Critic 网络
    # 定义Actor-Critic模型
    class ActorCritic(nn.Module):
        def __init__(self, input_dim, output_dim, hidden_size):
            super(ActorCritic, self).__init__()

            # 定义神经网络的层：两层全连接层（fc1, fc2），用于提取特征
            # 一个用于生成策略输出的输出层（policy_head），一个用于生成值函数输出的输出层（value_head）
            self.fc1 = nn.Linear(input_dim, hidden_size)
            self.fc2 = nn.Linear(hidden_size, hidden_size)
            self.policy_head = nn.Linear(hidden_size, output_dim)  # 策略输出：每个动作的概率
            self.value_head = nn.Linear(hidden_size, 1)  # 值函数输出：每个状态的值

        def forward(self, x):
            # 定义前向传播过程：输入通过两层全连接层，最后分别得到策略和值
            x = torch.relu(self.fc1(x))  # 使用ReLU激活函数
            x = torch.relu(self.fc2(x))  # 使用ReLU激活函数
            policy = self.policy_head(x)  # 策略输出
            value = self.value_head(x)  # 值函数输出
            return policy, value  # 返回策略和状态值

    # 初始化网络和优化器
    model = ActorCritic(input_dim, output_dim, hidden_size).to(device)  # 创建Actor-Critic模型
    optimizer = optim.Adam(model.parameters(), lr=lr)  # 使用Adam优化器

    # 选择动作的函数
    def select_action(state):
        # 将输入状态转换为Tensor并传入设备
        state = torch.tensor(state, dtype=torch.float32).to(device)

        # 获取策略输出和状态值
        policy, _ = model(state)

        # 通过softmax函数计算动作的概率分布
        prob = torch.softmax(policy, dim=-1)

        # 使用Categorical分布从概率分布中采样动作
        dist = Categorical(prob)

        # 从分布中采样一个动作
        action = dist.sample()

        # 返回选定的动作、动作的对数概率以及该分布的熵（用于entropy regularization）
        return action.item(), dist.log_prob(action), dist.entropy()

    # 计算优势函数
    def compute_advantage(rewards, values, next_value, gamma):
        returns = []
        next_value = next_value.detach().item()  # 将下一状态的值转换为普通数值

        # 从最后一步往前计算每个时间步的返回值（优势函数）
        for reward, value in zip(rewards[::-1], values[::-1]):
            return_ = reward + gamma * next_value  # 计算返回值，gamma是折扣因子
            returns.append(return_)
            next_value = value.detach().item()  # 更新下一状态的值
        return returns[::-1]  # 反转返回列表，返回从第一个时间步开始的返回值

    # 训练过程
    for episode in range(n_episodes):
        state = env.reset()  # 重置环境，获取初始状态
        episode_reward = 0  # 初始化本轮的奖励
        done = False  # 初始化done标志
        log_probs = []  # 存储每个动作的对数概率
        values = []  # 存储每个状态的值
        rewards = []  # 存储每个时间步的奖励

        for t in range(max_timesteps):
            # 选择动作并计算动作的对数概率、熵
            action, log_prob, entropy = select_action(state)

            # 执行动作，得到下一个状态、奖励和done标志
            next_state, reward, done, _ = env.step(action)

            # 存储log概率和奖励
            log_probs.append(log_prob)
            rewards.append(reward)

            # 将当前状态转换为Tensor并传入设备
            state_tensor = torch.tensor(state, dtype=torch.float32).to(device)

            # 获取当前状态的值
            _, value = model(state_tensor)

            # 存储当前状态的值
            values.append(value)

            state = next_state  # 更新状态
            episode_reward += reward  # 累加奖励

            # 如果done为True，表示当前episode结束，跳出循环
            if done:
                break

        # 计算目标值（返回值）：计算下一个状态的值并得到优势函数
        next_state_tensor = torch.tensor(next_state, dtype=torch.float32).to(device)
        _, next_value = model(next_state_tensor)
        returns = compute_advantage(rewards, values, next_value, gamma)

        # 转换为Tensor并移动到设备
        returns = torch.tensor(returns, dtype=torch.float32).to(device)
        log_probs = torch.stack(log_probs).to(device)  # 将log概率转为Tensor
        values = torch.stack(values).squeeze().to(device)  # 将值函数转为Tensor并去除多余的维度

        # 计算损失
        advantage = returns - values  # 计算优势
        policy_loss = -log_probs * advantage.detach()  # 策略损失：取log概率与优势的乘积
        value_loss = advantage.pow(2)  # 值函数损失：优势的平方

        # 总损失：策略损失与值函数损失的平均
        loss = policy_loss.mean() + value_loss.mean()

        # 更新网络
        optimizer.zero_grad()  # 清空梯度
        loss.backward()  # 反向传播计算梯度
        optimizer.step()  # 更新参数

        # 打印进度
        if (episode + 1) % 100 == 0:
            print(f'Episode {episode + 1}/{n_episodes}, Reward: {episode_reward}')

    # 测试模型
    state = env.reset()
    done = False
    total_reward = 0
    while not done:
        action, _, _ = select_action(state)
        state, reward, done, _ = env.step(action)
        total_reward += reward
    print(f'Total Reward after training: {total_reward}')

    return model


# 训练 Actor-Critic 模型
model = train_actor_critic()