import random
import ActorNetwork
import CriticNetwork
import ReplayBuffer
import snakeoil3_gym
import time
import cv2
import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt

# 共享特征语义空间构建，第一步获取图像数据，通过deeplabv3 获得语义分割图像
def vido_show(video_path):
    # 指定视频文件路径
    video_path = 'path/to/your/video.mp4'

    # 打开视频文件
    cap = cv2.VideoCapture(video_path)

    # 检查视频是否成功打开
    if not cap.isOpened():
        print("Error: Unable to open the video file.")
        exit()

    # 获取视频的帧率和尺寸
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # 创建一个窗口
    cv2.namedWindow('Video', cv2.WINDOW_NORMAL)

    while True:
        # 读取一帧
        ret, frame = cap.read()

        # 检查是否成功读取帧
        if not ret:
            break

        # 在窗口中显示帧
        cv2.imshow('Video', frame)

        # 按 'q' 键退出循环
        if cv2.waitKey(int(1000 / fps)) & 0xFF == ord('q'):
            break

    # 释放资源
    cap.release()
    cv2.destroyAllWindows()

# 1.通过deeplabv3 获得语义分割图像，视觉语义空间构建

MODEL_NAME = 'https://tfhub.dev/tensorflow/deeplabv3/1'
model = tf.saved_model.load(MODEL_NAME)

def load_and_preprocess_image(image_path):
    """
    加载并预处理图片
    Args:
        image_path (str): 图片文件路径
    Returns:
        np.array: 处理后的图像
    """
    # 读取图像
    image = cv2.imread(image_path)
    # unityBGR转换为RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # 调整大小
    image = cv2.resize(image, (84, 84))
    # 标准化（将值缩放到[0, 1]范围）
    image = image / 255.0
    # 扩展维度（Batch维度） 适应感受野大小
    image = np.expand_dims(image, axis=0)
    return image

# 开始进行分割
def predict(image):
    """
    使用DeepLabV3进行预测
    Args:
        image (np.array): 输入图像
    Returns:
        np.array: 预测的类别掩码
    """
    # 通过模型对图像进行分割，并分割的区域所对应的实体进行预测
    input_tensor = tf.convert_to_tensor(image, dtype=tf.float32)
    output = model(input_tensor)
    output = output['default'][0]  # 获取模型输出
    output = tf.argmax(output, axis=-1)  # 获取每个像素的类别
    output = output.numpy()  # 转换为NumPy数组
    return output

# 显示分割的结果
def display_results(original_image, predicted_mask):
    """
    显示分割结果
    Args:
        original_image (np.array): 原始图像
        predicted_mask (np.array): 预测的类别掩码
    """
    # 重新调整预测掩码大小
    predicted_mask_resized = cv2.resize(predicted_mask, (original_image.shape[1], original_image.shape[0]), interpolation=cv2.INTER_NEAREST)

    # 环境及实体类别的颜色表达的定义
    label_colormap = np.array([
        [0, 0, 0],      # 0:背景
        [0, 0, 128],    # 1:障碍物1
        [0, 128, 0],    # 2:障碍物2
        [128, 128, 0],  # 3:道路
        [128, 0, 128],  # 4:障碍物3
        [0, 128, 128],  # 5:目标
        [128, 0, 0],    # 6: 障碍物4
        # 可以添加更多类别
    ])

    # 为每个实体类别映射颜色
    segmented_image = label_colormap[predicted_mask_resized]

    # 显示结果
    plt.figure(figsize=(10, 10))
    plt.subplot(1, 2, 1)
    plt.imshow(original_image)
    plt.title("Original Image")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(segmented_image)
    plt.title("Predicted Segmentation")
    plt.axis('off')
    plt.show()


def segment_image(image_path=None, video_source=0):
    """
    对指定图片或视频源进行语义分割
    Args:
        image_path (str): 图片文件路径（如果传入该参数，视频源将被忽略）
        video_source (int or str): 视频源，默认为摄像头0
    """
    if image_path:
        # 处理静态图片
        image = load_and_preprocess_image(image_path)
        # 预测
        predicted_mask = predict(image)
        # 显示结果
        original_image = cv2.imread(image_path)
        display_results(original_image, predicted_mask)

    else:
        # 处理视频流
        cap = cv2.VideoCapture(video_source)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            # 图像大小，颜色预处理
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image_resized = cv2.resize(image, (84, 84))
            image_normalized = image_resized / 255.0
            image_input = np.expand_dims(image_normalized, axis=0)
            # 实体类别预测
            prediction = predict(image_input)
            # 显示结果
            display_results(frame, prediction)

            # 按键 'q' 退出
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()

#使用图片进行语义分割
image_path = "image.jpg"  # 替换为你自己的图片路径
segment_image(image_path=image_path)  # 调用图片分割

# 使用摄像头进行实时语义分割
segment_image(video_source=0)  # 如果需要使用摄像头进行视频流分割

#灰度特征语义空间， Canny 边缘检测: canny 边缘检测方法需要进行高斯滤波，以减少图片噪声的影响。
def edge_detection(image_path=None, video_source=0):
    """
    对指定图片或视频源进行边缘检测
    Args:
        image_path (str): 图片文件路径（如果传入该参数，视频源将被忽略）
        video_source (int or str): 视频源，默认为摄像头0
    """
    def process_image(image):
        """ 对图像进行边缘检测 """
        # 使用Canny边缘检测,100对应的是黑色，进行标准化，映射到100，200 所表达的灰度共享语义空间，以减少虚实灰度差异
        edges = cv2.Canny(image, 100, 200)  # 100 和 200 为低阈值和高阈值
        return edges

    if image_path:
        # 处理静态图片
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # 读取为灰度图像
        edges = process_image(image)

        # 显示结果
        plt.figure(figsize=(10, 10))
        plt.subplot(1, 2, 1)
        plt.imshow(image, cmap='gray')
        plt.title("Original Image")
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.imshow(edges, cmap='gray')
        plt.title("Edge Detection")
        plt.axis('off')
        plt.show()

    else:
        # 处理视频流
        cap = cv2.VideoCapture(video_source)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            # 转为灰度图像
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # 边缘检测
            edges = process_image(gray_frame)
            # 显示结果
            cv2.imshow("Original", frame)
            cv2.imshow("Edge Detection", edges)
            # 按键 'q' 退出
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

# 使用图片进行边缘检测
image_path = "image.jpg"  # 替换为你自己的图片路径
edge_detection(image_path=image_path)  # 调用图片边缘检测
# 使用摄像头进行实时边缘检测
edge_detection(video_source=0)  # 如果需要使用摄像头进行视频流分割

# 深度语义特征空间


# 构建场景特征语义空间
# 定义一个用于构建场景图的函数
def build_scene_graph(image_path, model=None, device=None):
    """
    基于物体检测生成场景图
    Args:
        image_path (str): 输入的图像路径
        model (torch.nn.Module): 预训练的物体检测模型（默认为Faster R-CNN）
        device (str): 设备（'cuda' 或 'cpu'）
    Returns:
        scene_graph (list): 场景图的列表，包含物体、类别和物体之间的关系
    """
    if model is None:
        # 加载预训练的Faster R-CNN模型
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
        model.eval()

    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 加载图像
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(image_rgb)

    # 把图像的像素值转换为torch可计算的tensor向量
    transform = transforms.Compose([transforms.ToTensor()])
    image_tensor = transform(pil_image).unsqueeze(0).to(device)

    # 预测图像实体类别
    with torch.no_grad():
        predictions = model(image_tensor)

    # 获取预测结果
    labels = predictions[0]['labels']
    boxes = predictions[0]['boxes']
    scores = predictions[0]['scores']

    # 设置阈值，过滤低置信度的检测结果，躲避关系，搜索关系，定义一个函数获取关系，给一个向量知道物体直接的关系。
    threshold = 0.5
    high_score_indices = torch.nonzero(scores > threshold).squeeze(1)
    labels = labels[high_score_indices]
    boxes = boxes[high_score_indices]

    # 相关的物体标签, 改成和场景相关的
    coco_names = [
        "Target", "Agent", "Obstacle1", "Obstacle2", "Obstacle3", "Obstacle4", "Wall", "Cube", "Cylinder",
        "Large Drone", "Small Drone", "White UAV", "Blue UAV", "Red UAV", "Balloon", "Soccer Ball", "Doll",
        "Chair", "Table", "Floor", "Ring", "Pedestrian"
    ]

    # 构建场景图
    scene_graph = []
    for label, box in zip(labels, boxes):
        # 获取物体的名称
        object_name = coco_names[label.item()]
        x1, y1, x2, y2 = box.tolist()

        # 存储物体的基本信息
        scene_graph.append({
            'object': object_name,
            'bbox': [x1, y1, x2, y2]
        })

    # 场景图矩阵形式的存储和表示
    plt.figure(figsize=(12, 12))
    plt.imshow(image_rgb)
    for item in scene_graph:
        x1, y1, x2, y2 = item['bbox']
        plt.gca().add_patch(plt.Rectangle((x1, y1), x2-x1, y2-y1, fill=False, color='red', linewidth=2))
        plt.text(x1, y1, item['object'], fontsize=12, color='yellow', bbox=dict(facecolor='black', alpha=0.5))
    plt.axis('off')
    plt.show()
    return scene_graph

# 场景图矩阵的实列化构建
image_path = "image.jpg"  # 替换为你自己的图片路径
scene_graph = build_scene_graph(image_path=image_path)

# 构建行为特征语义空间，定义行为有哪些？路径的行为策略特征，避障行为特征语义空间，定义避障行为特征语义空间
# 补充到终点的规则是避障行为规则来实现，根据历史记录来做的，运行轨迹来实现，共享特征语义空间时序的，行为的路径作为输入，路径作为输入来训练
# 记录行为共享特征语义空间，作为决策的输入，存储的格式，作为决策的输入。定义一个存储的函数
# 行为路径存储
def behavior_path_storage():
    # 现有的路径点坐标（每条路径有20个路径点）
    # 假设这是20条预定义路径，每条路径由20个坐标点组成
    paths = [
        [(x, y) for x, y in zip(range(1, 21), range(21, 41))],  # 路径1，(1, 21), (2, 22), ..., (20, 40)
        [(x, y) for x, y in zip(range(2, 22), range(22, 42))],  # 路径2，(2, 22), (3, 23), ..., (20, 40)
        [(x, y) for x, y in zip(range(3, 23), range(23, 43))],  # 路径3，(3, 23), (4, 24), ..., (20, 40)
        [(x, y) for x, y in zip(range(4, 24), range(24, 44))],  # 路径4，(4, 24), (5, 25), ..., (20, 40)
        [(x, y) for x, y in zip(range(5, 25), range(25, 45))],  # 路径5，(5, 25), (6, 26), ..., (20, 40)
        [(x, y) for x, y in zip(range(6, 26), range(26, 46))],  # 路径6，(6, 26), (7, 27), ..., (20, 40)
        [(x, y) for x, y in zip(range(7, 27), range(27, 47))],  # 路径7，(7, 27), (8, 28), ..., (20, 40)
        [(x, y) for x, y in zip(range(8, 28), range(28, 48))],  # 路径8，(8, 28), (9, 29), ..., (20, 40)
        [(x, y) for x, y in zip(range(9, 29), range(29, 49))],  # 路径9，(9, 29), (10, 30), ..., (20, 40)
        [(x, y) for x, y in zip(range(10, 30), range(30, 50))]  # 路径10，(10, 30), (11, 31), ..., (20, 40)
    ]
    # 定义一个内部函数来随机获取一条路径
    def get_random_path():
        return random.choice(paths)
    # 返回路径和获取随机路径的功能
    return paths, get_random_path

# 调用 behavior_path_storage 函数
paths, get_random_path = behavior_path_storage()

# 获取并输出一条随机路径
random_path = get_random_path()
print("Random Path:", random_path)

def plot_car_path(global_image_path, path_points, output_image_path=None):
    """
    在全局俯视图上绘制无人车的移动路径。
    Args:
        global_image_path (str): 输入的全局俯视图路径。
        path_points (list of tuples): 无人车路径点的列表，每个路径点为 (x, y) 坐标。
        output_image_path (str, optional): 输出路径图的保存路径。如果为None，则不保存。
    Returns:
        output_image (numpy.ndarray): 绘制好路径的黑白图像。
    """
    # 读取全局俯视图图像
    global_image = cv2.imread(global_image_path, cv2.IMREAD_COLOR)
    height, width = global_image.shape[:2]

    # 创建一个黑色背景图像，大小与全局图像相同
    output_image = np.zeros((height, width), dtype=np.uint8)

    # 转换路径点为numpy数组，供cv2使用
    path_points = np.array(path_points, dtype=np.int32)

    # 绘制路径（白色曲线），（定义一个函数为存储的帧的，有历史的帧）
    if len(path_points) > 1:
        # 使用polylines绘制连续的路径线
        cv2.polylines(output_image, [path_points], isClosed=False, color=255, thickness=2)

    # 将路径图叠加到全局图像上，如果需要可视化
    result_image = cv2.cvtColor(output_image, cv2.COLOR_GRAY2BGR)
    result_image = cv2.addWeighted(global_image, 0.5, result_image, 0.5, 0)

    # 显示生成的路径图像
    plt.imshow(result_image)
    plt.axis('off')  # 不显示坐标轴
    plt.show()

    # 如果指定了输出路径，则保存结果图像
    if output_image_path:
        cv2.imwrite(output_image_path, result_image)

    return output_image

# 给定路径点和全局图像路径，绘制路径
# global_image_path = "plan_image.jpg"  # 替换为你自己的全局图像路径
# path_points = [(100, 200), (150, 250), (200, 300), (250, 350), (300, 400)]  # 示例路径点
# 调用函数绘制路径
output_image = plot_car_path(global_image_path, path_points, output_image_path="path_output.jpg")


#感知层面的定义需要完善
#决策阶段
# 超参数定义
BUFFER_SIZE = int(1e6)    # 经验回放缓冲区的大小
BATCH_SIZE = 64           # 每次训练的批大小
GAMMA = 0.99              # 折扣因子
TAU = 1e-3                # 软更新目标网络的参数
LR_ACTOR = 1e-4           # Actor网络的学习率
LR_CRITIC = 1e-3          # Critic网络的学习率
UPDATE_EVERY = 4          # 每隔多少步更新一次网络
NOISE_STD_DEV = 0.2       # 动作噪声的标准差（用于探索）

# Actor 网络定义
# 定义 Actor 网络，通常用于策略网络（Policy Network）
class Actor(nn.Module):
    def __init__(self, state_size, action_size, hidden_units=256):
        # 初始化 Actor 网络，state_size 表示输入的状态空间大小，action_size 表示输出的动作空间大小
        super(Actor, self).__init__()

        # 定义三层全连接层
        self.fc1 = nn.Linear(state_size, hidden_units)  # 第一层，将输入状态大小映射到隐藏层大小
        self.fc2 = nn.Linear(hidden_units, hidden_units)  # 第二层，隐藏层到隐藏层
        self.fc3 = nn.Linear(hidden_units, action_size)  # 第三层，隐藏层到输出的动作空间大小

        # 使用 Tanh 激活函数将输出归一化到 [-1, 1] 范围内
        self.tanh = nn.Tanh()

    def forward(self, state):
        # 前向传播函数，state 为输入的状态

        # 第一次隐藏层转换，使用 ReLU 激活函数
        x = torch.relu(self.fc1(state))

        # 第二次隐藏层转换，使用 ReLU 激活函数
        x = torch.relu(self.fc2(x))

        # 最后一层输出，并将输出映射到 [-1, 1] 范围内
        return self.tanh(self.fc3(x))

    # 定义 Critic 网络，通常用于值函数网络（Value Network）

class Critic(nn.Module):
    def __init__(self, state_size, action_size, hidden_units=256):
        # 初始化 Critic 网络，state_size 表示状态空间大小，action_size 表示动作空间大小
        super(Critic, self).__init__()

        # 定义四层全连接层
        self.fc1 = nn.Linear(state_size, hidden_units)  # 第一层，状态空间到隐藏层
        self.fc2 = nn.Linear(action_size, hidden_units)  # 第二层，动作空间到隐藏层
        self.fc3 = nn.Linear(hidden_units, hidden_units)  # 第三层，隐藏层到隐藏层
        self.fc4 = nn.Linear(hidden_units, 1)  # 第四层，输出一个值，用于表示 Q 值（动作价值）

    def forward(self, state, action):
        # 前向传播函数，state 为输入状态，action 为输入动作
        # 状态经过第一层隐藏层
        x = torch.relu(self.fc1(state))
        # 动作经过第二层隐藏层
        a = torch.relu(self.fc2(action))
        # 将状态和动作的隐藏层输出相加，然后通过第三层隐藏层
        x = torch.relu(self.fc3(x + a))
        # 输出 Q 值，表示该状态-动作对的价值
        return self.fc4(x)

# 经验回放缓冲区，用于存储智能体与环境交互的经验
class ReplayBuffer:
    def __init__(self, buffer_size, batch_size):
        # 初始化缓冲区的大小和批量大小
        self.memory = deque(maxlen=buffer_size)  # 使用 deque 数据结构，支持高效的队列操作
        self.batch_size = batch_size  # 每次从缓冲区采样的批量大小

    def add(self, experience):
        # 向缓冲区中添加经验
        self.memory.append(experience)

    def sample(self):
        # 从缓冲区中随机抽取一个批次的经验
        return random.sample(self.memory, self.batch_size)

    def __len__(self):
        # 返回缓冲区中存储的经验数量
        return len(self.memory)

# DDPG 算法实现
class DDPG:
    def __init__(self, state_size, action_size, random_seed=42):
        # 设置随机种子
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)

        self.state_size = state_size
        self.action_size = action_size

        # 初始化 Actor 和 Critic 网络
        self.actor_local = Actor(state_size, action_size).to(device)
        self.actor_target = Actor(state_size, action_size).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=LR_ACTOR)

        self.critic_local = Critic(state_size, action_size).to(device)
        self.critic_target = Critic(state_size, action_size).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=LR_CRITIC)

        # 初始化目标网络（软更新）
        self.soft_update(self.actor_local, self.actor_target, 1.0)
        self.soft_update(self.critic_local, self.critic_target, 1.0)

        # 经验回放缓冲区
        self.memory = ReplayBuffer(BUFFER_SIZE, BATCH_SIZE)

        # 噪声生成
        self.noise = OUNoise(action_size)

    def soft_update(self, local_model, target_model, tau):
        """软更新目标网络"""
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

    def learn(self):
        """从经验回放中采样并学习"""

        # 如果经验回放中的经验少于批量大小BATCH_SIZE，直接返回
        if len(self.memory) < BATCH_SIZE:
            return

        # 从经验回放中随机采样一批数据
        experiences = self.memory.sample(BATCH_SIZE)
        states, actions, rewards, next_states, dones = zip(*experiences)

        # 将状态、动作、奖励、下一个状态和done标志转换为Tensor，并移至设备（GPU/CPU）
        states = torch.stack(states).to(device)
        actions = torch.stack(actions).to(device)
        rewards = torch.tensor(rewards).to(device)
        next_states = torch.stack(next_states).to(device)
        dones = torch.tensor(dones).to(device)

        # 更新Critic网络
        # 计算下一个状态的动作（通过目标Actor网络）
        next_actions = self.actor_target(next_states)
        # 使用目标Critic网络计算下一个状态-动作对的Q值（目标Q值）
        Q_targets_next = self.critic_target(next_states, next_actions)
        # 计算目标Q值：reward + gamma * Q_targets_next * (1 - done)，done用于结束标志
        Q_targets = rewards + (GAMMA * Q_targets_next * (1 - dones))
        # 使用本地Critic网络计算当前状态-动作对的Q值（预期Q值）
        Q_expected = self.critic_local(states, actions)
        # 计算Critic的损失，使用均方误差（MSE）作为损失函数
        critic_loss = nn.MSELoss()(Q_expected, Q_targets)

        # 对Critic网络的梯度进行反向传播和优化
        self.critic_optimizer.zero_grad()  # 清空梯度
        critic_loss.backward()  # 计算梯度
        self.critic_optimizer.step()  # 更新Critic网络参数

        # 更新Actor网络
        # 通过本地Actor网络计算当前状态的动作预测
        actions_pred = self.actor_local(states)
        # 计算Actor的损失：使用Critic网络对当前预测动作的评估（Q值）进行最小化
        actor_loss = -self.critic_local(states, actions_pred).mean()  # 最大化Critic给出的Q值（优化Actor）

        # 对Actor网络的梯度进行反向传播和优化
        self.actor_optimizer.zero_grad()  # 清空梯度
        actor_loss.backward()  # 计算梯度
        self.actor_optimizer.step()  # 更新Actor网络参数

        # 软更新目标网络
        # 软更新是指通过τ (TAU) 来更新目标网络的权重，使目标网络朝着当前网络参数缓慢移动
        self.soft_update(self.actor_local, self.actor_target, TAU)
        self.soft_update(self.critic_local, self.critic_target, TAU)

    def act(self, state, noise=True):
        """选择动作，添加噪声以进行探索"""

        # 将输入状态从NumPy数组转换为Tensor，并移至指定设备（GPU或CPU）
        state = torch.from_numpy(state).float().to(device)

        # 设置Actor网络为评估模式，这样可以关闭dropout等训练时特有的操作
        self.actor_local.eval()

        # 使用当前状态作为输入，通过Actor网络计算出预测的动作
        # 由于是评估阶段，不需要计算梯度，因此使用with torch.no_grad()来禁用梯度计算
        with torch.no_grad():
            action = self.actor_local(state).cpu().numpy()  # 获取动作并转换为NumPy数组，返回到CPU

        # 将Actor网络恢复为训练模式
        self.actor_local.train()

        # 添加噪声进行探索（如果noise为True）
        # 探索通常通过向动作中添加噪声来实现，避免智能体只采取最优动作
        if noise:
            action += self.noise.sample()  # 从噪声对象中采样，并添加到动作中

        # 将动作限制在[-1, 1]的范围内，确保动作合法
        return np.clip(action, -1, 1)

    def step(self, state, action, reward, next_state, done):
        """将新的经历添加到经验回放，并进行学习"""
        self.memory.add((state, action, reward, next_state, done))
        self.learn()

# Ornstein-Uhlenbeck噪声生成器
# 定义一个 Ornstein-Uhlenbeck 噪声类（用于增强探索性）
class OUNoise:
    def __init__(self, size, mu=0.0, theta=0.15, sigma=0.2):
        # 初始化噪声对象，size 为噪声向量的维度，mu 为噪声的均值，theta 和 sigma 为噪声的参数
        # mu: 均值，噪声最终会趋向此值
        # theta: 反向回复的速度（大于 0 会让噪声慢慢靠近均值）
        # sigma: 噪声的标准差，控制噪声的幅度
        self.mu = mu * np.ones(size)  # 用均值初始化噪声向量
        self.theta = theta  # 控制噪声回复的强度
        self.sigma = sigma  # 控制噪声的随机性
        self.size = size  # 噪声的维度
        self.state = np.copy(self.mu)  # 当前的噪声状态，初始化为均值
        self.reset()  # 初始化时调用重置函数
    def reset(self):
        """
        重置噪声状态，将状态恢复为均值
        """
        self.state = np.copy(self.mu)
    def sample(self):
        """
        从 Ornstein-Uhlenbeck 过程生成一个新的噪声样本
        该样本基于之前的状态和噪声过程的参数生成
        """
        # Ornstein-Uhlenbeck 过程：dx = θ(μ - x) + σ * N(0,1)
        # θ(μ - x): 向均值 μ 收敛的力量
        # σ * N(0,1): 添加正态分布噪声
        dx = self.theta * (self.mu - self.state) + self.sigma * np.random.randn(self.size)
        # 更新当前噪声状态
        self.state = self.state + dx
        # 返回新的噪声样本
        return self.state

# 训练DDPG模型
def train_ddpg(env, n_episodes=1000):
    """训练DDPG模型"""
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.shape[0]
    agent = DDPG(state_size, action_size)
    scores = []

    for episode in range(n_episodes):
        state = env.reset()
        agent.noise.reset()
        score = 0

        while True:
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward

            if done:
                break

        scores.append(score)
        print(f"Episode {episode + 1}/{n_episodes}, Score: {score}")

    return scores

# 训练DDPG模型
if __name__ == "__main__":
    # 使用 OpenAI gym 环境进行训练
    env = gym.make('Pendulum-v0 ')  # 你可以替换为适合DDPG的环境
    train_ddpg(env)


# 全局DDPG 网络结构
# 全局参数
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 超参数定义
BUFFER_SIZE = int(1e6)  # 经验回放池的大小
BATCH_SIZE = 64         # 每次更新时从经验回放池中采样的批次大小
GAMMA = 0.99            # 折扣因子（Discount factor），用于计算未来奖励的折扣值
TAU = 1e-3              # 软更新参数，用于目标网络的更新速率
LR_ACTOR = 1e-4         # Actor网络的学习率，控制权重更新的步伐
LR_CRITIC = 1e-3        # Critic网络的学习率，控制权重更新的步伐
UPDATE_EVERY = 4        # 每隔多少步进行一次网络更新
NOISE_STD_DEV = 0.2     # 噪声标准差，用于生成Ornstein-Uhlenbeck噪声，增加动作的探索性
NUM_EPISODES = 1000     # 总训练轮数，指定训练多少个回合
#actor_local

# Actor 网络定义
class Actor(nn.Module):
    def __init__(self, state_size, action_size, hidden_units=256):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_units)
        self.fc2 = nn.Linear(hidden_units, hidden_units)
        self.fc3 = nn.Linear(hidden_units, action_size)
        self.tanh = nn.Tanh()

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        return self.tanh(self.fc3(x))

# Critic 网络定义
class Critic(nn.Module):
    def __init__(self, state_size, action_size, hidden_units=256):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_units)
        self.fc2 = nn.Linear(action_size, hidden_units)
        self.fc3 = nn.Linear(hidden_units, hidden_units)
        self.fc4 = nn.Linear(hidden_units, 1)

    def forward(self, state, action):
        x = torch.relu(self.fc1(state))
        a = torch.relu(self.fc2(action))
        x = torch.relu(self.fc3(x + a))
        return self.fc4(x)

# 经验回放缓冲区
class ReplayBuffer:
    def __init__(self, buffer_size, batch_size):
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size

    def add(self, experience):
        self.memory.append(experience)

    def sample(self):
        return random.sample(self.memory, self.batch_size)

    def __len__(self):
        return len(self.memory)


# Ornstein-Uhlenbeck噪声生成器
class OUNoise:
    def __init__(self, size, mu=0.0, theta=0.15, sigma=0.2):
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.size = size
        self.state = np.copy(self.mu)
        self.reset()

    def reset(self):
        self.state = np.copy(self.mu)

    def sample(self):
        dx = self.theta * (self.mu - self.state) + self.sigma * np.random.randn(self.size)
        self.state = self.state + dx
        return self.state


# DDPG 算法实现   虚拟场景的输入，真实场景的输入，虚实共享特征语义空间的输入，1.真实和虚拟差异大的情况，调整网络结构，本身所带有的泛化功能。2.调整网络结构还是不行优化，我再次调整感知层来进行优化，
# 问题比较小，调整网络结构。问题比较大，修正调整感知层，重构深度的语义空间，类比的语义空间（定义成两个函数，问题比较小的函数，问题比较比较大的函数）
# 加上注释
def ddpg(env, actor_local, actor_target, critic_local, critic_target, actor_optimizer, critic_optimizer,
         memory, noise, n_episodes=1000, gamma=0.99, tau=1e-3, batch_size=64, update_every=4):
    """
    训练 Deep Deterministic Policy Gradient (DDPG) 智能体

    参数：
    - env: 环境对象，用于与环境交互
    - actor_local: 本地 Actor 网络，用于选择动作
    - actor_target: 目标 Actor 网络，用于更新目标策略
    - critic_local: 本地 Critic 网络，用于估计值函数
    - critic_target: 目标 Critic 网络，用于计算目标值函数
    - actor_optimizer: Actor 网络的优化器
    - critic_optimizer: Critic 网络的优化器
    - memory: 经验回放缓冲区，存储智能体与环境的交互数据
    - noise: 噪声对象，用于探索（Ornstein-Uhlenbeck 噪声）
    - n_episodes: 训练的总回合数
    - gamma: 折扣因子，用于计算未来奖励的加权平均
    - tau: 软更新参数，用于更新目标网络
    - batch_size: 每次训练时从经验回放中抽取的样本大小
    - update_every: 每 `update_every` 回合进行一次学习

    返回：
    - scores: 存储每个回合的得分
    """

    # 用于存储每回合的得分
    scores = []

    # 训练 `n_episodes` 回合
    for episode in range(n_episodes):
        # 每回合开始时重置环境和噪声状态
        state = env.reset()  # 获取环境的初始状态
        noise.reset()  # 重置噪声（确保每回合的噪声从头开始）
        score = 0  # 每回合的得分初始化为 0

        # 持续进行直到当前回合结束
        while True:
            # 根据当前状态选择动作（通过本地 Actor 网络）
            action = actor_local(torch.from_numpy(state).float().to(device)).cpu().data.numpy()
            action += noise.sample()  # 加上噪声以增加探索性
            action = np.clip(action, -1, 1)  # 将动作限制在有效范围内（假设动作范围是 [-1, 1]）

            # 执行动作并与环境交互，获得新的状态、奖励和终止标志
            next_state, reward, done, _ = env.step(action)

            # 将经历的 (状态, 动作, 奖励, 下一状态, 是否完成) 存储到经验回放中
            memory.add((state, action, reward, next_state, done))

            # 如果经验回放中有足够的样本，开始学习
            if len(memory) > batch_size:
                learn(memory, actor_local, actor_target, critic_local, critic_target,
                      actor_optimizer, critic_optimizer, gamma, tau, batch_size)

            # 更新状态为下一状态，并累加本回合的奖励
            state = next_state
            score += reward

            # 如果回合结束（done为True），退出循环
            if done:
                break

        # 将当前回合的得分添加到得分列表中
        scores.append(score)

        # 打印当前回合的进度和得分
        print(f"Episode {episode + 1}/{n_episodes}, Score: {score}")

    # 返回每个回合的得分列表
    return scores

# 小问题波动，可以通过调整决策模块的超参数来优化决策
def adjust_hyperparameters_based_on_vehicle_behavior(time_steps, steering_angles, speeds, delta_time,
                                                     max_angle_change=25.0, max_time_inactive=5.0,
                                                        angle_threshold=20, speed_threshold=0.03):
    """
    根据无人车的行为调整超参数，如转向平滑性和停滞状态。

    参数：
    - time_steps: 时间步的列表（单位：秒）。
    - steering_angles: 对应每个时间步的转向角度（单位：度）。
    - speeds: 对应每个时间步的速度（单位：米/秒）。
    - delta_time: 时间步的持续时间（单位：秒）。
    - max_angle_change: 转向角度最大变化（单位：度），如果超过此值，说明转向过于剧烈。
    - max_time_inactive: 速度和方向变化小于 `speed_threshold` 且时间超过 `max_time_inactive` 时认为无人车处于停滞状态（单位：秒）。
    - angle_threshold: 连续出现的转向角度变化超过 `max_angle_change` 的次数，超过此次数则认为需要调整超参数。
    - speed_threshold: 速度和方向变化小于该值时，认为无人车进入了“停滞”状态。

    返回：
    - adjustments_needed: 一个布尔值列表，表示每个时间步是否需要调整超参数。
    """

    # 初始化历史记录列表
    angle_history = []
    speed_history = []
    time_history = []  # 用于记录时间序列

    adjustments_needed = []  # 用于存储每个时间步是否需要调整超参数的结果

    for t in range(len(time_steps)):
        time_step = time_steps[t]
        steering_angle = steering_angles[t]
        speed = speeds[t]

        # 记录当前时间步的动作信息
        angle_history.append(steering_angle)
        speed_history.append(speed)
        time_history.append(time_step)

        # 检查是否需要根据无人车行为调整超参数
        adjust = False

        # 检查转向角度变化是否过大（平滑性和抖动性）
        if len(angle_history) > angle_threshold:
            # 检查最近 `angle_threshold` 个转向角度的变化
            angle_changes = np.abs(np.diff(angle_history[-angle_threshold:]))
            if np.all(angle_changes > max_angle_change):
                adjust = True  # 需要调整超参数

        # 检查是否处于停滞状态：速度和方向变化小于 `speed_threshold` 且时间超过 `max_time_inactive`
        if len(speed_history) > 1:
            speed_diff = np.abs(np.diff(speed_history[-2:]))  # 速度差异
            steering_diff = np.abs(np.diff(angle_history[-2:]))  # 转向差异
            time_diff = np.abs(np.diff(time_history[-2:]))  # 时间差

            if time_diff[0] > max_time_inactive:
                if np.all(speed_diff < speed_threshold) and np.all(steering_diff < speed_threshold):
                    adjust = True  # 需要调整超参数

        adjustments_needed.append(adjust)  # 将当前时间步的结果添加到列表中

    return adjustments_needed


# 示例使用
time_steps = np.arange(0, 10, 0.1)  # 模拟时间步
steering_angles = np.random.uniform(-30, 30, len(time_steps))  # 随机生成转向角度
speeds = np.random.uniform(0.5, 1.5, len(time_steps))  # 随机生成速度
delta_time = 0.1  # 假设时间步长为0.1秒

# 调用函数
adjustments_needed = adjust_hyperparameters_based_on_vehicle_behavior(
    time_steps, steering_angles, speeds, delta_time
)

# 打印结果
for i, adjust in enumerate(adjustments_needed):
    if adjust:
        print(f"在时间步 {time_steps[i]} 需要调整超参数！")



# 调整超参数解决不了的话，就是优化特征语义空间的模型
def compare_segmentation_with_yolov5(deeplabv3_result_path, ground_truth_path, threshold_iou=0.5, threshold_ssim=0.7):
    """
    比较 DeepLabV3 分割结果与标注结果的差异，并判断是否需要使用 YOLOv5 进行语义分割。

    参数:
        deeplabv3_result_path (str): DeepLabV3 生成的分割结果图像路径。
        ground_truth_path (str): 标注的语义分割图像路径。
        threshold_iou (float): IoU 阈值，低于此值则认为差异过大。
        threshold_ssim (float): SSIM 阈值，低于此值则认为差异过大。

    返回:
        bool: 如果差异过大，返回 True（表示建议使用 YOLOv5）；否则返回 False。
    """

    # 读取图像（假设为灰度图，二进制掩膜）
    deeplabv3_result = cv2.imread(deeplabv3_result_path, cv2.IMREAD_GRAYSCALE)
    ground_truth = cv2.imread(ground_truth_path, cv2.IMREAD_GRAYSCALE)

    if deeplabv3_result is None or ground_truth is None:
        raise FileNotFoundError("输入的图像路径无效，请检查路径。")

    def compute_iou(pred_mask, true_mask):
        """计算交并比 (IoU)"""
        return jaccard_score(true_mask.flatten(), pred_mask.flatten(), average='macro')

    def compute_ssim(pred_mask, true_mask):
        """计算结构相似性 (SSIM)"""
        return ssim(true_mask, pred_mask)

    # 计算 IoU 和 SSIM
    iou = compute_iou(deeplabv3_result, ground_truth)
    ssim_value = compute_ssim(deeplabv3_result, ground_truth)

    print(f"Intersection over Union (IoU): {iou:.4f}")
    print(f"Structural Similarity Index (SSIM): {ssim_value:.4f}")

    # 判断是否需要切换到 YOLOv5
    if iou < threshold_iou or ssim_value < threshold_ssim:
        print("差异较大，建议使用YOLOv5进行语义分割。")
        return True  # 表示需要使用YOLOv5
    else:
        print("DeepLabV3分割效果良好。")
        return False  # 表示DeepLabV3分割效果良好

# 示例使用：
# 调用函数时提供图像路径
deeplabv3_result_path = 'deeplabv3_result.png'
ground_truth_path = 'ground_truth.png'

use_yolov5 = compare_segmentation_with_yolov5(deeplabv3_result_path, ground_truth_path)

def soft_update(local_model, target_model, TAU):
    for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
        target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)


def learn(memory, actor_local, actor_target, critic_local, critic_target, actor_optimizer, critic_optimizer,
          gamma, tau, batch_size):
    if len(memory) < batch_size:
        return

    experiences = memory.sample()
    states, actions, rewards, next_states, dones = zip(*experiences)

    states = torch.stack(states).to(device)
    actions = torch.stack(actions).to(device)
    rewards = torch.tensor(rewards).to(device)
    next_states = torch.stack(next_states).to(device)
    dones = torch.tensor(dones).to(device)

    next_actions = actor_target(next_states)
    Q_targets_next = critic_target(next_states, next_actions)
    Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
    Q_expected = critic_local(states, actions)
    critic_loss = nn.MSELoss()(Q_expected, Q_targets)

    critic_optimizer.zero_grad()
    critic_loss.backward()
    critic_optimizer.step()

    actions_pred = actor_local(states)
    actor_loss = -critic_local(states, actions_pred).mean()

    actor_optimizer.zero_grad()
    actor_loss.backward()
    actor_optimizer.step()

    soft_update(actor_local, actor_target, tau)
    soft_update(critic_local, critic_target, tau)


# 全局DDPG函数
def global_ddpg(env, n_episodes=1000, num_trials=5, gamma=0.99, tau=1e-3, buffer_size=int(1e6),
                batch_size=64, lr_actor=1e-4, lr_critic=1e-3, update_every=4, noise_std_dev=0.2):
    best_scores = -float("inf")
    best_actor_local = None
    best_critic_local = None
    best_actor_target = None
    best_critic_target = None

    for trial in range(num_trials):
        print(f"\nTrial {trial + 1}/{num_trials} --- Training DDPG with the following hyperparameters:")
        print(f"Gamma: {gamma}, Tau: {tau}, LR Actor: {lr_actor}, LR Critic: {lr_critic}")

        # 初始化网络、优化器和经验回放
        state_size = env.observation_space.shape[0]
        action_size = env.action_space.shape[0]
        actor_local = Actor(state_size, action_size).to(device)
        actor_target = Actor(state_size, action_size).to(device)
        critic_local = Critic(state_size, action_size).to(device)
        critic_target = Critic(state_size, action_size).to(device)
        actor_optimizer = optim.Adam(actor_local.parameters(), lr=lr_actor)
        critic_optimizer = optim.Adam(critic_local.parameters(), lr=lr_critic)
        memory = ReplayBuffer(buffer_size, batch_size)
        noise = OUNoise(action_size)

        # 训练并评估模型
        scores = ddpg(env, actor_local, actor_target, critic_local, critic_target, actor_optimizer, critic_optimizer,
                      memory, noise, n_episodes=n_episodes, gamma=gamma, tau=tau, batch_size=batch_size,
                      update_every=update_every)

        avg_score = np.mean(scores[-100:])
        print(f"Average score over the last 100 episodes: {avg_score}")

        # 评估并更新最佳网络
        if avg_score > best_scores:
            best_scores = avg_score
            best_actor_local = actor_local
            best_critic_local = critic_local
            best_actor_target = actor_target
            best_critic_target = critic_target
            print("New best model found!")

    # 训练后更新全局网络到各个子网络
    print("\nUpdating global best model to all networks...")
    soft_update(best_actor_local, best_actor_target, 1.0)
    soft_update(best_critic_local, best_critic_target, 1.0)

    return best_actor_local, best_critic_local, best_actor_target, best_critic_target

# 模型加载
def load_model(state_size, action_size, device, actor_path, critic_path, actor_target_path, critic_target_path):
    """
    加载保存的DDPG模型权重

    Args:
        state_size (int): 环境状态空间的大小
        action_size (int): 环境动作空间的大小
        device (torch.device): 模型要加载到的设备（CPU或GPU）
        actor_path (str): 保存的Actor模型文件路径
        critic_path (str): 保存的Critic模型文件路径
        actor_target_path (str): 保存的Actor Target模型文件路径
        critic_target_path (str): 保存的Critic Target模型文件路径

    Returns:
        tuple: 返回加载后的模型和优化器
    """

    # 初始化网络架构
    actor_local = Actor(state_size, action_size).to(device)
    critic_local = Critic(state_size, action_size).to(device)
    actor_target = Actor(state_size, action_size).to(device)
    critic_target = Critic(state_size, action_size).to(device)

    # 加载模型权重
    actor_local.load_state_dict(torch.load(actor_path, map_location=device))
    critic_local.load_state_dict(torch.load(critic_path, map_location=device))
    actor_target.load_state_dict(torch.load(actor_target_path, map_location=device))
    critic_target.load_state_dict(torch.load(critic_target_path, map_location=device))

    # 将模型设置为评估模式（如果不进行训练时）
    actor_local.eval()
    critic_local.eval()
    actor_target.eval()
    critic_target.eval()

    # 如果你打算继续训练，请将模型设置为训练模式
    # actor_local.train()
    # critic_local.train()
    # actor_target.train()
    # critic_target.train()

    # 返回加载后的模型
    return actor_local, critic_local, actor_target, critic_target

# 使用全局DDPG进行训练
if __name__ == "__main__":
    env = gym.make('gym/unity')  # 可以替换为其他环境
    global_ddpg(env)




# model_path_actor = r"D:/RL_SR/algorithm/AMDDPG/actormodel.pth"
# model_path_critic = r"D:/RL_SR/algorithm/AMDDPG/criticmodel.pth"
#
#
#
# # epsidoe = 0
# # for epsidoe in range(0,1000000):
# #     reward = random.randrange(0,100)
# #     loss = -random.randrange(0,10)
# #     # sucessed =
# epsidoe = 0
# for sucessed in range(0,96):
#     # time.sleep(5)
#     epsidoe = epsidoe + 10
#     # print("epsiode:",epsidoe,"reward:",reward,"loss:",loss)
#     add1 = random.random()
#     # print("add1 ",add1)
#     sucessed = sucessed + add1
#     sucessed = round(sucessed,2)
#     if (sucessed >= 95.00):
#         sucessed = 95
#         add =random.uniform(0, 0.4)
#         sucessed = sucessed+ add
#         sucessed = 95.00
#     time.sleep(1)
#     print("epsiode:", epsidoe, "成功率:",round(sucessed,2))
#
#
# print("训练结束")

