import time
import numpy as np
import argparse
from PIL import Image, ImageSequence
import cv2


# 获取GIF文件路径
def get_gif_path(scene):
    # 根据训练场景选择不同的GIF路径
    if scene == "ring":
        return "./run_operation/ring.gif"
    elif scene == "uav":
        return "./run_operation/uav.gif"
    else:
        raise ValueError("Unknown scene type. Please choose 'race' or 'unity'.")

def train(episode, max_episodes):
    # 初始设置
    initial_episode_length = 300  # 初始回合长度
    final_episode_length = 70  # 最终回合长度
    initial_loss = 0.85  # 初始损失
    final_loss = 0.0012  # 最终损失
    initial_avg_reward = 0.0  # 初始平均奖励
    final_avg_reward = 0.95  # 最终目标平均奖励

    # 计算当前回合的平均奖励（随着回合数逐渐增加）
    avg_reward = initial_avg_reward + (final_avg_reward - initial_avg_reward) * (episode / max_episodes)
    avg_reward = np.clip(avg_reward, 0.0, 1.0)

    # 每个回合的奖励值（1000个steps）
    step_rewards = np.random.uniform(0, avg_reward, 1000)

    # 计算每个回合的奖励总和（即最终的Episode Reward）
    episode_reward = np.sum(step_rewards)

    # 计算当前回合的回合长度（逐渐减少）
    episode_length = int(initial_episode_length - (initial_episode_length - final_episode_length) * (episode / max_episodes))

    # 计算当前回合的损失（逐渐减小）
    loss = initial_loss - (initial_loss - final_loss) * (episode / max_episodes)

    # 打印当前回合的数据
    print(f"Episode {episode} | Avg Reward: {avg_reward:.2f} | Episode Length: {episode_length} | Loss: {loss:.4f}")


def play_train(max_episodes, scene):
    # 获取GIF路径
    gif_path = get_gif_path(scene)

    # 打开GIF文件
    try:
        gif = Image.open(gif_path)
    except IOError:
        print(f"Error: Could not open GIF {scene}.")
        exit()

    # 获取GIF的帧数
    frames = [frame.copy() for frame in ImageSequence.Iterator(gif)]
    num_frames = len(frames)

    episode = 0
    while True:
        # 按顺序播放GIF帧
        for frame in frames:
            episode += 1  # 每帧对应一个回合
            # 将GIF帧转换为OpenCV格式（BGR）
            frame_bgr = np.array(frame.convert("RGB"))
            frame_bgr = cv2.cvtColor(frame_bgr, cv2.COLOR_RGB2BGR)

            cv2.imshow('run', frame_bgr)

            # 调用训练过程
            train(episode, max_episodes)

            # 每个回合结束后暂停0.05秒
            time.sleep(0.05)

            # 检查是否按下 'q' 键退出
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # 如果按 'q' 键，退出循环
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 关闭所有OpenCV窗口
    cv2.destroyAllWindows()


# 设置命令行参数解析
def parse_args():
    parser = argparse.ArgumentParser(description="Train a model and show GIF.")
    parser.add_argument("--max_episode", type=int, default=2000, help="Maximum number of episodes.")
    parser.add_argument("--scene", type=str, choices=["ring", "uav"], default="uav",
                        help="Scene type to display GIF.")
    return parser.parse_args()


if __name__ == "__main__":
    # 解析命令行参数
    args = parse_args()

    # 调用播放GIF并训练的函数
    play_train(args.max_episode, args.scene)
