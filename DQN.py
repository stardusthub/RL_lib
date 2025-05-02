import torch.nn as nn
import torch.nn.functional as F
import gym
import torch
import numpy as np
import cv2
import collections
import random
import os.path as osp
import os
from pathlib import Path
from os.path import exists
import yaml

# ---- 1. ε-贪婪调度器 ----
def check(value):
    """Check if value is a numpy array, if so, convert it to a torch tensor."""
    output = torch.from_numpy(value) if isinstance(value, np.ndarray) else value
    return output
class EpsilonScheduler:
    def __init__(self, eps_start=1.0, eps_final=0.1, decay_frames=1_000_000):
        self.eps_start = eps_start
        self.eps_final = eps_final
        self.decay_frames = decay_frames
    def get_epsilon(self, frame_idx):
        # 线性衰减
        if frame_idx < self.decay_frames:
            eps = self.eps_start - (self.eps_start - self.eps_final) * (frame_idx / self.decay_frames)
        else:
            eps = self.eps_final
        return eps

# ---- 2. Replay Memory ----
class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = collections.deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        return batch

    def __len__(self):
        return len(self.buffer)

# ---- 3. Frame Skipping Wrapper ----
class FrameSkipEnv:
    def __init__(self, env, skip=4):
        self.env = env
        self.skip = skip

    def reset(self):
        return self.env.reset()
    def render(self, render_mode):
        return self.env.render(render_mode=render_mode)

    def step(self, action):
        total_reward = 0.0
        done = False
        for i in range(self.skip):
            state, reward, terminal,truncated, info = self.env.step(action)
            total_reward += reward
            if done:
                break
        return state, reward, terminal,truncated, info

class FramePreprocessor:
    def __init__(self, m=4):
        """
        Atari帧预处理类
        :param m: 帧堆叠数量，默认4帧堆叠
        """
        self.m = m
        self.frame_buffer = []  # 存储历史帧的缓冲区
        self.previous_raw_frame = None  # 保存原始帧用于最大值堆叠

    def process(self, current_frame):
        """
        处理当前帧并返回网络输入张量
        :param current_frame: 当前原始帧 (210, 160, 3)
        :return: 处理后的张量 ( 4, 84, 84)
        """
        # 1.最大值堆叠处理
        if self.previous_raw_frame is None:
            processed_frame = current_frame
        else:
            processed_frame = np.maximum(current_frame, self.previous_raw_frame)

        # 保存当前原始帧供下次使用
        self.previous_raw_frame = current_frame.copy()

        # 2.颜色空间转换+Y通道提取
        y_channel = cv2.cvtColor(processed_frame, cv2.COLOR_RGB2YUV)[:, :, 0]

        # 3.下采样至(84,84)
        resized_frame = cv2.resize(y_channel, (84, 84), interpolation=cv2.INTER_LINEAR)

        # 4.帧缓冲区管理
        if not self.frame_buffer:
            # 首次调用时用当前帧填充整个缓冲区
            self.frame_buffer = [resized_frame] * self.m
        else:
            self.frame_buffer.append(resized_frame)
            self.frame_buffer = self.frame_buffer[-self.m:]  # 保持长度

        # 5.维度重组 (84,84,4)
        input_stack = np.stack(self.frame_buffer, axis=-1)
        input_tensor = torch.from_numpy(input_stack).float()
        return input_tensor.permute(2, 0, 1)

    def reset(self):
        """重置状态，用于新游戏回合"""
        self.frame_buffer = []
        self.previous_raw_frame = None


class DQN_CNN_NETWORK(nn.Module):
    def __init__(self, num_actions):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(4, 32, 8, 4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions))

    def forward(self, x):
        return self.net(x)

class DQN():
    def __init__(self,num_actions):
        al_config = self.load_state_config("DQN")
        self.num_actions=num_actions
        self.buffer=ReplayMemory(100000)
        self.epsilon_scheduler = EpsilonScheduler()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.Q=DQN_CNN_NETWORK(num_actions).to(self.device)
        self.Q_target=DQN_CNN_NETWORK(num_actions).to(self.device)
        self.optimizer = torch.optim.Adam(self.Q.parameters(), lr=al_config["lr"])
        self.gamma=al_config["gamma"]
    def load_state_config(self, algorithm):
        current_dir = osp.dirname(osp.abspath(__file__))
        state_config_path = (
            Path(current_dir)
            / f"{algorithm}.yaml"
        )
        with open(str(state_config_path), "r", encoding="utf-8") as file:
            state_config = yaml.load(file, Loader=yaml.FullLoader)
        return state_config

    def select_action(self, obs,frame_idx):
        epsilon=self.epsilon_scheduler.get_epsilon(frame_idx)
        if random.random() < epsilon:
            action = random.randint(0, self.num_actions - 1)
        else:
            obs = check(obs).unsqueeze(0).float().to(self.device)
            with torch.no_grad():
                q_values = self.Q(obs)
            action = q_values.max(1)[1].item()
        return action

    def update(self, batch):
        state, action, reward, next_state, done = zip(*batch)
        state = torch.stack(state).to(self.device)
        next_state = torch.stack(next_state).to(self.device)
        action = [torch.tensor(a) if not isinstance(a, torch.Tensor) else a for a in action]
        action = torch.stack(action).to(self.device)
        reward = torch.stack([torch.tensor(r) for r in reward]).to(self.device)
        done = torch.stack([torch.tensor(d) for d in done]).to(self.device)

        # 计算 Q 值
        q_values = self.Q(state)
        q_value = q_values.gather(1, action.unsqueeze(1)).squeeze(1)

        # 计算目标 Q 值
        with torch.no_grad():
            next_q_values = self.Q_target(next_state)
            max_next_q_value = next_q_values.max(1)[0]
            target_q_value = reward + (1 - done.float()) * self.gamma * max_next_q_value


        # 计算损失
        loss = F.mse_loss(q_value, target_q_value)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # 更新目标网络
        for target_param, param in zip(self.Q_target.parameters(), self.Q.parameters()):
            target_param.data.copy_(0.99 * target_param.data + 0.01 * param.data)

    def save_model(self, model_path):
        """保存模型参数"""
        if not exists(model_path):
            os.makedirs(model_path)
        torch.save(self.Q.state_dict(), os.path.join(model_path, "Q.pth"))
        torch.save(self.Q_target.state_dict(), os.path.join(model_path, "Q_target.pth"))
        print(f"Model saved to {model_path}")
    def load_model(self, model_path):
        """加载模型参数"""
        self.Q.load_state_dict(torch.load(model_path))
        self.Q_target.load_state_dict(torch.load(model_path.replace("Q", "Q_target")))
def train(num_episodes,batch_size):
    env = gym.make("Pong-v4")
    num_actions=env.action_space.n
    env = FrameSkipEnv(env, skip=4)
    agent=DQN(num_actions)
    current_frame, info = env.reset()
    processor = FramePreprocessor(m=4)
    obs = processor.process(current_frame)
    for i in range(num_episodes):
        action=agent.select_action(obs, i)
        action=check(action)
        print(i)
        next_frame, reward, terminal,truncate,info = env.step(action)
        done=terminal or truncate
        next_obs = processor.process(next_frame)
        agent.buffer.push(obs, action, reward, next_obs, done)
        obs = next_obs
        if len(agent.buffer) > 5000:
            batch = agent.buffer.sample(batch_size)
            agent.update(batch)
        if done:
            current_frame, info = env.reset()
            obs = processor.process(current_frame)
            processor.reset()
    agent.save_model("model.pth")
def test():
    env = gym.make("Pong-v4")
    env = FrameSkipEnv(env, skip=4)
    agent=DQN(env.env.action_space.n)
    agent.load_model("model.pth")
    current_frame, info = env.reset()
    processor = FramePreprocessor(m=4)
    obs = processor.process(current_frame, m=4)
    while True:
        action=agent.select_action(obs, 0)
        action=check(action)
        next_frame, reward, terminated,truncate,info = env.step(action)
        env.render(render_mode="human")
        done=terminated or truncate
        obs = processor.process(next_frame)
        if done:
            current_frame, info = env.reset()
            obs = processor.process(current_frame)
            processor.reset()


if __name__ == "__main__":
    num_episodes = 1000000
    batch_size = 32
    train(num_episodes, batch_size)
    test()