import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import gym
import cv2
from collections import deque, namedtuple
from dataclasses import dataclass
import random


# ----------------------
# 数据结构定义
# ----------------------
@dataclass
class Transition:
    state: np.ndarray  # (84,84,4)
    action: int
    reward: float
    next_state: np.ndarray
    done: bool


# ----------------------
# 环境封装
# ----------------------
class FrameSkipEnv(gym.Wrapper):
    def __init__(self, env, skip=4):
        super().__init__(env)
        self._skip = skip

    def step(self, action):
        total_reward = 0.0
        for _ in range(self._skip):
            state, reward, done, truncated, info = self.env.step(action)
            total_reward += reward
            if done or truncated:
                break
        return state, total_reward, done, truncated, info


# ----------------------
# 预处理模块
# ----------------------
class FrameProcessor:
    def __init__(self, frame_stack=4):
        self.frame_stack = frame_stack
        self._buffer = deque(maxlen=frame_stack)
        self._last_frame = None

    def process(self, frame):
        # 最大值堆叠
        if self._last_frame is not None:
            frame = np.maximum(frame, self._last_frame)
        self._last_frame = frame

        # 颜色空间转换
        yuv = cv2.cvtColor(frame, cv2.COLOR_RGB2YUV)
        y_channel = yuv[..., 0]

        # 下采样
        resized = cv2.resize(y_channel, (84, 84), interpolation=cv2.INTER_AREA)

        # 帧堆叠
        if len(self._buffer) == 0:
            self._buffer.extend([resized] * self.frame_stack)
        else:
            self._buffer.append(resized)

        return np.stack(self._buffer, axis=-1)  # (84,84,4)

    def reset(self):
        self._buffer.clear()
        self._last_frame = None


# ----------------------
# 神经网络
# ----------------------
class DQN(nn.Module):
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


# ----------------------
# Agent核心逻辑
# ----------------------
class DQNAgent:
    def __init__(self, num_actions, device):
        # 超参数设置
        self.batch_size = 32
        self.memory_size = 100000
        self.gamma = 0.99
        self.tau = 0.005

        # 设备配置
        self.device = device

        # 神经网络
        self.online_net = DQN(num_actions).to(device)
        self.target_net = DQN(num_actions).to(device)
        self.optimizer = optim.Adam(self.online_net.parameters(), lr=1e-4)

        # 经验回放
        self.memory = deque(maxlen=self.memory_size)

    def act(self, state, epsilon):
        if np.random.random() < epsilon:
            return np.random.randint(self.num_actions)

        state_t = torch.from_numpy(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.online_net(state_t)
        return q_values.argmax().item()

    def update(self):
        if len(self.memory) < self.batch_size:
            return

        # 从内存中采样
        transitions = random.sample(self.memory, self.batch_size)
        batch = Transition(*zip(*transitions))

        # 转换为张量
        state_batch = torch.tensor(np.array(batch.state), dtype=torch.float32, device=self.device).permute(0, 3, 1, 2)
        action_batch = torch.tensor(batch.action, dtype=torch.long, device=self.device)
        reward_batch = torch.tensor(batch.reward, dtype=torch.float32, device=self.device)
        next_state_batch = torch.tensor(np.array(batch.next_state), dtype=torch.float32, device=self.device).permute(0,                                                                                   3,
                                                                                                                     1,
                                                                                                                     2)
        done_batch = torch.tensor(batch.done, dtype=torch.float32, device=self.device)

        # 计算Q值
        q_values = self.online_net(state_batch).gather(1, action_batch.unsqueeze(1))

        # 计算目标值（双Q学习）
        with torch.no_grad():
            next_actions = self.online_net(next_state_batch).argmax(1)
            next_q = self.target_net(next_state_batch).gather(1, next_actions.unsqueeze(1))
            target_q = reward_batch + (1 - done_batch) * self.gamma * next_q.squeeze()

        # 计算损失
        loss = F.mse_loss(q_values.squeeze(), target_q)

        # 反向传播
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online_net.parameters(), 10)
        self.optimizer.step()

        # 软更新目标网络
        for t_param, o_param in zip(self.target_net.parameters(), self.online_net.parameters()):
            t_param.data.copy_(self.tau * o_param.data + (1 - self.tau) * t_param.data)


# ----------------------
# 训练流程
# ----------------------
def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env = gym.make("PongNoFrameskip-v4")
    env = FrameSkipEnv(env)

    processor = FrameProcessor()
    agent = DQNAgent(env.action_space.n, device)

    epsilon = 1.0
    eps_decay = 100000
    min_epsilon = 0.01

    state = processor.process(env.reset())

    for step in range(1, 1000001):
        # 选择动作
        action = agent.act(state, epsilon)

        # 执行动作
        next_frame, reward, done, _, _ = env.step(action)
        next_state = processor.process(next_frame)

        # 存储经验
        agent.memory.append(Transition(state, action, reward, next_state, done))

        # 更新状态
        state = next_state if not done else processor.process(env.reset())

        # 衰减epsilon
        epsilon = max(min_epsilon, 1.0 - step / eps_decay)

        # 更新网络
        agent.update()

        # 定期保存模型
        if step % 10000 == 0:
            torch.save(agent.online_net.state_dict(), f"dqn_{step}.pth")


if __name__ == "__main__":
    train()