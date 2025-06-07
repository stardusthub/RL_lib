import math
import numpy as np
import cv2
import torch


class EpsilonScheduler:
    """
    ε-greedy 调度器，用于在训练过程中控制探索—利用权衡。

    :param args: 包含 ε 起始值、ε 最终值、衰减帧数和衰减类型的字典
        - epsilon_start (float): 初始 ε 值
        - epsilon_final (float): 最终 ε 值
        - epsilon_decay (int): 从初始到最终 ε 所需的帧数
        - epsilon_type (str): 衰减策略，"linear" 或 "exponential"
    """

    def __init__(self, args):
        self.eps_start = args["epsilon_start"]
        self.eps_final = args["epsilon_final"]
        self.decay_frames = args["epsilon_decay"]
        self.epsilon_type = args["epsilon_type"]

    def get_epsilon(self, frame_idx: int) -> float:
        """
        根据当前帧索引返回 ε 值。

        :param frame_idx: 当前帧编号
        :return: 线性或指数衰减后的 ε 值
        """
        if self.epsilon_type == "linear":
            if frame_idx < self.decay_frames:
                return self.eps_start - (self.eps_start - self.eps_final) * (frame_idx / self.decay_frames)
            return self.eps_final

        if self.epsilon_type == "exponential":
            return self.eps_final + (self.eps_final - self.eps_start) * math.exp(-frame_idx / self.decay_frames)

        raise ValueError(f"Unknown epsilon_type: {self.epsilon_type}")


class FramePreprocessor:
    """
    输入帧预处理器，支持帧差分、灰度转换、降采样和帧堆叠。

    :param m: 要堆叠的历史帧数量，默认 4
    """

    def __init__(self, m: int = 4):
        self.m = m
        self.frame_buffer = []             # 最近 m 帧的缓存
        self.previous_raw_frame = None     # 上一原始帧

    def process(self, current_frame: np.ndarray) -> torch.Tensor:
        """
        对 RGB 帧进行预处理并返回 (m*84*84,) 的扁平化张量。

        步骤：
          1. 帧差分：若有上一帧，则对素最大值合并
          2. 转为 YUV，并提取 Y 通道（亮度）
          3. 缩放至 84×84
          4. 更新帧缓冲区（FIFO）
          5. 将 m 帧沿深度堆叠，归一化到 [0,1]
          6. 扁平化为 (m*84*84,) 向量

        :param current_frame: 当前 RGB 原始帧，shape=(H,W,3)
        :return: 归一化并扁平化的 PyTorch 张量
        """
        # 1) 帧差分 / 最大值合并
        if self.previous_raw_frame is None:
            processed = current_frame
        else:
            processed = np.maximum(current_frame, self.previous_raw_frame)
        self.previous_raw_frame = current_frame.copy()

        # 2) 转为亮度图
        y = cv2.cvtColor(processed, cv2.COLOR_RGB2YUV)[:, :, 0]

        # 3) 缩放
        resized = cv2.resize(y, (84, 84), interpolation=cv2.INTER_LINEAR)

        # 4) 帧缓冲区管理
        if not self.frame_buffer:
            self.frame_buffer = [resized] * self.m
        else:
            self.frame_buffer.append(resized)
            self.frame_buffer = self.frame_buffer[-self.m:]

        # 5) 堆叠 + 归一化
        stack = np.stack(self.frame_buffer, axis=-1)
        tensor = torch.from_numpy(stack).float().div(255.0)

        # 6) 扁平化
        return tensor.flatten()

    def prepro(self, I: np.ndarray) -> torch.Tensor:
        """
        Atari Pong 专用预处理，将 210×160×3 uint8 帧裁剪、二值化、降采样为 80×80 的 1D 张量。

        :param I: 输入原始帧
        :return: 6400 维度的浮点张量
        """
        I = I[35:195]             # 裁剪上下
        I = I[::2, ::2, 0]        # 每隔一像素降采样并取单通道
        I[I == 144] = 0           # 背景1
        I[I == 109] = 0           # 背景2
        I[I != 0] = 1             # 其他前景部分
        return torch.from_numpy(I.astype(np.float32).ravel())

    def reset(self) -> None:
        """
        重置内部状态（帧缓冲与前一帧），用于环境重置时调用。
        """
        self.frame_buffer = []
        self.previous_raw_frame = None
