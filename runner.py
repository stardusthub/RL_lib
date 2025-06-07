from dqn import DQN
from env_wrapper import FrameSkipEnv
from common import FramePreprocessor
import gymnasium as gym
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import datetime
import torch
import os
# 1. 让日志文件夹名字里有当前日期时间，格式：YYYYMMDD-HHMMSS
start_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
print(start_time)
base_log_dir =  f"runs/DQN-logs{start_time}"


D=84*84

class Runner:
    def __init__(self, env_name):
        self.env = gym.make(f"{env_name}")
        self.num_actions = self.env.action_space.n
        self.env = FrameSkipEnv(self.env, skip=4)
        self.agent = DQN(D, self.num_actions)
        self.writer = SummaryWriter(base_log_dir)
        self.args = self.agent.args
    def train(self,steps):
        save_interval = self.agent.save_interval
        current_frame, info = self.env.reset()
        processor = FramePreprocessor(m=1)
        #这里使用了使用差分帧
        obs = processor.process(current_frame)
        obs=np.zeros(D)
        obs=torch.from_numpy(obs).float()
        total_reward = []
        episode_reward = 0
        episode=0
        pre_obs = obs
        for step in range(steps):
            action = self.agent.select_action(obs, step)
            next_frame, reward, done, info = self.env.step(action)
            cur_obs = processor.process(next_frame)
            next_obs = cur_obs-pre_obs
            pre_obs = cur_obs
            self.agent.buffer.push(obs, action, reward, next_obs, done)
            obs=next_obs
            episode_reward += reward

            if len(self.agent.buffer) > 10000 :
                batch = self.agent.buffer.sample(self.args["batch_size"])
                self.agent.update(step,batch)

            if done:
                episode+=1
                total_reward.append(episode_reward)
                np.savetxt("episode_reward.txt", total_reward, fmt='%d')
                print(f"step {step}/{steps} | Total Reward: {episode_reward} | episode: {episode}")
                episode_reward=0
                current_frame, info = self.env.reset()
                obs = processor.process(current_frame)
                processor.reset()
                obs = torch.from_numpy(np.zeros(D)).float()
            if step % save_interval == 0:
                self.agent.save_model("dqn_model")

    def test(self,episode):
        env = gym.make("PongNoFrameskip-v4",render_mode="human")
        num_actions = env.action_space.n
        env = FrameSkipEnv(env, skip=4)
        agent = DQN(num_actions)
        agent.load_model("dqn_model")

        current_frame, info = env.reset()
        processor = FramePreprocessor(m=4)
        obs = processor.prepro(current_frame)


        total_reward = 0
        for i in range(episode):
            action = agent.select_action(obs, 0, test_mode=True)
            print(action)
            next_frame, reward, done, info = env.step(action)
            next_obs = processor.process(next_frame)
            obs = next_obs-obs
            total_reward += reward

            if done:
                print(f"Total Reward: {total_reward}")
                current_frame, info = env.reset()
                obs = processor.prepro(current_frame)
                obs = np.zeros(84 * 84)
                processor.reset()
                total_reward = 0


if __name__ == "__main__":
  runner = Runner(env_name="PongNoFrameskip-v4")
  runner.train(1000000)

