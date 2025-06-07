import torch.nn.functional as F
import torch
import random
import os.path as osp
import os
from pathlib import Path
from os.path import exists
import yaml
from net_work import CNN_NETWORK,MLP_Network
from common import EpsilonScheduler
from buffer import ReplayBuffer
import datetime
start_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")


class DQN:
    def __init__(self,input_size, num_actions):
        self.args = self.load_state_config("DQN")
        self.num_actions = num_actions
        self.save_interval = self.args["save_interval"]
        self.buffer = ReplayBuffer(self.args)
        self.epsilon_scheduler = EpsilonScheduler(self.args)
        if torch.cuda.is_available():
            print("use gpu")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.Q = MLP_Network(self.args,input_size,num_actions).to(self.device)
        self.Q_target = MLP_Network(self.args,input_size,num_actions).to(self.device)
        # self.Q = CNN_NETWORK(4,num_actions).to(self.device)
        # self.Q_target = CNN_NETWORK(4,num_actions).to(self.device)
        self.optimizer = torch.optim.Adam(self.Q.parameters(), lr=self.args["lr"])
        #原论文采用的优化器,实验结果表明，Adam优化器效果更好
        # self.optimizer = torch.optim.RMSprop(self.Q.parameters(), lr=al_config ["lr"], alpha=0.95, eps=0.01)
        self.gamma = self.args["gamma"]
        self.target_update_freq=self.args["target_update_freq"]

        # 同步目标网络
        self.Q_target.load_state_dict(self.Q.state_dict())
        #目标网络（Target Network）不是用来“学习”参数的，每次更新也只是从 Q 网络“复制”一份权重。调用eval()防止dropout输出噪声，防止Batch—norm导致被batch影响
        self.Q_target.eval()

    def load_state_config(self, algorithm):
        current_dir = osp.dirname(osp.abspath(__file__))
        state_config_path = Path(current_dir) / f"{algorithm}.yaml"
        with open(str(state_config_path), "r", encoding="utf-8") as file:
            state_config = yaml.load(file, Loader=yaml.FullLoader)
        return state_config

    def select_action(self, obs, frame_idx, test_mode=False):
        if test_mode:
            obs = torch.tensor(obs).unsqueeze(0).float().to(self.device)
            with torch.no_grad():
                q_values = self.Q(obs)
            return q_values.max(1)[1].item()

        epsilon = self.epsilon_scheduler.get_epsilon(frame_idx)
        if random.random() < epsilon:
            return random.randint(0, self.num_actions - 1)
        else:
            obs = torch.tensor(obs).unsqueeze(0).float().to(self.device)
            with torch.no_grad():
                q_values = self.Q(obs)
            return q_values.max(1)[1].item()

    def update(self,step,batch):
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.stack(states).to(self.device)
        next_states = torch.stack(next_states).to(self.device)
        actions = torch.tensor(actions, dtype=torch.int64).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32).to(self.device)

        # 计算Q值
        q_values = self.Q(states)
        q_value = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        # 计算目标Q值，目标Q值是拷贝Q网络参数所以不用梯度
        with torch.no_grad():
            next_q_values = self.Q_target(next_states)
            max_next_q_value = next_q_values.max(1)[0]
            target_q_value = rewards + (1 - dones) * self.gamma * max_next_q_value

        # 计算损失
        loss = F.smooth_l1_loss(q_value, target_q_value)

        # 优化
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # 更新目标网络
        if step % self.target_update_freq == 0:
            self.Q_target.load_state_dict(self.Q.state_dict())

    def save_model(self, model_path):
        """保存模型参数"""
        model_path=os.path.join(model_path, start_time)
        if not exists(model_path):
            os.makedirs(model_path)
        torch.save(self.Q.state_dict(), os.path.join(model_path, "Q.pth"))
        torch.save(self.Q_target.state_dict(), os.path.join(model_path, "Q_target.pth"))
        print(f"Model saved to {model_path}")

    def load_model(self):
        """加载模型参数"""
        self.Q.load_state_dict(torch.load(os.path.join(self.args["model_path"], "Q.pth")))
        self.Q_target.load_state_dict(torch.load(os.path.join(self.args["model_path"], "Q_target.pth")))
        print(f"Model loaded from {self.args['model_path']}")


