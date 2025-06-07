import torch.nn as nn
import torch.nn.functional as F
# CNN_Network
class CNN_NETWORK(nn.Module):
    def __init__(self,in_channels,num_actions):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, 8, 4)
        self.conv2 = nn.Conv2d(32, 64, 4, 2)
        self.conv3 = nn.Conv2d(64, 64, 3, 1)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * 7 * 7, 512)
        self.fc2 = nn.Linear(512, num_actions)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

# MLP_Network
class MLP_Network(nn.Module):
    "[batch_size, 4, 84, 84]"
    def __init__(self, args,input_size,num_actions):
        super().__init__()
        self.hidden_dim = args["hidden_size"]
        self.fc1 = nn.Linear(input_size, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.fc3 = nn.Linear(self.hidden_dim, num_actions)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x =  F.relu(self.fc2(x))
        return self.fc3(x)