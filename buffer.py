import collections
import random


class ReplayBuffer:
    def __init__(self, args):
        self.capacity = args["buffer_size"]
        self.buffer = collections.deque(maxlen=self.capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        return batch

    def __len__(self):
        return len(self.buffer)
