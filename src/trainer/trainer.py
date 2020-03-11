import torch

from src.utils.memoryutil import ReplayMemory


class GraphTrainer(torch.nn.Module):
    def __init__(self, ):
        super(GraphTrainer, self).__init__()
        self.memory = ReplayMemory(capacity=10000)

    def forward(self, observation):
        pass

    def push(self, s, a, n_s, r, t):
        self.memory.push(s, a, n_s, r, t)
