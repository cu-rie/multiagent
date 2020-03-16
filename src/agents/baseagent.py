import torch

from src.utils.memoryutil import ReplayMemory


class BaseAgent(torch.nn.Module):
    def __init__(self, observation_space, action_space, batch_size=500, lr=0.001):
        super(BaseAgent, self).__init__()
        self.observation_space = observation_space
        self.action_space = action_space
        self.batch_size = batch_size
        self.lr = lr

        self.memory = ReplayMemory(capacity=10000)

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def fit(self, *args, **kwargs):
        raise NotImplementedError

    def push(self, *args):
        self.memory.push(args)

    @staticmethod
    def update_target(src, target, tau=0.9):
        if tau == 0.0:
            target.load_state_dict(src.state_dict())
        else:
            for source_param, target_param in zip(src.parameters(), target.parameters()):
                target_param.data.copy_(tau * source_param.data + (1.0 - tau) * target_param.data)
