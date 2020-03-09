import random
import torch

from torch.distributions.multinomial import Multinomial
from src.modules.nn.MLP import MultiLayerPerceptron as MLP
from src.utils.memoryutil import ReplayMemory


class MLP_Agent(torch.nn.Module):
    def __init__(self, observation_space, action_space, batch_size=500, lr=0.001):
        super(MLP_Agent, self).__init__()

        self.brain = MLP(input_dimension=observation_space.shape[0], output_dimension=action_space.n)
        self.target = MLP(input_dimension=observation_space.shape[0], output_dimension=action_space.n)
        self.target.load_state_dict(self.brain.state_dict())

        self.memory = ReplayMemory(capacity=10000)
        random_prob = torch.Tensor([1 / action_space.n for _ in range(action_space.n)])
        self.random_action = torch.distributions.Multinomial(1, probs=random_prob)
        self.n_actions = action_space.n
        self.batch_size = batch_size
        self.eps = 0.99
        self.eps_decay = 0.995
        self.gamma = 0.99
        self.eps_min = 0.01

        self.optimizer = torch.optim.Adam(self.brain.parameters(), lr=lr)

    def forward(self, observation):
        # random action
        if random.random() >= self.eps:
            action = self.random_action.sample()
        else:
            obs = torch.Tensor(observation)
            Q = self.brain(obs)
            action = torch.zeros_like(Q)
            argmax_action = torch.argmax(Q)
            action[argmax_action] = 1
        return action

    def push(self, state, action, n_state, reward, terminal):
        self.memory.push(state, action, n_state, reward, terminal)
        if terminal:
            self.eps *= max(self.eps * self.eps_decay, self.eps_min)

    def fit(self):
        transitions = self.memory.sample(self.batch_size)

        state = []
        action = []
        reward = []
        next_state = []
        terminal = []

        for sample in transitions:
            state.append(sample.state)
            action.append(sample.action.nonzero())
            reward.append(sample.reward)
            next_state.append(sample.next_state)
            terminal.append(sample.terminal)

        s = torch.Tensor(state)
        a = torch.stack(action).squeeze(dim=-1)
        r = torch.Tensor(reward)
        ns = torch.Tensor(next_state)
        terminal = torch.BoolTensor(terminal)

        state_action_value = self.brain(s).gather(1, a)

        next_state_value = torch.zeros_like(state_action_value)

        next_state_value[~terminal] = self.target(ns).max(1)[0].detach()[~terminal].reshape(-1, 1)

        expected_sa_val = next_state_value * self.gamma + r.reshape(-1, 1)

        loss = torch.nn.functional.smooth_l1_loss(state_action_value, expected_sa_val)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
