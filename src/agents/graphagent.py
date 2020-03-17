import torch
import dgl
import random

from src.networks.RelationalGraphNetwork import RelationalGraphNetwork, RelationalGraphNetworkConfig
from ..agents.baseagent import BaseAgent


class GraphAgent(BaseAgent):
    def __init__(self, observation_space, action_space, batch_size=500, lr=0.001):
        super(GraphAgent, self).__init__(observation_space, action_space, batch_size, lr)

        conf = RelationalGraphNetworkConfig()
        conf.gnn['input_node_dim'] = observation_space[0].shape[0]
        conf.gnn['init_node_dim'] = observation_space[0].shape[0]
        conf.gnn['output_node_dim'] = action_space[0].n

        self.brain = RelationalGraphNetwork(**conf.gnn)
        self.target = RelationalGraphNetwork(**conf.gnn)

        self.n_agents = len(action_space)
        self.n_actions = action_space[0].n

        random_prob = torch.Tensor([1 / self.n_actions for _ in range(self.n_actions)])
        self.random_action = torch.distributions.Multinomial(1, probs=random_prob)

        self.update_target(src=self.brain, target=self.target)
        self.eps = 0.99
        self.eps_decay = 0.996
        self.gamma = 0.99
        self.eps_min = 0.01

        self.optimizer = torch.optim.Adam(self.brain.parameters(), lr=lr)

    def forward(self, graph):
        node_feature = graph.ndata['init_node_feature']
        Q = self.brain(graph, node_feature)

        # random action
        if random.random() <= self.eps:
            action = [self.random_action.sample() for _ in range(self.n_agents)]
            argmax_action = [a.argmax() for a in action]
            argmax_action = torch.stack(argmax_action)
        else:
            argmax_action = Q.argmax(dim=1)
            action = self.convert_action_to_one_hot(self.n_actions, argmax_action)
        return action, argmax_action

    def push(self, state, action, n_state, reward, terminal):
        self.memory.push(state, action, n_state, reward, terminal)

    def fit(self, device=None):
        transitions = self.sample_from_memory()

        curr_g = []
        action = []
        reward = []
        next_g = []
        terminal = []

        for sample in transitions:
            curr_g.append(sample.state)
            action.append(sample.action)
            reward.append(sample.reward)
            next_g.append(sample.next_state)
            terminal.extend([sample.terminal for _ in range(self.n_agents)])

        s = dgl.batch(curr_g)
        a = torch.stack(action).reshape(-1, 1).to(device)
        r = torch.Tensor(reward).to(device)
        ns = dgl.batch(next_g)
        terminal = torch.BoolTensor(terminal).to(device)

        nf_before = s.ndata['init_node_feature']
        state_action_value = self.brain(s, nf_before).gather(1, a)

        next_state_value = torch.zeros_like(state_action_value)

        nf_next = ns.ndata['init_node_feature']
        next_state_value[~terminal] = self.target(ns, nf_next).max(1)[0].detach()[~terminal].reshape(-1, 1)

        expected_sa_val = next_state_value * self.gamma + r.reshape(-1, 1)

        loss = torch.nn.functional.smooth_l1_loss(state_action_value, expected_sa_val)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.update_eps()

    @staticmethod
    def convert_action_to_one_hot(num_actions, argmax_action):
        num_agents = len(argmax_action)
        out_action = torch.zeros(num_agents, num_actions)
        for agent_idx, action in enumerate(argmax_action):
            out_action[agent_idx, action] = 1

        return out_action
