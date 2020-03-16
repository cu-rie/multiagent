import torch

from src.networks.RelationalGraphNetwork import RelationalGraphNetwork, RelationalGraphNetworkConfig
from ..agents.baseagent import BaseAgent


class GraphAgent(BaseAgent):
    def __init__(self, observation_space, action_space, batch_size=500, lr=0.001):
        super(GraphAgent, self).__init__(observation_space, action_space, batch_size, lr)

        conf = RelationalGraphNetworkConfig()
        conf.gnn['input_node_dim'] = observation_space[0].shape[0]
        conf.gnn['output_node_dim'] = action_space[0].n

        self.brain = RelationalGraphNetwork(**conf.gnn)
        self.target = RelationalGraphNetwork(**conf.gnn)

        self.update_target(src=self.brain, target=self.target)

    def fit(self):
        pass


