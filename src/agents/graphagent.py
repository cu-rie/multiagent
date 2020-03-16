import torch


from ..agents.baseagent import BaseAgent


class GraphAgent(BaseAgent):
    def __init__(self, observation_space, action_space, batch_size=500, lr=0.001):
        super(GraphAgent, self).__init__(observation_space, action_space, batch_size, lr)

        self.brain = None
        self.target = None

        self.update_target(src=self.brain, target=self.target)
