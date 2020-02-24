from multiagent.environment import MultiAgentEnv
from scenarios.simple_tag import Scenario
import numpy as np


def make_env():
    # load scenario from script
    scenario = Scenario()
    # create world
    world = scenario.make_world()
    # create multiagent environment
    env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation)
    return env


if __name__ == '__main__':

    env = make_env()

    max_episodes = 100
    max_t = 25
    obs = env.reset()
    t = 0

    for e in range(max_episodes):
        ep_reward = 0
        while True:

            action = [[1, 1, 1, 0, 0] for _ in range(len(obs))]
            # action = [np.random.multinomial(1, [1/5]*5) for _ in range(len(obs))]
            obs, reward, _done, info = env.step(action)
            t += 1
            ep_reward += sum(reward)

            done = all(_done)
            terminal = (t >= max_t)

            if done or terminal:
                obs = env.reset()
                t = 0
                print("EP:{}, REW:{:.2f}".format(e, ep_reward))
                break
