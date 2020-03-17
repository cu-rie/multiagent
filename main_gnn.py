import torch
import matplotlib.pyplot as plt

from multiagent.environment import MultiAgentEnv
from scenarios.simple_tag import Scenario
from src.agents.graphagent import GraphAgent
from src.utils.graph_func import state2graphfunc


def make_env_tag():
    # load scenario from script
    scenario = Scenario()
    # create world
    world = scenario.make_world()
    # create multi-agent environment
    env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation)
    return env


if __name__ == '__main__':

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    env = make_env_tag()

    max_episodes = 10000
    max_t = 25
    curr_state = env.reset()
    curr_g = state2graphfunc(env, curr_state, device)
    t = 0
    fit_interval = 10
    batch_size = 100

    ally_rewards = []
    enemy_rewards = []
    epsilons = []

    agent = GraphAgent(env.observation_space, env.action_space).to(device)

    for e in range(max_episodes):
        ep_ally_reward = 0
        ep_adversary_reward = 0
        while True:
            # env.render()

            action, argmax_action = agent(curr_g)
            next_state, reward, _done, info = env.step(action)
            next_g = state2graphfunc(env, next_state, device)
            t += 1

            for rwd, ag in zip(reward, env.agents):
                if ag.adversary:
                    ep_adversary_reward += rwd
                else:
                    ep_ally_reward += rwd

            done = all(_done)
            terminal = (t >= max_t)

            agent.push(curr_g, argmax_action, next_g, reward, terminal)

            curr_g = next_g

            if done or terminal:
                curr_state = env.reset()
                t = 0
                epsilons.append(agent.eps)
                print("EP:{}, Ally_RWD:{:.2f}, Enemy_RWD:{:.2f}, EPS:{:.2f}".format(e, ep_ally_reward,
                                                                                    ep_adversary_reward, agent.eps))
                ally_rewards.append(ep_ally_reward)
                enemy_rewards.append(ep_adversary_reward)
                break

        if e % fit_interval == 0 and len(agent.memory) > agent.batch_size:
            agent.fit()

    fig = plt.figure()
    ax = fig.add_subplot(2, 1, 1)
    ax.plot(ally_rewards, label='ally')
    ax.plot(enemy_rewards, label='enemy')
    ax.legend()

    ax2 = fig.add_subplot(2, 1, 2)
    ax2.plot(epsilons)

    plt.savefig('reward_graph.png')
