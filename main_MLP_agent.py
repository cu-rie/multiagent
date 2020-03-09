from multiagent.environment import MultiAgentEnv
from scenarios.simple_tag import Scenario
from agents.MLPAgent import MLP_Agent
import matplotlib.pyplot as plt


def make_env():
    # load scenario from script
    scenario = Scenario()
    # create world
    world = scenario.make_world()
    # create multi-agent environment
    env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation)
    return env


if __name__ == '__main__':

    env = make_env()

    max_episodes = 1000
    max_t = 25
    curr_state = env.reset()
    t = 0
    fit_interval = 10
    batch_size = 100

    rewards = []
    epsilons = []

    agents = [MLP_Agent(o, a, batch_size=batch_size) for o, a in zip(env.observation_space, env.action_space)]

    for e in range(max_episodes):
        ep_reward = 0
        while True:

            action = [agent(curr_s) for agent, curr_s in zip(agents, curr_state)]
            next_state, reward, _done, info = env.step(action)
            t += 1
            ep_reward += sum(reward)

            done = all(_done)
            terminal = (t >= max_t)

            for ag, st, ac, ns, rw in zip(agents, curr_state, action, next_state, reward):
                ag.push(st, ac, ns, rw, terminal)

            curr_state = next_state

            if done or terminal:
                curr_state = env.reset()
                t = 0
                rewards.append(ep_reward)
                epsilons.append(agents[0].eps)
                print("EP:{}, REW:{:.2f}, EPS:{:.2f}".format(e, ep_reward, agents[0].eps))
                break

        if e % fit_interval == 0 and len(agents[0].memory) > batch_size:
            for agent in agents:
                agent.fit()

    fig = plt.figure()
    ax = fig.add_subplot(2, 1, 1)
    ax.plot(rewards)

    ax2 = fig.add_subplot(2, 1, 2)
    ax2.plot(epsilons)

    plt.savefig('reward_MLP.png')
