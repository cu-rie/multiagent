from multiagent.environment import MultiAgentEnv
from scenarios.simple_tag import Scenario
from src.agents.graphagent import GraphAgent
import matplotlib.pyplot as plt


def make_env_tag():
    # load scenario from script
    scenario = Scenario()
    # create world
    world = scenario.make_world()
    # create multi-agent environment
    env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation)
    return env


if __name__ == '__main__':

    env = make_env_tag()

    max_episodes = 1000
    max_t = 25
    curr_state = env.reset()
    t = 0
    fit_interval = 10
    batch_size = 100

    ally_rewards = []
    enemy_rewards = []
    epsilons = []

    agent = GraphAgent(env.observation_space, env.action_space)

    for e in range(max_episodes):
        ep_ally_reward = 0
        ep_adversary_reward = 0
        while True:
            env.render()

            action = [agent(curr_s) for agent, curr_s in zip(agents, curr_state)]
            next_state, reward, _done, info = env.step(action)
            t += 1

            for rwd, agent in zip(reward, env.agents):
                if agent.adversary:
                    ep_adversary_reward += rwd
                else:
                    ep_ally_reward += rwd

            done = all(_done)
            terminal = (t >= max_t)

            for ag, st, ac, ns, rw in zip(agents, curr_state, action, next_state, reward):
                ag.push(st, ac, ns, rw, terminal)

            curr_state = next_state

            if done or terminal:
                curr_state = env.reset()
                t = 0
                epsilons.append(agents[0].eps)
                print("EP:{}, Ally_RWD:{:.2f}, Enemy_RWD:{:.2f}, EPS:{:.2f}".format(e, ep_ally_reward,
                                                                                    ep_adversary_reward, agents[0].eps))
                ally_rewards.append(ep_ally_reward)
                enemy_rewards.append(ep_adversary_reward)
                break

        if e % fit_interval == 0 and len(agents[0].memory) > batch_size:
            for agent in agents:
                agent.fit()

    fig = plt.figure()
    ax = fig.add_subplot(2, 1, 1)
    ax.plot(ally_rewards, label='ally')
    ax.plot(enemy_rewards, label='enemy')
    ax.legend()

    ax2 = fig.add_subplot(2, 1, 2)
    ax2.plot(epsilons)

    plt.savefig('reward_MLP.png')
