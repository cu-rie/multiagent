from multiagent.environment import MultiAgentEnv
from scenarios.simple_tag import Scenario
from src.utils.policy_interactive import InteractivePolicy


def make_env_tag():
    # load scenario from script
    scenario = Scenario()
    # create world
    world = scenario.make_world()
    # create multi-agent environment
    env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation, info_callback=None, shared_viewer=False)
    return env


if __name__ == '__main__':

    env = make_env_tag()

    env.render()

    policies = [InteractivePolicy(env, i) for i in range(env.n)]
    # execution loop
    obs_n = env.reset()
    while True:
        # query for action from each agent's policy
        act_n = []
        for i, policy in enumerate(policies):
            act_n.append(policy.action(obs_n[i]))
        # step environment
        obs_n, reward_n, done_n, _ = env.step(act_n)
        # render all agent views
        env.render()
        # display rewards
        # for agent in env.world.agents:
        #    print(agent.name + " reward: %0.3f" % env._get_reward(agent))
