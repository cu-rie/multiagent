import matplotlib.pyplot as plt
from multiagent.environment import MultiAgentEnv
from matplotlib.patches import Rectangle

color_agent = [0.35, 0.85, 0.35]
color_enemy = [0.85, 0.35, 0.35]
color_landmark = [0.25, 0.25, 0.25]


def plot_env(env: MultiAgentEnv, ep, step, max_size=1.5, save=True):
    fig = plt.figure()
    fig.tight_layout()
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlim([-max_size, max_size])
    ax.set_ylim([-max_size, max_size])
    ax.set_aspect('equal', 'box')
    ax.axis('off')

    for ag in env.agents:
        pos = ag.state.p_pos
        size = ag.size
        color = color_enemy if ag.adversary else color_agent

        circle = make_circle(pos, size, color)
        ax.add_artist(circle)

    for landmark in env.world.landmarks:
        pos = landmark.state.p_pos
        size = landmark.size
        color = color_landmark

        circle = make_circle(pos, size, color)
        ax.add_artist(circle)

    currentAxis = plt.gca()
    currentAxis.add_patch(Rectangle((-1, -1), 2, 2, fill=None, edgecolor='red', zorder=10))

    if save:
        plt.savefig('fig/ep{}_step{}.png'.format(ep, step))

    plt.close()


def make_circle(pos, size, color):
    return plt.Circle((pos[0], pos[1]), size, color=color)
