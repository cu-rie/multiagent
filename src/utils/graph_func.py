import dgl
import torch
import itertools
import numpy as np
from multiagent.environment import MultiAgentEnv

VERY_LARGE_NUMBER = -99999

NODE_ALLY = 0
NODE_ENEMY = 1

EDGE_ALLY = 0
EDGE_ENEMY = 1


def minus_large_num_initializer(shape, dtype, ctx, id_range):
    return torch.ones(shape, dtype=dtype, device=ctx) * - VERY_LARGE_NUMBER


def state2graphfunc(env: MultiAgentEnv, obs, device=None):
    feat_shape = env.observation_space

    adversary_type = [a.adversary for a in env.agents]

    g = dgl.DGLGraph()

    g.set_n_initializer(minus_large_num_initializer)
    g.set_e_initializer(dgl.init.zero_initializer)

    num_agents = len(env.agents)

    g.add_nodes(num_agents)

    return g


def cartesian_product(*iterables, return_1d=False, self_edge=False):
    if return_1d:
        xs = []
        ys = []
        if self_edge:
            for ij in itertools.product(*iterables):
                xs.append(ij[0])
                ys.append(ij[1])
        else:
            for ij in itertools.product(*iterables):
                if ij[0] != ij[1]:
                    xs.append(ij[0])
                    ys.append(ij[1])
        ret = (xs, ys)
    else:
        ret = [i for i in itertools.product(*iterables)]
    return ret
