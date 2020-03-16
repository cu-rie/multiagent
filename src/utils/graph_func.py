import dgl
import torch
import itertools

from functools import partial
from multiagent.environment import MultiAgentEnv

VERY_LARGE_NUMBER = -99999

NODE_ALLY = 0
NODE_ENEMY = 1

EDGE_ALLY = 0
EDGE_ENEMY = 1
EDGE_SELF = 2


def minus_large_num_initializer(shape, dtype, ctx, id_range):
    return torch.ones(shape, dtype=dtype, device=ctx) * - VERY_LARGE_NUMBER


def state2graphfunc(env: MultiAgentEnv, obs, device=None):
    ally_idx = []
    enemy_idx = []
    node_type = []

    for i, a in enumerate(env.agents):
        if a.adversary:
            node_type.append(NODE_ENEMY)
            enemy_idx.append(i)
        else:
            node_type.append(NODE_ALLY)
            ally_idx.append(i)

    g = dgl.DGLGraph()

    g.set_n_initializer(minus_large_num_initializer)
    g.set_e_initializer(dgl.init.zero_initializer)

    num_agents = len(env.agents)
    num_ally = len(ally_idx)
    num_enemy = len(enemy_idx)

    g.add_nodes(num_agents)
    # g.ndata['node_feature'] = torch.Tensor(obs).to(device)
    g.ndata['init_node_feature'] = torch.Tensor(obs).to(device)
    g.ndata['node_type'] = torch.Tensor(node_type).reshape(-1, 1).to(device)

    a2a_edge = cartesian_product(ally_idx, ally_idx)
    e2a_edge = cartesian_product(enemy_idx, ally_idx)

    len_a2a = len(a2a_edge[0])
    len_e2a = len(e2a_edge[0])

    g.add_edges(a2a_edge[0], a2a_edge[1], {'edge_type': torch.Tensor(data=(EDGE_ALLY,)).repeat(len_a2a)})
    g.add_edges(e2a_edge[0], e2a_edge[1], {'edge_type': torch.Tensor(data=(EDGE_ENEMY,)).repeat(len_e2a)})
    g.add_edges(range(num_agents), range(num_agents), {'edge_type': torch.Tensor(data=(EDGE_SELF,)).repeat(num_agents)})

    return g


def cartesian_product(*iterables, return_1d=True, self_edge=False):
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


def get_filtered_node_index_by_type(graph, ntype_idx):
    filter_func = partial(filter_by_node_type_idx, ntype_idx=ntype_idx)
    node_idx = graph.filter_nodes(filter_func)
    return node_idx


def filter_by_node_type_idx(nodes, ntype_idx):
    return nodes.data['node_type'] == ntype_idx
