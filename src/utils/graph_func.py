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

    nfs = []

    if len(allies_pos_x) == 0:
        allies_center_pos_x = 0
        allies_center_pos_y = 0
    else:
        allies_center_pos_x = np.mean(allies_pos_x)
        allies_center_pos_y = np.mean(allies_pos_y)

    edge_in_attack_range = torch.Tensor(data=(EDGE_ENEMY,)).to(device)
    edge_in_attack_range = edge_in_attack_range.reshape(-1)

    node_types = []
    alive_ally_idx = []

    g.add_nodes(n_allies + n_enemies)

    for ally_idx, ally_unit in allies.items():
        nf = []
        x = ally_unit.pos.x
        y = ally_unit.pos.y

        nf.append(x)
        nf.append(y)
        nf.append(x - allies_center_pos_x)
        nf.append(y - allies_center_pos_y)
        nf.extend(env.get_avail_agent_actions(ally_idx)[2:6])
        nf.append(ally_unit.health / ally_unit.health_max)
        if ally_unit.health > 0:
            node_types.append(NODE_ALLY)
            alive_ally_idx.append(ally_idx)

            # EDGE_ENEMY
            sight_range = env.unit_sight_range(ally_idx)
            for enemy_idx, enemy_unit in enemies.items():
                e_x = enemy_unit.pos.x
                e_y = enemy_unit.pos.y
                dist = distance(x, y, e_x, e_y)
                if dist < sight_range and enemy_unit.health > 0:
                    g.add_edge(enemy_idx + n_allies, ally_idx, {'edge_type': edge_in_attack_range})
        else:
            node_types.append(NODE_ALLY)

        nfs.append(nf)

    enemies_pos_x = [unit.pos.x for unit in enemies.values() if unit.health > 0]
    enemies_pos_y = [unit.pos.y for unit in enemies.values() if unit.health > 0]

    if len(enemies_pos_x) == 0:
        enemies_center_pos_x = 0
        enemies_center_pos_y = 0
    else:
        enemies_center_pos_x = np.mean(enemies_pos_x)
        enemies_center_pos_y = np.mean(enemies_pos_y)

    for enemy_idx, enemy_unit in enemies.items():
        nf = []
        x = enemy_unit.pos.x
        y = enemy_unit.pos.y

        nf.append(x)
        nf.append(y)
        nf.append(x - enemies_center_pos_x)
        nf.append(y - enemies_center_pos_y)
        nf.extend(np.zeros(4))
        nf.append(enemy_unit.health / enemy_unit.health_max)

        if enemy_unit.health > 0:
            node_types.append(NODE_ENEMY)
        else:
            node_types.append(NODE_DEAD)

        nfs.append(nf)

    nfs = torch.Tensor(nfs).to(device)
    node_types = torch.Tensor(node_types).reshape(-1).to(device)
    num_nodes = n_allies + n_enemies

    g.ndata['node_feature'] = nfs
    g.ndata['node_type'] = node_types
    g.ndata['init_node_feature'] = nfs

    # EDGE_SELF
    self_edge_type = torch.Tensor(data=(EDGE_SELF,)).to(device)

    g.add_edges(range(num_nodes), range(num_nodes), {'edge_type': self_edge_type.repeat(num_nodes)})

    # EDGE_ALLY : fully connected edge between allies
    if len(alive_ally_idx) > 1:
        allies_edge_indices = cartesian_product(alive_ally_idx, alive_ally_idx, return_1d=True)
        allies_edge_type = torch.Tensor(data=(EDGE_ALLY,)).to(device)
        num_allies_edges = len(allies_edge_indices[0])
        g.add_edges(allies_edge_indices[0], allies_edge_indices[1],
                    {'edge_type': allies_edge_type.repeat(num_allies_edges)})

    e_to_a_edge_indices = cartesian_product(range(n_allies, n_allies + n_enemies), range(n_allies), return_1d=True)
    e_to_a_edge_type = torch.Tensor(data=(EDGE_ENEMY_TO_ALLY,)).to(device)
    num_e_to_a_edges = len(e_to_a_edge_indices[0])

    g.add_edges(e_to_a_edge_indices[0], e_to_a_edge_indices[1],
                {'edge_type': e_to_a_edge_type.repeat(num_e_to_a_edges)})

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
