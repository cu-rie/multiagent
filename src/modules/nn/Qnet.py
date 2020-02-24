from copy import deepcopy as dc

import torch
import torch.nn as nn
import numpy as np

from src.modules.nn.ActionModules import MoveModule, AttackModule
from src.utils.graph_func import get_filtered_node_index_by_type
from src.utils.ConfigBase import ConfigBase
from src.utils.ally_graph import NODE_ALLY, EDGE_ENEMY_TO_ALLY
from src.modules.nn.MLP import MLPConfig


class QnetConfig(ConfigBase):

    def __init__(self, name='qnet', qnet_conf=None, move_module_conf=None, attack_module_conf=None, move_only=None):
        super(QnetConfig, self).__init__(name=name, qnet=qnet_conf, move_module=move_module_conf,
                                         attack_module=attack_module_conf, move_only=move_only)

        mlp_conf = MLPConfig().mlp

        self.qnet = {'attack_edge_type_index': EDGE_ENEMY_TO_ALLY,
                     'ally_node_type_index': NODE_ALLY,
                     'exploration_method': 'eps_greedy'}

        self.move_module = dc(mlp_conf)
        self.move_module['normalization'] = 'layer'
        self.move_module['out_activation'] = None

        self.attack_module = dc(mlp_conf)
        self.attack_module['normalization'] = None
        self.attack_module['out_activation'] = None

        self.move_only = True


class Qnet(nn.Module):

    def __init__(self, conf, move_only=False):
        super(Qnet, self).__init__()
        self.conf = conf
        self.move_only = conf.move_only
        self.exploration_method = conf.qnet['exploration_method']

        if self.move_only:
            move_dim = 13
        else:
            move_dim = 5
        self.move_module = MoveModule(self.conf.move_module, move_dim=move_dim)
        self.attack_module = AttackModule(self.conf.attack_module)

    def forward(self, graph, node_feature, maximum_num_enemy):
        if self.move_only:
            # compute move qs
            move_argument = self.move_module(graph, node_feature)
            hold_argument = torch.zeros(size=(move_argument.shape[0], 1)).to(move_argument.device)
            qs = torch.cat((hold_argument, move_argument), dim=-1)

        else:
            # compute move qs
            move_argument = self.move_module(graph, node_feature)

            # compute attack qs
            attack_edge_type_index = self.conf.qnet['attack_edge_type_index']
            attack_argument = self.attack_module(graph, node_feature, maximum_num_enemy, attack_edge_type_index)

            hold_argument = torch.zeros(size=(move_argument.shape[0], 1)).to(attack_argument.device)

            qs = torch.cat((hold_argument, move_argument, attack_argument), dim=-1)

        return qs

    def compute_qs(self, graph, node_feature, maximum_num_enemy):

        # get qs of actions of all units including enemies

        qs = self(graph, node_feature, maximum_num_enemy)

        ally_node_type_index = self.conf.qnet['ally_node_type_index']
        ally_node_indices = get_filtered_node_index_by_type(graph, ally_node_type_index)
        qs = qs[ally_node_indices, :]  # of only ally units

        return qs
