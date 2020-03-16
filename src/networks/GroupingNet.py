import dgl
import torch
import torch.nn as nn

from src.networks.RelationalGraphNetwork import RelationalGraphNetwork, RelationalGraphNetworkConfig
from src.modules.nn.AttentionGrouping import AttentionGrouping
from src.modules.nn.LinearGrouping import LinearGrouping

from src.modules.nn.Qnet import Qnet, QnetConfig
from src.utils.ConfigBase import ConfigBase

from src.utils.graph_func import NODE_ALLY
from src.utils.graph_util import get_filtered_node_index_by_type, get_largest_number_of_enemy_nodes


class GroupingNetConfig(ConfigBase):
    def __init__(self, name='groupingqnet', encoder_conf=None, grouping_conf=None, qnet_conf=None):
        super(GroupingNetConfig, self).__init__(name=name, encoder=encoder_conf, grouping=grouping_conf,
                                                qnet=qnet_conf)
        gnn_conf = RelationalGraphNetworkConfig().gnn
        self.encoder = gnn_conf
        self.qnet = QnetConfig()


class GroupingNet(nn.Module):

    def __init__(self, conf, num_groups=3, use_gnn=True, grouping='incidence'):
        super(GroupingNet, self).__init__()

        self.use_gnn = use_gnn
        if use_gnn:
            self.gnn = RelationalGraphNetwork(**conf.encoder)
            embed_dim = conf.encoder['output_node_dim']
        else:
            embed_dim = conf.encoder['input_node_dim']

        self.grouping = grouping

        if grouping == 'incidence':
            self.attention_module = AttentionGrouping(embed_dim=embed_dim, num_heads=num_groups)
            self.grouping_module = LinearGrouping(embed_dim=embed_dim, num_groups=num_groups)
            conf.qnet.move_module['input_dimension'] = embed_dim * 2
            conf.qnet.attack_module['input_dimension'] = embed_dim * 2 + conf.encoder['init_node_dim']
        elif grouping == 'attention':
            self.grouping_module = AttentionGrouping(embed_dim=embed_dim, num_heads=num_groups)
            conf.qnet.move_module['input_dimension'] = embed_dim * 2
            conf.qnet.attack_module['input_dimension'] = embed_dim * 2 + conf.encoder['init_node_dim']
        else:
            conf.qnet.move_module['input_dimension'] = embed_dim
            conf.qnet.attack_module['input_dimension'] = embed_dim + conf.encoder['init_node_dim']

        self.qnet = Qnet(conf.qnet)

    def forward(self, graph, node_feature):
        if self.use_gnn:
            out_node_feature = self.gnn(graph, node_feature)
        else:
            out_node_feature = node_feature

        if self.grouping is not None:
            node_indices = get_filtered_node_index_by_type(graph, NODE_ALLY)

            # remove self edges for computational reason.
            ally_only_graph = graph.subgraph(node_indices)
            ally_node_feature = out_node_feature[node_indices]

            if self.grouping == 'incidence':
                # group_feat using incidence matrix
                group_feat, weight = self.attention_module(ally_only_graph, ally_node_feature)

                group_feat = self.grouping_module(ally_only_graph, group_feat)

            elif self.grouping == 'attention':
                # attention
                group_feat, weight = self.grouping_module(ally_only_graph, ally_node_feature)
            else:
                raise NotImplementedError('support only grouping mechanism : incidence, attention')

            q_input = torch.cat([ally_node_feature, group_feat], dim=1)

            # assign group feature on the input graph
            out_feat = torch.zeros(size=(node_feature.size(0), q_input.size(1)), device=node_feature.device)
            out_feat[node_indices] = q_input

        else:
            out_feat = out_node_feature

        if isinstance(graph, dgl.BatchedDGLGraph):
            maximum_num_enemy = get_largest_number_of_enemy_nodes(dgl.unbatch(graph))
        else:
            maximum_num_enemy = get_largest_number_of_enemy_nodes([graph])

        qs = self.qnet.compute_qs(graph, out_feat, maximum_num_enemy=maximum_num_enemy)
        return qs
