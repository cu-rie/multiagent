import math
from functools import partial

import dgl
import torch


class LinearGrouping(torch.nn.Module):

    def __init__(self, embed_dim, num_groups=3, use_hard_assign=False, T=1.0):
        super(LinearGrouping, self).__init__()

        self.fc = torch.nn.Linear(embed_dim, num_groups)
        self.use_hard_assign = use_hard_assign
        self.T = T

    def forward(self, graph, node_feature):
        unnormalizd_score = self.fc(node_feature)
        incidence_coeff = torch.softmax(unnormalizd_score / self.T, dim=1)

        if self.use_hard_assign:
            raise NotImplementedError('you have to reduce softmax temperature to replicate hard assignment')

        weighted_individual_feat = (node_feature.unsqueeze(1) * incidence_coeff.unsqueeze(2))
        graph.ndata['weighted_individual_feat'] = weighted_individual_feat
        graph.ndata['incidence_coeff'] = incidence_coeff

        graph.update_all(message_func=self.message_func, reduce_func=self.reduce_func)

        weighted_group_feat = graph.ndata.pop('weighted_group_feat')

        graph.ndata.pop('weighted_individual_feat')
        graph.ndata.pop('incidence_coeff')

        return weighted_group_feat

    def message_func(self, edges):
        weighted_feat = edges.src['weighted_individual_feat']
        return {'weighted_individual_feat': weighted_feat}

    def reduce_func(self, nodes):
        weighted_feat = nodes.mailbox['weighted_individual_feat']

        group_feat = weighted_feat.mean(1)

        incidence_coeff = nodes.data['incidence_coeff'].unsqueeze(-1)
        weighted_group_feat = (group_feat * incidence_coeff).sum(1)

        return {'weighted_group_feat': weighted_group_feat}
