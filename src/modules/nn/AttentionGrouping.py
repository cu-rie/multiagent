import math
from functools import partial

import dgl
import torch

from src.modules.nn.SparseMax import Sparsemax


class AttentionGrouping(torch.nn.Module):

    def __init__(self, embed_dim, num_heads=2):
        super(AttentionGrouping, self).__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads

        self.WK = torch.nn.Linear(embed_dim, num_heads * embed_dim, bias=False)
        self.WV = torch.nn.Linear(embed_dim, num_heads * embed_dim, bias=False)
        self.WQ = torch.nn.Linear(embed_dim, num_heads * embed_dim, bias=False)
        self.sparsemax = Sparsemax(dim=1)

    def forward(self, graph, node_feature):
        """
        :param graph: (DGLGraph) assume graph is either a fully connected graph or a batch of fully connected graphs
        those contains ONLY ALLY NODES!
        :param node_feature:
        :return:
        """

        graph.ndata['node_feature'] = node_feature

        if isinstance(graph, dgl.BatchedDGLGraph):
            max_num_allies = max(graph.num_nodes())
        else:
            max_num_allies = graph.number_of_nodes()

        reduce_function = partial(self.reduce_function, max_num_allies=max_num_allies)
        graph.send_and_recv(graph.edges(), message_func=self.message_function, reduce_func=reduce_function)
        # graph.send(graph.edges(), message_func=self.message_function)

        _ = graph.ndata.pop('node_feature')
        output = graph.ndata.pop('output')
        weight = graph.ndata.pop('weight')
        return output, weight

    def message_function(self, edges):
        src_node_feats = edges.src['node_feature']
        key = self.WK(src_node_feats)
        value = self.WV(src_node_feats)

        return {'key': key, 'value': value}

    def reduce_function(self, nodes, max_num_allies):
        node_feats = nodes.data['node_feature']
        device = node_feats.device

        query = self.WQ(node_feats)  # [ #. nodes x (num_heads x #. embed_dim)]
        key = nodes.mailbox['key']  # [ #.nodes x #.incoming edges x (num_heads x #. embed_dim)]
        value = nodes.mailbox['value']  # [ #.nodes x #.incoming edges x (num_heads x #. embed_dim)]

        nn, ne = key.size(0), key.size(1)

        key = key.view(nn, ne, self.num_heads,
                       self.embed_dim)  # [ #.nodes x #.incoming edges x num_heads x #. embed_dim]
        value = value.view(nn, ne, self.num_heads,
                           self.embed_dim)  # [ #.nodes x #.incoming edges x num_heads x #. embed_dim]
        query = query.view(nn, 1, self.num_heads, self.embed_dim)  # [ #.nodes x 1 x num_heads x #. embed_dim]

        score = (key * query).sum(dim=-1)  # [ #.nodes x #.incoming edges x num_heads]
        score = score / math.sqrt(self.embed_dim * self.num_heads)  # [ #.nodes x #.incoming edges x num_heads]

        # _weight = F.softmax(score, dim=1)  # softmax over incoming edges, [ #.nodes x #.incoming edges x num_heads]
        _weight = self.sparsemax(score)  # sparsemax over incoming edges, [ #.nodes x #.incoming edges x num_heads]

        # for handling variable #. incoming edges in DGL framework
        weight = torch.zeros(size=(nn, max_num_allies, self.num_heads), device=device)
        weight[:, :ne, :] = _weight

        # # [ #.nodes x (num_heads x #. embed_dim)]
        # output = (_weight.unsqueeze(dim=-1) * value).sum(dim=1).view(nn, self.num_heads * self.embed_dim)

        # [ #.nodes x num_heads #. embed_dim]
        output = (_weight.unsqueeze(dim=-1) * value).sum(dim=1).view(nn, self.num_heads, self.embed_dim)

        # averaging the features over attention heads [ #.nodes x #. embed_dim]
        output = output.mean(dim=1)
        return {'output': output, 'weight': weight}
