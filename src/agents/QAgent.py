import torch.nn as nn


class GNNAgent(nn.Module):
    def __init__(self, use_gnn=True, use_rnn=False, *args):
        super(GNNAgent, self).__init__()

        conf = GroupingNetConfig()
        conf.qnet.move_only = True

        self.use_rnn = use_rnn

        self.fc1 = nn.Linear(in_features=conf.encoder['init_node_dim'], out_features=conf.encoder['init_node_dim'])
        self.rnn_hidden_dim = conf.encoder['init_node_dim']

        if use_rnn:
            self.rnn = nn.GRUCell(self.rnn_hidden_dim, conf.encoder['init_node_dim'])

        self.grouping_net = GroupingNet(conf, use_gnn=use_gnn, grouping='incidence')

    def forward(self, graph, hidden_state):

        node_feature = graph.ndata.pop('node_feature')

        qs = self.grouping_net(graph, node_feature)
        h = hidden_state

        return qs, h
