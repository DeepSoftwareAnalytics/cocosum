# -*- coding: utf-8 -*-
# @Author : Lun
# @Created Time: Wed 24 Feb 2020 04:37:13 PM UTC

import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv

class GAT(torch.nn.Module):
    def __init__(self, num_features, num_hidden, dropout = None, heads = None, **kwargs):
        super(GAT, self).__init__()
        if dropout is None:
            dropout = [0.0 for _ in num_hidden]
            dropout.append(0.0)
        if heads is None:
            heads = [8 for _ in num_hidden]
            heads[-1] = 1
        self.conv1 = GATConv(num_features, num_hidden[0], heads = heads[0], dropout = dropout[0])
        self.conv2 = GATConv(num_hidden[0] * heads[0], num_hidden[1], heads = heads[1], dropout = dropout[1])

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.dropout(x, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, training=self.training)
        x = F.elu(self.conv2(x, edge_index))
        x = F.dropout(x, training=self.training)

        return x


