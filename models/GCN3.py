# -*- coding: utf-8 -*-
# @Author : Lun
# @Created Time: Wed 24 Feb 2020 04:37:13 PM UTC

import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class GCN(torch.nn.Module):
    def __init__(self, num_features, num_hidden, **kwargs):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(num_features, num_hidden[0])
        self.conv2 = GCNConv(num_hidden[0], num_hidden[1])
        self.conv3 = GCNConv(num_hidden[1], num_hidden[2])

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.conv2(x, edge_index))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.conv3(x, edge_index))
        x = F.dropout(x, training=self.training)

        return x


