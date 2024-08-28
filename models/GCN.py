# -*- coding: utf-8 -*-
# @Author : Lun
# @Created Time: Wed 24 Feb 2020 04:37:13 PM UTC

import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class GCN(torch.nn.Module):
    def __init__(self, num_features, num_hidden, dropout = [0.5, 0.5, 0.5], **kwargs):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(num_features, num_hidden[0])
        self.conv2 = GCNConv(num_hidden[0], num_hidden[1])
        self.dropout = dropout

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        
        x = F.dropout(x, p = self.dropout[0], training=self.training)
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p = self.dropout[1], training=self.training)
        x = F.relu(self.conv2(x, edge_index))
        x = F.dropout(x, p = self.dropout[2], training=self.training)

        return x


