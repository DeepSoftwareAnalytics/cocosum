# -*- coding: utf-8 -*-
# @Author : Lun
# @Created Time: Wed 24 Feb 2020 04:37:13 PM UTC

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data


import pdb

NUM_EDGE_TYPE = 4
class HGCN(nn.Module):
    def __init__(self, num_features, num_hidden, dropout = None, aggregation = "Mean"):
        """
        params:
            aggregation support "Mean" and "Concat"
        """
        super(HGCN, self).__init__()
        if dropout is None:
            dropout = [0.0 for _ in num_hidden]
            dropout.append(0.0)
        
        self.dropout = dropout
        self.aggregation = aggregation
        assert(self.aggregation in {"Mean", "Concat"})
        
        self.conv1_list = nn.ModuleList([GCNConv(num_features, num_hidden[0]) for _ in range(NUM_EDGE_TYPE + 1)])
        self.conv2_list = nn.ModuleList([GCNConv(num_hidden[0], num_hidden[1]) for _ in range(NUM_EDGE_TYPE + 1)])
        
        if self.aggregation == "Concat":
            self.fussion1 = nn.Linear((NUM_EDGE_TYPE + 1) * num_hidden[0], num_hidden[0], bias=True)
            self.fussion2 = nn.Linear((NUM_EDGE_TYPE + 1) * num_hidden[1], num_hidden[1], bias=True)
        

    def forward(self, data):
        x, full_edge_index, edge_type = data.x, data.edge_index, data.edge_attr
        with torch.no_grad():
            single_type_list = [full_edge_index.T[edge_type.T[i]==1].T for i in range(NUM_EDGE_TYPE)]
            single_type_list.append(full_edge_index)
        
        x = F.dropout(x, training=self.training, p = self.dropout[0])
        
        if self.aggregation == "Mean":
            output_list = [F.relu(self.conv1_list[i](x = x, edge_index = single_type_list[i])) for i in range(NUM_EDGE_TYPE + 1)]
            x = torch.stack(output_list, dim = 0).mean(dim = 0)
            x = F.dropout(x, training=self.training, p = self.dropout[1])
        
            output_list = [F.relu(self.conv2_list[i](x = x, edge_index = single_type_list[i])) for i in range(NUM_EDGE_TYPE + 1)]
            x = torch.stack(output_list, dim = 0).mean(dim = 0)
            x = F.dropout(x, training=self.training, p = self.dropout[2])
        
        elif self.aggregation == "Concat":
            output_list = [F.relu(self.conv1_list[i](x = x, edge_index = single_type_list[i])) for i in range(NUM_EDGE_TYPE + 1)]
            x = self.fussion1(torch.cat(output_list, dim = 1))
            x = F.dropout(F.relu(x), training=self.training, p = self.dropout[1])
        
            output_list = [F.relu(self.conv2_list[i](x = x, edge_index = single_type_list[i])) for i in range(NUM_EDGE_TYPE + 1)]
            x = self.fussion2(torch.cat(output_list, dim = 1))
            x = F.dropout(F.elu(x), training=self.training, p = self.dropout[2])


        return x


