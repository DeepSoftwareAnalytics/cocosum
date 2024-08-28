# -*- coding: utf-8 -*-
# @Author : Lun
# @Created Time: Wed 24 Feb 2020 04:37:13 PM UTC

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.data import Data

from models.GAT import GAT

import pdb

NUM_EDGE_TYPE = 4
class HGAT(nn.Module):
    def __init__(self, num_features, num_hidden, dropout = None, heads = None):
        super(HGAT, self).__init__()
        if dropout is None:
            dropout = [0.0 for _ in num_hidden]
            dropout.append(0.0)
        if heads is None:
            heads = [8 for _ in num_hidden]
            heads[-1] = 1
        
        self.num_layer = len(heads)
        assert(self.num_layer == len(dropout)-1)
        assert(self.num_layer == len(num_hidden))
        
        """
        num_hidden.append(num_features)
        heads.append(1)
        def construct_gat_sequential():
            layer_list = []
            for i in range(len(heads)):
                layer_list.append(GATConv(num_hidden[i-1] * heads[i-1], num_hidden[i], heads = heads[i], dropout = dropout[i]))
                layer_list.append(nn.Elu())
                layer_list.append(nn.Dropout())
            return nn.Sequential(*layer_list)
        
        self.GAT_list = nn.ModuleList([construct_gat_sequential() for _ in range(NUM_EDGE_TYPE)])
        num_hidden.pop()
        heads.pop()
        """
        
        self.GAT_list = nn.ModuleList([GAT(num_features, num_hidden, dropout, heads) for _ in range(NUM_EDGE_TYPE + 1)])
        self.fussion = nn.Linear((NUM_EDGE_TYPE + 1) * num_hidden[-1], num_hidden[-1], bias=True)
        
        #self.conv1 = GATConv(num_features, num_hidden[0], heads = heads[0], dropout = dropout[0])
        #self.conv2 = GATConv(num_hidden[0] * heads[0], num_hidden[1], heads = heads[1], dropout = dropout[1])

    def forward(self, data):
        x, full_edge_index, edge_type = data.x, data.edge_index, data.edge_attr
        with torch.no_grad():
            single_type_list = [full_edge_index.T[edge_type.T[i]==1].T for i in range(NUM_EDGE_TYPE)]
        
        y = self.GAT_list[-1](data)
        
        # for i in range(NUM_EDGE_TYPE):
        #    y += self.GAT_list[i](Data(x = x, edge_list = single_type_list[i]))
        
        #output_list = torch.stack([self.GAT_list[i](Data(x = x, edge_list = single_type_list[i]))], dim = 0)
        
        # pdb.set_trace()
        output_list = [self.GAT_list[i](Data(x = x, edge_index = single_type_list[i])) for i in range(NUM_EDGE_TYPE)]
        output_list.append(y)
        
        y = torch.cat(output_list, dim = 1)
        y = F.elu(self.fussion(y))
        

        return y


