import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, Batch

class FCN(nn.Module):
    def __init__(self, num_features, num_hidden, dropout = None):
        super(FCN, self).__init__()
        if dropout is None:
            dropout = [0.0 for _ in num_hidden]
            dropout.append(0.0)
        
        self.num_layer = len(num_hidden)
        num_hidden.insert(0, num_features)
        
            
        layer_list = [nn.Dropout(p = dropout[0])]
        for i in range(self.num_layer):
            layer_list.append(nn.Linear(num_hidden[i], num_hidden[i+1]))
            layer_list.append(nn.ReLU())
            layer_list.append(nn.Dropout(p = dropout[i+1]))
        
        self.model = nn.Sequential(*layer_list)
        
    def forward(self, x):
        if type(x) in [Data, Batch]:
            x = x.x
        x = self.model(x)
        return x