import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import matplotlib.pyplot as plt
import math

class GraphConvolution(nn.Module):                            
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):             # 这里代码做了简化如 3.2节。
        support = torch.mm(input, self.weight) # (2708, 16) = (2708, 1433) X (1433, 16)
        output = torch.spmm(adj, support)      # (2708, 16) = (2708, 2708) X (2708, 16)
        if self.bias is not None:
            return output + self.bias          # 加上偏置 (2708, 16)
        else:
            return output                      # (2708, 16)

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

class GCN(nn.Module):                                             # 定义两层GCN
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()
        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.dropout = dropout

    def forward(self, x, adj):
        x = torch.nn.functional.relu(self.gc1(x, adj))
        x = torch.nn.functional.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return torch.nn.functional.log_softmax(x, dim=1)       # 对每一个节点做softmax


class HierarchyBuilder(nn.Module):
    def __init__(self, config, input_slots, output_slots):
        super().__init__()
        num_unary_predicates = 4
        num_binary_predicates = 0
        self.num_unary_predicates = num_unary_predicates
        self.num_binary_predicates = num_binary_predicates
        self.graph_conv = GraphConvolution(input_slots,output_slots)

    def forward(self, x, concept_builder):
        """
        input: 
            x: feature to agglomerate [B,N,D]
        """
        factored_features = x

        # [Perform Convolution on Factored States]
        graph_conv_masks = self.graph_conv(factored_features)

        # [Build Connection Between Input Features and Conv Features]
        #graph_conv_masks = torch.einsum("bnd,bmd->bnm")
        graph_conv_masks = F.softmax(graph_conv_masks, dim = 1)
        return graph_conv_masks #[B,N,M]