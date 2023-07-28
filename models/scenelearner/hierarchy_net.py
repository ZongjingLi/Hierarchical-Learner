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

    def forward(self, inputs, adj):
        B = inputs.shape[0]

        support = torch.matmul(inputs, self.weight.unsqueeze(0).repeat(B,1,1))
        output = torch.matmul(adj, support)     
        if self.bias is not None:
            return output + self.bias         
        else:
            return output                      # (2708, 16)

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class HierarchyBuilder(nn.Module):
    def __init__(self, config, output_slots, nu = 8):
        super().__init__()
        num_unary_predicates = nu
        num_binary_predicates = 0
        spatial_feature_dim = 0
        input_dim = num_unary_predicates + spatial_feature_dim
        self.num_unary_predicates = num_unary_predicates
        self.num_binary_predicates = num_binary_predicates
        self.graph_conv = GraphConvolution(input_dim,output_slots)

    def forward(self, x, executor):
        """
        input: 
            x: feature to agglomerate [B,N,D]
        """
        B, N, D = x.shape
        predicates = executor.concept_vocab


        factored_features = [executor.entailment(
            x,executor.get_concept_embedding(predicate)
            ).unsqueeze(-1) for predicate in predicates]

        factored_features = torch.cat(factored_features, dim = -1)

        # [Perform Convolution on Factored States]
        adjs = torch.ones([B, N, N])
        graph_conv_masks = self.graph_conv(factored_features, adjs).permute([0,2,1])

        # [Build Connection Between Input Features and Conv Features]
        #graph_conv_masks = torch.einsum("bnd,bmd->bnm")
        graph_conv_masks = F.softmax(graph_conv_masks, dim = 1)
        return graph_conv_masks #[B,N,M]