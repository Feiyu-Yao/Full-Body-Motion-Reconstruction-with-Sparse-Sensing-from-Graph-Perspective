from __future__ import absolute_import, division

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def pairwise_distances(x, y=None):
    '''
    Input: x is a Nxd matrix
            y is an optional Mxd matirx
    Output: dist is a NxM matrix where dist[i,j] is the square norm between x[i,:] and y[j,:]
            if y is not given then use 'y=x'.
    i.e. dist[i,j] = ||x[i,:]-y[j,:]||^2
    '''
    x_norm = (x ** 2).sum(2).view(x.shape[0], x.shape[1], 1)
    if y is not None:
        y_t = torch.transpose(y, 0, 1)
        y_norm = (y ** 2).sum(1).view(1, -1)
    else:
        y_t = torch.transpose(x, 2, 1)
        y_norm = x_norm.view(x.shape[0], 1,x.shape[1])
    dist = x_norm + y_norm - 2.0 * torch.bmm(x, y_t)
    dist[dist<0] = 0
    # Ensure diagonal is zero if x=y
    # if y is None:
    #     dist = dist - torch.diag(dist.diag)8
    return dist

def square_norm(x):
    """
    Helper function returning square of the euclidean norm.
    Also here we clamp it since it really likes to die to zero.
    """
    norm = torch.norm(x, dim=-1, p=2) ** 2
    return norm

def poincare_distance(x):
    '''
    Calculate pair-wise poincare distance between each row in two input tensors
    
    See equation (1) in this paper for mathematical expression:
    https://arxiv.org/abs/1705.08039
    '''
    (batch_shape, node, D) = x.shape
    a = (1 - square_norm(x)).view(batch_shape,node, 1)
    b = (1 - square_norm(x)).view(batch_shape,1, node)
    return torch.acosh(1 + 2 * pairwise_distances(x) / torch.matmul(a, b))


def generate_materials(x):
    return F.normalize(torch.cosh(poincare_distance(x)).pow(2), p = 1, dim = 2) 


def generate_materials_euc(x):
    return F.normalize((pairwise_distances(x)).pow(2), p = 1, dim = 2) 

class ModulatedGraphConv(nn.Module):
    """
    Semantic graph convolution layer
    """

    def __init__(self, in_features, out_features, adj, bias=True):
        super(ModulatedGraphConv, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.W = nn.Parameter(torch.zeros(size=(2, in_features, out_features), dtype=torch.float))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)

        self.M = nn.Parameter(torch.zeros(size=(adj.size(0), out_features), dtype=torch.float))
        nn.init.xavier_uniform_(self.M.data, gain=1.414)

        self.adj = adj

        self.m = (self.adj > 0)

        for i in range(0,22):
            for j in range(0,i):
                self.m[i,j] = False

        self.m_inv = (self.adj <= 0)

        for i in range(0,22):
            for j in range(0,i+1):
                self.m_inv[i,j] = False

        self.e_1 = nn.Sequential(
                            nn.Linear(132, 256),
                            nn.ReLU(),
                            nn.Dropout(0.1),
                            nn.Linear(256, 43)
            )
        self.e_2 = nn.Sequential(
                            nn.Linear(22*192, 22*96),
                            nn.ReLU(),
                            nn.Dropout(0.1),
                            nn.Linear(22*96, 43)
            ) 
        self.e_3 = nn.Sequential(
                            nn.Linear(132, 256),
                            nn.ReLU(),
                            nn.Dropout(0.1),
                            nn.Linear(256, 210)
            )
        self.e_4 = nn.Sequential(
                            nn.Linear(22*192, 22*96),
                            nn.ReLU(),
                            nn.Dropout(0.1),
                            nn.Linear(22*96, 210)
            ) 

        torch.nn.init.kaiming_uniform_(self.e_1[0].weight.data, a=0, mode='fan_in', nonlinearity='relu')
        self.e_1[0].weight.data.mul_(1)
        torch.nn.init.kaiming_uniform_(self.e_2[0].weight.data, a=0, mode='fan_in', nonlinearity='relu')
        self.e_2[0].weight.data.mul_(1)
        torch.nn.init.kaiming_uniform_(self.e_3[0].weight.data, a=0, mode='fan_in', nonlinearity='relu')
        self.e_3[0].weight.data.mul_(1)
        torch.nn.init.kaiming_uniform_(self.e_4[0].weight.data, a=0, mode='fan_in', nonlinearity='relu')
        self.e_4[0].weight.data.mul_(1)

        torch.nn.init.kaiming_uniform_(self.e_1[3].weight.data, a=0, mode='fan_in', nonlinearity='relu')
        self.e_1[3].weight.data.mul_(1)
        torch.nn.init.kaiming_uniform_(self.e_2[3].weight.data, a=0, mode='fan_in', nonlinearity='relu')
        self.e_2[3].weight.data.mul_(1)
        torch.nn.init.kaiming_uniform_(self.e_3[3].weight.data, a=0, mode='fan_in', nonlinearity='relu')
        self.e_3[3].weight.data.mul_(1)
        torch.nn.init.kaiming_uniform_(self.e_4[3].weight.data, a=0, mode='fan_in', nonlinearity='relu')
        self.e_4[3].weight.data.mul_(1)


        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features, dtype=torch.float))
            stdv = 1. / math.sqrt(self.W.size(2))
            self.bias.data.uniform_(-stdv, stdv)
        else:
            self.register_parameter('bias', None)

    def forward(self, input):

        batch_size, _, hid_dim_node = input.shape

        h0 = torch.matmul(input, self.W[0])
        h1 = torch.matmul(input, self.W[1])
        
        adj_1 = self.adj.repeat(batch_size,1,1).to(input.device)
        if hid_dim_node == 6:
            adj_1[:,self.m_inv] = self.e_3(input.view(-1,132))
            adj_1[:,self.m] = adj_1[:,self.m] + self.e_1(input.view(-1,132))
            adj = adj_1.transpose(1,2) + adj_1
        else:
            
            adj_1[:,self.m_inv] = self.e_4(input.reshape(-1,22*192))
            adj_1[:,self.m] = adj_1[:,self.m] + self.e_2(input.reshape(-1,22*192))
            adj = adj_1.transpose(1,2) + adj_1
        E = torch.eye(self.adj.size(0), dtype=torch.float).to(input.device)
        
        output = torch.matmul(adj * E, self.M*h0) + torch.matmul(adj * (1 - E), self.M*h1)
        if self.bias is not None:
            return output + self.bias.view(1, 1, -1)
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'
