import torch
import torch.nn as nn
from einops.layers.torch import Rearrange
import numpy as np
from torch.nn import functional as F
import time
from torch.nn.parameter import Parameter
import math


class GraphConvolution(nn.Module):
    """
    adapted from : https://github.com/tkipf/gcn/blob/92600c39797c2bfb61a508e52b88fb554df30177/gcn/layers.py#L132
    """

    def __init__(self, in_features, out_features, bias=True, node_n=48):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features), requires_grad=True)
        self.att = Parameter(torch.FloatTensor(node_n, node_n), requires_grad=True)
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features), requires_grad=True)
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        self.att.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input):
        support = torch.matmul(input, self.weight)
        output = torch.matmul(self.att, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class GC_Block(nn.Module):
    def __init__(self, in_features, p_dropout, bias=True, node_n=48):
        """
        Define a residual block of GCN
        """
        super(GC_Block, self).__init__()
        self.in_features = in_features
        self.out_features = in_features

        self.gc1 = GraphConvolution(in_features, in_features, node_n=node_n, bias=bias)
        self.bn1 = nn.BatchNorm1d(node_n * in_features)

        self.gc2 = GraphConvolution(in_features, in_features, node_n=node_n, bias=bias)
        self.bn2 = nn.BatchNorm1d(node_n * in_features)

        self.do = nn.Dropout(p_dropout)
        self.act_f = nn.Tanh()

    def forward(self, x):
        y = self.gc1(x)
        b, n, f = y.shape
        y = self.bn1(y.view(b, -1)).view(b, n, f)
        y = self.act_f(y)
        y = self.do(y)

        y = self.gc2(y)
        b, n, f = y.shape
        y = self.bn2(y.view(b, -1)).view(b, n, f)
        y = self.act_f(y)
        y = self.do(y)

        return y + x

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class GCN(nn.Module):
    def __init__(self, input_feature, hidden_feature, p_dropout, num_stage=1, node_n=48):
        """

        :param input_feature: num of input feature
        :param hidden_feature: num of hidden feature
        :param p_dropout: drop out prob.
        :param num_stage: number of residual blocks
        :param node_n: number of nodes in graph
        """
        super(GCN, self).__init__()
        self.num_stage = num_stage

        self.gc1 = GraphConvolution(input_feature, hidden_feature, node_n=node_n)
        self.bn1 = nn.BatchNorm1d(node_n * hidden_feature)

        self.gcbs = []
        for i in range(num_stage):
            self.gcbs.append(GC_Block(hidden_feature, p_dropout=p_dropout, node_n=node_n))

        self.gcbs = nn.ModuleList(self.gcbs)

        self.gc7 = GraphConvolution(hidden_feature, input_feature, node_n=node_n)

        self.do = nn.Dropout(p_dropout)
        self.act_f = nn.Tanh()

    def forward(self, x):
        y = self.gc1(x)
        b, n, f = y.shape
        y = self.bn1(y.view(b, -1)).view(b, n, f)
        y = self.act_f(y)
        y = self.do(y)

        for i in range(self.num_stage):
            y = self.gcbs[i](y)

        y = self.gc7(y)
        y = y + x

        return y
    

def get_dct_matrix(N):
    dct_m = np.eye(N)
    for k in np.arange(N):
        for i in np.arange(N):
            w = np.sqrt(2 / N)
            if k == 0:
                w = np.sqrt(1 / N)
            dct_m[k, i] = w * np.cos(np.pi * (i + 1 / 2) * k / N)
    idct_m = np.linalg.inv(dct_m)
    return dct_m, idct_m
        
    
class Futuremotion(nn.Module):

    def __init__(self):
        super().__init__()
        
        self.gcn = GCN(input_feature=30, node_n=39, hidden_feature=256, p_dropout=0.1, num_stage=12)

        dct_m,idct_m = get_dct_matrix(30)
        self.dct_m = torch.tensor(dct_m).float().to("cuda")
        self.idct_m = torch.tensor(idct_m).float().to("cuda")
    def _prepare_in(self, x):

        N_KPS = 13
        INPUT_LEN, OUTPUT_LEN, dct_n = 16, 14, 30
        SL = INPUT_LEN + OUTPUT_LEN

        i_idx = np.append(np.arange(0, INPUT_LEN), np.repeat([INPUT_LEN - 1], OUTPUT_LEN))
        x = x.clone().reshape(-1, x.shape[1], x.shape[2]*x.shape[3]).float()

        x = x.transpose(0, 1).reshape(x.shape[1], -1)
        
        x = torch.matmul(self.dct_m[0:dct_n, :SL], x[i_idx, :])
           
        x = x.transpose(0, 1).reshape(-1, N_KPS*3, dct_n)

        x = x.transpose(1, 2)

        return x
    
    def _prepare_out(self, x):
        N_KPS = 13
        INPUT_LEN, OUTPUT_LEN, dct_n = 16, 14, 30
        SL = INPUT_LEN + OUTPUT_LEN

        y = x.transpose(1, 2)

        y = y.reshape(-1, dct_n).transpose(0, 1)

        y = torch.matmul(self.idct_m[:SL, :dct_n], y)
        
        y = y.transpose(0, 1).contiguous().view(-1, N_KPS*3, SL).transpose(1, 2)            
        
        return y
            
    def forward(self, x, y):
        
        return self.fwd(x), self.fwd(y)
    
    def fwd(self, x):
        
        orig = x.clone()
        
        x = x - x[:, 0:1, 0:1] # substract first frame, midhip [coordinate transform]
        
        x = self._prepare_in(x)
        
        x = self.gcn(x.transpose(1, 2)).transpose(1, 2)
            
        x = self._prepare_out(x)
        
        x = x.reshape(-1, 30, 13, 3)
            
        x = x + orig[:, 0:1, 0:1] # add first frame, midhip [coordinate transform]
        
        return x
    
    
class Futuremotion_ICCV21(nn.Module):
    def __init__(self):
        super().__init__()

        self.m = Futuremotion().float()

    def forward(self, data, train=True):
        
        x = data['keypoints0'].reshape(-1, 16, 13, 3).float()
        y = data['keypoints1'].reshape(-1, 16, 13, 3).float()
        
        _x, _y = self.m(x, y)
        _x, _y = _x.reshape(-1, 30, 13, 3), _y.reshape(-1, 30, 13, 3)

        _x, _y = _x[:, -14:], _y[:, -14:]
        
        data.update({'z0': _x, 'z1': _y})
        
        return data    