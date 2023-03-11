import torch
from torch import nn as nn
from torch.nn import functional as F



class Linear(nn.Module):  #B*H*L -> B*H'*L adjusted the columns in each batch, no touch rows directly
    def __init__(self, d_in, d_out, bias = False, dropout = 0.5):
        super().__init__()
        self.M = torch.randn(d_out, d_in)/d_in**0.5 ### Xavier initialization
        if not bias:
            self.b = torch.zeros(d_out, 1)
        self.bias = bias    
        self.dropout = nn.Dropout(p = dropout)

    def forward(self, x):
        res = self.M @ x
        if self.bias:
            res += self.b
        res = self.dropout(res)            
        return res 


class single_head_attention(nn.Module): 
    def __init__(self, d_in, d_out, dropout = 0.2, causal = True):
        super().__init__()
        self.L_Q = Linear(d_in, d_out, dropout = dropout)
        self.L_K = Linear(d_in, d_out, dropout = dropout)
        self.L_V = Linear(d_in, d_out, dropout = dropout)
        
    def forward(self, x): #BxHxL -> BxH'xL
        Q, K, V = x
        Q_d = self.L_Q(Q)  #BxHxL -> BxH'xL
        K_d = self.L_K(K)  #BxHxL -> BxH'xL
        V_d = self.L_V(V)  #BxHxL -> BxH'xL

        
        
        
        pass
    
class multi_head_attention(nn.Module):
    def __init__(self, ):  ## We will borrow some notation from tensorflow --- thank you tensorflow, we appreciate your effort, you can sit down now!
        super().__init__()
        self.compiled = False
        pass
    def __build__(self, shape):
        pass
    def forward(self, x): #BxHxL -> concat[BxH'xL for i in range(H/H')] = BxHxL
        Q, K, T = x
        pass
