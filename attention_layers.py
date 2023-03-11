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
        
        ### Simple Stuff ###
        self.__compiled__ = None
    
    @property
    def compiled(self):
        return self.__compiled__
    
    
    def __build__(self, shape):
        self.dims = None
        self.__compiled__ = True
    
    def forward(self, x): #BxHxL -> BxH'xL
        
        if not self.compiled: 
            # check if the layer 
            # compiled to see if 
            # the dims are set 
            # correctly
            self.__build__(x.shape)
            

        Q, K, V = x
        Q_d = self.L_Q(Q)  #BxHxL -> BxH'xL
        K_d = self.L_K(K)  #BxHxL -> BxH'xL
        V_d = self.L_V(V)  #BxHxL -> BxH'xL
        
        corr_mat = (Q_d.transpose(-1, -2) @ K_d)/self.d_k**0.5
        
        if self.causal:
            corr_mat += 1
        
        softmaxed = F.softmax(corr_mat, 0)
        return V_d @ softmaxed #BxH'xL
# 
    
class multi_head_attention(nn.Module):
    def __init__(self, n_heads = 5, dropout = 0.5):  
        # We will borrow some notation from tensorflow 
        # Thank you tensorflow, 
        # We appreciate your effort, 
        # You can sit down now!
        super().__init__()
        self.compiled = False
        self.n_heads = n_heads
        self.dropout = dropout
        pass
    def __build__(self, shape):
        #
        #
        #
        #
        pass 
    def forward(self, x): #concat[BxH'xL for i in range(H/H')] -> BxHxL
        if not self.compiled:
            pass ## do sth here
        
        Q, K, T = x
        pass
