import torch
from torch import nn as nn
from torch.nn import functional as F



class Linear(nn.Module):  #B*H*L -> B*H'*L adjusted the columns in each batch, no touch rows directly
    def __init__(self, d_in, d_out, bias = False, dropout = 0.5):
        super().__init__()
        self.M = torch.randn(d_out, d_in)/d_in**0.5 ### Xavier initialization
        if bias:
            self.b = torch.ones(d_out, 1)
        self.bias = bias    
        self.dropout = nn.Dropout(p = dropout)

    def forward(self, x):
        res = self.M @ x
        if self.bias:
            res += self.b
        res = self.dropout(res)            
        return res


class single_head_attention(nn.Module): 
    def __init__(self, d_in, d_out, dropout = 0.2, causal = True, lag = 512):
        super().__init__()
        
        assert (d_in/d_out).is_integer(), f"d_in/d_out should be integer while yours {d_in/d_out} "
        
        self.d_in = d_in  
        self.d_out = d_out  
        self.causal = causal
        self.W = lag
        ## EXP: d_in = H, d_out = H'    
        self.L_Q = Linear(d_in, d_out, dropout = dropout)  
        self.L_K = Linear(d_in, d_out, dropout = dropout)
        self.L_V = Linear(d_in, d_out, dropout = dropout)
        
        if self.causal:
            self.causal_factor = torch.tril(-torch.inf*torch.ones(self.W,self.W), diagonal = -1)
    
    def forward(self, x): #BxHxL -> BxH'xL
        Q, K, V = x
        Q_d = self.L_Q(Q)  #BxHxL -> BxH'xW
        K_d = self.L_K(K)  #BxHxL -> BxH'xW
        V_d = self.L_V(V)  #BxHxL -> BxH'xW
        ## Correlation
        corr_mat = (Q_d.transpose(-1, -2) @ K_d)/self.d_out**0.5 #B'xWxW
        
        if self.causal:
            corr_mat += self.causal_factor
        
        softmaxed = F.softmax(corr_mat, 1) #BxWxW
        
        return V_d @ softmaxed #BxH'xW, BxWxW -> BxH'xW

    
class multi_head_attention(nn.Module):
    def __init__(self, n_heads = 5, dropout = 0.5, causal = True):  
        # We will borrow some lazyness of TensorFlow 
        # Thank you TensorFlow, 
        # We appreciate your effort, 
        # You can sit down now!
        super().__init__()
        self.__compiled__ = False
        self.n_heads = n_heads
        self.dropout = dropout
        self.causal = causal
        
    @property
    def compiled(self):
        return self.__compiled__
    
    @compiled.setter
    def compiled(self, x):
        assert False, "You should first compile the model by doing at least one forward pass!!!"
    
    def __build__(self, shape):
        _, d_in, lags = shape
        assert (d_in/self.n_heads).is_integer(), f"{d_in/self.n_heads} is not an integer"
        self.heads = [single_head_attention(d_in, d_in//self.n_heads, self.dropout, self.causal, lags) for i in range(self.n_heads)]
        ## !!! Yeah !!! ##                        
        self.__compiled__ = True
        self.final_linear = Linear(d_in, d_in, dropout = self.dropout, bias = True)
                
    def forward(self, x): #concat[BxH'xL for i in range(H/H')] -> BxHxL
        if not self.compiled:
            assert len(x) == 3, "True shape can not be referred!"
            shape_ = x[-1].shape
            self.__build__(shape_)
            
        Q, K, V = x            
        Forward_heads = [self.heads[i]([Q, K, V]) for i in range(self.n_heads)] #[BxH'xW for i in range(H/H')] 
        concatted_heads = torch.concat(Forward_heads, 1) #[BxH'xW for i in range(H/H')]  -> BxHxW
        return self.final_linear(concatted_heads)  #BxHxW -> BxHxW


