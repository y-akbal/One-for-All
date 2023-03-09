import torch
from torch import nn as nn
from torch.nn import functional as F


class Linear(nn.Module):
    def __init__(self, d_in, d_out, bias = False):
        super().__init__()
        self.M = torch.randn(d_out, d_in)/d_in**0.5 ### Xavier initialization
        if not bias:
            self.b = torch.zeros(d_out, 1)
        self.bias = bias    
    
        
    def forward(self, x):
        res = self.M @ x + self.b
        if self.bias:
            return res
        return res



class attention(nn.Module):
    def __init__(self, d_in, d_out = None, causal = True):
        super().__init__()
        self.softmax = nn.Softmax(1)
        self.causal_factor = torch.triu(-torch.inf*torch.ones(d_in,d_in), diagonal = 1)
        self.d_out = d_out
        self.causal = causal
        #self.dense = nn.Linear(d_in, d_in, bias = False)
        

    def forward(self, x):
            
        x_ = self.dense(x)
        correlation_matrix = (x_ @ x_.transpose(-1, -2))/self.d_out**(0.5)

        if self.causal:
            return self.softmax(correlation_matrix+self.causal_factor) @ x
        return self.softmax(correlation_matrix) @ x



class Upsampling(nn.Module):
    def __init__(self, d_out = 128, kernel_size = 8, bias = False):
        Conv = nn.Conv1d(in_channels = 1, 
                         out_channels = d_out, 
                         kernel_size = kernel_size,
                         bias = bias
                        )
        #Activation = F.gelu
        ##
        ##
        LN = nn.LayerNorm(d_out) ####### layer normalization ###################################
        Dense = nn.Linear(d_out, d_out, bias = False)  ### dense layer #########################
        attention = attention(d_in = d_out, d_in = d_out, causal = True) ### atention layer ####
        ##
        ##
        
        
    def forward(self, x):
        time_series, positional_embeddings, ts_embedding = x
        pass
    
    