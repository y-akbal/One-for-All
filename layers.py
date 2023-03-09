import torch
from torch import nn as nn
from torch.nn import functional as F


class attention(nn.Module):
    def __init__(self, d_in, d_out, causal = True):
        super().__init__()
        self.softmax = nn.Softmax(1)
        self.causal_factor = torch.triu(-torch.inf*torch.ones(d_in,d_in), diagonal = 1)
        self.d_out = d_out
        self.causal = causal
    def forward(self, x):
        correlation_matrix = (x @ x.transpose(-1, -2))/self.d_out**(0.5)
        if self.causal:
            return correlation_matrix, self.softmax(correlation_matrix+self.causal_factor) 
        return correlation_matrix, self.softmax(correlation_matrix)   

    

class Linear(nn.Module):
    def __init__(self, d_in, d_out, bias = False):
        super().__init__()
        self.M = torch.randn(d_out, d_in)
        if not bias:
            self.b = torch.zeros(d_out, 1)
        self.bias = bias    
        
    def forward(self, x):
        if self.bias:
            return self.M @ x + self.b
        return self.M @ x
    
    
class Upsampling(nn.Module):
    def __init__(self, d_out = 128, kernel_size = 8, bias = False):
        Conv = nn.Conv1d(in_channels = 1, 
                         out_channels = d_out, 
                         kernel_size = kernel_size,
                         bias = bias
                        )
        #Activation = F.gelu
        LN = nn.LayerNorm(128)
        Dense = nn.Linear(d_out, d_out, bias = False)
    def forward(self, x):
        time_series, positional_embeddings, ts_embedding = x
        pass
