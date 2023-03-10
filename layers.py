import torch
from torch import nn as nn
from torch.nn import functional as F
from torch import vmap



class Linear(nn.Module):  #B*H*L -> B*H'*L adjusted the columns in each batch, no touch rows directly
    def __init__(self, d_in, d_out, bias = False):
        super().__init__()
        self.M = torch.randn(d_out, d_in)/d_in**0.5 ### Xavier initialization
        if not bias:
            self.b = torch.zeros(d_out, 1)
        self.bias = bias    

    def forward(self, x):
        res = self.M @ x
        if self.bias:
            res += self.b
        return res 

class single_att_head(nn.Module):
    def __init__(self, d_in, causal = True,**kwargs):
        super().__init__(**kwargs)
        self.d_in = d_in
        self.causal = causal
        self.W = torch.randn(d_in, d_in)/(d_in**0.5) ### Xavier init, HxH
        if self.causal:
            self.causal_factor = torch.tril(-torch.inf*torch.ones(d_in,d_in), diagonal = -1)

    def forward(self, x):
        x_ = self.W @ x # BxHxH, BxHxW -> BxHxW
        corr_mat = (x_.transpose(-1, -2) @ x_)/self.d_in**0.5  #BxWxH,BxHxW -> BxWxW        
        if self.causal: ## killing att. to future if asked
            corr_mat += self.causal_factor
        
        softmaxed = F.softmax(corr_mat, 1) #BxWxW -> BxWxW, softmaxed along axis 1
        return x @ softmaxed #BxWxW, BxHxW

class multi__att_head(nn.Module):
    def __init__(self, d_in, num_heads = 5, causal = True, **kwargs):
        super().__init__(**kwargs)
        self.d_in = d_in
        self.num_heads = num_heads
        self.heads = [single_att_head(d_in) for i in range(num_heads)]
        self.aggregator = nn.Conv2d(num_heads, 1, 1,1, bias = True)
        nn.init.constant_((self.aggregator).weight, 1/num_heads) ### here we start by averaging
        nn.init.constant_((self.aggregator).bias, 0) ### here we start by averaging

    def forward(self, x):
        heads_x = [torch.unsqueeze(self.heads[i](x),0) for i in range(self.num_heads)]
        concatted_heads = torch.concat(heads_x, 1)
        aggregated_heads = self.aggregator(concatted_heads)
        return torch.squeeze(aggregated_heads, dim = 1)


class Upsampling(nn.Module):
    def __init__(self, d_out = 128, kernel_size = 8, conv_bias = False, att_heads = 5):
        self.Conv = nn.Conv1d(in_channels = 1, 
                         out_channels = d_out, 
                         kernel_size = kernel_size,
                         bias = conv_bias
                        )
        self.heads = multi__att_head(d_out, num_heads = att_heads)
        self.activation = F.gelu
        self.normalization = nn.LayerNorm(d_out)
        self.dense = Linear(d_out, d_out)
        
    def forward(self, x):
        time_series, positional_embeddings, ts_embedding = x
        pass
