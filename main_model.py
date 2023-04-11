import torch
from torch import nn as nn
from torch.nn import functional as F

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

import attention_layers
from attention_layers import block, Upsampling, layernorm

mod = Upsampling(pool_size=4, d_out=4)
mod = mod.to("cuda")


s = torch.tensor([[24], [2], [5]]).to(device)
l = torch.randn(3, 1, 512).to(device)
model = nn.Sequential(mod)
model.to("cuda")
mod([l,s])


class main_model(nn.Module):
    def __init__(self, lags = 512, 
                 embedding_dim = 48, 
                 n_blocks = 3, 
                 pool_size = 4, 
                 dropout = 0.3,
                 number_of_heads = 3*4, 
                 number_ts = 25,
                 device = device
                 ):
        assert (lags/pool_size).is_integer(), "Lag size should be divisible by pool_size"
        super().__init__()
        self.width = (lags//pool_size)
        self.embedding_dim = embedding_dim
        ### 
        self.blocks = nn.ModuleList([block(embedding_dim, width = self.width, dropout = dropout, n_heads = number_of_heads) for _ in range(n_blocks)] )
        self.up_sampling = Upsampling(lags = lags, d_out = self.embedding_dim, pool_size = pool_size, num_of_ts = number_ts, att_heads = number_of_heads, device = device)
        ###
    def forward(self, x,y):
        x = self.up_sampling((x,y))
        for layer in self.blocks:
            x = layer(x)
        return x


mod = main_model()
mod.to(device)
mod(l,s)

s = torch.tensor([[1]]).to("cuda")
s.shape
l = torch.randn(1, 1,512).to("cuda")
l.shape
mod = main_model()
mod.to("cuda")
mod(l,s).shape


import numpy as np
lay = 0
for i in mod.parameters():
   
    lay += np.array(i.shape).prod()
lay