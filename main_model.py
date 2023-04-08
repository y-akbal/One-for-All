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


from attention_layers import block, Upsampling
mod = Upsampling(pool_size=4, d_out=4)
mod = mod.to("cuda")


s = torch.tensor([[24], [2], [5]]).to(device)
l = torch.randn(3, 1, 512).to(device)

model = nn.Sequential(mod)

s.device


class main_model(nn.Module):
    def __init__(self, lags = 512, 
                 embedding_dim = 768, 
                 n_blocks = 30, 
                 pool_size = 4, 
                 dropout = 0.3,
                 number_of_heads = 4, 
                 number_ts = 25
                 ):
        assert (lags/pool_size).is_integer(), "Lag size should be divisible by pool_size"
        super().__init__()
        self.width = (lags//pool_size)
        self.embedding_dim = embedding_dim
        ### 
        self.blocks = nn.ModuleList([block(embedding_dim, width = self.width, n_heads = 4, dropout = dropout, n_heads = number_of_heads) for _ in range(n_blocks)] )
        self.up_sampling = Upsampling(lags = 512, d_out = self.embedding_dim, pool_size = pool_size, num_of_ts = number_ts, att_heads = number_of_heads)
        ###
    def forward(self, x):
        x = self.up_sampling(x)

        return x

s