import torch
from torch import nn as nn
from torch.nn import functional as F


class Linear(nn.Module):  # B*H*L -> B*H'*L adjusted the columns in each batch, no touch rows directly
    def __init__(self, d_in, d_out, bias=False, dropout=0.5):
        super().__init__()
        self.M = torch.randn(d_out, d_in)/d_in**0.5  # Xavier initialization
        if not bias:
            self.b = torch.zeros(d_out, 1)
        self.bias = bias
        self.dropout = dropout

    def forward(self, x):
        res = self.M @ x
        if self.bias:
            res += self.b
        return res


class single_att_head(nn.Module):
    def __init__(self, d_in, causal=True, **kwargs):
        super().__init__(**kwargs)
        self.d_in = d_in
        self.causal = causal
        self.W = torch.randn(d_in, d_in)/(d_in**0.5)  # Xavier init, HxH
        if self.causal:
            self.causal_factor = torch.tril(-torch.inf *
                                            torch.ones(128, 128), diagonal=-1)

    def forward(self, x):
        x_ = self.W @ x  # BxHxH, BxHxW -> BxHxW

        corr_mat = (x_.transpose(-1, -2) @ x_) / \
            self.d_in**0.5  # BxWxH,BxHxW -> BxWxW
        if self.causal:  # killing att. to future if asked
            corr_mat += self.causal_factor
        # BxWxW -> BxWxW, softmaxed along axis 1
        softmaxed = F.softmax(corr_mat, 1)
        return x @ softmaxed  # BxHxW


class multi__att_head(nn.Module):
    def __init__(self, d_in, num_heads=5, causal=True, **kwargs):
        super().__init__(**kwargs)
        self.d_in = d_in
        self.num_heads = num_heads
        self.heads = [single_att_head(d_in, causal=causal)
                      for i in range(num_heads)]
        self.aggregator = nn.Conv2d(num_heads, 1, 1, 1, bias=True)
        # here we start by averaging
        nn.init.constant_((self.aggregator).weight, 1/num_heads)
        nn.init.constant_((self.aggregator).bias, 0)

    def forward(self, x):
        heads_x = [self.heads[i](x)
                   for i in range(self.num_heads)]

        concatted_heads = torch.stack(heads_x, 1)
        aggregated_heads = self.aggregator(concatted_heads)
        return torch.squeeze(aggregated_heads)


class Upsampling(nn.Module):

    def __init__(self, ts_used: int = 5,
                 lags: int = 512,
                 d_out=128,
                 pool_size=4,
                 conv_bias=False,
                 dense_bias=False,
                 att_heads=5,
                 activation=F.gelu,
                 num_of_ts=25,
                 **kwargs):
        super().__init__(**kwargs)

        assert (
            lags/pool_size).is_integer(), "Make sure that lag size is divisible by pool_size"
        self.num_pools = int(lags/pool_size)
        self.num_of_ts = num_of_ts
        self.lags = lags
        self.Conv = nn.Conv1d(in_channels=1,
                              out_channels=d_out,
                              kernel_size=pool_size,
                              stride=pool_size,
                              bias=conv_bias
                              )

        self.heads = multi__att_head(4,
                                     num_heads=att_heads)
        self.activation = activation
        self.normalization = nn.LayerNorm(self.num_pools)
        self.dense = Linear(d_out,
                            d_out,
                            bias=dense_bias)

        ## -- Begining of Embedding Layers -- ##
        self.num_enum = torch.tensor([i for i in range(self.num_pools)])
        # positional embedding of pools
        self.pe_embedding = nn.Embedding(self.num_pools, d_out)
        # positional embeddings of time series
        self.ts_embedding = nn.Embedding(self.num_of_ts, d_out)
        ## -- End of Embedding Layers -- ##

        ## Attention Part ##
        self.att = multi__att_head(d_in=d_out, num_heads=5, causal=True)

    def forward(self, x: tuple) -> torch.Tensor:
        ts, te = x
        assert ts.shape[-1] == self.lags, f"{self.lags} is not equal to {ts.shape[-1]}"

        # ts: Bx1xW (W here is used for Lags) the raw time series,
        # pe: (BxHxW) positional embeddings of time series,
        # te: (Embedding (geospatial) of the time series depending).

        convolved_ts = self.Conv(ts)  # Bx1xW -> BxHxW

        # BxHxW += #BxHxW (WxH -> HxW)   # Position embedding of pools
        convolved_ts += self.pe_embedding(self.num_enum).transpose(-1, -2)
        activated = self.activation(convolved_ts)  # BxHxW -> BxHxW
        normalized = self.normalization(activated)  # BxHxW -> BxHxW
        # BxHxW -> BxHxW (Dense layer is applied H dim)
        dense_applied = self.dense(normalized)
        # BxHxW += #BxHx1 (WxH -> HxW)   # Position embedding of time series
        dense_applied += self.ts_embedding(te).transpose(-1, -2)
        attention_calcd = self.att(dense_applied)
        attention_calcd += convolved_ts
        normalized = self.normalization(attention_calcd)  # BxHxW -> BxHxW
        return normalized


layer = Upsampling(pool_size=4, d_out=4)

Upsampling(pool_size=4, d_out=4)(
    (torch.randn(3, 1, 512), torch.tensor([[23], [2], [5]]))).shape


x.shape
x.shape
