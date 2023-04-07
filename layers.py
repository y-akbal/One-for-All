import torch
from torch import nn as nn
from torch.nn import functional as F


class Linear(nn.Module):  # B*H*W -> B*H'*W adjusted the columns in each batch, no touch to rows directly
    def __init__(self, d_in, d_out, bias=False, dropout = 0.5):
        super().__init__()
        self.M = nn.Parameter(torch.randn(d_out, d_in, requires_grad = True)*d_in**(-0.5))  # Xavier initialization
        self.bias = bias
        if self.bias:
            self.b = nn.Parameter(torch.zeros(d_out, 1, requires_grad = True)) 
        
        self.dropout = nn.Dropout(dropout, inplace = True)
    def forward(self, x): # Bxd_inxW -> Bxd_outxW
        x = self.dropout(x)
        res = self.M @ x
        if self.bias:
            res += self.b
        return res

    

class old_school_att_head(nn.Module): ## There is no decrease in dimension here!
    """_summary_
    self attention --- no dimensionality reduction we have here!!!!
    # BxHxH, BxHxW -> BxHxW 
    """
    def __init__(self, d_in, d_out = 128, causal=True, **kwargs):
        super().__init__(**kwargs)
        self.d_in = d_in
        self.causal = causal
        self.W = nn.Parameter(torch.randn(d_in, d_in, requires_grad = True)*(d_in**-0.5))  # Xavier init, HxH
        if self.causal:
            self.causal_factor = torch.tril(-torch.inf *
                                            torch.ones(d_out, d_out), diagonal=-1)

    def forward(self, x):
        x_ = self.W @ x  # BxHxH, BxHxW -> BxHxW 

        corr_mat = (x_.transpose(-1, -2) @ x_)*self.d_in**-0.5  # BxWxH,BxHxW -> BxWxW
        if self.causal:  # killing att. to future if asked
            corr_mat += self.causal_factor 
        softmaxed = F.softmax(corr_mat, 1) # BxWxW -> BxWxW, softmaxed along axis 1
        return x @ softmaxed  # BxHxW


class old_school_multi__att_head(nn.Module): 
    def __init__(self, d_in, d_out = 128, num_heads=5, causal=True, **kwargs):
        super().__init__(**kwargs)
        self.d_in = d_in
        self.num_heads = num_heads
        self.heads = [old_school_att_head(d_in, d_out = d_out, causal=causal)
                      for i in range(num_heads)]
        self.aggregator = nn.Conv2d(num_heads, 1, 1, 1, bias=True)
        # here we start by averaging (Conv2D setting weights and biases as averaging initially).
        nn.init.constant_((self.aggregator).weight, 1/num_heads)
        nn.init.constant_((self.aggregator).bias, 0)

    def forward(self, x):  # BxHxW ->  Bx(num_heads*H)xW -> Conv2D (aggreagate) -> BxHxW 
        heads_x = [self.heads[i](x)
                   for i in range(self.num_heads)]

        concatted_heads = torch.stack(heads_x, 1)
        aggregated_heads = self.aggregator(concatted_heads)
        return torch.squeeze(aggregated_heads, dim = 1)



class Upsampling(nn.Module):
    def __init__(self, 
                 lags: int = 512, ### input dimension (width)
                 d_out = 128, ## output dimension (height)
                 pool_size = 4, ## pool_sizes
                 conv_bias = True, 
                 dense_bias = False, 
                 att_heads = 5, ### attention heads to be used
                 activation = F.gelu, 
                 num_of_ts=25, ### number of time series to be used
                 **kwargs):
        super().__init__(**kwargs)

        assert (lags/pool_size).is_integer(), "Make sure that lag size is divisible by pool_size"
        
        self.num_pools = int(lags/pool_size)
        self.num_of_ts = num_of_ts
        self.lags = lags
        self.Conv = nn.Conv1d(in_channels=1,
                              out_channels=d_out,
                              kernel_size=pool_size,
                              stride=pool_size,
                              bias=conv_bias
                              )

        self.heads = old_school_multi__att_head(4,
                                     num_heads=att_heads)
        self.activation = activation
        self.normalization_1 = nn.LayerNorm(self.num_pools)
        self.normalization_2 = nn.LayerNorm(self.num_pools)
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
        self.att = old_school_multi__att_head(d_in = d_out, num_heads = 5, causal = True)

    def forward(self, x: tuple) -> torch.Tensor:
        ts, te = x ## split 
        assert ts.shape[-1] == self.lags, f"{self.lags} is not equal to {ts.shape[-1]}"

        # ts: Bx1xW (W here is used for Lags) the raw time series,
        # pe: (BxHxW) positional embeddings of time series,
        # te: (Embedding (geospatial) of the time series depending).

        convolved_ts = self.Conv(ts)  # Bx1xW -> BxHxW

        # BxHxW += #BxHxW (WxH -> HxW)   # Position embedding of pools
        convolved_ts += self.pe_embedding(self.num_enum).transpose(-1, -2)
        activated = self.activation(convolved_ts)  # BxHxW -> BxHxW
        normalized = self.normalization_1(activated)  # BxHxW -> BxHxW
        # BxHxW -> BxHxW (Dense layer is applied H dim)
        dense_applied = self.dense(normalized)
        # BxHxW += #BxHx1 (WxH -> HxW)   # Position embedding of time series
        dense_applied += self.ts_embedding(te).transpose(-1, -2)
        attention_calcd = self.att(dense_applied)
        attention_calcd += convolved_ts
        normalized = self.normalization_2(attention_calcd)  # BxHxW -> BxHxW
        return normalized


Upsampling(pool_size=4, d_out=4)((torch.randn(3, 1, 512), torch.tensor([[24], [2], [5]]))).shape

