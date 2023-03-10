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
            self.causal_factor = torch.tril(-torch.inf*torch.ones(512,512), diagonal = -1)
    
    def forward(self, x):
        x_ = self.W @ x # BxHxH, BxHxW -> BxHxW
        
        corr_mat = (x_.transpose(-1, -2) @ x_)/self.d_in**0.5  #BxWxH,BxHxW -> BxWxW        
        
        if self.causal: ## killing att. to future if asked
            corr_mat += self.causal_factor
        
        softmaxed = F.softmax(corr_mat, 1) #BxWxW -> BxWxW, softmaxed along axis 1
        return x @ softmaxed #BxHxW


single_att_head(10)(torch.randn(1,10,512))




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
    def __init__(self, ts_used: int = 5, 
                 lags: int = 512, 
                 d_out = 128, 
                 pool_size = 4, 
                 conv_bias = False, 
                 att_heads = 5,
                 activation = F.gelu, 
                 **kwargs):
        super().__init__(**kwargs)
        assert (lags/pool_size).is_integer(), "Make sure that lag size is divisible by pool_size"
        self.num_pools = int(lags/pool_size)
        self.lags = lags
        self.Conv = nn.Conv1d(in_channels = 1, 
                         out_channels = d_out, 
                         kernel_size = pool_size,
                         stride = pool_size,
                         bias = conv_bias
                        )
        
        self.heads = multi__att_head(self.num_pools, num_heads = att_heads)
        self.activation = activation
        self.normalization = nn.LayerNorm(self.num_pools)
        self.dense = Linear(d_out, d_out)
        
        ## -- Embedding Layers -- ##
        self.ts_embedding = nn.Embedding(self.num_pools, self.num_pools)
        
        
        
        
    def forward(self, x :tuple) -> torch.Tensor:
        ts, pe, te = x 
        assert ts.shape[-1] == self.lags, f"{self.lags} is not equal to {ts.shape[-1]}"
        # ts: Bx1xW (W here is used for Lags), 
        # pe: (BxHxW) positional embeddings of time series, 
        # te: (Embedding (geospatial) of the time series depending)
        convolved_ts = self.Conv(ts)
        convolved_ts += self.ts_embedding(pe)  ## we add embeddings 
        activated = self.activation(convolved_ts)
        normalized = self.normalization(activated) ##Layer normalization
        dense_applied = self.dense(normalized)    
        #dense_applied += self.ts_emmebding(te)
        
            
        return dense_applied



m = Upsampling(pool_size = 4, d_out = 4)((torch.randn(2,1,512), torch.tensor([[1],[2]]), torch.tensor([[2],[3]])))

m.shape

multi__att_head(4)

m.shape

m[0].shape


m[1].shape
