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
        
        self.dropout = nn.Dropout(dropout, inplace = False)
    def forward(self, x): # Bxd_inxW -> Bxd_outxW
        x = self.dropout(x) ## apply dropout!!!
        res = self.M @ x
        if self.bias:
            res += self.b
        return res




class FFN(nn.Module):
    def __init__(self, d_in, expansion_size = 2, dropout = 0.2, activation = nn.ReLU()) -> None:
        super().__init__()
        self.linear = nn.Sequential(
            Linear(d_in= d_in, d_out = expansion_size*d_in, dropout = dropout),
            activation,
            Linear(d_in= expansion_size*d_in, d_out = d_in, dropout = dropout),
        )
    def forward(self, x):
        return self.linear(x)



class single_head_attention(nn.Module): 
    def __init__(self, d_in, d_out, lag = 512, dropout = 0.2, causal = True):
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
            self.causal_factor = nn.Parameter(torch.tril(-torch.inf*torch.ones(self.W,self.W), diagonal = -1))
    
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
    def __init__(self, d_in, lags, n_heads = 5, dropout = 0.5, causal = True):  
        super().__init__()
        self.__compiled__ = False
        self.n_heads = n_heads
        self.dropout = dropout
        self.causal = causal
        assert (d_in/self.n_heads).is_integer(), f"{d_in/self.n_heads} is not an integer"
        ## -- ##
        self.heads = nn.ModuleList([single_head_attention(d_in, d_in//self.n_heads, lag = lags, dropout = self.dropout, causal = self.causal) for i in range(self.n_heads)])
        ## !!! Yeah !!! ##                        
        self.__compiled__ = True
        ### This next layer is FFN after the attention layer (used for memorization):
        self.final_linear = Linear(d_in, d_in, dropout = self.dropout, bias = True)

    def forward(self, x): #concat[BxH'xL for i in range(H/H')] -> BxHxL
        Q, K, V = x            
        Forward_heads = [self.heads[i]([Q, K, V]) for i in range(self.n_heads)] #[BxH'xW for _ in range(H/H')] 
        concatted_heads = torch.concat(Forward_heads, 1) #[BxH'xW for _ in range(H/H')]  -> BxHxW
        return self.final_linear(concatted_heads) #BxHxW -> BxHxW
    
    


class layernorm(nn.Module): # We noemalize the local copies not along time dimension
    def __init__(self, dim, eps = 1e-5):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(dim))
        self.beta = nn.Parameter(torch.zeros(dim))
    def forward(self,x):
        mean = x.mean(1, keepdims = True)
        var = x.var(1, keepdims = True)
        unit = (x-mean)/torch.sqrt(self.eps + var)
        return self.gamma*unit + self.beta ### B*H*W -> B*H*W 
    
    def parameters(self):
        return [self.gamma, self.beta]


# m = layernorm(512)
# m.to("cuda")
# torch.compile(m)
# x = torch.randn(1, 100, 512)
# x = x.to("cuda")
# m(x)


class block(nn.Module):
    def __init__(self, 
                 d_in, ### intermediate dimension
                 width = 128,  ### width of time series to be used
                 n_heads = 4, 
                 dropout = 0.5, 
                 causal = True,
                 expansion_size = 2, 
                 activation = nn.GELU()):
        super().__init__()
        self.att_head = multi_head_attention(n_heads = n_heads, 
                                             dropout = dropout,
                                             lags = width,
                                             d_in = d_in,
                                            causal = causal)
        self.FFN = FFN(d_in, 
                       expansion_size= expansion_size,
                       dropout = dropout,
                       activation = activation)
        self.ln1 = layernorm(width)
        self.ln2 = layernorm(width)
        
    def forward(self, x): #B*H*W -> B*H*W
        y = self.ln1(x)  
        y = self.att_head([y,y,y])
        y += x
        
        x = self.ln2(y)
        x = self.FFN(x)
        x = self.ln2(x)
        x += y
        
        
        return y
    
# block_ = block(128, 512)    
# block_.to("cuda")
# x = torch.randn(1, 128, 512).to("cuda")
# block_(x)


class Upsampling(nn.Module):
    def __init__(self, 
                 lags: int = 512, ### input dimension (width)
                 d_out = 128, ## output dimension (height)
                 pool_size = 4, ## pool_sizes
                 conv_bias = True, 
                 dense_bias = False, 
                 att_heads = 4, ### attention heads to be used
                 activation = F.gelu, 
                 num_of_ts=25, ### number of time series to be used
                 device = "cuda",
                 channel_shuffle = False, ### we add channel shuffle to trick on early attention layers
                 **kwargs):
        super().__init__(**kwargs)

        assert (lags/pool_size).is_integer(), "Make sure that lag size is divisible by pool_size"
        
        self.num_pools = int(lags/pool_size)
        self.num_of_ts = num_of_ts
        self.lags = lags
        self.channel_shuffle = channel_shuffle
        self.Conv = nn.Conv1d(in_channels=1,
                              out_channels=d_out,
                              kernel_size=pool_size,
                              stride=pool_size,
                              bias=conv_bias
                              )

        self.activation = activation
        self.normalization_1 = layernorm(self.num_pools) ### This part is important!!!
        self.normalization_2 = layernorm(self.num_pools) ### Fix this part, mind the error messages!!!
        self.dense = Linear(d_out,
                            d_out,
                            bias=dense_bias)

        ## -- Begining of Embedding Layers -- ##
        self.num_enum = torch.arange(self.num_pools, device = device if not None else "cuda")
        # positional embedding of pools
        self.pe_embedding = nn.Embedding(self.num_pools, d_out)
        # positional embeddings of time series
        self.ts_embedding = nn.Embedding(self.num_of_ts, d_out)
        ## -- End of Embedding Layers -- ##

        ## Attention Part ##
        self.att = multi_head_attention(n_heads = att_heads, causal = True, d_in = d_out, lags = self.num_pools)
        ## Channel Shuffle ##
        if channel_shuffle: 
            self.shuffle = nn.ChannelShuffle(2)

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
        if self.channel_shuffle:
            normalized = self.channel_shuffle(normalized)
            
        # BxHxW -> BxHxW (Dense layer is applied H dim)
        dense_applied = self.dense(normalized)
        # BxHxW += #BxHx1 (WxH -> HxW)   # Position embedding of time series
        dense_applied += self.ts_embedding(te).transpose(-1, -2)
        
        attention_calcd = self.att([dense_applied, dense_applied, dense_applied])
        attention_calcd += convolved_ts
        normalized = self.normalization_2(attention_calcd)  # BxHxW -> BxHxW
        
        return normalized
x = torch.randn(1, 6,2)

x
lay = nn.ChannelShuffle(3)
lay(x)

x = (torch.randn(3, 1, 512).to("cuda"), torch.tensor([[24], [2], [5]]).to("cuda"))
mod = Upsampling(pool_size=4, lags = 512, d_out=4)
mod(x[0], x[1])
torch.compile(mod)
mod(x)
mod.to("cuda")
for i in mod.parameters():
    print(i.device, i.shape)
mod.state_dict()
mod.to("cuda")
mod(x)