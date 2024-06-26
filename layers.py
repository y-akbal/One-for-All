import torch
from torch import nn as nn
from torch.nn import functional as F
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import time

#### Convention here we use: BxHxW ---- W here refers to the lags of the time series,
#### H refers to population of lags via layers
#### On time permitting https://pytorch.org/docs/stable/nn.init.html
#### will look at the above initializations as He initialization may sound better, for the first layers
#### If possible do torch.compile(model), things go damn fast!!! and furious....


class Linear(
    nn.Module
):  ## B*H*W -> B*H'*W adjusted the columns in each batch, no touch to rows directly
    ## This layer just mixes H, no touch to lags
    ### Motivated by Pytorch original Linear Layer
    ### BTW if we had the information of static shapes, then we would change the order of multiplication
    ## To further
    def __init__(self, d_in, 
                 d_out, 
                 bias=False, 
                 device=None, 
                 dtype=None, 
                 dropout=0.1):
        kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.M = nn.Parameter(
            torch.randn((d_in, d_out), requires_grad=True, **kwargs)/d_in
        )  # Kaiming init
        
        self.bias = bias
        if self.bias:
            self.b = nn.Parameter(
                # torch.zeros(d_out, requires_grad=True, dtype=torch.float32)
                torch.zeros(d_out, requires_grad=True,**kwargs)
            )
         
        
        ## Let's do some functional stuff
        self.bias_f = lambda x: x + self.b if self.bias else x
        self.dropout = nn.Dropout(dropout, inplace=False) if dropout > 0 else lambda x: x
        
    def forward(self, x):  # Bxd_inxW -> Bxd_outxW
        ### changed the order and saved some time
        ### because CUDA likes it!!!
        res = x.transpose(-1, -2) @ self.M
        res = self.bias_f(res)
        return self.dropout(res.transpose(-1, -2))
    

class FFN(nn.Module):
    def __init__(
        self, 
        d_in, 
        expansion_size=2, 
        dropout=0.2, 
        activation=nn.ReLU(), 
        bias=True
    ) -> None:
        assert d_in * expansion_size > 1, "Do not squeeze too much buddy!!!!"
        d_temp_out = int(d_in * expansion_size)
        ### This dude is FFN part as given in the all you need paper, we use nn.ReLU, as we may.
        super().__init__()
        self.linear = nn.Sequential(
            Linear(d_in=d_in, d_out=d_temp_out, dropout=0.0, bias=bias),
            activation,
            Linear(d_in=d_temp_out, d_out=d_in, dropout=dropout, bias=bias),
        )

    def forward(self, x):
        return self.linear(x)


class layernorm(nn.Module):
    ## We normalize the local copies not along time dimension
    ## standard layer norm guy.
    ## We are closely approaching real Pytorch's implementation
    def __init__(self, dim, eps=1e-5, **kwargs):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones((dim, 1), requires_grad=True, **kwargs))
        self.beta = nn.Parameter(torch.zeros((dim, 1), requires_grad=True, **kwargs))

    def forward(self, x):
        mean = x.mean(1, keepdims=True)
        var = x.var(1, keepdims=True)
        unit = (x - mean) / torch.sqrt(self.eps + var)
        return self.gamma * unit + self.beta  ### B*H*W -> B*H*W



class Upsampling(nn.Module):
    def __init__(
        self,
        lags: int = 512,  ### input dimension (width)
        d_out=128,  ## output dimension (height)
        pool_size=4,  ## pool_sizes
        conv_bias=True, ## Convolutional layer bias
        dense_bias=True, ## FFN bias
        conv_activation=None,
        FFN_activation=nn.GELU("tanh"),
        num_of_ts=25,  ### number of time series to be used
        num_of_clusters=None,  ### number of clusters of times series
        dropout_FFN=0.2,  ## droput of FFN layer,
        dropout_linear=0.2,  ##dropout of linear layer,
        FFN_expansion_size = 2,
        **kwargs,
    ):
        ## This layers uses no positional embedding at all!!!!
        super().__init__(**kwargs)
        assert (
            lags / pool_size
        ).is_integer(), "Make sure that lag size is divisible by pool_size"

        self.num_pools = int(lags / pool_size)
        self.num_of_ts = num_of_ts
        self.lags = lags

        ### ---- Beginning of layers ---- ###
        ## Convolution layer first,
        self.Conv = nn.Conv1d(
            in_channels=1,
            out_channels=d_out,
            kernel_size=pool_size,
            stride=pool_size,
            bias=conv_bias,
        )
        self.conv_activation = (
            conv_activation if conv_activation != None else torch.nn.Identity()
        )
        ## Normalization layer
        self.normalization = layernorm(d_out)
        ## FFN part
        
        self.FFN = FFN(
            d_in=d_out, 
            bias=dense_bias, 
            dropout=dropout_FFN, 
            activation=FFN_activation,
            expansion_size= FFN_expansion_size,
        )
        ## Linear Layer
        self.linear = Linear(d_out, d_out, bias=True, dropout=dropout_linear)
        ### Embedding enumeration
        
        ## ts_embedding
        self.ts_embedding = nn.Embedding(self.num_of_ts, d_out)
        ## cluster embedding of time series
        self.num_of_clusters = num_of_clusters
        if num_of_clusters != None:
            self.cls_embedding = nn.Embedding(num_of_clusters, d_out)
        ## -- End of Embedding Layers -- ##

    def forward(self, x: tuple) -> torch.Tensor:
        if self.num_of_clusters != None:
            ts, te, tc = x  ## split
        else:
            ts, te = x  ## split

        # ts: Bx1xW (W here is used for Lags) the raw time series,
        # pe: (BxHxW) positional embeddings of time series,
        # te: Enumeration for Embedding (geospatial) of the time series depending.
        # tc: Clustered time series enumeration for depending on geospatial data

        convolved_ts = self.Conv(ts)  # Bx1xW -> BxHxW/pool_size
        
        ## From now on we convey W = W/pool_size
        # BxHxW += #BxHxW (WxH -> HxW)   # Position embedding of pools
        activated = self.conv_activation(convolved_ts)  # BxHxW -> BxHxW
        normalized = self.normalization(activated)  # BxHxW -> BxHxW
        
        # BxHxW -> BxHxW (Dense layer is applied H dim)
        dense_applied = self.FFN(normalized)
        # BxHxW += #BxHxW (WxH -> HxW) + #BxHx1 -> BxHxW   # Position embedding of time series
        if self.num_of_clusters != None:
            dense_applied += (
                convolved_ts  ### Residual connection here
                + self.ts_embedding(te).transpose(-1, -2)  ### time series empeedings
                + self.cls_embedding(tc).transpose(
                    -1, -2
                )  ### cluster embeddings to help the model
            )
        else:
            dense_applied += normalized
        ## Final linear layer two mix the channels
        return self.linear(dense_applied)  + self.ts_embedding(te).transpose(-1, -2) # BxHxW-> BxHxW
        # Bx1xW-> BxHxW/pool_size (this what happens finally)




class PUpsampling(nn.Module):
    def __init__(
        self,
        lags: int = 256,  ### input dimension (width)
        d_out=128,  ## output dimension (height)
        pool_size=4,  ## pool_sizes
        conv_bias=True, ## Convolutional layer bias
        dense_bias=True, ## FFN bias
        conv_activation = None,
        FFN_activation = nn.GELU("tanh"),
        num_of_ts=25,  ### number of time series to be used
        num_of_clusters=None,  ### number of clusters of times series
        dropout_FFN=0.2,  ## droput of FFN layer,
        dropout_linear=0.2,  ##dropout of linear layer,
        FFN_expansion_size = 2,
        **kwargs,
    ):
        super().__init__(**kwargs)
        ## The difference between this layer and the above upsampling layer: stride = 1, instead of stride = pool_size
        ## We also pad the inputs from right!!!!
        self.pool_size = pool_size
        self.num_pools = lags
        self.num_of_ts = num_of_ts
        self.lags = lags
        ### ---- Beginning of layers ---- ###
        ## Convolution layer first,
        self.Conv = nn.Conv1d(
            in_channels=1,
            out_channels=d_out-50,
            kernel_size=pool_size,
            bias=conv_bias,
        )
        self.conv_activation = (
            conv_activation if conv_activation != None else torch.nn.Identity()
        )
        ## Normalization layer
        self.normalization = layernorm(d_out)
        ## FFN part
        self.FFN = FFN(
            d_in=d_out, 
            bias=dense_bias, 
            dropout=dropout_FFN, 
            activation=FFN_activation,
            expansion_size= FFN_expansion_size,
        )
        ## Linear Layer
        self.linear = Linear(d_out, d_out, bias=True, dropout=dropout_linear)
        ### Embedding enumeration
        self.register_buffer(
            "num_enum",
            torch.tensor(
                [i for i in range(self.num_pools)],
                dtype=torch.int,
                requires_grad=False,
            ),
        )
        ## positional embedding of pools ##
        self.pe_embedding = nn.Embedding(self.num_pools, 50)
        ## positional embeddings of time series
        self.ts_embedding = nn.Embedding(self.num_of_ts, d_out)
        ## cluster embedding of time series
        self.num_of_clusters = num_of_clusters
        if num_of_clusters != None:
            self.cls_embedding = nn.Embedding(num_of_clusters, d_out)
        ## -- End of Embedding Layers -- ##

    def forward(self, x: tuple) -> torch.Tensor:
        if self.num_of_clusters != None:
            ts, te, tc = x  ## split
        else:
            ts, te = x  ## split
        
        # ts: Bx1xW (W here is used for Lags) the raw time series,
        # pe: (BxHxW) positional embeddings of time series,
        # te: Enumeration for Embedding (geospatial) of the time series depending.
        # tc: Clustered time series enumeration for depending on geospatial data
        ts_padded = F.pad(ts, (self.pool_size-1, 0), "constant")

        convolved_ts = self.Conv(ts_padded)  # Bx1xW -> BxHxW

        B, H, W = convolved_ts.shape
        # BxHxW += #BxHxW (WxH -> HxW)   # Position embedding of pools
        pe_embeddings = self.pe_embedding(self.num_enum[:convolved_ts.shape[-1]]).transpose(-1, -2)
        convolved_ts = torch.concat([convolved_ts, pe_embeddings.unsqueeze(0).repeat(B,1,1)], axis = 1)

        
        activated = self.conv_activation(convolved_ts)  # BxHxW -> BxHxW
        normalized = self.normalization(activated)  # BxHxW -> BxHxW
        
        # BxHxW -> BxHxW (Dense layer is applied H dim)
        dense_applied = self.FFN(normalized)
        
        # BxHxW += #BxHxW (WxH -> HxW) + #BxHx1 -> BxHxW   # Position embedding of time series
        if self.num_of_clusters != None:
            dense_applied += (
                convolved_ts  ### Residual connection here
                + self.ts_embedding(te).transpose(-1, -2)  ### time series empeedings
                + self.cls_embedding(tc).transpose(
                    -1, -2
                )  ### cluster embeddings to help the model
            )
        else:
            dense_applied += normalized
        ## Final linear layer two mix the channels
        return self.linear(dense_applied) + self.ts_embedding(te).transpose(-1, -2)
        # Bx1xW-> BxHxW/pool_size (this what happens finally)
    
"""
torch.manual_seed(0)
PUpsampling(d_out = 64,pool_size=10)([torch.randn(2, 1, 4), torch.tensor([[7],[2]])])

a,b,c =x.shape
y = y.unsqueeze(0).repeat(a,1,1)
torch.concat([x,y], axis = 1).shape
"""

class LUpsampling(nn.Module):
    def __init__(
        self,
        lags: int = 256,  ### input dimension (width)
        d_out=128,  ## output dimension (height)
        pool_size=4,  ## pool_sizes
        conv_bias=True, ## Convolutional layer bias
        dense_bias=True, ## FFN bias
        conv_activation = None,
        FFN_activation = nn.GELU("tanh"),
        num_of_ts=25,  ### number of time series to be used
        num_of_clusters=None,  ### number of clusters of times series
        dropout_FFN=0.2,  ## droput of FFN layer,
        dropout_linear=0.2,  ##dropout of linear layer,
        FFN_expansion_size = 2,
        **kwargs,
    ):
        super().__init__(**kwargs)
        ## The difference between this layer and the above upsampling layer: stride = 1, instead of stride = pool_size
        ## We also pad the inputs from right!!!!
        self.pool_size = pool_size
        self.num_pools = lags
        self.num_of_ts = num_of_ts
        self.lags = lags
        ### ---- Beginning of layers ---- ###
        ## Convolution layer first,
        self.Conv = nn.Conv1d(
            in_channels=1,
            out_channels=d_out,
            kernel_size=pool_size,
            bias=conv_bias,
        )
        self.conv_activation = (
            conv_activation if conv_activation != None else torch.nn.Identity()
        )
        ## Normalization layer
        self.normalization = layernorm(d_out)
        ## FFN part
        self.FFN = FFN(
            d_in=d_out, 
            bias=dense_bias, 
            dropout=dropout_FFN, 
            activation=FFN_activation,
            expansion_size= FFN_expansion_size,
        )
        ## Linear Layer
        self.linear = Linear(d_out, d_out, bias=True, dropout=dropout_linear)
        ### Embedding enumeration
        self.register_buffer(
            "num_enum",
            torch.tensor(
                [i for i in range(self.num_pools)],
                dtype=torch.int,
                requires_grad=False,
            ),
        )
        ## positional embedding of pools ##
        #self.pe_embedding = nn.Embedding(self.num_pools, d_out)
        ## positional embeddings of time series
        self.ts_embedding = nn.Embedding(self.num_of_ts, d_out)
        ## cluster embedding of time series
        self.num_of_clusters = num_of_clusters
        if num_of_clusters != None:
            self.cls_embedding = nn.Embedding(num_of_clusters, d_out)
        ## -- End of Embedding Layers -- ##

    def forward(self, x: tuple) -> torch.Tensor:
        if self.num_of_clusters != None:
            ts, te, tc = x  ## split
        else:
            ts, te = x  ## split

        # ts: Bx1xW (W here is used for Lags) the raw time series,
        # pe: (BxHxW) positional embeddings of time series,
        # te: Enumeration for Embedding (geospatial) of the time series depending.
        # tc: Clustered time series enumeration for depending on geospatial data
        ts_padded = F.pad(ts, (self.pool_size-1, 0), "constant")

        convolved_ts = self.Conv(ts_padded)  # Bx1xW -> BxHxW
        # BxHxW += #BxHxW (WxH -> HxW)   # Position embedding of pools
        #convolved_ts += self.pe_embedding(self.num_enum[:convolved_ts.shape[-1]]).transpose(-1, -2)


        activated = self.conv_activation(convolved_ts)  # BxHxW -> BxHxW
        normalized = self.normalization(activated)  # BxHxW -> BxHxW
        # BxHxW -> BxHxW (Dense layer is applied H dim)
        dense_applied = self.FFN(normalized)
        # BxHxW += #BxHxW (WxH -> HxW) + #BxHx1 -> BxHxW   # Position embedding of time series
        if self.num_of_clusters != None:
            dense_applied += (
                convolved_ts  ### Residual connection here
                + self.ts_embedding(te).transpose(-1, -2)  ### time series empeedings
                + self.cls_embedding(tc).transpose(
                    -1, -2
                )  ### cluster embeddings to help the model
            )
        else:
            dense_applied += normalized + self.ts_embedding(te).transpose(-1, -2)
        ## Final linear layer two mix the channels
        return self.linear(dense_applied)  # BxHxW-> BxHxW
        # Bx1xW-> BxHxW/pool_size (this what happens finally)

""""
LUpsampling(pool_size=10)([torch.randn(2, 1, 256), torch.tensor([[2],[1]])]).shape


q = F.pad(torch.ones(1, 1, 4), (2,0),"constant")
layer = nn.Conv1d(1, 1, kernel_size = 2, padding = "valid")
layer.state_dict()
layer(q)
"""


"""

torch.manual_seed(0)
x = LUpsampling(d_out=768, lags = 512, pool_size=8)
q = torch.torch.distributions.Uniform(low=-1, high=1).sample((1, 1, 512))
x.eval()
x = x([q, torch.tensor([13])])

"""




class multi_head_attention(nn.Module):
    """This dude is a bit faster than the original provided that we do not use flash attention!!!"""

    def __init__(self, 
                 embedding_dim=768, 
                 heads=4, 
                 lag=512, 
                 projection_dropout=0.2, 
                 att_head_dropout = 0.1,
                 causal=True,
                 flash_attention = True):
        super().__init__()

        assert (
            embedding_dim / heads
        ).is_integer(), (
            f"embedding_dim/heads should be integer while yours {embedding_dim/heads} "
        )

        self.embedding_dim = embedding_dim

        self.linear = Linear(embedding_dim, 3*embedding_dim, dropout = 0.0, bias = False)

        self.heads = heads
        self.causal = causal
        self.flash_attention = flash_attention
        self.att_dropout_rate = att_head_dropout
        #self.W = lag  ### Here W stands for width (or lags), redundant will not be needed really!!!!
        if not self.flash_attention:
            self.att_dropout = nn.Dropout(p=att_head_dropout)
        ## output linear
        self.projection = Linear(embedding_dim, embedding_dim, bias=True, dropout = projection_dropout)

        ### blocking attenting to Future ###
        if self.causal:
            mask = torch.triu(torch.ones(lag,lag, dtype = torch.bool),1)
            self.register_buffer(
                "mask", mask)

    def forward(self, x:torch.Tensor):  # BxHxL -> BxHxL

        B, embedding_dim, L = x.shape
        K, Q, V = self.linear(x).split(self.embedding_dim, -2) #BxHxL -> Bx3*HxL
        K, Q, V = map(lambda x : x.view(B, self.heads, embedding_dim//self.heads, L), [K, Q, V])
        
        if self.flash_attention:
            K, Q, V = map(lambda x: x.transpose(-1, -2), [K, Q, V])
            t = F.scaled_dot_product_attention(Q, K, V, is_causal = self.causal, dropout_p = self.att_dropout_rate)
        else:
            ## Need to add here causal factor!!!
            t = (Q.transpose(-1,-2) @ K)/(Q.shape)[-2]**0.5
            t = t.masked_fill(self.mask[:L,:L], float("-inf")) if self.causal else t
            t = F.softmax(t,  -1) @ V.transpose(-1, -2)
            t = self.att_dropout(t)
        t = t.transpose(-1, -2).contiguous().view(B, embedding_dim, L)
        t = self.projection(t)
        return t
""""
torch.manual_seed(0)
model = multi_head_attention(embedding_dim=12)
model.eval()
model.state_dict()
model(torch.randn(1, 10, 12))""""


class attention_block(nn.Module):
    def __init__(
        self,
        d_in=128,  ### embedding dimension
        width=128,  ### width of time series to be used
        n_heads=4,
        dropout_FFN=0.5,  ## dropout of FFN
        bias_FFN = True,
        att_head_dropout=0.2,  ## dropout of attention heads
        projection_dropout = 0.2,
        causal=True,
        expansion_size=2,  ### expansion size of FFN
        activation=nn.GELU("tanh"),  ### this is used
    ):
        super().__init__()
        ### Multihead attention
        self.att_head = multi_head_attention(
            embedding_dim=d_in,
            lag=width,
            heads=n_heads,
            att_head_dropout = att_head_dropout,
            projection_dropout = projection_dropout,
            causal=causal,
        )
        ### FFN layer
        self.FFN = FFN(
            d_in=d_in,
            expansion_size=expansion_size,
            dropout=dropout_FFN,
            activation=activation,
            bias=bias_FFN
        )
        ### Normalization layers
        self.ln1 = layernorm(d_in)
        self.ln2 = layernorm(d_in)

    def forward(self, x):  # B*H*W -> B*H*W
        x = x + self.att_head(self.ln1(x))
        x = x + self.FFN(self.ln2(x))
        return x



#attention_block(768, 504)(x)
"""
torch.set_float32_matmul_precision("high")
lay = block(n_heads=8)

lay = nn.Sequential(*[lay for _ in range(10)]).cuda(0)
lay.eval()


x = torch.randn(100, 128, 128).cuda(0)
import time

a = time.time()
torch.cuda.synchronize()
for i in range(100):
    q = lay(x)
torch.cuda.synchronize()
print(time.time() - a)


lay_ = torch.nn.MultiheadAttention(128, 4).cuda()
lay_.state_dict()
y = x.transpose(-1, -2)
y = y.cuda()

import time

a = time.time()
torch.cuda.synchronize()
for i in range(100):
    q = lay_(y, y, y)
torch.cuda.synchronize()
print(time.time() - a)

"""
if __name__ == "__main__":
    pass
