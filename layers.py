import torch
from torch import nn as nn
from torch.nn import functional as F
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import time

#### Convention here we use: BxHxW ---- W here refers to the lags of the time series,
#### H refers to population of lags via layers
#### On time permitting https://pytorch.org/docs/stable/nn.init.html
### will look at the above initializations as He initialization may sound better, for the first layers
## If possible do torch.compile(model), things go damn fast!!! furious....


class Linear(
    nn.Module
):  ## B*H*W -> B*H'*W adjusted the columns in each batch, no touch to rows directly
    ## This layer just mixes H, no touch to lags
    ### Motivated by Pytorch original Linear Layer
    ### BTW if we had the information of static shapes, then we would change the order of multiplication
    ## To further
    def __init__(self, d_in, d_out, bias=False, device=None, dtype=None, dropout=0.1):
        kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.M = Parameter(
            torch.randn(d_in, d_out, requires_grad=True, **kwargs)
            * ((d_in + d_out) / 2) ** (-0.5)
        )  # Kaiming init
        self.bias = bias
        if self.bias:
            self.b = Parameter(
                # torch.zeros(d_out, requires_grad=True, dtype=torch.float32)
                torch.randn(d_out, requires_grad=True, **kwargs)
                / d_out**2
            )
        ## Let's do some functional stuff
        self.bias_f = lambda x: x + self.b if self.bias else x
        if dropout > 0:
            self.dropout = nn.Dropout(dropout, inplace=False)
        else:
            self.dropout = lambda x: x

    def forward(self, x):  # Bxd_inxW -> Bxd_outxW
        ### changed the order and saved some time
        ### because CUDA likes it!!!
        x = self.dropout(x)
        res = x.transpose(-1, -2) @ self.M
        res = self.bias_f(res)
        return res.transpose(-1, -2)


class FFN(nn.Module):
    def __init__(
        self, d_in, expansion_size=2, dropout=0.2, activation=nn.ReLU(), bias=True
    ) -> None:
        assert d_in * expansion_size > 1, "Do not squeeze too much buddy!!!!"
        d_temp_out = int(d_in * expansion_size)
        ### This dude is FFN part as given in the all you need paper, we use nn.ReLU, as we may.
        super().__init__()
        self.linear = nn.Sequential(
            Linear(d_in=d_in, d_out=d_temp_out, dropout=0, bias=bias),
            activation,
            Linear(d_in=d_temp_out, d_out=d_in, dropout=dropout, bias=bias),
        )

    def forward(self, x):
        return self.linear(x)


class layernorm(nn.Module):
    # We normalize the local copies not along time dimension
    ## standard layer norm guy, horoko!!!
    ## We are closely approaching real Pytorch's implementation
    ## This layer seems to beat Pytorch's default layer norm with transpose
    ## If the layer is compiled.
    def __init__(self, dim, eps=1e-5, **kwargs):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones((dim), requires_grad=True, **kwargs))
        self.beta = nn.Parameter(torch.zeros((dim), requires_grad=True, **kwargs))

    def forward(self, x):
        mean = x.mean(1, keepdims=True)
        var = x.var(1, keepdims=True)
        unit = (x - mean) / torch.sqrt(self.eps + var)
        return self.gamma * unit + self.beta  ### B*H*W -> B*H*W


def channel_shuffleF(x, groups):
    ## Grabbed this from https://github.com/jaxony/ShuffleNet/blob/master/model.py
    ## Asdjusted properly
    ## Do not know, why the hell we have this layer though, it serves some purpose I believe,
    batchsize, height, width = x.shape

    channels_per_group = height // groups

    # reshape
    x = x.view(batchsize, groups, channels_per_group, width)

    x = torch.transpose(x, 1, 2).contiguous()

    # flatten
    x = x.view(batchsize, -1, width)

    return x


class Upsampling(nn.Module):
    def __init__(
        self,
        lags: int = 512,  ### input dimension (width)
        d_out=128,  ## output dimension (height)
        pool_size=4,  ## pool_sizes
        conv_bias=True,
        dense_bias=True,
        conv_activation=None,
        FFN_activation=nn.GELU("tanh"),
        num_of_ts=25,  ### number of time series to be used
        channel_shuffle=True,  ### we add channel shuffle to trick
        num_of_clusters=None,  ### number of clusters of times series
        channel_shuffle_group=2,  ## active only and only when channel_shuffle is True
        dropout_FFN=0.2,  ## droput of FFN layer,
        dropout_linear=0.2,  ##dropout of linear layer,
        **kwargs,
    ):
        super().__init__(**kwargs)
        assert (
            lags / pool_size
        ).is_integer(), "Make sure that lag size is divisible by pool_size"

        self.num_pools = int(lags / pool_size)
        self.num_of_ts = num_of_ts
        self.lags = lags
        self.channel_shuffle = channel_shuffle

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
        self.normalization = layernorm(self.num_pools)
        ## FFN part
        self.FFN = FFN(
            d_in=d_out, bias=dense_bias, dropout=dropout_FFN, activation=FFN_activation
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

        ## positional embedding of pools

        self.pe_embedding = nn.Embedding(self.num_pools, d_out)
        ## positional embeddings of time series
        self.ts_embedding = nn.Embedding(self.num_of_ts, d_out)
        ## cluster embedding of time series
        self.num_of_clusters = num_of_clusters
        if num_of_clusters != None:
            self.cls_embedding = nn.Embedding(num_of_clusters, d_out)
        ## -- End of Embedding Layers -- ##

        self.shuffle = (
            lambda x: channel_shuffleF(x, channel_shuffle_group)
            if self.channel_shuffle
            else x
        )

    def forward(self, x: tuple) -> torch.Tensor:
        if self.num_of_clusters != None:
            ts, te, tc = x  ## split
        else:
            ts, te = x  ## split

        assert ts.shape[-1] == self.lags, f"{self.lags} is not equal to {ts.shape[-1]}"

        # ts: Bx1xW (W here is used for Lags) the raw time series,
        # pe: (BxHxW) positional embeddings of time series,
        # te: (Embedding (geospatial) of the time series depending).
        # tc: Clustered time series, depending on geospatial data

        convolved_ts = self.Conv(ts)  # Bx1xW -> BxHxW/pool_size
        ## From now on we convey W = W/pool_size
        # BxHxW += #BxHxW (WxH -> HxW)   # Position embedding of pools
        convolved_ts += self.pe_embedding(self.num_enum).transpose(-1, -2)
        activated = self.conv_activation(convolved_ts)  # BxHxW -> BxHxW
        normalized = self.normalization(activated)  # BxHxW -> BxHxW

        ###
        # if self.channel_shuffle:
        normalized = self.shuffle(normalized)

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
            dense_applied += convolved_ts + self.ts_embedding(te).transpose(-1, -2)
        ## Final linear layer two mix the channels
        final_linear = self.linear(dense_applied)  # BxHxW-> BxHxW
        return final_linear
        # Bx1xW-> BxHxW/pool_size (this what happens finally)


class multi_head_attention(nn.Module):
    """This dude is a bit faster than the original"""

    def __init__(self, embedding_dim=128, heads=4, lag=512, dropout=0.2, causal=True):
        super().__init__()

        assert (
            embedding_dim / heads
        ).is_integer(), (
            f"embedding_dim/heads should be integer while yours {embedding_dim/heads} "
        )

        self.embedding_dim = embedding_dim
        self.linear = Linear(embedding_dim, 3 * embedding_dim)
        self.heads = heads
        self.causal = causal
        self.W = lag  ### Here W stands for width (or lags)
        self.Dropout = nn.Dropout(p=dropout)
        ## output linear
        self.dense = Linear(embedding_dim, embedding_dim, bias=True)

        ### blocking attenting to Future ###
        if self.causal:
            self.register_buffer(
                "causal_factor",
                torch.tril(
                    torch.ones(self.W, self.W, dtype=torch.bool),
                    diagonal=-1,
                ),
            )

    def forward(self, x):  # BxHxL -> BxHxL
        B, _, _ = x.shape
        ## Apply Dropout ##
        KQV = self.linear(x)
        KQV = x.view(B, self.heads, int(self.embedding_dim / self.heads), self.W)

        attention_scores = KQV.transpose(-2, -1) @ KQV

        if self.causal:
            attention_scores = attention_scores.masked_fill(
                self.causal_factor, -torch.inf
            )
        softmaxed_att = nn.Softmax(-2)(attention_scores * self.embedding_dim ** (-0.5))

        t = KQV @ softmaxed_att
        t = self.Dropout(
            t,
        )
        t = t.view(B, self.embedding_dim, self.W)

        return self.dense(t)


class block(nn.Module):
    def __init__(
        self,
        d_in=128,  ### embedding dimension
        width=128,  ### width of time series to be used
        n_heads=4,
        dropout_FFN=0.5,  ## dropout of FFN
        att_head_dropout=0.2,  ## dropout of attention heads
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
            dropout=att_head_dropout,
            causal=causal,
        )
        ### FFN layer
        self.FFN = FFN(
            d_in=d_in,
            expansion_size=expansion_size,
            dropout=dropout_FFN,
            activation=activation,
        )
        ### Normalization layers
        self.ln1 = layernorm(width)
        self.ln2 = layernorm(width)

    def forward(self, x):  # B*H*W -> B*H*W
        y = self.ln1(x)
        y = self.att_head(y)
        y += x  ## Residual connection
        x = self.ln2(y)
        x = self.FFN(x)
        x += y  ## Residual Connection
        return x


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
