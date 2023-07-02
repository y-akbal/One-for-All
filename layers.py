import torch
from torch import nn as nn
from torch.nn import functional as F
from torch.nn.parameter import Parameter, UninitializedParameter

#### Convention here we use: BxHxW ---- W here refers to the lags of the time series,
#### H refers to population of lags via layers


class Linear(
    nn.Module
):  ## B*H*W -> B*H'*W adjusted the columns in each batch, no touch to rows directly
    ## This layer just mixes H, no touch to lags
    ### Motivated by Pytorch original Linear Layer
    def __init__(self, d_in, d_out, bias=False, dropout=0.1, device=None, dtype=None):
        kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.M = Parameter(
            torch.randn(d_out, d_in, requires_grad=True, **kwargs)
            * ((1 / (d_in + d_out)) ** (0.5))
        )  # Kaiming init
        self.bias = bias
        if self.bias:
            self.b = Parameter(torch.zeros(d_out, 1, requires_grad=True, **kwargs))

        self.dropout = nn.Dropout(dropout, inplace=False)

    def forward(self, x):  # Bxd_inxW -> Bxd_outxW
        x = self.dropout(x)  ## apply dropout!!!
        res = self.M @ x
        if self.bias:
            res += self.b
        return res


class FFN(nn.Module):
    def __init__(
        self, d_in, expansion_size=2, dropout=0.2, activation = nn.ReLU(), bias=True
    ) -> None:
        assert expansion_size > 1, "You must have an expansion size greater than one"
        assert isinstance(expansion_size, int), "Expansion size must be an integer"
        ### This dude is FFN part as given in the all you need paper, we use nn.ReLU, we may
        ## change this later, depending on needs.
        super().__init__()
        self.linear = nn.Sequential(
            Linear(d_in=d_in, d_out=expansion_size * d_in, dropout=dropout, bias=bias),
            activation,
            Linear(d_in=expansion_size * d_in, d_out=d_in, dropout=dropout, bias=bias),
        )

    def forward(self, x):
        return self.linear(x)


class layernorm(nn.Module):
    # We noemalize the local copies not along time dimension
    ## standard layer norm guy, horoko!!!
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(dim, requires_grad=True))
        self.beta = nn.Parameter(torch.zeros(dim, requires_grad=True))

    def forward(self, x):
        mean = x.mean(1, keepdims=True)
        var = x.var(1, keepdims=True)
        unit = (x - mean) / torch.sqrt(self.eps + var)
        return self.gamma * unit + self.beta  ### B*H*W -> B*H*W

    def parameters(self):
        return [self.gamma, self.beta]


class Upsampling(nn.Module):
    def __init__(
        self,
        lags: int = 512,  ### input dimension (width)
        d_out=128,  ## output dimension (height)
        pool_size=4,  ## pool_sizes
        conv_bias=True,
        dense_bias=False,
        activation=F.gelu,
        num_of_ts=25,  ### number of time series to be used
        num_of_clusters = None,  ### number of clusters of times series
        device="cuda",
        channel_shuffle=False,  ### we add channel shuffle to trick
        channel_shuffle_group=2,  ## active only and only when channel_shuffle is True
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
        self.Conv = nn.Conv1d(
            in_channels=1,
            out_channels=d_out,
            kernel_size=pool_size,
            stride=pool_size,
            bias=conv_bias,
        )

        self.activation = activation
        self.normalization = layernorm(self.num_pools)  ### This part is important!!!
        ##Dense part
        self.FFN = FFN(d_in=d_out, bias=dense_bias)  ## Feed forward layer
        self.dense = Linear(
            d_out, d_out, bias=True, dropout=0.2
        )  ### is is for final layer

        ## -- Begining of Embedding Layers -- ##
        self.num_enum = torch.arange(
            self.num_pools, device=device if not None else "cuda"
        )
        # positional embedding of pools
        self.pe_embedding = nn.Embedding(self.num_pools, d_out)
        # positional embeddings of time series
        self.ts_embedding = nn.Embedding(self.num_of_ts, d_out)

        self.num_of_clusters = num_of_clusters
        if num_of_clusters != None:
            
            self.cls_embedding = nn.Embedding(num_of_clusters, d_out)
        ## -- End of Embedding Layers -- ##

        if channel_shuffle:
            self.shuffle = nn.ChannelShuffle(channel_shuffle_group)

    def forward(self, x: tuple) -> torch.Tensor:
        if self.num_of_clusters:
            ts, te, tc = x  ## split
        else:
            ts, te = x  ## split
            
        assert ts.shape[-1] == self.lags, f"{self.lags} is not equal to {ts.shape[-1]}"

        # ts: Bx1xW (W here is used for Lags) the raw time series,
        # pe: (BxHxW) positional embeddings of time series,
        # te: (Embedding (geospatial) of the time series depending).
        # tc: Clustered time series, depending on geospatial data

        convolved_ts = self.Conv(ts)  # Bx1xW -> BxHxW/pool_size

        # BxHxW += #BxHxW (WxH -> HxW)   # Position embedding of pools
        convolved_ts += self.pe_embedding(self.num_enum).transpose(-1, -2)
        activated = self.activation(convolved_ts)  # BxHxW -> BxHxW
        normalized = self.normalization(activated)  # BxHxW -> BxHxW

        if self.channel_shuffle:
            normalized = self.shuffle(normalized)

        # BxHxW -> BxHxW (Dense layer is applied H dim)
        dense_applied = self.FFN(normalized)
        # BxHxW += #BxHxW (WxH -> HxW) + #BxHx1 -> BxHxW   # Position embedding of time series
        if not self.num_of_clusters:
            dense_applied += (
            convolved_ts
            + self.ts_embedding(te).transpose(-1, -2)
            + self.cls_embedding(tc).transpose(-1, -2)
            )
        else:
            dense_applied += (
            convolved_ts
            + self.ts_embedding(te).transpose(-1, -2)
            )
        final_linear = self.dense(dense_applied)  # BxHxW-> BxHxW

        return final_linear  # Bx1xW-> BxHxW/pool_size


class multi_head_attention(nn.Module):
    def __init__(self, 
                 embedding_dim=128, 
                 heads=4, lag=512, 
                 dropout=0.2, 
                 causal=True):
        super().__init__()

        assert (embedding_dim / heads).is_integer(), (
            f"embedding_dim/heads should be integer while yours {embedding_dim/heads} "
        )

        self.embedding_dim = embedding_dim
        self.heads = heads
        self.causal = causal
        self.W = lag
        ### Attention Part ###
        ### Frist Dropout layers
        self.dropout_Q = nn.Dropout(p=dropout)
        self.dropout_K = nn.Dropout(p=dropout)
        self.dropout_V = nn.Dropout(p=dropout)
        ### Weights ###
        self.L_Q = nn.Parameter(
            torch.randn(embedding_dim, embedding_dim) * (embedding_dim) ** (-0.5)
        )
        self.L_K = nn.Parameter(
            torch.randn(embedding_dim, embedding_dim) * (embedding_dim) ** (-0.5)
        )
        self.L_V = nn.Parameter(
            torch.randn(embedding_dim, embedding_dim) * (embedding_dim) ** (-0.5)
        )
        ### Final Linear Layer with no activation
        self.dense = Linear(embedding_dim, embedding_dim, bias = True)
        ### --- End of weights --- ###
        if self.causal:
            self.causal_factor = nn.Parameter(
                torch.tril(-torch.inf * torch.ones(self.W, self.W), diagonal=-1)
            )
            ## No gradient is required in the above layer....
            self.causal_factor.requires_grad = False

    def forward(self, x):  # BxHxL -> BxHxL
        K, Q, V = x[0], x[1], x[2]
        K = self.dropout_K(self.L_K @ K)
        Q = self.dropout_Q(self.L_Q @ Q)
        V = self.dropout_V(self.L_V @ V)
        K_v = K.view(-1, self.heads, int(self.embedding_dim / self.heads), self.W)
        Q_v = Q.view(-1, self.heads, int(self.embedding_dim / self.heads), self.W)
        V_v = V.view(-1, self.heads, int(self.embedding_dim / self.heads), self.W)

        attention_scores = Q_v.transpose(-2, -1) @ K_v
        if self.causal:
            attention_scores += self.causal_factor

        scores = nn.Softmax(-2)(attention_scores) / self.embedding_dim**0.5
        attention_output = (V_v @ scores).view(-1, self.embedding_dim, self.W)

        return self.dense(attention_output)




class block(nn.Module):
    def __init__(
        self,
        d_in,  ### embedding dimension
        width=128,  ### width of time series to be used
        n_heads=4,
        dropout=0.5,## dropout of FFN
        att_head_dropout = 0.2, ## dropout of attention heads
        causal=True,
        expansion_size=2, ### expansion size of FFN
        activation=nn.GELU("tanh"), ### this is used 
    ):
        super().__init__()
        self.att_head = multi_head_attention(
            heads=n_heads, dropout=att_head_dropout, 
            lag=width, 
            embedding_dim=d_in, 
            causal = causal
        )
        self.FFN = FFN(
            d_in, 
            expansion_size=expansion_size, 
            dropout=dropout, 
            activation=activation
        )
        ### Normalization layers
        self.ln1 = layernorm(width)
        self.ln2 = layernorm(width)

    def forward(self, x):  # B*H*W -> B*H*W
        y = self.ln1(x)
        y = self.att_head([y, y, y])
        y += x
        x = self.ln2(y)
        x = self.FFN(x)
        x += y
        return x
    

