### Here I will introduce some experimental attention layers
import torch
import torch
from torch import nn as nn
from torch.nn import functional as F
from torch.nn.parameter import Parameter
import torch.nn.functional as F


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
            * ((d_in + d_out) / 2) ** (-0.5)
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


class multi_head_attention(nn.Module):
    def __init__(self, embedding_dim=128, heads=4, lag=512, dropout=0.2, causal=True):
        super().__init__()

        assert (
            embedding_dim / heads
        ).is_integer(), (
            f"embedding_dim/heads should be integer while yours {embedding_dim/heads} "
        )

        self.embedding_dim = embedding_dim
        self.heads = heads
        self.causal = causal
        self.W = lag  ### Here W stands for width
        ### --- Attention Part --- ###
        ## ------------------------####
        ### Frist Dropout layers  ####
        self.dropout_Q = nn.Dropout(p=dropout)
        self.dropout_K = nn.Dropout(p=dropout)
        self.dropout_V = nn.Dropout(p=dropout)
        ### ----------------------####
        ###  -----Weights ------ Xavier Reyizzzz was here!!!!! Istırırız ###
        ### https://www.youtube.com/watch?v=kFmwBtlOLV8 ###
        self.L_Q = nn.Parameter(
            torch.randn(embedding_dim, embedding_dim) * (embedding_dim) ** (-0.5)
        )
        self.L_K = nn.Parameter(
            torch.randn(embedding_dim, embedding_dim) * (embedding_dim) ** (-0.5)
        )
        self.L_V = nn.Parameter(
            torch.randn(embedding_dim, embedding_dim) * (embedding_dim) ** (-0.5)
        )
        ### Final Linear Layer with no activation ###
        self.dense = Linear(embedding_dim, embedding_dim, bias=True)
        ### --- End of weights --- ###

        ### Attenting to Future ###
        if self.causal:
            self.causal_factor = nn.Parameter(
                torch.tril(-torch.inf * torch.ones(self.W, self.W), diagonal=-1)
            )
            ## This dude will not be affected by gradient updates ---
            self.causal_factor.requires_grad = False

    def forward(self, x, return_scores=False):  # BxHxL -> BxHxL
        K, Q, V = x[0], x[1], x[2]
        ## Apply Dropout ##
        K, Q, V = (
            self.dropout_K(self.L_K @ K),
            self.dropout_Q(self.L_Q @ Q),
            self.dropout_V(self.L_V @ V),
        )
        # -- # Reshaoe the arrays
        K_v, Q_v, V_v = (
            K.view(-1, self.heads, int(self.embedding_dim / self.heads), self.W),
            Q.view(-1, self.heads, int(self.embedding_dim / self.heads), self.W),
            V.view(-1, self.heads, int(self.embedding_dim / self.heads), self.W),
        )
        ## Grab the attention scores
        attention_scores = Q_v.transpose(-2, -1) @ K_v
        if self.causal:
            attention_scores += self.causal_factor

        scores = nn.Softmax(-2)(attention_scores) * self.embedding_dim ** (-0.5)
        attention_output = (V_v @ scores).view(-1, self.embedding_dim, self.W)
        if return_scores:
            return self.dense(attention_output), attention_scores, scores
        return self.dense(attention_output)
