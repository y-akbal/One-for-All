import torch
from torch import nn as nn
from torch.nn import functional as F
from layers import block, Upsampling

device = "cuda" if torch.cuda.is_available() else "cpu"


class main_model(nn.Module):
    def __init__(
        self,
        lags: int = 512,
        embedding_dim: int = 64,
        n_blocks: int = 10,
        pool_size: int = 4,
        number_of_heads=4,
        number_ts=25,
    ):
        assert (
            lags / pool_size
        ).is_integer(), "Lag size should be divisible by pool_size"
        super().__init__()
        self.width = lags // pool_size
        self.embedding_dim = embedding_dim
        ###
        self.blocks = nn.ModuleList(
            [
                block(
                    embedding_dim,
                    width=self.width,
                    n_heads=number_of_heads,
                )
                for _ in range(n_blocks)
            ]
        )
        self.up_sampling = Upsampling(
            lags=lags,
            d_out=self.embedding_dim,
            pool_size=pool_size,
            num_of_ts=number_ts,
            conv_activation=F.gelu,
        )
        ###

    def forward(self, x, y):
        x = self.up_sampling((x, y))
        for layer in self.blocks:
            x = layer(x)
        return x


t = main_model()
t.cuda(1)
torch.set_float32_matmul_precision("high")
t.state_dict()
# t = torch.compile(m)

x = torch.randn(5, 1, 512, device="cuda:1")
y = torch.randn(5, 1, 128, device="cuda:1")
embedding = torch.tensor([[1] for i in range(5)], device="cuda:1")

optimizer = torch.optim.SGD(t.parameters(), lr=0.01, momentum=0.9)
t(x, embedding).mean(1, keepdims=True)
