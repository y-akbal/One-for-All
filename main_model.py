import torch
from torch import nn as nn
from torch.nn import functional as F
from layers import block, Upsampling, Linear
from torch.utils.data import DataLoader, Dataset

## This is important in the case that you compile the model!!!!
torch.set_float32_matmul_precision("high")

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
        num_of_clusters=None,  ### number of clusters of times series
        channel_shuffle_group=2,  ## active only and only when channel_shuffle is True
    ):
        assert (
            lags / pool_size
        ).is_integer(), "Lag size should be divisible by pool_size"
        super().__init__()
        self.width = lags // pool_size
        self.embedding_dim = embedding_dim
        ###
        self.cluster_used = True if num_of_clusters is not None else False
        ###
        self.blocks = nn.Sequential(
            *(
                block(
                    embedding_dim,
                    width=self.width,
                    n_heads=number_of_heads,
                )
                for _ in range(n_blocks)
            )
        )
        self.up_sampling = Upsampling(
            lags=lags,
            d_out=self.embedding_dim,
            pool_size=pool_size,
            num_of_ts=number_ts,
            conv_activation=F.gelu,
            num_of_clusters=num_of_clusters,
            channel_shuffle_group=channel_shuffle_group,
        )

        ###
        self.Linear = Linear(self.embedding_dim, 1)
        ###

    def forward(self, x):
        ## Here we go with upsampling layer
        if self.cluster_used:
            x, tse_embedding, cluster_embedding = x[0], x[1], x[2]
            x = self.up_sampling((x, tse_embedding, cluster_embedding))
        else:
            x, tse_embedding = x[0], x[1]
            x = self.up_sampling((x, tse_embedding))
        ## Concatted transformer blocks
        ###
        x = self.blocks(x)
        return self.Linear(x)


torch.manual_seed(0)
t = main_model(num_of_clusters=5, number_ts=10)
t.cuda("cuda:0")
t.state_dict()
# t = torch.compile(m)
# t((torch.randn(1, 1, 512), torch.tensor([[2]]), torch.tensor([[3]])))


## Fake dataset here we create to see if the model is doing good
class fake_data(Dataset):
    def __init__(self):
        self.x = torch.randn(5000, 1, 512)
        self.y = torch.randn(5000, 1, 128)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        gen = torch.manual_seed(i)

        return (
            self.x[i],
            self.y[i],
            torch.randint(0, 10, size=(1,), generator=gen),
            torch.randint(0, 5, size=(1,), generator=gen),
        )


####
data = fake_data()
train_dataloader = DataLoader(data, batch_size=64, shuffle=True)
optimizer = torch.optim.SGD(t.parameters(), lr=0.00001, momentum=0.9)

for i in range(1000):
    temp_loss = 0.1
    counter = 0
    for x, y, tse, cls in train_dataloader:
        x, y, tse, cls = map(lambda x: x.to("cuda:0"), [x, y, tse, cls])
        output = t((x, tse, cls))
        optimizer.zero_grad()
        loss = nn.MSELoss()(output, y)
        loss.backward()
        optimizer.step()
        ### mean loss calculation ###
        counter += 1
        temp_loss -= (temp_loss - loss.item()) / counter

    print(f"The loss is {temp_loss} and epoch {i}")
