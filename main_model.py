import torch
from torch import nn as nn
from torch.nn import functional as F
from layers import block, Upsampling, Linear
from torch.utils.data import DataLoader
from memmap_arrays import ts_concatted
import numpy as np
import time

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

def main():
    torch.manual_seed(0)
    model = main_model(number_ts=264, embedding_dim=256, n_blocks=6)
    t = model.cuda(1)
    t = torch.compile(t)
    try:
        t.load_state_dict(torch.load("model_on_train_one_epoch0.trc"))
    except Exception as exp:
        print("Something went wrong bro!!!", exp)
    #print(t((torch.randn(1, 1, 512, device=device), torch.tensor([[2]], device=device))))


    memmap_data = np.memmap("array_train.dat", dtype=np.float32)
    memmap_lengths = np.memmap("lengthsarray_train.dat", dtype=np.int32)
    lags = [513 for _ in memmap_lengths]

    data = ts_concatted(array=memmap_data, lengths=memmap_lengths, lags=lags)
    train_dataloader = DataLoader(data, batch_size=256, shuffle=True)
    optimizer = torch.optim.SGD(t.parameters(), lr=0.00001)
    try:
        optimizer.load_state_dict(torch.load("model_on_train_state0.trc"))
    except Exception as exp:
        print(exp)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=15)
    # Creates a GradScaler once at the beginning of training.
    scaler = torch.cuda.amp.GradScaler()
    ### 

    for j in range(1,3):

          
        temp_loss = 0.1
        counter = 0
        a = time.time()
        for i, (x, y, tse) in enumerate(train_dataloader):
            m = time.time()
            x, y, tse = map(lambda x: x.cuda(1).unsqueeze(1), [x, y, tse])
            
            loss = torch.tensor([0], dtype=torch.float32).cuda(0)
            
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                output = t((x, tse))
                loss = nn.MSELoss()(output, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()

            ### mean loss calculation ###
            counter += 1
            temp_loss -= (temp_loss - loss.item()) / counter
            q = time.time() - m
            if i % 20 == 0:
                print(
                    f"Batch num {i}, The loss is {temp_loss:0.2f}, time to pass a single batch {q}, current lr is {scheduler.get_last_lr()}"
                )
        torch.save(t.state_dict(), f"model_on_train_one_epoch{j}.trc")
        torch.save(optimizer.state_dict(), f"model_on_train_state{j}.trc")
            
        ### One batch takes at most 0.31 seconds with size 256 -- this number is the same as picking a batch from random array
        ### so no bottleneck in the pipeline!!!!
        ### If we do not compile the model then it takes .47 (if you do so .32 seconds on A2000 gpu (a bit less on 3060)) seconds to pass a batch!!!!
        ### 12773226/256 = 49896 batches, therefore one epoch will take 49896*0.32/60 = 266 (float32) minutes (182.95 in mixed precision)
        ### therefore one epoch will take 4.4 hours (3 hours mixed precision) which is good I believe. torch.compile saves %35 accelation.
        print(
            f"The loss is {temp_loss:0.2f} and epoch {j}, {time.time() - a}   seconds to pass,"
        )
    
    ### For testing purposes only
    memmap_data = np.memmap("array_test.dat", dtype=np.float32)
    memmap_lengths = np.memmap("lengthsarray_test.dat", dtype=np.int32)
    lags = [513 for _ in memmap_lengths]

    data = ts_concatted(array=memmap_data, lengths=memmap_lengths, lags=lags)
    test_dataloader = DataLoader(data, batch_size=256, shuffle=False, num_workers=5)

    temp_loss = 0.1
    counter = 0

    with torch.no_grad():
        for i, (x, y, tse) in enumerate(test_dataloader):
            x, y, tse = map(lambda x: x.cuda(1).unsqueeze(1), [x, y, tse])
            loss = torch.tensor([0], dtype=torch.float32).cuda(1)
            output = t((x, tse))
            loss = nn.MSELoss()(output, y)
            counter += 1
            temp_loss -= (temp_loss - loss) / counter
            if i % 100 == 0:
                print(temp_loss.item())

if __name__ == '__main__':
    main()