import torch
from torch import nn as nn
from torch.nn import functional as F
from layers import block, Upsampling, Linear
from torch.utils.data import DataLoader
from memmap_arrays import ts_concatted
import numpy as np
import time
from main_model import Model




## This is important in the case that you compile the model!!!!
torch.set_float32_matmul_precision("high")
device = "cuda" if torch.cuda.is_available() else "cpu"



def data_loader(**kwargs):
    pass

def train_model(model:nn.Module, train_data:DataLoader, optimizer:torch.Optimizer):
    pass

def validate_model(model:nn.Module, validate_data:DataLoader):
    pass



def main():
    ## Here we create the model!!!
    torch.manual_seed(0)
    model = Model(number_ts=264, embedding_dim=256, n_blocks=6, number_of_heads = 4)
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
            if i % 30 == 0:
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

    temp_loss = 1e-5
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