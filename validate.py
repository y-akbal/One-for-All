import os
import torch
from torch import nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from main_model import Model
from memmap_arrays import ts_concatted
import numpy as np
import tqdm
import pandas as pd
import time


model = nn.Sequential(*[
    nn.Linear(100,10),
    nn.GELU(),
    nn.Linear(10, 10)
])

X = np.random.normal(size = (1000, 100)).astype(np.float32)
with torch.no_grad():
    y = model(torch.tensor(X, dtype = torch.float32)).numpy()

class dd(Dataset):
    def __init__(self):
        self.X = X
        self.y = y
    def __getitem__(self, i):
        return torch.tensor(self.X[i], dtype = torch.float32), torch.tensor(self.y[i], dtype = torch.float32)
    def __len__(self):
        return len(self.X)
d = dd()
data = DataLoader(d, batch_size = 32, shuffle = False)

y_out = np.zeros(d.y.shape)
with torch.no_grad():
    for i, (x,y) in enumerate(data):
        y_output = model(x).numpy()
        y_out[32*i:32*(i+1)] = y_output
## okito dokito        

np.isclose(y_out, d.y) ### great succeesss!!!!



def return_dataset(**kwargs):
    memmap_data = np.memmap(kwargs["file"], dtype=np.float32)
    memmap_lengths = np.memmap(kwargs["length_file"], dtype=np.int32)
    lags = kwargs["lags"]
    lags = [lags for _ in memmap_lengths]
    data_ = ts_concatted(**{"array":memmap_data, "lengths": memmap_lengths, "lags": lags})
    return data_


def main(**kwargs):
    """
    things to do here
    1) Create the model -- ok
    2) Load the data -- ok
    2.5) Save the prediceted and the ground truth
    3) Create a csv file
    4) Compare it with others
    """
    ### First grab the data:
    data = return_dataset(**kwargs)
    batch_size = kwargs["batch_size"]
    batched_data = DataLoader(data, batch_size = batch_size, shuffle = False)
    ## -- ##
    ## Let's load the model from trained file ##    
    file_name = kwargs["file_name"]   
    device =  kwargs["gpu"]
    try:
        model = Model.from_pretrained(file_name).cuda(device)
    except Exception as exp:
        print(f"Something went wrong with {exp}!!!")
    ### If we come so far everything shoud be good ## Now let's give a try!!!!
    
    
    
        

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Validation of model on a dataset")

    parser.add_argument(
        "--array_name",
        type=str,
        help="concatted time series",
    )
    parser.add_argument(
        "--lengths_array",
        type=str,
        help="lenghts of concatted time series",
    )
    parser.add_argument(
        "--lags",
        default=513,
        type=str,
        help="lags to be used",
        ### Here 512 + 1 --- here 1 is saved for the next days prediction!!!
    )
    parser.add_argument(
        "--batch_size",
        default = 256,
        type=str,
        help="lenghts of concatted time series",
    )
    parser.add_argument(
        "--model_file",
        type=str,
        help = "config file to create the model that has been already trained", 
    )
    parser.add_argument(
        "--report_file",
        default = "report.csv",
        type = str,
        help = "Report file to be written"
    )
    parser.add_argument(
        "--gpu",
        default = 0,
        type = int,
        help = "CUDA device to use"
    )
    args = parser.parse_args()
    ### --- ###
    kwargs = vars(args)
    main(**kwargs)


    