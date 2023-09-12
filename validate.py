import os
import torch
from torch import nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from memmap_arrays import ts_concatted
import numpy as np
import tqdm
import pandas as pd


def return_dataset(**kwargs):
    memmap_data = np.memmap(kwargs["file"], dtype=np.float32)
    memmap_lengths = np.memmap(kwargs["length_file"], dtype=np.int32)
    lags = kwargs["lags"]
    lags = [lags for _ in memmap_lengths]
    data_ = ts_concatted(**{"array":memmap_data, "lengths": memmap_lengths, "lags": lags})
    return data_

def data_loader(data, **kwargs):
    return DataLoader(data, **kwargs)


def main(**kwargs):
    """
    things to do here
    1) Create the model
    2) Adjust the data
    3) Create a csv file
    4) Compare it with others
    """
    print(kwargs)


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
        help="lenghts of concatted time series",
        ### Here 512 +1 --- 1 is saved for the next days prediction!!!
    )
    parser.add_argument(
        "--batch_size",
        default = 256,
        type=str,
        help="lenghts of concatted time series",
        ### Here 512 +1 --- 1 is saved for the next days prediction!!!
    )
    parser.add_argument(
        "--config_file",
        type=str,
        help = "config file for th emodel"
    )
    parser.add_argument(
        "--repot_file",
        type = str,
        help = "Report file to be written"
    )
    args = parser.parse_args()
    ### --- ###
    kwargs = vars(args)
    main(**kwargs)


    