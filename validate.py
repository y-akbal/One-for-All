import os
import torch
from torch import nn as nn
#from torch.nn import functional as F
from torch.utils.data import DataLoader#, Dataset
from main_model import Model
from dataset_generator import ts_concatted, data_set
import numpy as np
from tqdm import tqdm
import pandas as pd
import argparse

torch.set_float32_matmul_precision('high')

DATA_DIR = "data"
DATA_FILES = "array_test.dat", "lengthsarray_test.dat", "names_array_test.txt"
BATCH_SIZE = 768
LAGSIZE = 512


def return_dataset(data_dir:str = DATA_DIR,
                    data_files:tuple[str, str, str] = DATA_FILES,
                    batch_size:int = BATCH_SIZE,
                    lag_size = LAGSIZE)-> DataLoader:
    
    file_name, lengths_name, names_file  = map(lambda x: os.path.join(data_dir, x), data_files)
     
    data = data_set(**{"file":file_name, "length_file":lengths_name, "lags":lag_size, "file_names":names_file})
    batched_data = DataLoader(data, batch_size = batch_size, shuffle=False, num_workers= 10, prefetch_factor = 3)
    print(f"Data set loaded successfully, it has {len(batched_data)} many batches!!!")
    return batched_data

"""
for x, cls, file_name in return_dataset():
    print(x.shape, cls.shape, file_name)
"""

def preprocess_model_file(model_state:dict) -> dict:
    ## 1) This dude moves to states to cpu, in which case you would like to do stuff on cpu
    ## 2) Since the model is trained using DDP, the weights are somehow has weird names, this is the place
    ## - that we change the names!!!
    new_model_state_dict = {}
    for keys, values in model_state.items():
        if "module" in keys:
            keys = keys.replace("module.", "")
        new_model_state_dict[keys] = values.cpu()
    return new_model_state_dict
    
def return_model(**kwargs)->tuple[nn.Module, int]:
    file_name = os.path.join(kwargs["model_dir"], kwargs["model_file"])
    states = torch.load(file_name)
    ## -- determine the device -- ##
    gpu_0 = kwargs["gpu"]
    ## -- ##
    device = (
        f"cuda:{gpu_0}"
        if torch.cuda.is_available() and kwargs["gpu"] is not None
        else "cpu"
    )
    device = torch.device(device)
    print(f"The device to be used is {device}")
    ## -- ## ok we now load the model!!!
    model_state_dict = preprocess_model_file(states["model_state_dict"])
    model_config = states["model_config"]
    print(model_config)
    ## create the model
    model = Model.from_data_class(model_config)
    model.load_state_dict(model_state_dict),
    model.eval()
    model = model.to(device)
    if kwargs["compile_model"] == "True":
        model = torch.compile(model)
        print("The model loaded and compiled successfully!!!")
        return model, device
   
    print("Model loaded successfully!!!")
    return model, device


def main(**kwargs):
    """
    things to do here
    1) Create the model -- ok
    2) Load the data -- ok
    2.5) Save the prediceted and the ground truth
    3) Create a csv file
    4) Do the metric computations
    """
    ### First grab the data:
    batched_data = return_dataset()
    ## -- ##
    ## Let's load the model from trained file ##
    model, device = return_model(**kwargs)
    ## -- ##
    Y_output = []
    Y = []
    TSE = []
    data_ = tqdm(batched_data)
    with torch.inference_mode():
        for source, cls_, file_name in data_:
            source, cls_ = map(lambda x: x.to(device, non_blocking=True), [source, cls_])
            N = source.shape[1]
            X, y = source[:, :-1], source[:,4:N:4]
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                y_output = model([X.unsqueeze(-2), cls_.unsqueeze(-1)])
            y_output = y_output.to("cpu").numpy()
            Y_output.append(y_output[:, -1])
            Y.append(y[:, -1].to("cpu").numpy())
            TSE.append(file_name)
            

    ## -- ##
    ## We next convert everything into a np array--!!!
    map_array = map(lambda x: np.array(x).flatten(), [Y_output, Y, TSE])
    # for l in map_array:
    #   print(l.shape)
    dict_ = {name: array for name, array in zip(["TSE", "Y_output", "Y"], map_array)}
    ### -- ###

    data_frame = pd.DataFrame().from_dict(dict_)
    try:
        data_frame.to_csv("results.csv")
    except Exception as e:
        print(f"Something went wrong with {e}")
 
 
 
    ### We have now converted everything into a csv file--!!!
    ### -- ###
    ### -- ###
    ###Our metrics will be R^2, MAE, MSE ###
    ### We shall give R^2 for each different city, as this is important to mention ###
 

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Validation of model on a dataset")

    parser.add_argument(
        "--model_dir",
        default="model",
        type=str,
        help="directory of model files",
    )

    parser.add_argument(
        "--model_file",
        default="small_model_",
        type=str,
        help="config file to create the model that has been already trained",
    )

    parser.add_argument(
        "--compile_model",
        default="False",
        type=str,
        help="Compile model for fast inference!!!",
    )

    parser.add_argument(
        "--gpu",
        default = 0,
        type=int,
        help="GPU to use for inference!!!",
    )

    parser.add_argument(
        "--report_file",
        default="report.csv",
        type=str,
        help="Report file to be written",
    )
    args = parser.parse_args()
    ### --- ###
    kwargs = vars(args)
    main(**kwargs)
