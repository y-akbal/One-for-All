import os
import torch
from torch import nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from dataset_generator import ts_concatted
import numpy as np
import time
from main_model import Model
## import hydra now
import hydra
from omegaconf import DictConfig, OmegaConf
from ddp_trainer import Trainer
from dataset_generator import return_dataset, data_loader


## Needed really?
torch.set_float32_matmul_precision("high")
## 

def return_optimizer(**kwargs):
    pass
def return_scheduler(**kwargs):
    pass




@hydra.main(version_base=None, config_path=".", config_name="model_config")
def main(cfg : DictConfig):
    ## model configuration ##
    model_config, optimizer_config, scheduler_config = cfg["model_config"], cfg["optimizer_config"], cfg["scheduler_config"]
    snapshot_path = cfg["snapshot_path"]
    save_every = cfg["save_every"]
    ## model_config -- optimizer config -- scheduler config ##
    torch.manual_seed(0)
    model = Model(**model_config)
    os.environ["LOCAL_RANK"] = cfg["local_rank"]
    ## -- ##
    
    ### We now do some data_stuff ###
    train_dataset, val_dataset = cfg["data"]["train_path"], cfg["data"]["val_path"]
    train_data_kwargs, val_data_kwargs = cfg["data"]["train_data_details"], cfg["data"]["val_data_details"]
    train_data = return_dataset(**train_dataset)
    validation_data = return_dataset(**val_dataset)
    train_dataloader = data_loader(train_data, **train_data_kwargs)
    val_dataloader = data_loader(validation_data, **val_data_kwargs)
    print(f"Note that the train dataset contains {len(train_dataloader)}! batches!!")
    ### --- End of data grabbing --- ###
    
    ### Optimizer ###
    optimizer = torch.optim.SGD(model.parameters(), **optimizer_config)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, **scheduler_config)
    ### Training Stuff Here it comes ###
    
    trainer = Trainer(model = model, 
            train_data= train_dataloader,
            val_data = val_dataloader,
            optimizer = optimizer, 
            scheduler = scheduler,
            save_every = save_every,
            snapshot_path=snapshot_path,
            compile_model=cfg["compile_model"],
            
    )
    
    trainer.train(max_epochs = 2)
    trainer.validate()

    
if __name__ == '__main__':
    main()