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
from trainer import Trainer
from dataset_generator import data_set, data_loader


## Needed really?
torch.set_float32_matmul_precision("high")
## 

def return_dataset(**kwargs):
    train_dataset, val_dataset = kwargs["train_path"], kwargs["val_path"]
    train_data_kwargs, val_data_kwargs = kwargs["train_data_details"], kwargs["val_data_details"]
    train_data = data_set(**train_dataset)
    validation_data = data_set(**val_dataset)
    train_dataloader = data_loader(train_data, **train_data_kwargs)
    val_dataloader = data_loader(validation_data, **val_data_kwargs)
    return train_dataloader, val_dataloader


def return_training_stuff(seed = 0, **cfg):
    keys = ["model_config", "optimizer_config","scheduler_config"]
    model_config, optimizer_config, scheduler_config = map(lambda x: cfg.__getitem__(x), keys)
    torch.manual_seed(0)
    model = Model(**model_config)
    optimizer = torch.optim.SGD(model.parameters(), **optimizer_config)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, **scheduler_config)
    return model, optimizer, scheduler



@hydra.main(version_base=None, config_path=".", config_name="model_config")
def main(cfg : DictConfig):

    train_dataloader, val_dataloader = return_dataset(cfg["data_config"])
    trainer_config = cfg["trainer_config"]
    model, optimizer, scheduler = return_training_stuff(**cfg)
    
    trainer = Trainer(model = model, 
            train_data= train_dataloader,
            val_data = val_dataloader,
            optimizer = optimizer, 
            scheduler = scheduler,
            **trainer_config,                        
    )
    trainer.train()
if __name__ == '__main__':
    main()
    print("Şifa olsun!!!")