import os
os.environ["TORCH_DISTRIBUTED_DEBUG"] = "INFO"


import torch
from torch import nn as nn
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.distributed import init_process_group, destroy_process_group

from main_model import Model
import hydra
from omegaconf import DictConfig, OmegaConf
from trainer import Trainer
from training_tools import loss_track, wandb_loss_logger
from dataset_generator import data_set

## Needed really?
torch.set_float32_matmul_precision("high")
## 

class ddp_setup(object):
    def __init__(self):
        pass
    def __enter__(self):
        init_process_group(backend="nccl")
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        return 1
    def __exit__(self, *args):
        destroy_process_group()
        return 0


def return_dataset(**kwargs):
    train_kwargs, val_kwargs = kwargs["train_path"], kwargs["val_path"]
    train_data, validation_data = data_set(**train_kwargs), data_set(**val_kwargs)
    
    train_sampler = DistributedSampler(train_data, shuffle = True)
    validation_sampler = DistributedSampler(validation_data, shuffle = False)
    
    train_data_kwargs, val_data_kwargs = kwargs["train_data_details"], kwargs["val_data_details"]

    train_dataloader = DataLoader(dataset = train_data, 
                                  sampler = train_sampler,
                                  **train_data_kwargs)
    val_dataloader = DataLoader(dataset = validation_data, 
                                sampler = validation_sampler,
                                **val_data_kwargs)

    return train_dataloader, val_dataloader 


def return_training_stuff(seed = 3, **cfg):
    keys = ["model_config", "optimizer_config", "scheduler_config"]
    model_config, optimizer_config, scheduler_config = map(lambda x: cfg.__getitem__(x), keys)
    
    torch.manual_seed(seed)
    model = Model(**model_config)
    
    optimizer = torch.optim.AdamW(model.parameters(), **optimizer_config)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, **scheduler_config)
    
    return model, optimizer, scheduler



@hydra.main(version_base=None, config_path=".", config_name="model_config")
def main(cfg : DictConfig):

    with ddp_setup():
        train_dataloader, val_dataloader = return_dataset(**cfg["data"])

        print(f"There are a total number of {len(train_dataloader)} number of training batches, and {len(val_dataloader)} validation batches!!!")

        trainer_config = cfg["trainer_config"]
        model, optimizer, scheduler = return_training_stuff(**cfg)
        local_gpu_id = int(os.environ["LOCAL_RANK"])
        
        trainer = Trainer(model = model, 
            train_data= train_dataloader,
            val_data = val_dataloader, 
            optimizer = optimizer, 
            scheduler = scheduler,
            train_loss_logger = loss_track(gpu_id = local_gpu_id),
            val_loss_logger = loss_track(gpu_id = local_gpu_id),
            wandb_loss_logger = wandb_loss_logger(**cfg),
            **trainer_config,                        
        )

        trainer.train()

if __name__ == '__main__':
    main()
    
    