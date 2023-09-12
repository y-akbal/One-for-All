import os
import torch
from torch import nn as nn
from torch.nn import functional as F
from layers import block, Upsampling, Linear
from torch.utils.data import DataLoader
from memmap_arrays import ts_concatted
import numpy as np
import time
from main_model import Model
## import hydra now
import hydra
from omegaconf import DictConfig, OmegaConf


## This is important in the case that you compile the model!!!!
torch.set_float32_matmul_precision("high")


def return_dataset(**kwargs):
    memmap_data = np.memmap(kwargs["file"], dtype=np.float32)
    memmap_lengths = np.memmap(kwargs["length_file"], dtype=np.int32)
    lags = kwargs["lags"]
    lags = [lags for _ in memmap_lengths]
    data_ = ts_concatted(**{"array":memmap_data, "lengths": memmap_lengths, "lags": lags})
    return data_

def data_loader(data, **kwargs):
    return DataLoader(data, **kwargs)

class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        train_data: DataLoader,
        val_data: DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler,
        save_every: int,
        snapshot_path: str,
        loss_loger = None,
        compile_model = False,
    ) -> None:
        self.gpu_id = int(os.environ["LOCAL_RANK"])
        self.model = model.to(self.gpu_id)
        if compile_model:
            self.model = torch.compile(self.model)
        ## -- ##
        self.train_data = train_data
        self.val_data = val_data
        ## -- ##
        self.optimizer = optimizer
        self.scaler = torch.cuda.amp.GradScaler(enabled= True)
        self.scheduler = scheduler
        ## -- ##
        self.save_every = save_every
        ## -- ##
        self.snapshot_path = snapshot_path
        if os.path.exists(snapshot_path):
            print("Loading snapshot")
            self._load_snapshot(snapshot_path)

    def _run_epoch(self):
        for i, (x, y, tse) in enumerate(self.train_data):
            self.model.train()
            temp_loss = 0.1
            counter = 0
            m = time.time()
            x, y, tse = map(lambda x: x.cuda(self.gpu_id).unsqueeze(1), [x, y, tse])
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                output = self.model((x, tse))
                loss = F.mse_loss(output, y)
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()
            self.optimizer.zero_grad()
            ### Let's start the training ###
            counter += 1
            temp_loss -= (temp_loss - loss.item()) / counter
            q = time.time() - m
            if i % 10 == 0:
                print(f"{i}th batch passed, it takes {q} to pass a batch!!!, the loss is {temp_loss}, lr is {self.scheduler.get_last_lr()}")

    def _save_snapshot(self, epoch):
        snapshot = {
            "MODEL_STATE": self.model.module.state_dict(),
            "OPTIMIZATION_STATE": self.optimizer.state_dict(),
            "EPOCHS_RUN": epoch,
        }
        torch.save(snapshot, self.snapshot_path)
        print(f"Epoch {epoch} | Training snapshot saved at {self.snapshot_path}")

    def train(self, max_epochs: int):
        for epoch in range(max_epochs):
            self._run_epoch()
            if epoch % self.save_every == 0:
                self._save_snapshot(epoch)
                
    def validate(self):
        with torch.no_grad():
            temp_loss = 0
            counter = 0
            for i, (x, y, tse) in enumerate(self.val_data):
                x, y, tse = map(lambda x: x.cuda(self.gpu_id).unsqueeze(1), [x, y, tse])
                output = self.model((x, tse))
                loss = nn.MSELoss()(output, y)
                counter += 1
                temp_loss -= (temp_loss - loss) / counter
                if i % 10 == 0:
                    print(temp_loss.item())       
        return temp_loss


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