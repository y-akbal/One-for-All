import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from torch.utils.data import Dataset, DataLoader
from datautils import MyTrainDataset
from loss_logger import distributed_loss_track
import pickle


## This is important in the case that you compile the model!!!!
torch.set_float32_matmul_precision("high")
###


def ddp_setup(rank, world_size):
    """
    Args:
        rank: Unique identifier of each process
        world_size: Total number of processes
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

## We grabbed this from the official pytorch github repository.
class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        train_data: DataLoader,
        val_data: DataLoader,
        optimizer: torch.optim.Optimizer,
        gpu_id: int,
        save_every: int,
        val_loss_logger=None,
        train_loss_logger=None,
        compile=True
        # tracker ## this dude is for tracking stuff
    ) -> None:
        self.gpu_id = gpu_id
        self.model = model.to(gpu_id)

        self.model = DDP(self.model, device_ids=[gpu_id])
        if compile:
            self.model = torch.compile(self.model, backend="inductor")

        self.train_data = train_data
        self.val_data = val_data
        self.optimizer = optimizer
        # self.scheduler = scheduler
        self.save_every = save_every
        self.val_loss_logger = val_loss_logger
        self.train_loss_logger = train_loss_logger
        self.autocast = torch.autocast
        self.scaler = torch.cuda.amp.GradScaler()

    def _run_batch(self, source, targets):
        ### All the things like low precision training will happen dude!!!
        self.optimizer.zero_grad()
        with self.autocast(device_type="cuda", dtype=torch.bfloat16):
            output = self.model(source)
            loss = F.mse_loss(output, targets)
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()

        ### We log the loss,

    def _run_epoch(self, epoch, report_in_every = 100):
        # b_sz = len(next(iter(self.train_data))[0])
        if epoch % report_in_every == 0:
            print(f"[GPU{self.gpu_id}] Epoch {epoch}")
        self.train_data.sampler.set_epoch(epoch)
        for source, targets in self.train_data:
            source = source.to(self.gpu_id, non_blocking=True)
            targets = targets.to(self.gpu_id, non_blocking=True)
            self._run_batch(source, targets)
        self.validate()

    def _save_checkpoint(self, epoch):
        ckp = self.model.module.state_dict()
        PATH = "checkpoint.pt"
        torch.save(ckp, PATH)
        print(f"Epoch {epoch} | Training checkpoint saved at {PATH}")
    
    

    def train(self, max_epochs: int):
        for epoch in range(max_epochs):
            self._run_epoch(epoch)
            # if self.gpu_id == 0 and epoch % self.save_every == 0:
            #   self._save_checkpoint(epoch)

    def validate(self):
        self.model.eval()
        with torch.no_grad():  ## block tracking gradients
            for source, targets in self.val_data:
                source = source.to(self.gpu_id)
                targets = targets.to(self.gpu_id)
                output = self.model(source)  
                """"""
                loss = F.mse_loss(output, targets)
                self.val_loss_logger.update(loss.item())
            self.val_loss_logger.all_reduce()
            if self.gpu_id == 0:
                # print(self.val_loss_logger.get_avg_loss())
                self.val_loss_logger.reset()


def load_train_objs(seed=0, model_config=None):
    train_set = MyTrainDataset(2048)
    val_set = MyTrainDataset(2048)

    torch.manual_seed(seed)
    if model_config is not None:
        pass
    else:
        model = nn.Sequential(
            torch.nn.Linear(20, 300),
            torch.nn.GELU(),
            torch.nn.Linear(300, 300),
            torch.nn.GELU(),
            torch.nn.Linear(300, 300),
            torch.nn.GELU(),
            torch.nn.Linear(300, 1),
        )
    # load your model
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-5)
    ### Here you gotta add lr_scheduler this is pretty important
    return train_set, val_set, model, optimizer


def train_dataloader(dataset: Dataset, batch_size: int):
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=True,
        shuffle=False,
        sampler=DistributedSampler(dataset),
    )


def val_dataloader(dataset: Dataset, batch_size: int):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=True,
        shuffle=False,
        sampler=DistributedSampler(dataset),
        drop_last=True,  ### drop the last dude to avoid confusion
    )


def main(
    rank: int, world_size: int, save_every: int, total_epochs: int, batch_size: int
):
    ddp_setup(rank, world_size)
    dataset, valset, model, optimizer = load_train_objs()
    
    ## -- ##
    train_data = train_dataloader(dataset, batch_size)
    val_data = val_dataloader(valset, batch_size)
    loss_tracker = distributed_loss_track()
    trainer = Trainer(
        model,
        train_data,
        val_data,
        optimizer,
        rank,
        save_every,
        val_loss_logger=loss_tracker,
    )
    trainer.train(total_epochs)
    
    destroy_process_group()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="simple distributed training job")
    parser.add_argument(
        "--total_epochs",
        type=int,
        default=3000,
        help="Total epochs to train the model",
    )
    parser.add_argument(
        "--save_every", type=int, default=100000, help="How often to save a snapshot"
    )
    parser.add_argument(
        "--batch_size",
        default=256,
        type=int,
        help="Input batch size on each device (default: 32)",
    )
    args = parser.parse_args()
    import time

    ###  ---------------------------------------------------------------- ###
    world_size = torch.cuda.device_count()
    a = time.time()
    mp.spawn(
        main,
        args=(world_size, args.save_every, args.total_epochs, args.batch_size),
        nprocs=world_size,
    )
    print(time.time() - a)
