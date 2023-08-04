### training details are here!!!
### The ideas to be used here are:
### Here you should add here simiarity loss of layers ###
### Mixed predicison training details are here!!!
### Maybe abit accumulation gradients so forth so on!!!
### Adjust learning rate, maybe a bit weight decay would be needed
### Take snapshots some number of times, or after some callbacks.....
import os
import torch
import torch.nn.functional as F
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from torch.utils.data import Dataset, DataLoader
from loss_logger import loss_track
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
#       test_data: DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.Scheduler,
        gpu_id: int,
        save_every: int,
        loss_logger = loss_track(),
        #tracker ## this dude is for tracking stuff
    ) -> None:
        self.gpu_id = gpu_id
        self.model = model.to(gpu_id)
        self.tracker = loss_logger
        self.train_data = train_data
        #self.test_data = test_data
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.save_every = save_every
        self.model = DDP(model, device_ids=[gpu_id])
        

    def _run_batch(self, source, targets):
        self.optimizer.zero_grad()
        output = self.model(source)
        loss = F.mse_loss(output, targets)
        loss.backward()
        self.optimizer.step()
        ### We log the loss, 
        if self.gpu_id == 0:
            loss_ = loss.item()
            self.tracker.update(loss_)

    def _run_epoch(self, epoch):
        b_sz = len(next(iter(self.train_data))[0])
        print(
            f"[GPU{self.gpu_id}] Epoch {epoch} | Batchsize: {b_sz} | Steps: {len(self.train_data)}"
        )
        self.train_data.sampler.set_epoch(epoch)
        for source, targets in self.train_data:
            source = source.to(self.gpu_id)
            targets = targets.to(self.gpu_id)
            self._run_batch(source, targets)

    def _save_checkpoint(self, epoch):
        ckp = self.model.module.state_dict()
        PATH = "checkpoint.pt"
        torch.save(ckp, PATH)
        print(f"Epoch {epoch} | Training checkpoint saved at {PATH}")

    def train(self, max_epochs: int):
        for epoch in range(max_epochs):
            self._run_epoch(epoch)
            if self.gpu_id == 0 and epoch % self.save_every == 0:
                self._save_checkpoint(epoch)
            if self.gpu_id == 0:
                

def load_train_objs():
    train_set = torch.rand(10,10)
    model = nn.Sequential(
        torch.nn.Linear(500, 300),
        torch.nn.GELU(),
        torch.nn.Linear(300, 300),
        torch.nn.GELU(),
        torch.nn.Linear(300, 100),
        torch.nn.Linear(100, 1),
    )
    # load your model
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-10)
    return train_set, model, optimizer


def prepare_dataloader(dataset: Dataset, batch_size: int):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=True,
        shuffle=False,
        sampler=DistributedSampler(dataset),
    )


def main(
    rank: int, world_size: int, save_every: int, total_epochs: int, batch_size: int
):
    ddp_setup(rank, world_size)
    dataset, model, optimizer = load_train_objs()
    train_data = prepare_dataloader(dataset, batch_size)
    trainer = Trainer(model, train_data, optimizer, rank, save_every)
    trainer.train(total_epochs)
    destroy_process_group()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="simple distributed training job")
    parser.add_argument(
        "total_epochs", type=int, help="Total epochs to train the model"
    )
    parser.add_argument("save_every", type=int, help="How often to save a snapshot")
    parser.add_argument(
        "--batch_size",
        default=32,
        type=int,
        help="Input batch size on each device (default: 32)",
    )
    args = parser.parse_args()
    ###  ---------------------------------------------------------------- ###
    world_size = torch.cuda.device_count()
    mp.spawn(
        main,
        args=(world_size, args.save_every, args.total_epochs, args.batch_size),
        nprocs=world_size,
    )
