### training details are here!!!
### The ideas to be used here are:
### Here you should add here simiarity loss of layers ###
### Mixed predicison training details are here!!!
### Maybe abit accumulation gradients so forth so on!!!
### Adjust learning rate, maybe a bit weight decay would be needed
### Take snapshots some number of times, or after some callbacks.....

import torch
import torch.nn.functional as F
import os
from torch.utils.data import Dataset, DataLoader
import pickle

# from datautils import MyTrainDataset

import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import os


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


class loss_track:
    """
    summary: A class to keep track the loss of the training process,
    probably I will pickle the list keeping track of the loss list.
    """

    def __init__(self, file_name="logger.log"):
        ## File name for logging loss logs ##
        self.file_name = file_name
        ## --------------- ##
        self.__temp_loss__ = 1e-10
        self.counter = 1
        L = []

    def update(self, loss):
        self.__temp_loss__ -= (self.__temp_loss__ - loss) / (self.counter + 1)
        self.counter += 1

    def reset(self):
        self.__temp_loss__ = 1e-10
        self.counter = 1

    @property
    def loss(self):
        return self.__temp_loss__

    @loss.getter
    def loss(self):
        return self.__temp_loss__

    @loss.setter
    def loss(self, value):
        self.__temp_loss__ = value

    def load(self):
        pass

    def save(Self):
        pass


## We grabbed this from the official pytorch github repository.
class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        train_data: DataLoader,
        optimizer: torch.optim.Optimizer,
        gpu_id: int,
        save_every: int,
    ) -> None:
        self.gpu_id = gpu_id
        self.model = model.to(gpu_id)
        self.train_data = train_data
        self.optimizer = optimizer
        self.save_every = save_every
        self.model = DDP(model, device_ids=[gpu_id])
        ### To keep track of the local loss
        if gpu_id == 0:
            self.local_loss = 1e-2
            self.counter = 1

    def _run_batch(self, source, targets):
        self.optimizer.zero_grad()
        output = self.model(source)
        loss = F.mse_loss(output, targets)
        loss.backward()
        self.optimizer.step()

        if self.gpu_id == 0:
            self.local_loss -= (self.local_loss - loss.item()) / self.counter
            self.counter += 1

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
                print(self.local_loss)


def load_train_objs():
    train_set = MyTrainDataset(20480)  # load your dataset
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
