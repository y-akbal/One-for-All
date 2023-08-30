import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import (
    init_process_group,
    destroy_process_group,
    all_reduce,
    ReduceOp,
)
from loss_logger import distributed_loss_track


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


def main(rank: int, world_size: int, t):
    ddp_setup(rank, world_size)
    loss_tracker = distributed_loss_track()
    for i in t.tolist():
        loss_tracker.update(i)
    print(
        loss_tracker.temp_loss,
        loss_tracker.counter,
    )
    loss_tracker.all_reduce()
    print(loss_tracker.temp_loss, loss_tracker.get_avg_loss())
    destroy_process_group()


if __name__ == "__main__":
    ###  ---------------------------------------------------------------- ###
    world_size = torch.cuda.device_count()
    import time
    from time import sleep

    torch.manual_seed(0)
    t = torch.randn(size=(100000000,))
    print(f"{t}")
    mp.spawn(
        main,
        args=(world_size, t),
        nprocs=world_size,
    )
