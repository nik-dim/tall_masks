import os

import torch


def setup_ddp(rank, world_size, port=12357):
    """
    Set up the Distributed Data Parallel (DDP) environment for multi-GPU training.

    Args:
        rank (int): The rank of the current process.
        world_size (int): The total number of processes.
        port (int, optional): The port number for communication. Defaults to 12357.
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)

    # initialize the process group
    torch.distributed.init_process_group(
        "nccl",
        rank=rank,
        world_size=world_size,
    )
    torch.cuda.set_device(rank)
    torch.distributed.barrier()


def cleanup_ddp():
    """
    Cleans up the distributed data parallel (DDP) process group.

    This function is responsible for cleaning up the DDP process group after training or inference is complete.
    It ensures that all resources used by the DDP process group"""
    torch.distributed.destroy_process_group()


def is_main_process():
    return torch.distributed.get_rank() == 0


def distribute_loader(loader):
    return torch.utils.data.DataLoader(
        loader.dataset,
        batch_size=loader.batch_size // torch.distributed.get_world_size(),
        sampler=torch.utils.data.distributed.DistributedSampler(
            loader.dataset,
            num_replicas=torch.distributed.get_world_size(),
            rank=torch.distributed.get_rank(),
        ),
        num_workers=loader.num_workers,
        pin_memory=loader.pin_memory,
    )
