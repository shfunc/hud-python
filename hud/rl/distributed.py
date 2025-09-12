"""Distributed training utilities for GRPO."""

import os
import torch
import torch.distributed as dist
from typing import Any, Optional


def setup_distributed() -> tuple[int, int, int]:
    """Initialize distributed training environment.
    
    Returns:
        local_rank: GPU index for this process
        global_rank: Global process index
        world_size: Total number of processes
    """
    if "RANK" in os.environ:
        # Launched with torchrun
        local_rank = int(os.environ["LOCAL_RANK"])
        global_rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
    else:
        # Single GPU mode
        local_rank = 0
        global_rank = 0
        world_size = 1
        
    # Set device
    torch.cuda.set_device(local_rank)
    
    # Initialize process group if multi-GPU
    if world_size > 1:
        dist.init_process_group("nccl")
        
    return local_rank, global_rank, world_size


def cleanup_distributed() -> None:
    """Clean up distributed environment."""
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process() -> bool:
    """Check if this is the main process (rank 0)."""
    if not dist.is_initialized():
        return True
    return dist.get_rank() == 0


def synchronize() -> None:
    """Synchronize all processes."""
    if dist.is_initialized():
        dist.barrier()


def all_reduce_mean(tensor: torch.Tensor) -> torch.Tensor:
    """Average a tensor across all processes."""
    if not dist.is_initialized():
        return tensor
        
    world_size = dist.get_world_size()
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    tensor /= world_size
    return tensor


def broadcast_object(obj: Any, src: int = 0) -> Any:
    """Broadcast a Python object from src rank to all ranks."""
    if not dist.is_initialized():
        return obj
        
    obj_list = [obj] if dist.get_rank() == src else [None]
    dist.broadcast_object_list(obj_list, src=src)
    return obj_list[0]


def gather_tensors(tensor: torch.Tensor) -> Optional[list[torch.Tensor]]:
    """Gather tensors from all ranks to rank 0.
    
    Returns:
        List of tensors on rank 0, None on other ranks
    """
    if not dist.is_initialized():
        return [tensor]
        
    world_size = dist.get_world_size()
    
    if dist.get_rank() == 0:
        gathered = [torch.zeros_like(tensor) for _ in range(world_size)]
        dist.gather(tensor, gathered, dst=0)
        return gathered
    else:
        dist.gather(tensor, None, dst=0)
        return None
