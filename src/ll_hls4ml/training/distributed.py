"""Distributed training helpers for DDP / torchrun."""

from __future__ import annotations

import os

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


def is_distributed() -> bool:
    return dist.is_available() and dist.is_initialized() and get_world_size() > 1


def get_rank() -> int:
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank()
    return 0


def get_world_size() -> int:
    if dist.is_available() and dist.is_initialized():
        return dist.get_world_size()
    return 1


def get_local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", get_rank()))


def is_main_process() -> bool:
    return get_rank() == 0


def setup_from_env() -> tuple[int, int, int]:
    """
    Initialize process group from torchrun environment variables.

    Returns (rank, world_size, local_rank). No-op when not launched via torchrun.
    """
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        return 0, 1, 0

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", rank))

    if not dist.is_initialized():
        use_cuda = torch.cuda.is_available() and torch.cuda.device_count() > local_rank
        if use_cuda:
            torch.cuda.set_device(local_rank)
            backend = "nccl"
        else:
            backend = "gloo"
        dist.init_process_group(backend=backend)

    return rank, world_size, local_rank


def cleanup_ddp() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if isinstance(model, DDP) else model


def wrap_ddp(model: torch.nn.Module, device: torch.device) -> torch.nn.Module:
    if not is_distributed():
        return model
    if isinstance(model, DDP):
        return model
    local_rank = get_local_rank()
    return DDP(model, device_ids=[local_rank] if device.type == "cuda" else None)
