"""PyG DataLoader helpers."""

import os

from torch.utils.data.distributed import DistributedSampler
from torch_geometric.loader import DataLoader as PyGDataLoader


def _default_num_workers(distributed: bool) -> int:
    if distributed or os.environ.get("KAGGLE_KERNEL_RUN_TYPE"):
        return 0
    cpu_cores = os.cpu_count() or 2
    return max(2, min(4, cpu_cores))


def make_loader(ds, batch_size, shuffle=True, num_workers=None, distributed=False):
    """
    Build a PyG DataLoader, optionally sharded with DistributedSampler.

    When ``distributed=True``, the train loader uses a DistributedSampler and
    ``shuffle`` is disabled (call ``loader.sampler.set_epoch(epoch)`` each epoch).
    """
    sampler = None
    if distributed:
        sampler = DistributedSampler(ds, shuffle=shuffle)
        shuffle = False

    if num_workers is None:
        num_workers = _default_num_workers(distributed)

    loader_kwargs = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": num_workers,
        "pin_memory": False,
        "persistent_workers": num_workers > 0,
    }
    if sampler is not None:
        loader_kwargs["sampler"] = sampler

    return PyGDataLoader(ds, **loader_kwargs)
