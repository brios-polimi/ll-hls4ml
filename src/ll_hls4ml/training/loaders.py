"""PyG DataLoader helpers."""

import os
from pathlib import Path
import platform
import tempfile
from concurrent.futures import ThreadPoolExecutor

import torch
from torch.utils.data.distributed import DistributedSampler
from torch_geometric.loader import DataLoader as PyGDataLoader


class ThreadPrefetchLoader:
    """Overlap one CPU load/collation with the current GPU training step."""

    def __init__(self, loader):
        self.loader = loader

    def __len__(self):
        return len(self.loader)

    def __getattr__(self, name):
        return getattr(self.loader, name)

    def __iter__(self):
        iterator = iter(self.loader)
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(next, iterator, None)
            while True:
                batch = future.result()
                if batch is None:
                    return
                future = executor.submit(next, iterator, None)
                yield batch


def _default_num_workers(distributed: bool) -> int:
    if distributed or os.environ.get("KAGGLE_KERNEL_RUN_TYPE"):
        return 0
    if "microsoft" in platform.release().lower():
        return 0
    # Python 3.14 uses a forkserver on POSIX. Unix-domain sockets cannot be
    # created in WSL's Windows-mounted temp directories, so worker startup
    # fails before the first batch. Users can still override this explicitly
    # after pointing TMPDIR at a native Linux path.
    if Path(tempfile.gettempdir()).as_posix().startswith("/mnt/"):
        return 0
    cpu_cores = os.cpu_count() or 2
    return max(2, min(4, cpu_cores))


def make_loader(
    ds,
    batch_size,
    shuffle=True,
    num_workers=None,
    distributed=False,
    sampler=None,
    pin_memory=None,
    prefetch_factor=2,
    thread_prefetch=False,
):
    """
    Build a PyG DataLoader, optionally sharded with DistributedSampler.

    When ``distributed=True``, the train loader uses a DistributedSampler and
    ``shuffle`` is disabled (call ``loader.sampler.set_epoch(epoch)`` each epoch).
    """
    if distributed:
        if sampler is not None:
            raise ValueError("custom sampler is not supported with distributed loading")
        sampler = DistributedSampler(ds, shuffle=shuffle)
        shuffle = False
    elif sampler is not None:
        shuffle = False

    if num_workers is None:
        num_workers = _default_num_workers(distributed)
    if pin_memory is None:
        pin_memory = torch.cuda.is_available()

    loader_kwargs = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "persistent_workers": num_workers > 0,
    }
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = prefetch_factor
    if sampler is not None:
        loader_kwargs["sampler"] = sampler

    loader = PyGDataLoader(ds, **loader_kwargs)
    return ThreadPrefetchLoader(loader) if thread_prefetch else loader
