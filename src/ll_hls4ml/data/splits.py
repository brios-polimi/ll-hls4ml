"""Train/validation splits and target statistics."""

from __future__ import annotations

import torch
from torch.utils.data import Subset, random_split
from ll_hls4ml.io.schema import LABEL_KEYS


def random_train_val_split(dataset, val_fraction: float = 0.2, seed: int = 42):
    """Random train/val split. Returns (train_subset, val_subset)."""
    n = len(dataset)
    n_val = int(n * val_fraction)
    n_train = n - n_val
    generator = torch.Generator().manual_seed(seed)
    return random_split(dataset, [n_train, n_val], generator=generator)


def compute_target_stats(dataset) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute mean and std of log1p(targets) over dataset

    Returns tensors of shape (len(LABEL_KEYS)) for mean and std.
    """
    log_ys = torch.log1p(torch.stack([graph.y for graph in dataset], dim=0))
    return log_ys.mean(dim=0), log_ys.std(dim=0)  