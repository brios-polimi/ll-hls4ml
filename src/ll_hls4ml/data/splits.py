"""Train/validation splits and target statistics."""

from __future__ import annotations

import torch
from torch.utils.data import Subset, random_split


def random_train_val_test_split(dataset, val_fraction: float = 0.15, test_fraction: float = 0.15, seed: int = 42):
    """Random train/val split. Returns (train_subset, val_subset)."""
    n = len(dataset)
    n_val = int(n * val_fraction)
    n_test = int(n * test_fraction)
    n_train = n - n_val - n_test
    generator = torch.Generator().manual_seed(seed)
    return random_split(dataset, [n_train, n_val, n_test], generator=generator)


