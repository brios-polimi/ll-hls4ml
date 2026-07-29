"""Train/validation splits and target statistics."""

from __future__ import annotations

from collections import defaultdict
import random

import torch
from torch.utils.data import Subset, random_split


def random_train_val_test_split(
    dataset,
    val_fraction: float = 0.15,
    test_fraction: float = 0.15,
    seed: int = 42,
):
    """Return deterministic random train, validation, and test subsets."""
    if val_fraction < 0 or test_fraction < 0 or val_fraction + test_fraction >= 1:
        raise ValueError(
            "val_fraction and test_fraction must be non-negative and sum to less than 1"
        )
    n = len(dataset)
    n_val = int(n * val_fraction)
    n_test = int(n * test_fraction)
    n_train = n - n_val - n_test
    generator = torch.Generator().manual_seed(seed)
    return random_split(dataset, [n_train, n_val, n_test], generator=generator)


def benchmark_train_val_test_split(
    dataset,
    val_fraction: float = 0.15,
    test_fraction: float = 0.15,
    seed: int = 42,
):
    """Prefer official membership; otherwise group and stratify by family."""
    official = defaultdict(list)
    for index in range(len(dataset)):
        split = str(dataset.metadata_of(index).get("dataset_split", "")).lower()
        if split in {"train", "val", "validation", "test"}:
            official["validation" if split in {"val", "validation"} else split].append(
                index
            )
    if sum(map(len, official.values())) == len(dataset) and all(
        official[name] for name in ("train", "validation", "test")
    ):
        return tuple(
            Subset(dataset, official[name])
            for name in ("train", "validation", "test")
        )

    if val_fraction < 0 or test_fraction < 0 or val_fraction + test_fraction >= 1:
        raise ValueError(
            "val_fraction and test_fraction must be non-negative and sum to less than 1"
        )

    groups: dict[tuple[str, str], list[int]] = defaultdict(list)
    for index in range(len(dataset)):
        family = dataset.type_of(index)
        group_id = str(
            dataset.metadata_of(index).get("group_id")
            or dataset.paths[index].stem
        )
        groups[(family, group_id)].append(index)

    by_family: dict[str, list[list[int]]] = defaultdict(list)
    for (family, _group_id), indices in groups.items():
        by_family[family].append(indices)

    rng = random.Random(seed)
    split_indices = {"train": [], "validation": [], "test": []}
    for family_groups in by_family.values():
        rng.shuffle(family_groups)
        total = sum(map(len, family_groups))
        targets = {
            "validation": total * val_fraction,
            "test": total * test_fraction,
        }
        assigned = {"validation": 0, "test": 0}
        for indices in family_groups:
            candidates = [
                name
                for name in ("validation", "test")
                if assigned[name] < targets[name]
            ]
            if candidates:
                destination = max(
                    candidates,
                    key=lambda name: targets[name] - assigned[name],
                )
                split_indices[destination].extend(indices)
                assigned[destination] += len(indices)
            else:
                split_indices["train"].extend(indices)

    return tuple(
        Subset(dataset, sorted(split_indices[name]))
        for name in ("train", "validation", "test")
    )
