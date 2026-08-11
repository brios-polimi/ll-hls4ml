"""Train/validation splits and target statistics."""

from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import math
from pathlib import Path
import random
import re

import torch
from torch.utils.data import Subset, random_split


def saved_manifest_split(
    dataset,
    manifest: dict,
    tensor_root: str | Path,
    split_names: tuple[str, ...],
    *,
    require_all: bool = True,
):
    """Select exact saved split membership by relative tensor path."""
    tensor_root = Path(tensor_root)
    index_by_path = {
        path.relative_to(tensor_root).as_posix(): index
        for index, path in enumerate(dataset.paths)
    }
    manifest_paths = [
        row["tensor_path"]
        for name in split_names
        for row in manifest[name]
    ]
    duplicates = [
        path for path, count in Counter(manifest_paths).items() if count > 1
    ]
    if duplicates:
        raise ValueError(
            f"Saved split manifest contains duplicate paths: {duplicates[:5]}"
        )
    unexpected = sorted(set(index_by_path) - set(manifest_paths))
    if unexpected:
        raise ValueError(
            f"{len(unexpected)} indexed tensors are absent from the saved "
            f"manifest; first: {unexpected[:5]}"
        )

    subsets = []
    coverage = {}
    for name in split_names:
        requested = [row["tensor_path"] for row in manifest[name]]
        missing = [path for path in requested if path not in index_by_path]
        if require_all and missing:
            raise ValueError(
                f"Saved {name} split is missing {len(missing)} tensors; "
                f"first: {missing[:5]}"
            )
        indices = [index_by_path[path] for path in requested if path in index_by_path]
        subsets.append(Subset(dataset, indices))
        coverage[name] = {
            "requested": len(requested),
            "selected": len(indices),
            "missing": len(missing),
        }
    return (*subsets, coverage)


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


def _natural_key(value: str) -> tuple:
    return tuple(
        (1, int(part)) if part.isdigit() else (0, part.lower())
        for part in re.split(r"(\d+)", value)
    )


def _archive_name(dataset, index: int) -> str:
    return Path(dataset.paths[index]).parent.name


def _available_archives(dataset, family: str, indices: list[int]) -> list[str]:
    root = getattr(dataset, "root", None)
    family_dir = Path(root) / family if root is not None else None
    if family_dir is not None and family_dir.is_dir():
        names = [
            path.name
            for path in family_dir.iterdir()
            if path.is_dir()
        ]
    else:
        names = [
            _archive_name(dataset, index)
            for index in indices
            if dataset.type_of(index) == family
        ]
    return sorted(set(names), key=_natural_key)


def limit_subset_archives(
    dataset,
    subset,
    archives_per_family: int,
    *,
    strict: bool = True,
):
    """Keep a deterministic prefix of archive cohorts in each family."""
    if archives_per_family < 1:
        raise ValueError("archives_per_family must be positive")
    indices = list(subset.indices)
    families = sorted({dataset.type_of(index) for index in indices})

    selected: dict[str, list[str]] = {}
    for family in families:
        ordered = _available_archives(dataset, family, indices)
        if strict and len(ordered) < archives_per_family:
            raise ValueError(
                f"{family} has {len(ordered)} archive cohort(s), but "
                f"{archives_per_family} are required"
            )
        selected[family] = ordered[:archives_per_family]

    selected_sets = {family: set(names) for family, names in selected.items()}
    kept = [
        index
        for index in indices
        if _archive_name(dataset, index) in selected_sets[dataset.type_of(index)]
    ]
    return Subset(dataset, sorted(kept)), selected


def nested_group_train_subset(
    dataset,
    train_subset,
    scale: float,
    *,
    baseline_archives_per_family: int,
    seed: int = 42,
    strict: bool = True,
):
    """
    Select a nested family-stratified training subset without splitting groups.

    Fractions up to 1.0 are deterministic group prefixes of the baseline archive
    cohorts. Scales above 1.0 expand the archive prefix and retain all groups.
    """
    if scale <= 0:
        raise ValueError("train scale must be positive")
    required_archives = (
        baseline_archives_per_family
        if scale <= 1
        else math.ceil(baseline_archives_per_family * scale)
    )
    pool, selected_archives = limit_subset_archives(
        dataset,
        train_subset,
        required_archives,
        strict=strict,
    )

    groups: dict[str, dict[str, list[int]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for index in pool.indices:
        family = dataset.type_of(index)
        group_id = str(
            dataset.metadata_of(index).get("group_id")
            or dataset.paths[index].stem
        )
        groups[family][group_id].append(index)

    selected_indices = []
    family_report = {}
    for family in sorted(groups):
        family_groups = groups[family]
        ranked = sorted(
            family_groups,
            key=lambda group_id: hashlib.sha256(
                f"{seed}\0{family}\0{group_id}".encode()
            ).hexdigest(),
        )
        if scale < 1:
            selected_count = max(1, math.ceil(len(ranked) * scale))
            selected_group_ids = ranked[:selected_count]
        else:
            selected_group_ids = ranked
        family_indices = sorted(
            index
            for group_id in selected_group_ids
            for index in family_groups[group_id]
        )
        selected_indices.extend(family_indices)
        family_report[family] = {
            "archive_cohorts": selected_archives[family],
            "available_groups": len(ranked),
            "selected_groups": len(selected_group_ids),
            "available_samples": sum(map(len, family_groups.values())),
            "selected_samples": len(family_indices),
        }

    report = {
        "scale": scale,
        "baseline_archives_per_family": baseline_archives_per_family,
        "required_archives_per_family": required_archives,
        "subset_seed": seed,
        "families": family_report,
    }
    return Subset(dataset, sorted(selected_indices)), report
