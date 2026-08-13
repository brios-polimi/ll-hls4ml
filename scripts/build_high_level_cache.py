#!/usr/bin/env python3
"""Build an archives-prefix high-level feature cache from local wa-hls4ml data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))

from ll_hls4ml.data.high_level import build_high_level_cache


DEFAULT_FAMILIES = (
    "2layer",
    "3layer",
    "conv1d",
    "conv2d",
    "dense_latency",
    "dense_resource",
    "rule4ml",
)


def _archive_number(tensor_path: str) -> int:
    archive = Path(tensor_path).parts[1]
    prefix = "archive_"
    if not archive.startswith(prefix):
        raise ValueError(f"Unexpected archive directory in {tensor_path!r}")
    return int(archive.removeprefix(prefix))


def _unique_by_graph_id(paths) -> tuple[list[str], int]:
    """Match HeteroGraphDataset's deterministic global graph-ID deduplication."""
    selected = []
    seen = set()
    duplicate_count = 0
    for path in sorted(paths):
        graph_id = Path(path).stem
        if graph_id in seen:
            duplicate_count += 1
            continue
        selected.append(path)
        seen.add(graph_id)
    return selected, duplicate_count


def build_manifest(
    tensor_root: Path,
    archives: int,
    families: tuple[str, ...],
    exemplar_archives: int | None = None,
) -> tuple[dict, dict]:
    exemplar_archives = exemplar_archives or archives
    index_path = tensor_root / "labels.json"
    if not index_path.is_file():
        raise FileNotFoundError(f"Tensor index not found: {index_path}")
    index = json.loads(index_path.read_text())
    labels = index["labels"]
    metadata = index.get("metadata", {})

    main_paths, main_duplicates = _unique_by_graph_id(
        path
        for path in labels
        if Path(path).parts[0] in families
        and _archive_number(path) <= archives
    )
    exemplar_paths, exemplar_duplicates = _unique_by_graph_id(
        path
        for path in labels
        if Path(path).parts[0] == "exemplar"
        and _archive_number(path) <= exemplar_archives
    )
    main_ids = {Path(path).stem for path in main_paths}
    exemplar_ids = {Path(path).stem for path in exemplar_paths}
    collisions = sorted(main_ids & exemplar_ids)
    if collisions:
        raise ValueError(
            "Graph IDs collide between benchmark and exemplar data; first: "
            f"{collisions[:5]}"
        )

    manifest = {
        name: [] for name in ("train", "validation", "test")
    }
    group_splits: dict[tuple[str, str], str] = {}
    for path in main_paths:
        sample_metadata = metadata.get(path, {})
        split = str(sample_metadata.get("dataset_split", "")).lower()
        split = "validation" if split in {"val", "validation"} else split
        if split not in manifest:
            raise ValueError(f"Missing official train/val/test split for {path}")
        family = Path(path).parts[0]
        group_id = str(sample_metadata.get("group_id") or Path(path).stem)
        group_key = (family, group_id)
        previous_split = group_splits.setdefault(group_key, split)
        if previous_split != split:
            raise ValueError(
                f"Group {group_key!r} crosses {previous_split} and {split}"
            )
        manifest[split].append(
            {"kernel_family": family, "tensor_path": path}
        )

    manifest["exemplar"] = [
        {"kernel_family": "exemplar", "tensor_path": path}
        for path in exemplar_paths
    ]
    selected_paths = [
        row["tensor_path"]
        for rows in manifest.values()
        for row in rows
    ]
    missing = [
        path for path in selected_paths if not (tensor_root / path).is_file()
    ]
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} indexed tensors are absent; first: {missing[:5]}"
        )

    report = {
        "archives": archives,
        "exemplar_archives": exemplar_archives,
        "sizes": {name: len(rows) for name, rows in manifest.items()},
        "main_duplicate_graph_ids_removed": main_duplicates,
        "exemplar_duplicate_graph_ids_removed": exemplar_duplicates,
        "groups": len(group_splits),
    }
    return manifest, report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tensor-root", type=Path, default=Path("../data/tensors"))
    parser.add_argument(
        "--label-root",
        type=Path,
        default=Path("../data/labels/wa-hls4ml"),
    )
    parser.add_argument(
        "--wa-gnn-dir",
        type=Path,
        default=Path("../wa_hls4ml_models/GNN"),
    )
    parser.add_argument("--archives", type=int, default=8)
    parser.add_argument(
        "--exemplar-archives",
        type=int,
        help="Optional independent exemplar archive limit",
    )
    parser.add_argument("--families", nargs="+", default=list(DEFAULT_FAMILIES))
    parser.add_argument(
        "--cache",
        type=Path,
        default=Path("artifacts/cache/wa_high_level_archives1_8.pt"),
    )
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.archives < 1:
        parser.error("--archives must be positive")
    if args.exemplar_archives is not None and args.exemplar_archives < 1:
        parser.error("--exemplar-archives must be positive")
    for name in ("tensor_root", "label_root", "wa_gnn_dir", "cache"):
        setattr(args, name, getattr(args, name).resolve())
    manifest_path = (
        args.manifest.resolve()
        if args.manifest is not None
        else args.cache.with_name(f"{args.cache.stem}_manifest.json")
    )
    if not args.overwrite:
        existing = [path for path in (args.cache, manifest_path) if path.exists()]
        if existing:
            raise FileExistsError(
                f"Refusing to overwrite {existing}; pass --overwrite if intentional"
            )

    manifest, report = build_manifest(
        args.tensor_root,
        args.archives,
        tuple(args.families),
        args.exemplar_archives,
    )
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"Saved manifest to {manifest_path}")
    print(json.dumps(report, indent=2))

    cache = build_high_level_cache(
        manifest_path=manifest_path,
        label_root=args.label_root,
        tensor_root=args.tensor_root,
        wa_gnn_dir=args.wa_gnn_dir,
        cache_path=args.cache,
    )
    expected = sum(report["sizes"].values())
    actual = len(cache["samples"])
    if actual != expected:
        raise RuntimeError(f"Cache has {actual} samples; expected {expected}")
    print(f"Cache ready: {args.cache}")


if __name__ == "__main__":
    main()
