#!/usr/bin/env python3
"""Convert CDFG JSON graphs to PyG HeteroData .pt tensors."""

import argparse
import json
from pathlib import Path
import re
import shutil

from ll_hls4ml.config import load_config
from ll_hls4ml.data.tensorize import create_graph_tensors
from ll_hls4ml.data.vocab import load_vocab, save_vocab, vocab_scan


def _parse_archive_spec(spec: str) -> list[str]:
    """Expand archive numbers/ranges to the directory names used on disk."""
    archives = []
    for part in spec.split(","):
        part = part.strip()
        match = re.fullmatch(r"(?:archive_)?(\d+)(?:-(?:archive_)?(\d+))?", part)
        if match:
            start = int(match.group(1))
            end = int(match.group(2) or start)
            if end < start:
                raise ValueError(f"Invalid archive range: {part}")
            archives.extend(f"archive_{number}" for number in range(start, end + 1))
        else:
            archives.append(part)
    return list(dict.fromkeys(archives))


def main():
    parser = argparse.ArgumentParser(description="Build PyG tensor files from CDFG JSON")
    parser.add_argument("--config", default=None)
    parser.add_argument(
        "--kernel",
        nargs="+",
        action="extend",
        default=None,
        help="Kernel types to process (default: all)",
    )
    parser.add_argument(
        "--archive",
        default=None,
        help="Archive number/range (for example, 1-5) or directory name",
    )
    parser.add_argument("--max-archives", type=int, default=None)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--vocab", default=None, help="Vocab JSON path (default: from config)")
    parser.add_argument(
        "--metadata-index",
        default=None,
        help="Optional JSON mapping graph UUIDs to synthesis metadata",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    archive_subset = _parse_archive_spec(args.archive) if args.archive else None
    vocab_path = Path(args.vocab) if args.vocab else cfg.vocab_path

    if vocab_path.exists():
        vocab, max_pos, counts = load_vocab(vocab_path)
        print(f"Loaded vocab from {vocab_path}")
    else:
        print("Vocab not found; scanning graphs...")
        vocab, max_pos, counts = vocab_scan(
            cfg.graph_dir, kernel_subset=args.kernel
        )

    metadata_by_graph_id = None
    if args.metadata_index:
        with Path(args.metadata_index).open() as handle:
            metadata_by_graph_id = json.load(handle)

    create_graph_tensors(
        cfg.graph_dir,
        cfg.tensor_dir,
        vocab,
        max_pos,
        kernel_subset=args.kernel,
        archive_subset=archive_subset,
        max_archives=args.max_archives,
        n_workers=args.workers,
        metadata_by_graph_id=metadata_by_graph_id,
    )
    bundled_vocab = cfg.tensor_dir / "vocab.json"
    if vocab_path.exists():
        if vocab_path.resolve() != bundled_vocab.resolve():
            shutil.copy2(vocab_path, bundled_vocab)
    else:
        save_vocab(vocab, max_pos, bundled_vocab, counts)
    print(f"Tensors and labels index written under {cfg.tensor_dir}")


if __name__ == "__main__":
    main()
