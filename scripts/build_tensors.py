#!/usr/bin/env python3
"""Convert CDFG JSON graphs to PyG HeteroData .pt tensors."""

import argparse
from pathlib import Path

from ll_hls4ml.config import load_config
from ll_hls4ml.data.tensorize import create_graph_tensors
from ll_hls4ml.data.vocab import load_vocab, vocab_scan


def main():
    parser = argparse.ArgumentParser(description="Build PyG tensor files from CDFG JSON")
    parser.add_argument("--config", default=None)
    parser.add_argument("--kernel", default=None, help="Single kernel type to process")
    parser.add_argument(
        "--archive",
        default=None,
        help="Single archive directory to process (requires --kernel)",
    )
    parser.add_argument("--max-archives", type=int, default=None)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--vocab", default=None, help="Vocab JSON path (default: from config)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    vocab_path = Path(args.vocab) if args.vocab else cfg.vocab_path

    if vocab_path.exists():
        vocab, _max_pos, _counts = load_vocab(vocab_path)
        print(f"Loaded vocab from {vocab_path}")
    else:
        print("Vocab not found; scanning graphs...")
        vocab, _max_pos, _ = vocab_scan(cfg.graph_dir, kernel_subset=args.kernel)

    create_graph_tensors(
        cfg.graph_dir,
        cfg.tensor_dir,
        vocab,
        kernel_subset=args.kernel,
        archive_subset=args.archive,
        max_archives=args.max_archives,
        n_workers=args.workers,
    )
    print(f"Tensors written under {cfg.tensor_dir}")


if __name__ == "__main__":
    main()
