#!/usr/bin/env python3
"""Build a content-stable manifest for a tensor dataset."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))

from ll_hls4ml.data.fingerprint import (  # noqa: E402
    build_content_manifest,
    write_content_manifest,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tensor-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    tensor_dir = args.tensor_dir.resolve()
    paths = sorted(tensor_dir.rglob("*.pt"))
    paths.extend(
        path
        for path in (tensor_dir / "labels.json", tensor_dir / "vocab.json")
        if path.is_file()
    )
    if not paths:
        raise ValueError(f"No tensor files found under {tensor_dir}")

    def progress(index: int, total: int, _path: Path) -> None:
        if index == 1 or index == total or index % 250 == 0:
            print(f"Hashed {index}/{total} files", flush=True)

    manifest = build_content_manifest(paths, tensor_dir, progress=progress)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_content_manifest(args.output, manifest)
    print("Tensor snapshot SHA-256:", manifest["snapshot_sha256"])
    print("Wrote:", args.output)


if __name__ == "__main__":
    main()
