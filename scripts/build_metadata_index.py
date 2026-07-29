#!/usr/bin/env python3
"""Build a small synthesis-metadata overlay for the graphs present locally."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))

from ll_hls4ml.io.discovery import iter_graph_paths


def iter_json_array(path: Path, chunk_size: int = 1 << 20):
    """Stream objects from a top-level JSON array using only the stdlib."""
    decoder = json.JSONDecoder()
    with path.open() as handle:
        buffer = ""
        position = 0
        started = False
        eof = False
        while True:
            if position >= len(buffer) and not eof:
                buffer = handle.read(chunk_size)
                position = 0
                eof = not buffer
            while position < len(buffer) and (
                buffer[position].isspace() or buffer[position] == ","
            ):
                position += 1
            if not started:
                if position >= len(buffer):
                    if eof:
                        return
                    continue
                if buffer[position] != "[":
                    raise ValueError(f"{path} is not a top-level JSON array")
                position += 1
                started = True
                continue
            while position < len(buffer) and buffer[position].isspace():
                position += 1
            if position < len(buffer) and buffer[position] == "]":
                return
            try:
                value, end = decoder.raw_decode(buffer, position)
            except json.JSONDecodeError:
                if eof:
                    raise
                chunk = handle.read(chunk_size)
                buffer = buffer[position:] + chunk
                position = 0
                eof = not chunk
                continue
            yield value
            position = end
            if position > chunk_size:
                buffer = buffer[position:]
                position = 0


def _group_id(row: dict, source_name: str) -> str:
    """Group synthesis variants of the same generated model architecture."""
    model_name = Path(
        str((row.get("meta_data") or {}).get("model_name", ""))
    ).name
    family = source_name.removesuffix(".json")
    for prefix in ("train_", "val_", "test_"):
        family = family.removeprefix(prefix)
    family = family.removesuffix("_merged")
    identity = f"{family}:{model_name}"
    return hashlib.sha1(identity.encode()).hexdigest()[:16]


def _metadata(row: dict, split: str, source_name: str) -> dict:
    latency = row.get("latency_report") or {}
    return {
        "backend": str(row.get("backend") or "vitis"),
        "target_part": str(row.get("target_part") or ""),
        "vivado_version": str(row.get("vivado_version") or ""),
        "hls4ml_version": str(row.get("hls4ml_version") or ""),
        "target_clock": latency.get("target_clock"),
        "dataset_split": split,
        "group_id": _group_id(row, source_name),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph-dir", type=Path, default=Path("../data/graphs"))
    parser.add_argument(
        "--labels-dir",
        type=Path,
        default=Path("../data/labels/wa-hls4ml"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/metadata/graph_metadata.json"),
    )
    parser.add_argument(
        "--tensor-dir",
        type=Path,
        default=None,
        help="Optionally refresh metadata in an existing tensor labels.json",
    )
    parser.add_argument(
        "--reuse-output",
        action="store_true",
        help="Reuse the existing output mapping and only refresh tensor metadata",
    )
    args = parser.parse_args()

    graph_ids = {
        path.stem for _kernel, path in iter_graph_paths(args.graph_dir)
    }
    if args.reuse_output:
        found = json.loads(args.output.read_text())
    else:
        found: dict[str, dict] = {}
        label_paths = sorted(args.labels_dir.glob("*/*.json"))
        for path in label_paths:
            split = path.parent.name
            matched_before = len(found)
            for row in iter_json_array(path):
                if not isinstance(row, dict):
                    continue
                artifact = str(
                    (row.get("meta_data") or {}).get("artifacts_file", "")
                )
                graph_id = artifact.removesuffix(".tar.gz")
                if graph_id in graph_ids:
                    found[graph_id] = _metadata(row, split, path.name)
            print(
                f"{path.name}: +{len(found) - matched_before} matches",
                flush=True,
            )
            if len(found) == len(graph_ids):
                break

        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(found, indent=2, sort_keys=True) + "\n"
        )
    if args.tensor_dir is not None:
        tensor_index_path = args.tensor_dir / "labels.json"
        tensor_index = json.loads(tensor_index_path.read_text())
        tensor_index["metadata"] = {
            relative_path: found.get(Path(relative_path).stem, {})
            for relative_path in tensor_index.get("labels", {})
        }
        tensor_index_path.write_text(
            json.dumps(tensor_index, indent=2, sort_keys=True) + "\n"
        )
    missing = len(graph_ids - found.keys())
    print(
        f"Wrote {len(found)} metadata records to {args.output}; "
        f"{missing} local graphs unmatched"
    )


if __name__ == "__main__":
    main()
