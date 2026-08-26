#!/usr/bin/env python3
"""Extend wa-hls4ml paper Tables 4 and 5 with hls-surrogate-lab runs."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from ll_hls4ml.reporting import generate_wahls4ml_comparison


REPO_ROOT = Path(__file__).resolve().parents[1]


def _default_label_path() -> Path:
    data_root = Path(
        os.environ.get("LL_HLS4ML_DATA_ROOT", REPO_ROOT.parent / "data")
    )
    return data_root / "labels" / "wa-hls4ml" / "exemplar" / "exemplar_models.json"


def _parse_run(value: str) -> tuple[str, Path]:
    if "=" not in value:
        path = Path(value)
        return path.name, path
    name, path = value.split("=", 1)
    if not name.strip() or not path.strip():
        raise argparse.ArgumentTypeError("--run must be NAME=PATH or PATH")
    return name.strip(), Path(path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare run predictions with wa-hls4ml paper Tables 4 and 5"
    )
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        metavar="NAME=PATH",
        help="Run directory containing predictions.csv; repeat for more rows",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--exemplar-labels", type=Path, default=_default_label_path()
    )
    args = parser.parse_args()
    runs = dict(_parse_run(value) for value in args.run)
    if len(runs) != len(args.run):
        parser.error("Each --run display name must be unique")
    manifest = generate_wahls4ml_comparison(
        runs=runs,
        exemplar_labels=args.exemplar_labels,
        output_dir=args.output_dir,
    )
    print(f"Wrote wa-hls4ml comparison to {args.output_dir.resolve()}")
    for model, coverage in manifest["exemplar_coverage"].items():
        selected = sum(row["selected"] for row in coverage.values())
        total = sum(row["paper_total"] for row in coverage.values())
        print(f"  {model}: exemplar UUID coverage {selected}/{total}")


if __name__ == "__main__":
    main()
