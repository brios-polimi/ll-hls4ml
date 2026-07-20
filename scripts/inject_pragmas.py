#!/usr/bin/env python3
"""Inject Vitis compiler pragma nodes into one ProGraML graph JSON file."""

from __future__ import annotations

import argparse
import json

from ll_hls4ml.pragmas import inject_vitis_pragmas


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("graph_path")
    parser.add_argument("pragma_dump_path")
    args = parser.parse_args()
    print(json.dumps(inject_vitis_pragmas(args.graph_path, args.pragma_dump_path), indent=2))


if __name__ == "__main__":
    main()
