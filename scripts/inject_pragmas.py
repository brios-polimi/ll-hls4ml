#!/usr/bin/env python3
"""Inject HLS pragma nodes into one ProGraML graph JSON file."""

from __future__ import annotations

import argparse
import json

from ll_hls4ml.pragmas import inject_pragmas


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("project_path")
    parser.add_argument("ll_path")
    parser.add_argument("json_path")
    args = parser.parse_args()
    print(json.dumps(inject_pragmas(args.project_path, args.ll_path, args.json_path), indent=2))


if __name__ == "__main__":
    main()
