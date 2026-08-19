#!/usr/bin/env python3
"""Measure how conservative LLVM passes change one retained CDFG input.

This is a read-only diagnostic for the durable LLVM/graph artifacts.  Optimized
IR is written only to a temporary directory.  If a matching enriched graph is
provided, the report also checks whether its source-loop block names survive.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import tempfile
from pathlib import Path


PIPELINES = (
    ("raw", None),
    ("mem2reg", "mem2reg"),
    ("sroa+mem2reg", "sroa,mem2reg"),
    ("sroa+mem2reg+instcombine", "sroa,mem2reg,instcombine"),
    (
        "canonical-cfg",
        "sroa,mem2reg,instcombine,simplifycfg,loop-simplify,lcssa",
    ),
    ("O1", "default<O1>"),
)

FUNCTION_RE = re.compile(r"^define ", re.MULTILINE)
BLOCK_RE = re.compile(r"^[-$._A-Za-z0-9]+:\s*(?:;.*)?$", re.MULTILINE)
INSTRUCTION_RE = re.compile(
    r"^\s*(?:%[-$._A-Za-z0-9]+\s*=|call\b|br\b|ret\b|store\b|switch\b|invoke\b)",
    re.MULTILINE,
)
ALLOCA_RE = re.compile(r"\balloca\b")
VITIS_INTRINSIC_RE = re.compile(r"\bvitis\.fpga\.")
SIDE_EFFECT_RE = re.compile(r"\bllvm\.sideeffect\b")


def source_loop_names(graph_path: Path | None) -> set[str]:
    if graph_path is None:
        return set()
    graph = json.loads(graph_path.read_text())
    names = set()
    for node in graph["nodes"]:
        if int(node.get("type", -1)) != 4:
            continue
        features = node.get("features", {})
        loop_values = features.get("is_source_loop", [])
        name_values = features.get("name", [])
        if (
            len(loop_values) == 1
            and str(loop_values[0]).lower() == "true"
            and len(name_values) == 1
        ):
            names.add(str(name_values[0]))
    return names


def ir_stats(text: str, loop_names: set[str]) -> dict[str, int]:
    block_matches = list(BLOCK_RE.finditer(text))
    defined_blocks = {
        match.group(0).split(":", 1)[0] for match in block_matches
    }
    return {
        "bytes": len(text.encode()),
        "functions": len(FUNCTION_RE.findall(text)),
        "blocks": len(block_matches),
        "instructions": len(INSTRUCTION_RE.findall(text)),
        "allocas": len(ALLOCA_RE.findall(text)),
        "vitis_intrinsics": len(VITIS_INTRINSIC_RE.findall(text)),
        "side_effects": len(SIDE_EFFECT_RE.findall(text)),
        "loop_anchors": len(loop_names & defined_blocks),
    }


def table(rows: list[dict[str, object]], anchor_total: int) -> str:
    headers = (
        "pipeline",
        "instructions",
        "vs raw",
        "blocks",
        "functions",
        "allocas",
        "Vitis intrinsics",
        "pragma carriers",
        "loop anchors",
        "KiB",
    )
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    raw_instructions = int(rows[0]["instructions"])
    for row in rows:
        instructions = int(row["instructions"])
        reduction = 100.0 * (1.0 - instructions / raw_instructions)
        anchors = (
            f"{row['loop_anchors']}/{anchor_total}" if anchor_total else "not checked"
        )
        lines.append(
            "| "
            + " | ".join(
                (
                    str(row["pipeline"]),
                    str(instructions),
                    f"{reduction:+.1f}%",
                    str(row["blocks"]),
                    str(row["functions"]),
                    str(row["allocas"]),
                    str(row["vitis_intrinsics"]),
                    str(row["side_effects"]),
                    anchors,
                    f"{int(row['bytes']) / 1024:.1f}",
                )
            )
            + " |"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--graph", type=Path)
    parser.add_argument("--opt-binary", default="opt-16")
    parser.add_argument(
        "--analysis-triple",
        default="x86_64-unknown-linux-gnu",
        help="Temporary triple used because stock opt does not know fpga64.",
    )
    args = parser.parse_args()

    loop_names = source_loop_names(args.graph)
    rows = []
    raw_text = args.input.read_text()
    with tempfile.TemporaryDirectory(prefix="ll-hls4ml-ir-probe-") as directory:
        temp = Path(directory)
        for label, passes in PIPELINES:
            if passes is None:
                text = raw_text
            else:
                output = temp / f"{label}.ll"
                subprocess.run(
                    [
                        args.opt_binary,
                        "-S",
                        f"-mtriple={args.analysis_triple}",
                        f"-passes={passes}",
                        str(args.input),
                        "-o",
                        str(output),
                    ],
                    check=True,
                )
                text = output.read_text()
            rows.append({"pipeline": label, **ir_stats(text, loop_names)})

    print(f"Input: `{args.input}`")
    if args.graph:
        print(f"Loop-anchor source: `{args.graph}` ({len(loop_names)} unique names)")
    print()
    print(table(rows, len(loop_names)))


if __name__ == "__main__":
    main()
