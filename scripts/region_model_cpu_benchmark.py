#!/usr/bin/env python3
"""Build one real schema-v3 sample in memory and benchmark model ablations."""

from __future__ import annotations

import argparse
import copy
from pathlib import Path
import statistics
import sys
import time

import orjson
import torch
from torch_geometric.data import Batch


REPO_ROOT = Path(__file__).resolve().parents[1]
PIPELINE_SRC = REPO_ROOT.parent / "hls4ml_pipeline" / "src"
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(PIPELINE_SRC))

from hls4ml_pipeline.pipeline.loops import add_llvm_loop_hierarchy  # noqa: E402
from ll_hls4ml.data.tensorize import _json_to_hetero  # noqa: E402
from ll_hls4ml.data.vocab import load_vocab  # noqa: E402
from ll_hls4ml.models.registry import build  # noqa: E402


DEFAULT_ID = "b0051e09-ffb4-48e1-8991-9af53f5ee0e8"


def _region_graph(graph: dict, llvm_path: Path) -> tuple[dict, dict]:
    enriched = copy.deepcopy(graph)
    nodes = enriched["nodes"]
    links = enriched["links"]
    block_lookup = {
        (int(node["function"]), str(node["features"]["name"][0])): node
        for node in nodes
        if int(node.get("type", -1)) == 4
    }
    function_lookup = {
        int(node["function"]): node
        for node in nodes
        if int(node.get("type", -1)) == 5
    }
    source_labels = {
        str(node.get("features", {}).get("source_loop_label", [""])[0])
        for node in nodes
        if int(node.get("type", -1)) == 3
        and node.get("features", {}).get("source_loop_label", [""])[0]
    }
    loop_lookup, stats = add_llvm_loop_hierarchy(
        nodes,
        links,
        llvm_path,
        block_lookup,
        function_lookup,
        source_labels,
    )
    for link in links:
        if int(link.get("flow", -1)) == 6:
            link["relation"] = "contains"
            link.pop("flow", None)
            link.pop("key", None)
    for node in nodes:
        if int(node.get("type", -1)) != 3:
            continue
        features = node.get("features", {})
        label = str(features.get("source_loop_label", [""])[0])
        loop = loop_lookup.get((int(node.get("function", -1)), label))
        if loop is not None:
            links.append(
                {
                    "relation": "applies_to",
                    "position": 0,
                    "source": int(node["id"]),
                    "target": int(loop["id"]),
                }
            )
        if "attachment_schema_version" in features:
            features["attachment_schema_version"] = ["5"]
    enriched["hierarchy_enrichment"].update(stats)
    enriched["pragma_injection"]["attachment_schema_version"] = 5
    return enriched, stats


def _parameters(model) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def _benchmark(model, data, warmup: int, iterations: int) -> float:
    model.eval()
    with torch.inference_mode():
        for _ in range(warmup):
            model(data)
        timings = []
        for _ in range(iterations):
            started = time.perf_counter()
            model(data)
            timings.append(1000 * (time.perf_counter() - started))
    return statistics.median(timings)


def _table(headers, rows) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(map(str, row)) + " |" for row in rows)
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--graph",
        type=Path,
        default=Path(f"../data/graphs/3layer/archive_6/{DEFAULT_ID}.json"),
    )
    parser.add_argument(
        "--llvm",
        type=Path,
        default=Path(f"../data/ll/3layer/archive_6/{DEFAULT_ID}.ll"),
    )
    parser.add_argument(
        "--vocab", type=Path, default=Path("artifacts/vocab/vocab.json")
    )
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    torch.set_num_threads(args.threads)
    graph = orjson.loads(args.graph.read_bytes())
    region_graph, stats = _region_graph(graph, args.llvm)
    vocab, max_pos, _ = load_vocab(args.vocab)
    h0_data, _ = _json_to_hetero(graph, vocab, max_pos, inference_mode=True)
    region_data, _ = _json_to_hetero(
        region_graph, vocab, max_pos, inference_mode=True
    )
    h0_batch = Batch.from_data_list([h0_data])
    region_batch = Batch.from_data_list([region_data])
    common = {
        "instruction_vocab_size": len(vocab),
        "edge_pos_vocab_size": max_pos,
        "y_means": torch.zeros(6),
        "y_stds": torch.ones(6),
        "hidden_dim": 64,
        "num_layers": 3,
        "dropout": 0.15,
        "use_global_features": True,
        "use_context": True,
        "split_heads": True,
        "hurdle_heads": True,
    }
    specifications = [
        ("H0", "hierarchical", h0_batch, {}),
        (
            "region / mean messages / generic",
            "hierarchical_region",
            region_batch,
            {"cardinality_messages": False, "composition": "generic"},
        ),
        (
            "region / cardinality / generic",
            "hierarchical_region",
            region_batch,
            {"cardinality_messages": True, "composition": "generic"},
        ),
        (
            "region / mean messages / hardware",
            "hierarchical_region",
            region_batch,
            {
                "cardinality_messages": False,
                "composition": "hardware_aligned",
            },
        ),
        (
            "region / cardinality / hardware",
            "hierarchical_region",
            region_batch,
            {
                "cardinality_messages": True,
                "composition": "hardware_aligned",
            },
        ),
    ]
    rows = []
    for label, name, data, options in specifications:
        model = build(name, **common, **options)
        elapsed = _benchmark(model, data, args.warmup, args.iterations)
        rows.append((label, f"{_parameters(model):,}", f"{elapsed:.1f}"))

    report = [
        "# Region model CPU benchmark",
        "",
        f"One real `{args.graph.parent.parent.name}` validation graph, one graph "
        f"per batch, {args.threads} Torch CPU thread(s), median of "
        f"{args.iterations} timed forwards after {args.warmup} warmups.",
        "",
        f"Tensor sizes: {h0_data['instruction'].num_nodes:,} instructions, "
        f"{h0_data['block'].num_nodes:,} blocks, "
        f"{stats['loop_nodes_injected']:,} loops.",
        "",
        _table(("model", "parameters", "median ms"), rows),
        "",
        "This is a feasibility/accounting benchmark, not an accuracy result.",
        "",
    ]
    text = "\n".join(report)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    print(text)


if __name__ == "__main__":
    main()
