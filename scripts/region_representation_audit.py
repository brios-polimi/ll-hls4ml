"""Audit the semantic coverage of schema-v3 natural-loop tensors.

This is deliberately a read-only, no-GPU diagnostic.  It operates on a saved
split manifest so that conclusions refer to the exact cohort used by a run.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import torch

from ll_hls4ml.io.schema import PRAGMA_VOCAB


LOOP_SCOPE = ("pragma", "applies_to", "loop")
PRAGMA_SCOPES = (
    ("pragma", "applies_to", "instruction"),
    ("pragma", "applies_to", "variable"),
    ("pragma", "applies_to", "constant"),
    ("pragma", "applies_to", "block"),
    ("pragma", "applies_to", "function"),
    LOOP_SCOPE,
)
DIRECTIVE_NAMES = {value: key.removeprefix("pragma.") for key, value in PRAGMA_VOCAB.items()}
NUMERIC_PRAGMA_ARGUMENTS = frozenset({
    "ii", "factor", "dim", "depth", "min", "max", "avg", "latency",
    "interval", "num", "max_read_burst_length", "max_write_burst_length",
    "num_read_outstanding", "num_write_outstanding", "max_widen_bitwidth", "limit",
})
NUMBER_RE = re.compile(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$")


def _edge(data, edge_type: tuple[str, str, str]) -> torch.Tensor:
    return data[edge_type].edge_index if edge_type in data.edge_types else torch.empty((2, 0), dtype=torch.long)


def _new_stats() -> dict:
    return {
        "samples": 0,
        "loops": 0,
        "loops_with_known_trip_count": 0,
        "loops_with_source_anchor": 0,
        "samples_with_loops": 0,
        "samples_with_loop_pragma": 0,
        "loop_pragma_edges": 0,
        "loop_scoped_pragma_nodes": 0,
        "all_pragma_nodes": 0,
        "pragma_nodes_with_any_scope": 0,
        "loop_nodes_with_pragma": 0,
        "loop_block_edges": 0,
        "loop_child_edges": 0,
        "top_level_loop_edges": 0,
        "memory_like_variables": 0,
        "samples_with_memory_like_variable": 0,
        "pragma_dump_records": 0,
        "numeric_arguments_resolved": 0,
        "injected_resolution_provenance": 0,
        "injected_unresolved_numeric_arguments": 0,
        "unmatched_symbolic_numeric_arguments": 0,
        "samples_with_unmatched_symbolic_numeric": 0,
        "directives": Counter(),
        "loop_directives": Counter(),
        "loop_directive_argument_variants": defaultdict(set),
        "families": defaultdict(_new_stats),
    }


def _record(stats: dict, data) -> None:
    stats["samples"] += 1
    loops = data["loop"].x
    loop_count = int(loops.size(0))
    stats["loops"] += loop_count
    stats["samples_with_loops"] += bool(loop_count)
    if loop_count:
        # loop_embedding: depth, block count, latches, exits, trip count,
        # trip-known mask, source-anchor mask.
        stats["loops_with_known_trip_count"] += int(loops[:, 5].bool().sum())
        stats["loops_with_source_anchor"] += int(loops[:, 6].bool().sum())

    pragma = data["pragma"].x
    pragma_count = int(pragma.size(0))
    stats["all_pragma_nodes"] += pragma_count
    directive_ids = pragma[:, 0].long().tolist() if pragma_count else []
    stats["directives"].update(DIRECTIVE_NAMES.get(i, f"unknown_{i}") for i in directive_ids)

    scoped_pragma_ids: set[int] = set()
    for scope in PRAGMA_SCOPES:
        scoped_pragma_ids.update(_edge(data, scope)[0].tolist())
    stats["pragma_nodes_with_any_scope"] += len(scoped_pragma_ids)

    loop_scope = _edge(data, LOOP_SCOPE)
    loop_pragma_ids = loop_scope[0].tolist()
    stats["loop_pragma_edges"] += int(loop_scope.size(1))
    stats["loop_scoped_pragma_nodes"] += len(set(loop_pragma_ids))
    stats["loop_nodes_with_pragma"] += len(set(loop_scope[1].tolist()))
    stats["samples_with_loop_pragma"] += bool(loop_scope.size(1))
    for pragma_id in sorted(set(loop_pragma_ids)):
        name = DIRECTIVE_NAMES.get(directive_ids[pragma_id], f"unknown_{directive_ids[pragma_id]}")
        stats["loop_directives"][name] += 1
        # A compact measure of whether a directive has corpus variation beyond
        # its identity. The tensors already store normalized/masked arguments.
        stats["loop_directive_argument_variants"][name].add(
            tuple(round(float(v), 6) for v in pragma[pragma_id, 1:].tolist())
        )

    stats["loop_block_edges"] += int(_edge(data, ("loop", "contains", "block")).size(1))
    stats["loop_child_edges"] += int(_edge(data, ("loop", "contains", "loop")).size(1))
    stats["top_level_loop_edges"] += int(_edge(data, ("function", "contains", "loop")).size(1))

    # This reproduces the existing reporting proxy. It is not a memory-node
    # representation: schema-v3 has no memory node type.
    memory_like = data["variable"].x[:, 5:10].bool().any(dim=-1)
    memory_count = int(memory_like.sum())
    stats["memory_like_variables"] += memory_count
    stats["samples_with_memory_like_variable"] += bool(memory_count)


def _record_graph(stats: dict, graph: dict) -> None:
    """Record the same coverage directly from graph JSON, avoiding .pt I/O."""

    stats["samples"] += 1
    nodes = {int(node["id"]): node for node in graph["nodes"]}
    loops = [node for node in nodes.values() if int(node["type"]) == 6]
    pragmas = [node for node in nodes.values() if int(node["type"]) == 3]
    stats["loops"] += len(loops)
    stats["samples_with_loops"] += bool(loops)
    for loop in loops:
        features = loop.get("features", {})
        known = features.get("trip_count_known", ["false"])
        labels = features.get("source_loop_labels", [""])
        stats["loops_with_known_trip_count"] += str(known[0]).lower() == "true"
        stats["loops_with_source_anchor"] += bool(labels[0])

    pragma_by_id = {int(node["id"]): node for node in pragmas}
    stats["all_pragma_nodes"] += len(pragmas)
    stats["directives"].update(node["text"].removeprefix("pragma.") for node in pragmas)
    for pragma in pragmas:
        features = pragma.get("features", {})
        stats["injected_resolution_provenance"] += "numeric_resolution_json" in features
        arguments = json.loads(features.get("arguments_json", ["{}"])[0])
        for key, values in arguments.items():
            if key in NUMERIC_PRAGMA_ARGUMENTS:
                stats["injected_unresolved_numeric_arguments"] += sum(
                    not NUMBER_RE.fullmatch(str(value).strip()) for value in values
                )
    injection = graph.get("pragma_injection", {})
    stats["pragma_dump_records"] += int(injection.get("pragma_dump_records", 0))
    stats["numeric_arguments_resolved"] += int(injection.get("numeric_arguments_resolved", 0))
    unmatched_symbolic = 0
    for record in injection.get("unmatched_records", []):
        for key, values in record.get("arguments", {}).items():
            if key in NUMERIC_PRAGMA_ARGUMENTS:
                unmatched_symbolic += sum(
                    not NUMBER_RE.fullmatch(str(value).strip()) for value in values
                )
    stats["unmatched_symbolic_numeric_arguments"] += unmatched_symbolic
    stats["samples_with_unmatched_symbolic_numeric"] += bool(unmatched_symbolic)
    scoped_ids: set[int] = set()
    loop_pragma_ids: set[int] = set()
    loop_ids: set[int] = set()
    for link in graph["links"]:
        source, target = int(link["source"]), int(link["target"])
        if source not in pragma_by_id or str(link.get("relation")) != "applies_to":
            continue
        scoped_ids.add(source)
        if int(nodes[target]["type"]) == 6:
            loop_pragma_ids.add(source)
            loop_ids.add(target)
            stats["loop_pragma_edges"] += 1
    stats["pragma_nodes_with_any_scope"] += len(scoped_ids)
    stats["loop_scoped_pragma_nodes"] += len(loop_pragma_ids)
    stats["loop_nodes_with_pragma"] += len(loop_ids)
    stats["samples_with_loop_pragma"] += bool(loop_ids)
    for pragma_id in loop_pragma_ids:
        pragma = pragma_by_id[pragma_id]
        name = pragma["text"].removeprefix("pragma.")
        stats["loop_directives"][name] += 1
        stats["loop_directive_argument_variants"][name].add(
            tuple(pragma.get("features", {}).get("arguments_json", []))
        )
    for link in graph["links"]:
        source, target = int(link["source"]), int(link["target"])
        pair = (int(nodes[source]["type"]), int(nodes[target]["type"]))
        if str(link.get("relation")) == "contains" and pair == (6, 4):
            stats["loop_block_edges"] += 1
        elif str(link.get("relation")) == "contains" and pair == (6, 6):
            stats["loop_child_edges"] += 1
        elif str(link.get("relation")) == "contains" and pair == (5, 6):
            stats["top_level_loop_edges"] += 1


def _finalize(stats: dict) -> dict:
    samples = max(stats["samples"], 1)
    loops = max(stats["loops"], 1)
    return {
        "samples": stats["samples"],
        "loops": stats["loops"],
        "mean_loops_per_sample": stats["loops"] / samples,
        "loop_trip_count_known": {
            "count": stats["loops_with_known_trip_count"],
            "fraction_of_loops": stats["loops_with_known_trip_count"] / loops,
        },
        "source_anchored_loops": {
            "count": stats["loops_with_source_anchor"],
            "fraction_of_loops": stats["loops_with_source_anchor"] / loops,
        },
        "loop_pragma_coverage": {
            "samples_with_loop_pragma": stats["samples_with_loop_pragma"],
            "fraction_of_samples": stats["samples_with_loop_pragma"] / samples,
            "loop_nodes_with_pragma": stats["loop_nodes_with_pragma"],
            "fraction_of_loops": stats["loop_nodes_with_pragma"] / loops,
            "loop_scoped_pragma_nodes": stats["loop_scoped_pragma_nodes"],
            "loop_pragma_edges": stats["loop_pragma_edges"],
        },
        "pragma_scope_coverage": {
            "all_pragma_nodes": stats["all_pragma_nodes"],
            "nodes_with_any_semantic_scope": stats["pragma_nodes_with_any_scope"],
            "fraction_scoped": stats["pragma_nodes_with_any_scope"] / max(stats["all_pragma_nodes"], 1),
        },
        "containment_edges": {
            "loop_to_direct_block": stats["loop_block_edges"],
            "loop_to_child_loop": stats["loop_child_edges"],
            "function_to_top_level_loop": stats["top_level_loop_edges"],
        },
        "memory_proxy": {
            "memory_node_type_exists": False,
            "memory_like_variables": stats["memory_like_variables"],
            "samples_with_memory_like_variable": stats["samples_with_memory_like_variable"],
        },
        "symbolic_pragma_resolution": {
            "pragma_dump_records": stats["pragma_dump_records"],
            "resolved_numeric_arguments": stats["numeric_arguments_resolved"],
            "injected_nodes_with_resolution_provenance": stats["injected_resolution_provenance"],
            "injected_unresolved_numeric_arguments": stats["injected_unresolved_numeric_arguments"],
            "unmatched_symbolic_numeric_arguments": stats["unmatched_symbolic_numeric_arguments"],
            "samples_with_unmatched_symbolic_numeric": stats["samples_with_unmatched_symbolic_numeric"],
        },
        "directives": dict(stats["directives"].most_common()),
        "loop_scoped_directives": {
            name: {
                "pragma_nodes": count,
                "argument_variants": len(stats["loop_directive_argument_variants"][name]),
            }
            for name, count in stats["loop_directives"].most_common()
        },
    }


def _markdown(audit: dict) -> str:
    lines = [
        "# Region representation audit",
        "",
        "Read-only audit of tensors/graphs selected from the schema-v3 cohort named in the split manifest.",
        "",
        f"Selection: {audit['selection']}.",
        "",
        "## Core coverage",
        "",
        "| split | samples | loops | loops/sample | known trip counts | source-anchored loops | samples with loop pragma | loops with loop pragma |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for split, item in audit["splits"].items():
        trip = item["loop_trip_count_known"]
        anchor = item["source_anchored_loops"]
        pragma = item["loop_pragma_coverage"]
        lines.append(
            f"| {split} | {item['samples']} | {item['loops']} | {item['mean_loops_per_sample']:.2f} | "
            f"{trip['count']} ({trip['fraction_of_loops']:.1%}) | "
            f"{anchor['count']} ({anchor['fraction_of_loops']:.1%}) | "
            f"{pragma['samples_with_loop_pragma']} ({pragma['fraction_of_samples']:.1%}) | "
            f"{pragma['loop_nodes_with_pragma']} ({pragma['fraction_of_loops']:.1%}) |"
        )
    lines += [
        "",
        "## Interpretation",
        "",
        "`trip_count_known` is the tensor feature consumed by the region model. A zero count means the current experiment did not test trip-count-conditioned composition.",
        "The schema has no memory node type; `memory_like_variables` is only the existing reporting proxy, not a memory representation.",
        "",
        "## Symbolic pragma values",
        "",
        "| split | resolved numeric arguments | injected unresolved numeric arguments | unmatched symbolic numeric arguments | graphs with unmatched symbolic values |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for split, item in audit["splits"].items():
        resolution = item["symbolic_pragma_resolution"]
        lines.append(
            f"| {split} | {resolution['resolved_numeric_arguments']} | "
            f"{resolution['injected_unresolved_numeric_arguments']} | "
            f"{resolution['unmatched_symbolic_numeric_arguments']} | "
            f"{resolution['samples_with_unmatched_symbolic_numeric']} |"
        )
    lines += [
        "",
        "Injected unresolved numeric arguments would make tensorization fail; unmatched values are dump records whose source function was not represented in the final LLVM graph.",
        "",
        "## Loop-scoped directives (all splits)",
        "",
        "| directive | loop-scoped pragma nodes | distinct stored argument vectors |",
        "| --- | ---: | ---: |",
    ]
    all_directives = audit["all"]["loop_scoped_directives"]
    for name, item in all_directives.items():
        lines.append(f"| {name} | {item['pragma_nodes']} | {item['argument_variants']} |")
    lines += ["", "## Family detail (all splits)", "", "| family | samples | loops | known trip counts | loops with loop pragma |", "| --- | ---: | ---: | ---: | ---: |"]
    for family, item in sorted(audit["families"].items()):
        lines.append(
            f"| {family} | {item['samples']} | {item['loops']} | "
            f"{item['loop_trip_count_known']['count']} | {item['loop_pragma_coverage']['loop_nodes_with_pragma']} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--tensor-dir", type=Path)
    source.add_argument("--graph-dir", type=Path)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="Markdown output path; JSON is written beside it.")
    parser.add_argument(
        "--max-records-per-split", type=int,
        help="Optional smoke-test limit; omit for the full manifest audit.",
    )
    parser.add_argument(
        "--records-per-family",
        type=int,
        help="Optional deterministic stratified limit within each split/family.",
    )
    args = parser.parse_args()

    manifest = json.loads(args.split_manifest.read_text())
    split_stats = {split: _new_stats() for split in manifest}
    all_stats = _new_stats()
    for split, records in manifest.items():
        if args.max_records_per_split is not None:
            records = records[:args.max_records_per_split]
        if args.records_per_family is not None:
            selected = []
            per_family = Counter()
            for record in records:
                family = record["kernel_family"]
                if per_family[family] < args.records_per_family:
                    selected.append(record)
                    per_family[family] += 1
            records = selected
        for record in records:
            if args.tensor_dir:
                tensor_path = args.tensor_dir / record["tensor_path"]
                data = torch.load(tensor_path, map_location="cpu", weights_only=False)
                if int(data.hierarchy_schema_version) != 3:
                    raise ValueError(f"Expected schema 3: {tensor_path}")
                record_fn = lambda stats: _record(stats, data)
            else:
                graph_path = (args.graph_dir / record["tensor_path"]).with_suffix(".json")
                graph = json.loads(graph_path.read_text())
                record_fn = lambda stats: _record_graph(stats, graph)
            record_fn(split_stats[split])
            record_fn(all_stats)
            record_fn(all_stats["families"][record["kernel_family"]])

    audit = {
        "source": str(args.tensor_dir or args.graph_dir),
        "split_manifest": str(args.split_manifest),
        "selection": (
            f"first {args.records_per_family} records per split/family"
            if args.records_per_family is not None
            else (
                f"first {args.max_records_per_split} records per split"
                if args.max_records_per_split is not None
                else "full manifest"
            )
        ),
        "splits": {split: _finalize(stats) for split, stats in split_stats.items()},
        "all": _finalize(all_stats),
        "families": {family: _finalize(stats) for family, stats in all_stats["families"].items()},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(_markdown(audit))
    args.output.with_suffix(".json").write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
