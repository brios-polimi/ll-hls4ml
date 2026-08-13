"""Small, shared experiment-accounting helpers."""

from __future__ import annotations

from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np


HURDLE_TARGETS = ("dsp", "bram")
RESOURCE_TARGETS = ("lut", "ff", "dsp", "bram")
TIMING_TARGETS = ("cycles_max", "interval_max")
ALL_TARGETS = (*RESOURCE_TARGETS, *TIMING_TARGETS)
STRUCTURE_FEATURES = (
    "instruction_count",
    "block_count",
    "maximum_block_length",
    "loop_block_fraction",
    "block_scc_fraction",
    "call_depth",
    "memory_node_count",
)


def split_sha256(manifest: dict) -> str:
    membership = {
        split: sorted(
            ({
                "kernel_family": row["kernel_family"],
                "tensor_path": row["tensor_path"],
            } for row in rows),
            key=lambda row: (row["kernel_family"], row["tensor_path"]),
        )
        for split, rows in sorted(manifest.items())
    }
    canonical = json.dumps(
        membership, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode()
    return hashlib.sha256(canonical).hexdigest()


def cohort_membership(manifest: dict) -> dict[str, list[dict]]:
    result = {}
    for split, rows in manifest.items():
        counts = Counter()
        for row in rows:
            path = Path(row["tensor_path"])
            archive = next(
                (part for part in path.parts if part.startswith("archive_")),
                "unknown",
            )
            counts[(row["kernel_family"], archive)] += 1
        result[split] = [
            {"kernel_family": family, "archive": archive, "n_samples": count}
            for (family, archive), count in sorted(counts.items())
        ]
    return result


def hurdle_confusion_rows(prediction_rows: list[dict]) -> list[dict]:
    """Compute presence/absence confusion matrices from persisted predictions."""
    groups: dict[tuple[str, str], list[dict]] = {}
    for row in prediction_rows:
        groups.setdefault((row["split"], "all"), []).append(row)
        groups.setdefault((row["split"], row["kernel_family"]), []).append(row)
    output = []
    for (split, cohort), rows in sorted(groups.items()):
        for target in HURDLE_TARGETS:
            tn = fp = fn = tp = 0
            for row in rows:
                truth = float(row[f"target_{target}"]) > 0
                probability = row.get(f"presence_probability_{target}")
                predicted = (
                    float(probability) >= 0.5
                    if probability is not None
                    else float(row[f"prediction_{target}"]) > 0
                )
                if truth and predicted:
                    tp += 1
                elif truth:
                    fn += 1
                elif predicted:
                    fp += 1
                else:
                    tn += 1
            output.append(
                {
                    "split": split,
                    "cohort": cohort,
                    "target": target,
                    "n_samples": len(rows),
                    "tn": tn,
                    "fp": fp,
                    "fn": fn,
                    "tp": tp,
                    "accuracy": (tp + tn) / len(rows),
                    "precision": tp / (tp + fp) if tp + fp else None,
                    "recall": tp / (tp + fn) if tp + fn else None,
                }
            )
    return output


def hurdle_calibration_rows(
    prediction_rows: list[dict], bins: int = 10
) -> list[dict]:
    """Reliability-diagram rows for DSP/BRAM presence probabilities."""

    groups: dict[tuple[str, str], list[dict]] = {}
    for row in prediction_rows:
        groups.setdefault((row["split"], "all"), []).append(row)
        groups.setdefault((row["split"], row["kernel_family"]), []).append(row)
    output = []
    for (split, cohort), rows in sorted(groups.items()):
        for target in HURDLE_TARGETS:
            probability_key = f"presence_probability_{target}"
            if not all(probability_key in row for row in rows):
                continue
            probabilities = np.asarray(
                [
                    float(row[probability_key])
                    for row in rows
                ]
            )
            truth = np.asarray(
                [float(row[f"target_{target}"]) > 0 for row in rows],
                dtype=float,
            )
            bin_ids = np.minimum((probabilities * bins).astype(int), bins - 1)
            ece = 0.0
            records = []
            for bin_index in range(bins):
                selected = bin_ids == bin_index
                count = int(selected.sum())
                if not count:
                    continue
                mean_probability = float(probabilities[selected].mean())
                observed_frequency = float(truth[selected].mean())
                gap = abs(mean_probability - observed_frequency)
                ece += gap * count / len(rows)
                records.append(
                    {
                        "split": split,
                        "cohort": cohort,
                        "target": target,
                        "bin": bin_index,
                        "lower": bin_index / bins,
                        "upper": (bin_index + 1) / bins,
                        "n_samples": count,
                        "mean_probability": mean_probability,
                        "observed_frequency": observed_frequency,
                        "absolute_gap": gap,
                    }
                )
            for record in records:
                record["expected_calibration_error"] = ece
                output.append(record)
    return output


def macro_metric_rows(metric_rows: list[dict]) -> list[dict]:
    """Aggregate per-target metrics into overall/resource/timing scopes."""

    grouped: dict[tuple[str, str], list[dict]] = {}
    for row in metric_rows:
        grouped.setdefault((row["split"], row["kernel_family"]), []).append(row)
    output = []
    scopes = {
        "overall": ALL_TARGETS,
        "resource": RESOURCE_TARGETS,
        "timing": TIMING_TARGETS,
    }
    for (split, family), rows in sorted(grouped.items()):
        by_target = {row["target"]: row for row in rows}
        for scope, targets in scopes.items():
            selected = [by_target[target] for target in targets if target in by_target]
            if not selected:
                continue
            output.append(
                {
                    "split": split,
                    "kernel_family": family,
                    "scope": scope,
                    "n_samples": selected[0]["n_samples"],
                    "r2": float(np.mean([row["r2"] for row in selected])),
                    "smape": float(np.mean([row["smape"] for row in selected])),
                    "rmse": float(np.mean([row["rmse"] for row in selected])),
                }
            )
    return output


def _raw_metrics(predictions: np.ndarray, targets: np.ndarray) -> dict:
    residual = targets - predictions
    total = np.square(targets - targets.mean(axis=0)).sum(axis=0)
    residual_sum = np.square(residual).sum(axis=0)
    r2 = np.full(total.shape, np.nan, dtype=float)
    valid_r2 = total > np.finfo(float).eps
    r2[valid_r2] = 1 - residual_sum[valid_r2] / total[valid_r2]
    smape = (
        np.abs(residual) / (np.abs(targets) + np.abs(predictions) + 1.0)
    ).mean(axis=0) * 200
    rmse = np.sqrt(np.square(residual).mean(axis=0))
    return {"r2": r2, "smape": smape, "rmse": rmse}


def structural_error_rows(prediction_rows: list[dict]) -> list[dict]:
    """Quartile-sliced error over the structural burdens required by the spec."""

    output = []
    splits = sorted({row["split"] for row in prediction_rows})
    scopes = {
        "overall": ALL_TARGETS,
        "resource": RESOURCE_TARGETS,
        "timing": TIMING_TARGETS,
    }
    for split in splits:
        split_rows = [row for row in prediction_rows if row["split"] == split]
        for feature in STRUCTURE_FEATURES:
            if not split_rows or feature not in split_rows[0]:
                continue
            values = np.asarray([float(row[feature]) for row in split_rows])
            boundaries = np.unique(np.quantile(values, [0, 0.25, 0.5, 0.75, 1]))
            if boundaries.size == 1:
                boundaries = np.asarray([boundaries[0], boundaries[0]])
            for bin_index, (lower, upper) in enumerate(
                zip(boundaries[:-1], boundaries[1:])
            ):
                selected = [
                    row
                    for row, value in zip(split_rows, values)
                    if value >= lower
                    and (value <= upper if bin_index == len(boundaries) - 2 else value < upper)
                ]
                if not selected:
                    continue
                for scope, targets_for_scope in scopes.items():
                    predictions = np.asarray(
                        [
                            [float(row[f"prediction_{target}"]) for target in targets_for_scope]
                            for row in selected
                        ]
                    )
                    targets = np.asarray(
                        [
                            [float(row[f"target_{target}"]) for target in targets_for_scope]
                            for row in selected
                        ]
                    )
                    metrics = _raw_metrics(predictions, targets)
                    finite_r2 = metrics["r2"][np.isfinite(metrics["r2"])]
                    output.append(
                        {
                            "split": split,
                            "feature": feature,
                            "bin": bin_index,
                            "lower": float(lower),
                            "upper": float(upper),
                            "scope": scope,
                            "n_samples": len(selected),
                            "r2": (
                                float(finite_r2.mean())
                                if finite_r2.size
                                else None
                            ),
                            "smape": float(np.mean(metrics["smape"])),
                            "rmse": float(np.mean(metrics["rmse"])),
                        }
                    )
    return output


def read_prediction_rows(path: str | Path) -> list[dict]:
    with Path(path).open(newline="") as handle:
        return list(csv.DictReader(handle))


def paired_delta_rows(
    candidate_rows: list[dict], baseline_rows: list[dict]
) -> tuple[list[dict], list[dict]]:
    """Return per-sample and aggregate SMAPE-contribution deltas vs. H0."""

    baseline = {
        (row["split"], row["tensor_path"]): row for row in baseline_rows
    }
    per_sample = []
    for candidate in candidate_rows:
        key = (candidate["split"], candidate["tensor_path"])
        if key not in baseline:
            raise ValueError(f"Baseline predictions do not contain {key}")
        reference = baseline[key]
        row = {
            "split": candidate["split"],
            "kernel_family": candidate["kernel_family"],
            "tensor_path": candidate["tensor_path"],
        }
        for target in ALL_TARGETS:
            truth = float(candidate[f"target_{target}"])
            baseline_truth = float(reference[f"target_{target}"])
            if not math.isclose(truth, baseline_truth, rel_tol=1e-6, abs_tol=1e-6):
                raise ValueError(f"Baseline target mismatch for {key}: {target}")
            denominator = abs(truth) + 1.0
            baseline_error = 200 * abs(
                truth - float(reference[f"prediction_{target}"])
            ) / (abs(truth) + abs(float(reference[f"prediction_{target}"])) + 1.0)
            candidate_error = 200 * abs(
                truth - float(candidate[f"prediction_{target}"])
            ) / (abs(truth) + abs(float(candidate[f"prediction_{target}"])) + 1.0)
            row[f"baseline_smape_{target}"] = baseline_error
            row[f"candidate_smape_{target}"] = candidate_error
            row[f"delta_smape_{target}"] = candidate_error - baseline_error
            row[f"delta_absolute_error_{target}"] = (
                abs(truth - float(candidate[f"prediction_{target}"]))
                - abs(truth - float(reference[f"prediction_{target}"]))
            ) / denominator
        per_sample.append(row)

    summary = []
    groups: dict[tuple[str, str], list[dict]] = {}
    for row in per_sample:
        groups.setdefault((row["split"], "all"), []).append(row)
        groups.setdefault((row["split"], row["kernel_family"]), []).append(row)
    for (split, family), rows in sorted(groups.items()):
        for target in ALL_TARGETS:
            values = np.asarray([row[f"delta_smape_{target}"] for row in rows])
            summary.append(
                {
                    "split": split,
                    "kernel_family": family,
                    "target": target,
                    "n_samples": len(rows),
                    "mean_delta_smape": float(values.mean()),
                    "median_delta_smape": float(np.median(values)),
                    "fraction_improved": float((values < 0).mean()),
                }
            )
    return per_sample, summary


def _scc_burden(num_nodes: int, edges: list[tuple[int, int]]) -> tuple[int, int]:
    """Return cyclic-node count and largest SCC via iterative Kosaraju."""

    adjacency = [[] for _ in range(num_nodes)]
    reverse = [[] for _ in range(num_nodes)]
    self_loops = set()
    for source, target in edges:
        adjacency[source].append(target)
        reverse[target].append(source)
        if source == target:
            self_loops.add(source)
    visited = [False] * num_nodes
    order = []
    for start in range(num_nodes):
        if visited[start]:
            continue
        visited[start] = True
        stack = [(start, 0)]
        while stack:
            node, cursor = stack[-1]
            if cursor < len(adjacency[node]):
                neighbor = adjacency[node][cursor]
                stack[-1] = (node, cursor + 1)
                if not visited[neighbor]:
                    visited[neighbor] = True
                    stack.append((neighbor, 0))
            else:
                order.append(node)
                stack.pop()
    assigned = [False] * num_nodes
    cyclic = 0
    largest = 0
    for start in reversed(order):
        if assigned[start]:
            continue
        component = []
        stack = [start]
        assigned[start] = True
        while stack:
            node = stack.pop()
            component.append(node)
            for neighbor in reverse[node]:
                if not assigned[neighbor]:
                    assigned[neighbor] = True
                    stack.append(neighbor)
        largest = max(largest, len(component))
        if len(component) > 1 or component[0] in self_loops:
            cyclic += len(component)
    return cyclic, largest


def graph_structure_rows(data) -> list[dict]:
    """Extract per-graph structural burdens from a CPU PyG batch."""

    instruction_batch = data["instruction"].batch.cpu()
    block_batch = data["block"].batch.cpu()
    function_batch = data["function"].batch.cpu()
    variable_batch = data["variable"].batch.cpu()
    graph_count = data.num_graphs
    instruction_counts = torch_bincount(instruction_batch, graph_count)
    block_counts = torch_bincount(block_batch, graph_count)
    contains = data[("block", "contains", "instruction")].edge_index.cpu()
    block_lengths = torch_bincount(contains[0], data["block"].num_nodes)
    cfg = data[("block", "control", "block")].edge_index.cpu()
    call_depth = data["function"].call_depth.cpu()
    memory_like = data["variable"].x[:, 5:10].bool().any(dim=-1).cpu()
    memory_counts = torch_bincount(variable_batch[memory_like], graph_count)
    operands = data[("variable", "operand", "instruction")].edge_index.cpu()
    memory_access_batch = variable_batch[operands[0][memory_like[operands[0]]]]
    memory_access_counts = torch_bincount(memory_access_batch, graph_count)
    loop_flags = data["block"].x[:, 2].bool().cpu()

    output = []
    for graph in range(graph_count):
        blocks = (block_batch == graph).nonzero().flatten()
        functions = (function_batch == graph).nonzero().flatten()
        block_start = int(blocks[0])
        edge_mask = block_batch[cfg[0]] == graph
        local_edges = [
            (int(source) - block_start, int(target) - block_start)
            for source, target in cfg[:, edge_mask].t().tolist()
        ]
        cyclic, largest_scc = _scc_burden(len(blocks), local_edges)
        output.append(
            {
                "instruction_count": int(instruction_counts[graph]),
                "block_count": int(block_counts[graph]),
                "maximum_block_length": int(block_lengths[blocks].max()),
                "loop_block_fraction": float(loop_flags[blocks].float().mean()),
                "block_scc_fraction": cyclic / max(len(blocks), 1),
                "largest_block_scc": largest_scc,
                "call_depth": int(call_depth[functions].max()),
                "memory_node_count": int(memory_counts[graph]),
                "memory_access_count": int(memory_access_counts[graph]),
            }
        )
    return output


def torch_bincount(values, minlength: int):
    """Late torch import keeps basic report refreshes lightweight."""

    import torch

    return torch.bincount(values, minlength=minlength)


def format_cohort_table(membership: dict[str, list[dict]]) -> str:
    lines = [
        "| split | family | archive | samples |",
        "| --- | --- | --- | ---: |",
    ]
    for split, rows in membership.items():
        families: dict[str, list[dict]] = {}
        for row in rows:
            families.setdefault(row["kernel_family"], []).append(row)
        for family, family_rows in families.items():
            lines.append(
                f"| {split} | {family} | "
                f"{', '.join(row['archive'] for row in family_rows)} | "
                f"{sum(row['n_samples'] for row in family_rows)} |"
            )
    return "\n".join(lines)


def format_hurdle_table(rows: list[dict]) -> str:
    lines = [
        "| split | target | TN | FP | FN | TP | accuracy |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        if row["cohort"] != "all":
            continue
        lines.append(
            f"| {row['split']} | {row['target'].upper()} | {row['tn']} | "
            f"{row['fp']} | {row['fn']} | {row['tp']} | "
            f"{row['accuracy']:.3f} |"
        )
    return "\n".join(lines)
