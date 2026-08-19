#!/usr/bin/env python3
"""Data/result audit for deciding the next ll-hls4ml thesis experiments.

The script intentionally avoids loading graph tensors.  It reads the compact
``labels.json`` index and committed result CSV/JSON blobs, including blobs that
are deleted in the current worktree but still available at ``HEAD``.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
from collections import Counter, defaultdict
from pathlib import Path
import subprocess

import numpy as np


TARGETS = ("lut", "ff", "dsp", "bram", "cycles_max", "interval_max")
SYNTHETIC_FAMILIES = (
    "2layer",
    "3layer",
    "conv1d",
    "conv2d",
    "dense_latency",
    "dense_resource",
    "rule4ml",
)

RUNS = {
    "h0": "artifacts/results/ll_hls4ml_hierarchy_fusion_scale200_results/"
    "hierarchical_scale200_seed42",
    "high_level": "artifacts/results/ll_hls4ml_hierarchy_fusion_scale200_results/"
    "high_level_gatv2_scale200_seed42",
    "late_fusion": "artifacts/results/ll_hls4ml_hierarchy_fusion_scale200_results/"
    "hierarchical_high_level_fusion_scale200_seed42",
    "block_attention": "artifacts/results/ll_hls4ml_hx_architectures_scale200_results/"
    "hx_block_attention_scale200_seed42",
    "memory_dual": "artifacts/results/ll_hls4ml_hx_architectures_scale200_results/"
    "hx_memory_dual_scale200_seed42",
    "sequence_gru": "artifacts/results/ll_hls4ml_hx_architectures_scale200_results/"
    "hx_sequence_gru_scale200_seed42",
    "program": "artifacts/results/ll_hls4ml_hierarchical_program_scale200_results/"
    "hierarchical_program_scale200_seed42",
}


def git_blob(repo: Path, ref: str, path: str) -> str:
    result = subprocess.run(
        ["git", "show", f"{ref}:{path}"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout


def read_blob_json(repo: Path, ref: str, path: str) -> dict:
    return json.loads(git_blob(repo, ref, path))


def read_blob_csv(repo: Path, ref: str, path: str) -> list[dict[str, str]]:
    return list(csv.DictReader(io.StringIO(git_blob(repo, ref, path))))


def smape_components(target: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    return 200.0 * np.abs(target - prediction) / (
        np.abs(target) + np.abs(prediction) + 1.0
    )


def prediction_arrays(rows: list[dict[str, str]], split: str = "test"):
    selected = [row for row in rows if row["split"] == split]
    paths = [row["tensor_path"] for row in selected]
    families = np.asarray([row["kernel_family"] for row in selected])
    target = np.asarray(
        [[float(row[f"target_{name}"]) for name in TARGETS] for row in selected]
    )
    prediction = np.asarray(
        [
            [float(row[f"prediction_{name}"]) for name in TARGETS]
            for row in selected
        ]
    )
    return paths, families, target, prediction


def matched_arrays(
    left_rows: list[dict[str, str]],
    right_rows: list[dict[str, str]],
    split: str = "test",
):
    left = {row["tensor_path"]: row for row in left_rows if row["split"] == split}
    right = {row["tensor_path"]: row for row in right_rows if row["split"] == split}
    paths = sorted(set(left) & set(right))
    families = np.asarray([left[path]["kernel_family"] for path in paths])
    target = np.asarray(
        [
            [float(left[path][f"target_{name}"]) for name in TARGETS]
            for path in paths
        ]
    )
    left_prediction = np.asarray(
        [
            [float(left[path][f"prediction_{name}"]) for name in TARGETS]
            for path in paths
        ]
    )
    right_prediction = np.asarray(
        [
            [float(right[path][f"prediction_{name}"]) for name in TARGETS]
            for path in paths
        ]
    )
    return paths, families, target, left_prediction, right_prediction


def bootstrap_mean_interval(
    values: np.ndarray,
    rng: np.random.Generator,
    samples: int,
) -> tuple[float, float]:
    n = len(values)
    chunk = 500
    boot = []
    for start in range(0, samples, chunk):
        count = min(chunk, samples - start)
        indices = rng.integers(0, n, size=(count, n))
        boot.append(values[indices].mean(axis=1))
    distribution = np.concatenate(boot)
    return tuple(np.quantile(distribution, [0.025, 0.975]))


def rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(len(values), dtype=float)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1)
        start = end
    return ranks


def spearman(left: np.ndarray, right: np.ndarray) -> float:
    if len(left) < 3 or np.ptp(left) == 0 or np.ptp(right) == 0:
        return float("nan")
    return float(np.corrcoef(rankdata(left), rankdata(right))[0, 1])


def family_adjust(values: np.ndarray, families: np.ndarray) -> np.ndarray:
    adjusted = values.astype(float).copy()
    for family in np.unique(families):
        mask = families == family
        adjusted[mask] -= adjusted[mask].mean()
    return adjusted


def markdown_table(headers: list[str], rows: list[list[object]]) -> list[str]:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(map(str, row)) + " |" for row in rows)
    return lines


def archive_number(path: str) -> int:
    return int(Path(path).parts[1].removeprefix("archive_"))


def dataset_audit(index: dict, group_overlay: dict | None = None) -> list[str]:
    labels = index["labels"]
    metadata = index.get("metadata", {})
    selected_paths = [
        path
        for path in labels
        if Path(path).parts[0] in SYNTHETIC_FAMILIES
        and archive_number(path) <= 8
    ]
    exemplar_paths = [
        path
        for path in labels
        if Path(path).parts[0] == "exemplar" and archive_number(path) <= 9
    ]
    split_counts = Counter(
        (Path(path).parts[0], str(metadata.get(path, {}).get("dataset_split", "")))
        for path in selected_paths
    )

    lines = ["## Compact dataset audit", ""]
    count_rows = []
    for family in SYNTHETIC_FAMILIES:
        count_rows.append(
            [
                family,
                split_counts[(family, "train")],
                split_counts[(family, "validation")]
                + split_counts[(family, "val")],
                split_counts[(family, "test")],
                sum(
                    count
                    for (candidate, _split), count in split_counts.items()
                    if candidate == family
                ),
            ]
        )
    lines.extend(markdown_table(["family", "train", "val", "test", "total"], count_rows))
    lines.extend(["", f"Exemplar records through archive 9: {len(exemplar_paths)}.", ""])

    values = np.asarray([labels[path] for path in selected_paths], dtype=float)
    stat_rows = []
    for column, target in enumerate(TARGETS):
        vector = values[:, column]
        q10, median, q90, q99 = np.quantile(vector, [0.1, 0.5, 0.9, 0.99])
        stat_rows.append(
            [
                target,
                f"{np.mean(vector == 0):.3f}",
                f"{q10:.3g}",
                f"{median:.3g}",
                f"{q90:.3g}",
                f"{q99:.3g}",
                f"{vector.max():.3g}",
                len(np.unique(vector)),
            ]
        )
    lines.extend(
        markdown_table(
            ["target", "zero frac", "p10", "median", "p90", "p99", "max", "unique"],
            stat_rows,
        )
    )
    lines.extend(["", "Log1p target correlation:", ""])
    correlations = np.corrcoef(np.log1p(values), rowvar=False)
    correlation_rows = [
        [TARGETS[index], *[f"{value:.2f}" for value in correlations[index]]]
        for index in range(len(TARGETS))
    ]
    lines.extend(markdown_table(["target", *TARGETS], correlation_rows))

    cycles = values[:, TARGETS.index("cycles_max")]
    interval = values[:, TARGETS.index("interval_max")]
    ratio = cycles / interval
    ratio_quantiles = np.quantile(ratio, [0.1, 0.5, 0.9, 0.99])
    lines.extend(
        [
            "",
            "Timing-target relationship: "
            f"{np.mean(interval <= cycles):.3f} have interval <= cycles; "
            "cycles/interval p10, median, p90, p99 = "
            + ", ".join(f"{value:.3g}" for value in ratio_quantiles)
            + ".",
        ]
    )

    zero_rows = []
    for family in SYNTHETIC_FAMILIES:
        family_values = np.asarray(
            [labels[path] for path in selected_paths if Path(path).parts[0] == family]
        )
        zero_rows.append(
            [
                family,
                f"{np.mean(family_values[:, 2] == 0):.3f}",
                f"{np.mean(family_values[:, 3] == 0):.3f}",
            ]
        )
    lines.extend(["", "Zero prevalence by family:", ""])
    lines.extend(markdown_table(["family", "DSP zero", "BRAM zero"], zero_rows))

    groups: dict[tuple[str, str], set[str]] = defaultdict(set)
    group_sizes: Counter[tuple[str, str]] = Counter()
    uuid_paths: dict[tuple[str, str], list[str]] = defaultdict(list)
    contexts: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    metadata_keys = Counter()
    for path in selected_paths:
        meta = metadata.get(path, {})
        overlay_meta = (group_overlay or {}).get(Path(path).stem, {})
        metadata_keys.update(meta.keys())
        family = Path(path).parts[0]
        split = str(meta.get("dataset_split", ""))
        group = str(
            meta.get("group_id")
            or overlay_meta.get("group_id")
            or Path(path).stem
        )
        groups[(family, group)].add(split)
        group_sizes[(family, group)] += 1
        uuid_paths[(family, Path(path).stem)].append(path)
        for key in (
            "backend",
            "target_part",
            "clock_period",
            "tool_version",
            "hls4ml_version",
        ):
            if key in meta:
                contexts[key][family].add(str(meta[key]))
    crossing = sum(len(splits) > 1 for splits in groups.values())
    crossing_samples = sum(
        group_sizes[group] for group, splits in groups.items() if len(splits) > 1
    )
    repeated_uuid_paths = sum(
        len(paths) - 1 for paths in uuid_paths.values() if len(paths) > 1
    )
    lines.extend(
        [
            "",
            f"Architecture groups: {len(groups)}; groups crossing official splits: "
            f"{crossing} ({crossing_samples} samples); "
            f"repeated UUID paths removed by dataset-level deduplication: "
            f"{repeated_uuid_paths}.",
            "Most common metadata fields: "
            + ", ".join(f"{key} ({count})" for key, count in metadata_keys.most_common(12))
            + ".",
        ]
    )
    if contexts:
        context_rows = []
        for key, by_family in contexts.items():
            all_values = set().union(*by_family.values()) if by_family else set()
            family_signatures = {tuple(sorted(values)) for values in by_family.values()}
            context_rows.append([key, len(all_values), len(family_signatures)])
        lines.extend(["", "Context diversity/confounding proxy:", ""])
        lines.extend(
            markdown_table(
                ["metadata field", "unique values", "distinct family value-sets"],
                context_rows,
            )
        )
    return lines


def load_runs(repo: Path, ref: str):
    result = {}
    for name, directory in RUNS.items():
        result[name] = {
            "predictions": read_blob_csv(repo, ref, f"{directory}/predictions.csv"),
            "summary": read_blob_json(repo, ref, f"{directory}/summary.json"),
            "config": read_blob_json(repo, ref, f"{directory}/resolved_config.json"),
        }
        try:
            result[name]["curves"] = read_blob_csv(
                repo, ref, f"{directory}/learning_curves.csv"
            )
        except subprocess.CalledProcessError:
            result[name]["curves"] = []
    return result


def run_overview(runs: dict) -> list[str]:
    rows = []
    for name, run in runs.items():
        _, _, target, prediction = prediction_arrays(run["predictions"])
        config = run["config"]
        summary = run["summary"]
        curves = run["curves"]
        rows.append(
            [
                name,
                f"{smape_components(target, prediction).mean():.2f}",
                config.get("parameter_count", summary.get("parameter_count", "?")),
                summary.get("best_epoch", config.get("best_epoch", "?")),
                len(curves) or "legacy",
                config.get("tensor_source_revision", "?"),
                config.get("split_sha256", "?"),
            ]
        )
    return [
        "## Result overview",
        "",
        *markdown_table(
            [
                "run",
                "test SMAPE",
                "params",
                "best epoch",
                "curve epochs",
                "tensor revision",
                "split hash",
            ],
            rows,
        ),
    ]


def paired_audit(runs: dict, bootstrap_samples: int, seed: int) -> list[str]:
    rng = np.random.default_rng(seed)
    baseline = runs["h0"]["predictions"]
    rows = []
    family_rows = []
    target_rows = []
    for name in ("block_attention", "memory_dual", "sequence_gru", "program"):
        paths, families, target, h0_prediction, candidate = matched_arrays(
            baseline, runs[name]["predictions"]
        )
        h0_error = smape_components(target, h0_prediction).mean(axis=1)
        candidate_error = smape_components(target, candidate).mean(axis=1)
        delta = candidate_error - h0_error
        low, high = bootstrap_mean_interval(delta, rng, bootstrap_samples)
        rows.append(
            [
                name,
                len(paths),
                f"{h0_error.mean():.2f}",
                f"{candidate_error.mean():.2f}",
                f"{delta.mean():+.2f}",
                f"[{low:+.2f}, {high:+.2f}]",
                f"{np.mean(delta < 0):.3f}",
                f"{np.corrcoef(h0_error, candidate_error)[0, 1]:.3f}",
            ]
        )
        for family in np.unique(families):
            mask = families == family
            family_rows.append(
                [name, family, int(mask.sum()), f"{delta[mask].mean():+.2f}"]
            )
        h0_target_error = smape_components(target, h0_prediction).mean(axis=0)
        candidate_target_error = smape_components(target, candidate).mean(axis=0)
        for target_name, h0_value, candidate_value in zip(
            TARGETS, h0_target_error, candidate_target_error
        ):
            target_rows.append(
                [
                    name,
                    target_name,
                    f"{h0_value:.2f}",
                    f"{candidate_value:.2f}",
                    f"{candidate_value - h0_value:+.2f}",
                ]
            )
    lines = [
        "## Matched test-set architecture deltas",
        "",
        "Positive delta means worse than H0. Bootstrap intervals resample test projects and do not include training-seed variance.",
        "",
        *markdown_table(
            [
                "candidate",
                "N",
                "H0",
                "candidate",
                "delta",
                "project-bootstrap 95%",
                "fraction better",
                "error corr",
            ],
            rows,
        ),
        "",
        "Mean candidate-minus-H0 SMAPE by family:",
        "",
        *markdown_table(["candidate", "family", "N", "delta"], family_rows),
        "",
        "Per-target candidate-minus-H0 SMAPE:",
        "",
        *markdown_table(
            ["candidate", "target", "H0", "candidate", "delta"], target_rows
        ),
    ]
    return lines


def fixed_ensemble_audit(runs: dict, bootstrap_samples: int, seed: int) -> list[str]:
    rng = np.random.default_rng(seed + 1)
    experiments = (
        ("H0 + sequence GRU", "h0", "sequence_gru"),
        ("H0 + block attention", "h0", "block_attention"),
        ("H0 + memory dual", "h0", "memory_dual"),
        ("H0 + program", "h0", "program"),
        ("H0 + high-level", "h0", "high_level"),
    )
    rows = []
    for label, left_name, right_name in experiments:
        paths, _families, target, left, right = matched_arrays(
            runs[left_name]["predictions"], runs[right_name]["predictions"]
        )
        arithmetic = 0.5 * (left + right)
        geometric = np.expm1(0.5 * (np.log1p(left) + np.log1p(right)))
        left_error = smape_components(target, left).mean(axis=1)
        right_error = smape_components(target, right).mean(axis=1)
        for blend_name, blend in (("arithmetic", arithmetic), ("log-space", geometric)):
            blend_error = smape_components(target, blend).mean(axis=1)
            delta = blend_error - left_error
            low, high = bootstrap_mean_interval(delta, rng, bootstrap_samples)
            rows.append(
                [
                    label,
                    blend_name,
                    len(paths),
                    f"{left_error.mean():.2f}",
                    f"{right_error.mean():.2f}",
                    f"{blend_error.mean():.2f}",
                    f"{delta.mean():+.2f}",
                    f"[{low:+.2f}, {high:+.2f}]",
                ]
            )

    _, _, target, _h0, fusion = matched_arrays(
        runs["h0"]["predictions"], runs["late_fusion"]["predictions"]
    )
    fusion_smape = smape_components(target, fusion).mean()
    return [
        "## Fixed ensemble diagnostics",
        "",
        "These 50/50 blends are diagnostics, not test-tuned proposals. They ask whether independently trained representations contain complementary residual signal.",
        "",
        *markdown_table(
            [
                "pair",
                "blend",
                "N",
                "left",
                "right",
                "blend",
                "blend-left",
                "95% interval",
            ],
            rows,
        ),
        "",
        f"The learned late-fusion model scores {fusion_smape:.2f} SMAPE on the same 593-project test set.",
    ]


def external_probe_ensemble_audit(runs: dict, probe_rows: list[dict[str, str]]) -> list[str]:
    rows = []
    for left_name in ("h0", "late_fusion"):
        test = matched_arrays(runs[left_name]["predictions"], probe_rows, split="test")
        _, _, test_target, test_left, test_probe = test
        for mode in ("arithmetic", "log-space"):
            if mode == "arithmetic":
                test_prediction = 0.5 * test_left + 0.5 * test_probe
            else:
                test_prediction = np.expm1(
                    0.5 * np.log1p(test_left) + 0.5 * np.log1p(test_probe)
                )
            rows.append(
                [
                    left_name,
                    mode,
                    f"{smape_components(test_target, test_left).mean():.2f}",
                    f"{smape_components(test_target, test_probe).mean():.2f}",
                    f"{smape_components(test_target, test_prediction).mean():.2f}",
                ]
            )
    return [
        "## External tabular-probe blend diagnostic",
        "",
        "Fixed 50/50 blends test whether the CPU probe contains complementary residual signal; no weight was test-tuned.",
        "",
        *markdown_table(
            [
                "left model",
                "blend",
                "test left",
                "test probe",
                "test blend",
            ],
            rows,
        ),
    ]


def structural_audit(runs: dict) -> list[str]:
    structural_source = {
        row["tensor_path"]: row
        for row in runs["sequence_gru"]["predictions"]
        if row["split"] == "test"
    }
    features = (
        "instruction_count",
        "block_count",
        "maximum_block_length",
        "loop_block_fraction",
        "block_scc_fraction",
        "largest_block_scc",
        "call_depth",
        "memory_node_count",
        "memory_access_count",
    )
    rows = []
    for model_name in ("h0", "sequence_gru", "program"):
        paths, families, target, prediction = prediction_arrays(
            runs[model_name]["predictions"]
        )
        error = smape_components(target, prediction).mean(axis=1)
        valid = np.asarray([path in structural_source for path in paths])
        paths = [path for path, keep in zip(paths, valid) if keep]
        families = families[valid]
        error = error[valid]
        adjusted_error = family_adjust(error, families)
        correlations = []
        for feature in features:
            values = np.asarray(
                [float(structural_source[path][feature]) for path in paths]
            )
            correlations.append(
                (
                    feature,
                    spearman(values, error),
                    spearman(family_adjust(values, families), adjusted_error),
                )
            )
        correlations.sort(key=lambda item: abs(item[2]), reverse=True)
        for feature, overall, within_family in correlations[:4]:
            rows.append(
                [model_name, feature, f"{overall:+.3f}", f"{within_family:+.3f}"]
            )
    return [
        "## Structural error associations",
        "",
        "Top absolute Spearman associations per model. The within-family column removes family means before ranking and is the less-confounded diagnostic.",
        "",
        *markdown_table(
            ["model", "feature", "overall rho", "family-adjusted rho"], rows
        ),
    ]


def training_curve_audit(runs: dict) -> list[str]:
    rows = []
    for name in ("block_attention", "memory_dual", "sequence_gru", "program"):
        curves = runs[name]["curves"]
        if not curves:
            continue
        validation = np.asarray([float(row["val_smape"]) for row in curves])
        train_loss = np.asarray([float(row["train_loss"]) for row in curves])
        validation_loss = np.asarray([float(row["val_loss"]) for row in curves])
        learning_rates = np.asarray(
            [float(row["learning_rate"]) for row in curves if row.get("learning_rate")]
        )
        best = int(np.argmin(validation))
        rows.append(
            [
                name,
                len(curves),
                best + 1,
                f"{validation[best]:.2f}",
                f"{validation[-1]:.2f}",
                f"{train_loss[-1]:.3f}",
                f"{validation_loss[-1]:.3f}",
                len(np.unique(learning_rates)) if learning_rates.size else "not logged",
                f"{learning_rates[-1]:.1e}" if learning_rates.size else "not logged",
            ]
        )
    return [
        "## Optimization audit",
        "",
        *markdown_table(
            [
                "run",
                "epochs",
                "best epoch",
                "best val SMAPE",
                "last val SMAPE",
                "last train loss",
                "last val loss",
                "LR levels",
                "last LR",
            ],
            rows,
        ),
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument("--ref", default="HEAD")
    parser.add_argument(
        "--labels", type=Path, default=Path(__file__).resolve().parents[2] / "data/tensors/labels.json"
    )
    parser.add_argument(
        "--metadata-overlay",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "artifacts/metadata/graph_metadata.json",
    )
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260819)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--tabular-predictions", type=Path)
    args = parser.parse_args()

    with args.labels.open() as handle:
        index = json.load(handle)
    group_overlay = None
    if args.metadata_overlay.is_file():
        with args.metadata_overlay.open() as handle:
            group_overlay = json.load(handle)
    runs = load_runs(args.repo, args.ref)
    probe_rows = None
    if args.tabular_predictions:
        with args.tabular_predictions.open(newline="") as handle:
            probe_rows = list(csv.DictReader(handle))

    sections = [
        "# ll-hls4ml thesis-wall audit",
        "",
        f"Git result ref: `{args.ref}`. Local compact label index: `{args.labels}`.",
        "No graph tensor was loaded and no model was trained.",
        "",
        *dataset_audit(index, group_overlay),
        "",
        *run_overview(runs),
        "",
        *paired_audit(runs, args.bootstrap_samples, args.seed),
        "",
        *fixed_ensemble_audit(runs, args.bootstrap_samples, args.seed),
        "",
        *(
            external_probe_ensemble_audit(runs, probe_rows)
            if probe_rows is not None
            else []
        ),
        "",
        *structural_audit(runs),
        "",
        *training_curve_audit(runs),
        "",
    ]
    report = "\n".join(sections)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report)
    print(report)


if __name__ == "__main__":
    main()
