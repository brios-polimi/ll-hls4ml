"""Generate paper-style comparisons against the published wa-hls4ml tables.

The published values live in a plain JSON file beside this module so that paper
revisions or presentation changes do not require touching the reporting code.
Existing training artifacts are read-only inputs; comparison files are written
to a separate output directory.
"""

from __future__ import annotations

import csv
import json
import math
from collections import Counter
from pathlib import Path
from typing import Iterable, Mapping


PAPER_RESULTS_PATH = Path(__file__).with_name("wahls4ml_paper_results.json")
TARGET_DISPLAY = {
    "bram": "BRAM",
    "dsp": "DSP",
    "ff": "FF",
    "lut": "LUT",
    "cycles_max": "Cycles",
    "interval_max": "II",
}
METRIC_DISPLAY = {"r2": r"$R^2$", "smape": "SMAPE [%]", "rmse": "RMSE"}
TEST_COHORTS = {
    "All": lambda family: True,
    "Dense": lambda family: family not in {"conv1d", "conv2d"},
    "Conv1D": lambda family: family == "conv1d",
    "Conv2D": lambda family: family == "conv2d",
}
EXEMPLAR_PREFIXES = {
    "Jet": "Jet",
    "Quarks": "Quarks",
    "Anomaly": "Anomaly",
    "Bipc": "BiPC",
    "Cookie": "Cookie-box",
    "Automlp": "AutoMLP",
    "Particle": "Particle Tracking",
}


def _load_json(path: Path) -> dict | list:
    with path.open() as handle:
        return json.load(handle)


def _validate_paper_results(data: dict) -> None:
    targets = data["target_order"]
    metrics = data["metric_order"]
    models = data["model_order"]
    for split in ("test", "exemplar"):
        for cohort, cohort_data in data[split].items():
            for model in models:
                if model not in cohort_data:
                    raise ValueError(f"Missing {split}/{cohort}/{model} paper row")
                for metric in metrics:
                    values = cohort_data[model].get(metric)
                    if values is None or len(values) != len(targets):
                        raise ValueError(
                            f"Expected {len(targets)} values for "
                            f"{split}/{cohort}/{model}/{metric}"
                        )


def _paper_records(data: dict) -> list[dict]:
    records = []
    for split in ("test", "exemplar"):
        for cohort, cohort_data in data[split].items():
            for model in data["model_order"]:
                for metric in data["metric_order"]:
                    for target, value in zip(
                        data["target_order"], cohort_data[model][metric]
                    ):
                        records.append(
                            {
                                "split": split,
                                "cohort": cohort,
                                "source": "paper",
                                "model": model,
                                "n_samples": cohort_data["n_samples"],
                                "metric": metric,
                                "target": target,
                                "value": value,
                            }
                        )
    return records


def _load_prediction_rows(run_dir: Path) -> list[dict]:
    path = run_dir / "predictions.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing predictions file: {path}")
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"No prediction rows in {path}")
    return rows


def _run_metadata(run_dir: Path) -> dict:
    for filename in ("resolved_config.json", "summary.json"):
        path = run_dir / filename
        if not path.exists():
            continue
        data = _load_json(path)
        if filename == "summary.json":
            data = data.get("resolved_config", data)
        return {
            "experiment_name": data.get("experiment_name", run_dir.name),
            "model": data.get("model"),
            "seed": data.get("seed"),
            "scale_percent": data.get("scale_percent"),
            "tensor_source_revision": data.get("tensor_source_revision"),
            "split_manifest_path": data.get("split_manifest_path"),
        }
    return {"experiment_name": run_dir.name}


def _exemplar_label_map(label_path: Path) -> tuple[dict[str, dict], Counter]:
    records = _load_json(label_path)
    if not isinstance(records, list):
        raise ValueError(f"Expected a list in {label_path}")
    by_uuid = {}
    totals = Counter()
    for record in records:
        metadata = record.get("meta_data") or {}
        identifier = str(metadata.get("uuid", ""))
        model_name = str(metadata.get("model_name", ""))
        prefix = model_name.split("_", 1)[0]
        cohort = EXEMPLAR_PREFIXES.get(prefix)
        if not identifier or cohort is None:
            raise ValueError(
                f"Unrecognized exemplar identity: uuid={identifier!r}, "
                f"model_name={model_name!r}"
            )
        if identifier in by_uuid:
            raise ValueError(f"Duplicate exemplar UUID in labels: {identifier}")
        by_uuid[identifier] = {"cohort": cohort, "record": record}
        totals[cohort] += 1
    return by_uuid, totals


def _label_targets(record: dict) -> dict[str, float]:
    resource = record.get("resource_report") or {}
    latency = record.get("latency_report") or {}
    return {
        "bram": float(resource["bram"]),
        "dsp": float(resource["dsp"]),
        "ff": float(resource["ff"]),
        "lut": float(resource["lut"]),
        "cycles_max": float(latency["cycles_max"]),
        "interval_max": float(latency["interval_max"]),
    }


def _annotate_exemplar_rows(
    rows: list[dict], label_path: Path
) -> tuple[list[dict], dict]:
    by_uuid, label_totals = _exemplar_label_map(label_path)
    annotated = []
    seen = set()
    for row in rows:
        identifier = Path(row["tensor_path"]).stem
        if identifier in seen:
            raise ValueError(f"Duplicate exemplar prediction UUID: {identifier}")
        seen.add(identifier)
        match = by_uuid.get(identifier)
        if match is None:
            raise ValueError(
                f"Exemplar prediction UUID is absent from labels: {identifier}"
            )
        expected = _label_targets(match["record"])
        for target, expected_value in expected.items():
            observed = float(row[f"target_{target}"])
            if not math.isclose(observed, expected_value, rel_tol=0, abs_tol=1e-5):
                raise ValueError(
                    f"Exemplar target mismatch for {identifier}/{target}: "
                    f"prediction artifact has {observed}, labels have {expected_value}"
                )
        annotated.append({**row, "comparison_cohort": match["cohort"]})
    selected_totals = Counter(row["comparison_cohort"] for row in annotated)
    coverage = {
        cohort: {
            "selected": selected_totals[cohort],
            "paper_total": label_totals[cohort],
            "missing": label_totals[cohort] - selected_totals[cohort],
        }
        for cohort in EXEMPLAR_PREFIXES.values()
    }
    return annotated, coverage


def _metrics(rows: Iterable[dict], target: str) -> dict[str, float | None]:
    pairs = [
        (float(row[f"target_{target}"]), float(row[f"prediction_{target}"]))
        for row in rows
    ]
    if not pairs:
        raise ValueError(f"Cannot compute {target} metrics for an empty cohort")
    targets = [pair[0] for pair in pairs]
    predictions = [pair[1] for pair in pairs]
    residual_sum = sum((target_value - prediction) ** 2 for target_value, prediction in pairs)
    target_mean = sum(targets) / len(targets)
    total_sum = sum((target_value - target_mean) ** 2 for target_value in targets)
    r2 = None if total_sum == 0 else 1.0 - residual_sum / total_sum
    smape = (
        200.0
        * sum(
            abs(target_value - prediction)
            / (abs(target_value) + abs(prediction) + 1.0)
            for target_value, prediction in pairs
        )
        / len(pairs)
    )
    rmse = math.sqrt(residual_sum / len(pairs))
    return {"r2": r2, "smape": smape, "rmse": rmse}


def _run_records(
    model_name: str,
    rows: list[dict],
    targets: list[str],
    label_path: Path,
) -> tuple[list[dict], dict]:
    test_rows = [row for row in rows if row["split"] == "test"]
    exemplar_rows = [row for row in rows if row["split"] == "exemplar"]
    if not test_rows or not exemplar_rows:
        raise ValueError("predictions.csv must contain both test and exemplar rows")
    exemplar_rows, coverage = _annotate_exemplar_rows(exemplar_rows, label_path)
    output = []
    for cohort, predicate in TEST_COHORTS.items():
        selected = [row for row in test_rows if predicate(row["kernel_family"])]
        for target in targets:
            for metric, value in _metrics(selected, target).items():
                output.append(
                    {
                        "split": "test",
                        "cohort": cohort,
                        "source": "run",
                        "model": model_name,
                        "n_samples": len(selected),
                        "metric": metric,
                        "target": target,
                        "value": value,
                    }
                )
    for cohort in EXEMPLAR_PREFIXES.values():
        selected = [
            row for row in exemplar_rows if row["comparison_cohort"] == cohort
        ]
        for target in targets:
            for metric, value in _metrics(selected, target).items():
                output.append(
                    {
                        "split": "exemplar",
                        "cohort": cohort,
                        "source": "run",
                        "model": model_name,
                        "n_samples": len(selected),
                        "metric": metric,
                        "target": target,
                        "value": value,
                    }
                )
    return output, coverage


def _mark_best(records: list[dict]) -> None:
    grouped = {}
    for row in records:
        value = row["value"]
        if value is not None:
            grouped.setdefault(
                (row["split"], row["cohort"], row["metric"], row["target"]),
                [],
            ).append(row)
    for rows in grouped.values():
        metric = rows[0]["metric"]
        best = (
            max(row["value"] for row in rows)
            if metric == "r2"
            else min(row["value"] for row in rows)
        )
        for row in rows:
            row["is_best"] = math.isclose(
                row["value"], best, rel_tol=1e-12, abs_tol=1e-12
            )
    for row in records:
        row.setdefault("is_best", False)


def _write_comparison_csv(path: Path, records: list[dict]) -> None:
    fields = [
        "split",
        "cohort",
        "source",
        "model",
        "n_samples",
        "metric",
        "target",
        "value",
        "is_best",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)


def _format_value(metric: str, value: float | None) -> str:
    if value is None:
        return "N/A"
    return f"{value:.2f}" if metric == "r2" else f"{value:.1f}"


def _lookup(records: list[dict]) -> dict[tuple, dict]:
    return {
        (row["split"], row["cohort"], row["model"], row["metric"], row["target"]): row
        for row in records
    }


def _render_table(
    path: Path,
    split: str,
    records: list[dict],
    cohorts: list[str],
    models: list[str],
    targets: list[str],
    metrics: list[str],
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    lookup = _lookup(records)
    rows_per_cohort = len(models)
    row_count = len(cohorts) * rows_per_cohort
    widths = [1.75, 1.75, 0.72] + [0.78] * (len(targets) * len(metrics))
    x_edges = [0.0]
    for width in widths:
        x_edges.append(x_edges[-1] + width)
    row_height = 0.43
    header_height = 1.05
    total_height = header_height + row_count * row_height
    figure, axis = plt.subplots(
        figsize=(20, max(5.0, total_height * 0.48)), constrained_layout=True
    )
    axis.set_xlim(0, x_edges[-1])
    footnote_height = 0.32
    axis.set_ylim(-footnote_height, total_height)
    axis.axis("off")

    header_bottom = total_height - header_height
    axis.text((x_edges[0] + x_edges[1]) / 2, header_bottom + 0.28, "Arch.", ha="center", va="center", fontsize=9)
    axis.text((x_edges[1] + x_edges[2]) / 2, header_bottom + 0.28, "Model", ha="center", va="center", fontsize=9)
    axis.text((x_edges[2] + x_edges[3]) / 2, header_bottom + 0.28, "N", ha="center", va="center", fontsize=9)
    numeric_start = 3
    for metric_index, metric in enumerate(metrics):
        start = numeric_start + metric_index * len(targets)
        end = start + len(targets)
        axis.text(
            (x_edges[start] + x_edges[end]) / 2,
            header_bottom + 0.78,
            METRIC_DISPLAY[metric],
            ha="center",
            va="center",
            fontsize=10,
        )
        for target_index, target in enumerate(targets):
            column = start + target_index
            axis.text(
                (x_edges[column] + x_edges[column + 1]) / 2,
                header_bottom + 0.28,
                TARGET_DISPLAY[target],
                ha="center",
                va="center",
                fontsize=8,
            )

    y = header_bottom
    for cohort_index, cohort in enumerate(cohorts):
        cohort_top = y
        cohort_bottom = y - rows_per_cohort * row_height
        axis.text(
            (x_edges[0] + x_edges[1]) / 2,
            (cohort_top + cohort_bottom) / 2,
            cohort.replace(" ", "\n") if cohort == "Particle Tracking" else cohort,
            ha="center",
            va="center",
            fontsize=9,
        )
        for model in models:
            row_bottom = y - row_height
            sample = lookup[(split, cohort, model, metrics[0], targets[0])]
            if sample["source"] == "run":
                axis.add_patch(
                    Rectangle(
                        (x_edges[1], row_bottom),
                        x_edges[-1] - x_edges[1],
                        row_height,
                        facecolor="#eaf3fb",
                        edgecolor="none",
                        zorder=-1,
                    )
                )
            axis.text(
                (x_edges[1] + x_edges[2]) / 2,
                row_bottom + row_height / 2,
                model,
                ha="center",
                va="center",
                fontsize=8.5,
            )
            axis.text(
                (x_edges[2] + x_edges[3]) / 2,
                row_bottom + row_height / 2,
                f"{sample['n_samples']:,}",
                ha="center",
                va="center",
                fontsize=7.5,
            )
            for metric_index, metric in enumerate(metrics):
                start = numeric_start + metric_index * len(targets)
                for target_index, target in enumerate(targets):
                    cell = lookup[(split, cohort, model, metric, target)]
                    column = start + target_index
                    axis.text(
                        (x_edges[column] + x_edges[column + 1]) / 2,
                        row_bottom + row_height / 2,
                        _format_value(metric, cell["value"]),
                        ha="center",
                        va="center",
                        fontsize=7.3,
                        fontweight="bold" if cell["is_best"] else "normal",
                    )
            y = row_bottom
        axis.plot([x_edges[0], x_edges[-1]], [y, y], color="black", linewidth=0.55)

    axis.plot([x_edges[0], x_edges[-1]], [total_height, total_height], color="black", linewidth=0.9)
    axis.plot([x_edges[0], x_edges[-1]], [header_bottom, header_bottom], color="black", linewidth=0.9)
    for edge in (1, 2, 3, 3 + len(targets), 3 + 2 * len(targets), len(widths)):
        axis.plot([x_edges[edge], x_edges[edge]], [0, total_height], color="black", linewidth=0.55)
    title = (
        "wa-hls4ml Table 4 comparison: test set and subsets"
        if split == "test"
        else "wa-hls4ml Table 5 comparison: exemplar architectures"
    )
    axis.set_title(title, fontsize=13, pad=10, fontfamily="serif")
    axis.text(
        x_edges[-1] / 2,
        -0.22,
        "Bold = best displayed value within each cohort/metric/target. "
        "Blue rows are ll-hls4ml runs; N exposes partial-cohort comparisons.",
        ha="center",
        va="center",
        fontsize=8,
    )
    figure.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def _write_report(
    path: Path,
    paper: dict,
    runs: list[dict],
    coverage: dict,
) -> None:
    run_lines = "\n".join(
        f"- **{run['display_name']}**: `{run['run_dir']}` "
        f"(model `{run['metadata'].get('model')}`, seed {run['metadata'].get('seed')}, "
        f"scale {run['metadata'].get('scale_percent')}%)"
        for run in runs
    )
    coverage_lines = []
    for model, cohort_coverage in coverage.items():
        selected = sum(row["selected"] for row in cohort_coverage.values())
        total = sum(row["paper_total"] for row in cohort_coverage.values())
        details = ", ".join(
            f"{cohort} {values['selected']}/{values['paper_total']}"
            for cohort, values in cohort_coverage.items()
        )
        coverage_lines.append(f"- **{model}**: {selected}/{total} exemplars ({details})")
    source = paper["source"]
    report = f"""# wa-hls4ml paper comparison

This is an **additional** report generated from existing `predictions.csv` files;
the original run reports and metrics are unchanged.

![Test comparison](wahls4ml_test_comparison.png)

![Exemplar comparison](wahls4ml_exemplar_comparison.png)

## Inputs

{run_lines}

Static baselines are transcribed from Tables {source['test_table']} and
{source['exemplar_table']} of [{source['title']}]({source['url']})
({source['version']}). The editable transcription is
`src/ll_hls4ml/reporting/wahls4ml_paper_results.json`.

## Comparability boundary

The metric definitions are identical to the paper: per-target R², SMAPE with
`+1` in the denominator, and RMSE on original-scale absolute values. Cohorts are
also aligned: **All**, all five fully-connected families as **Dense**,
**Conv1D**, **Conv2D**, and the seven exemplar architectures.

The sample membership and training scale are not identical. The paper models
were trained on the full 478,220-sample training set and evaluated on all 102,484
synthetic test samples plus all 887 exemplars. The displayed ll-hls4ml runs use
their persisted prediction subsets. The `N` column must therefore accompany any
shared table; these are benchmark-context comparisons, not paired head-to-head
evaluations on identical samples.

## Exemplar identity audit

Every displayed exemplar prediction was joined losslessly by UUID to
`exemplar_models.json`, and its six ground-truth targets were checked against the
label record. Coverage is:

{chr(10).join(coverage_lines)}

Missing exemplars were not assigned by name heuristics or silently included.

## Files

- `wahls4ml_comparison.csv`: tidy, machine-readable values and best-cell flags
- `wahls4ml_test_comparison.png`: paper-style Table 4 extension
- `wahls4ml_exemplar_comparison.png`: paper-style Table 5 extension
- `comparison_manifest.json`: input provenance and coverage
"""
    path.write_text(report)


def generate_wahls4ml_comparison(
    runs: Mapping[str, Path],
    exemplar_labels: Path,
    output_dir: Path,
    paper_results_path: Path = PAPER_RESULTS_PATH,
) -> dict:
    """Generate CSV, PNG, Markdown, and provenance files for one or more runs."""
    paper = _load_json(paper_results_path)
    if not isinstance(paper, dict):
        raise ValueError(f"Expected an object in {paper_results_path}")
    _validate_paper_results(paper)
    reserved_names = set(paper["model_order"])
    conflicts = reserved_names.intersection(runs)
    if conflicts:
        raise ValueError(
            "Run display names conflict with paper model names: "
            + ", ".join(sorted(conflicts))
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    records = _paper_records(paper)
    coverage = {}
    run_manifest = []
    for display_name, raw_run_dir in runs.items():
        run_dir = Path(raw_run_dir).resolve()
        metadata = _run_metadata(run_dir)
        run_rows, run_coverage = _run_records(
            display_name,
            _load_prediction_rows(run_dir),
            paper["target_order"],
            exemplar_labels,
        )
        records.extend(run_rows)
        coverage[display_name] = run_coverage
        run_manifest.append(
            {
                "display_name": display_name,
                "run_dir": str(run_dir),
                "metadata": metadata,
            }
        )
    _mark_best(records)
    split_rank = {"test": 0, "exemplar": 1}
    cohort_rank = {
        split: {cohort: index for index, cohort in enumerate(paper[split])}
        for split in ("test", "exemplar")
    }
    model_order = paper["model_order"] + list(runs)
    model_rank = {model: index for index, model in enumerate(model_order)}
    metric_rank = {metric: index for index, metric in enumerate(paper["metric_order"])}
    target_rank = {target: index for index, target in enumerate(paper["target_order"])}
    records.sort(
        key=lambda row: (
            split_rank[row["split"]],
            cohort_rank[row["split"]][row["cohort"]],
            model_rank[row["model"]],
            metric_rank[row["metric"]],
            target_rank[row["target"]],
        )
    )
    _write_comparison_csv(output_dir / "wahls4ml_comparison.csv", records)
    _render_table(
        output_dir / "wahls4ml_test_comparison.png",
        "test",
        records,
        list(paper["test"]),
        model_order,
        paper["target_order"],
        paper["metric_order"],
    )
    _render_table(
        output_dir / "wahls4ml_exemplar_comparison.png",
        "exemplar",
        records,
        list(paper["exemplar"]),
        model_order,
        paper["target_order"],
        paper["metric_order"],
    )
    manifest = {
        "paper_source": paper["source"],
        "paper_results_path": str(paper_results_path.resolve()),
        "exemplar_labels": str(exemplar_labels.resolve()),
        "runs": run_manifest,
        "exemplar_coverage": coverage,
    }
    (output_dir / "comparison_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    _write_report(
        output_dir / "REPORT.md", paper, run_manifest, coverage
    )
    return manifest
