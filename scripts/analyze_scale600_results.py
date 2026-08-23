#!/usr/bin/env python3
"""Build matched scale-600 diagnostic tables and compact figures."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


TARGETS = ["lut", "ff", "dsp", "bram", "cycles_max", "interval_max"]
TARGET_LABELS = ["LUT", "FF", "DSP", "BRAM", "Cycles", "II"]
RUNS = {
    "Hierarchy": "hierarchical_scale600_seed42",
    "Fusion": "hierarchical_high_level_fusion_scale600_seed42",
    "Paper-GATv2": "paper_high_level_gatv2_scale600_seed42",
    "Paper-Transformer": "paper_transformer_scale600_seed42",
}


def _predictions(run_dir: Path, split: str) -> pd.DataFrame:
    rows = pd.read_csv(run_dir / "predictions.csv")
    return rows.loc[rows["split"] == split].set_index("tensor_path").sort_index()


def _sample_smape(rows: pd.DataFrame) -> np.ndarray:
    values = []
    for target in TARGETS:
        actual = rows[f"target_{target}"].to_numpy()
        predicted = rows[f"prediction_{target}"].to_numpy()
        values.append(
            200.0 * np.abs(actual - predicted)
            / (np.abs(actual) + np.abs(predicted) + 1.0)
        )
    return np.stack(values, axis=1).mean(axis=1)


def _bootstrap_mean(values: np.ndarray, seed: int = 42) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(10_000, len(values)))
    means = values[indices].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return float(low), float(high)


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _current_tables(root: Path, output: Path) -> dict[str, pd.DataFrame]:
    overview = []
    targets = []
    families = []
    predictions = {}
    split_hashes = set()
    for model, directory in RUNS.items():
        run_dir = root / directory
        config = json.loads((run_dir / "resolved_config.json").read_text())
        accounting = json.loads((run_dir / "experiment_accounting.json").read_text())
        split_hashes.add(config["split_sha256"])
        macro = pd.read_csv(run_dir / "macro_metrics.csv")
        metrics = pd.read_csv(run_dir / "metrics.csv")
        for split in ("test", "exemplar"):
            predictions[(model, split)] = _predictions(run_dir, split)
            overall = macro[
                (macro["split"] == split)
                & (macro["kernel_family"] == "all")
                & (macro["scope"] == "overall")
            ].iloc[0]
            overview.append({
                "model": model,
                "split": split,
                "n_samples": int(overall["n_samples"]),
                "macro_smape": float(overall["smape"]),
                "macro_r2": float(overall["r2"]),
                "parameters": int(config["parameter_count"]),
                "training_hours": accounting["cumulative_training_seconds"] / 3600,
                "peak_gpu_memory_mb": accounting["peak_gpu_memory_mb"],
                "best_epoch": json.loads((run_dir / "summary.json").read_text())["best_epoch"],
            })
            selected = metrics[
                (metrics["split"] == split) & (metrics["kernel_family"] == "all")
            ]
            for row in selected.itertuples():
                targets.append({
                    "model": model,
                    "split": split,
                    "target": row.target,
                    "smape": row.smape,
                    "r2": row.r2,
                    "rmse": row.rmse,
                })
        selected = macro[
            (macro["split"] == "test")
            & (macro["scope"] == "overall")
            & (macro["kernel_family"] != "all")
        ]
        for row in selected.itertuples():
            families.append({
                "model": model,
                "kernel_family": row.kernel_family,
                "n_samples": int(row.n_samples),
                "smape": row.smape,
                "r2": row.r2,
            })
    if len(split_hashes) != 1:
        raise ValueError(f"Runs do not share one split hash: {sorted(split_hashes)}")
    for split in ("test", "exemplar"):
        reference = predictions[("Hierarchy", split)]
        for model in RUNS:
            candidate = predictions[(model, split)]
            if not reference.index.equals(candidate.index):
                raise ValueError(f"Prediction membership differs for {model}/{split}")
            for target in TARGETS:
                if not np.array_equal(
                    reference[f"target_{target}"].to_numpy(),
                    candidate[f"target_{target}"].to_numpy(),
                ):
                    raise ValueError(f"Targets differ for {model}/{split}/{target}")
    tables = {
        "overview": pd.DataFrame(overview),
        "targets": pd.DataFrame(targets),
        "families": pd.DataFrame(families),
    }
    for name, table in tables.items():
        table.to_csv(output / f"{name}.csv", index=False)
    return {**tables, "predictions": predictions}


def _paired_current(predictions: dict, output: Path) -> pd.DataFrame:
    rows = []
    pairs = [
        ("Hierarchy", "Fusion"),
        ("Paper-GATv2", "Fusion"),
        ("Paper-Transformer", "Fusion"),
    ]
    for split in ("test", "exemplar"):
        for reference, candidate in pairs:
            old = _sample_smape(predictions[(reference, split)])
            new = _sample_smape(predictions[(candidate, split)])
            delta = new - old
            low, high = _bootstrap_mean(delta)
            rows.append({
                "split": split,
                "candidate": candidate,
                "reference": reference,
                "candidate_minus_reference_smape": delta.mean(),
                "bootstrap_95_low": low,
                "bootstrap_95_high": high,
                "candidate_win_fraction": float((delta < 0).mean()),
                "n_samples": len(delta),
            })
    table = pd.DataFrame(rows)
    table.to_csv(output / "paired_current.csv", index=False)
    return table


def _matched_scaling(repo_root: Path, root: Path, output: Path) -> pd.DataFrame:
    comparisons = [
        ("Hierarchy", 2742, repo_root / "artifacts/results/ll_hls4ml_hierarchy_fusion_scale200_results/hierarchical_scale200_seed42", root / RUNS["Hierarchy"]),
        ("Hierarchy", 5166, repo_root / "artifacts/results/hierarchical_bottom_up_v3_seed42", root / RUNS["Hierarchy"]),
        ("Fusion", 2742, repo_root / "artifacts/results/ll_hls4ml_hierarchy_fusion_scale200_results/hierarchical_high_level_fusion_scale200_seed42", root / RUNS["Fusion"]),
    ]
    rows = []
    for model, old_train_n, old_dir, new_dir in comparisons:
        for split in ("test", "exemplar"):
            old = _predictions(old_dir, split)
            new = _predictions(new_dir, split).loc[old.index]
            old_score = _sample_smape(old)
            new_score = _sample_smape(new)
            delta = new_score - old_score
            low, high = _bootstrap_mean(delta)
            rows.append({
                "model": model,
                "split": split,
                "old_train_n": old_train_n,
                "new_train_n": 8265,
                "matched_n": len(old),
                "old_smape": old_score.mean(),
                "new_smape_on_old_membership": new_score.mean(),
                "delta_smape": delta.mean(),
                "bootstrap_95_low": low,
                "bootstrap_95_high": high,
            })
    table = pd.DataFrame(rows)
    table.to_csv(output / "matched_scaling.csv", index=False)
    return table


def _heatmap(table: pd.DataFrame, output: Path) -> None:
    pivot = table.pivot(index="model", columns="kernel_family", values="smape").loc[list(RUNS)]
    figure, axis = plt.subplots(figsize=(11, 4.5), constrained_layout=True)
    image = axis.imshow(pivot.to_numpy(), cmap="YlOrRd", aspect="auto", vmin=0, vmax=90)
    axis.set_xticks(range(len(pivot.columns)), [name.replace("_", "\n") for name in pivot.columns])
    axis.set_yticks(range(len(pivot.index)), pivot.index)
    for row in range(len(pivot.index)):
        for column in range(len(pivot.columns)):
            value = pivot.iloc[row, column]
            axis.text(column, row, f"{value:.1f}", ha="center", va="center", color="white" if value > 48 else "black")
    axis.set_title("Scale-600 test SMAPE by kernel family (lower is better)")
    figure.colorbar(image, ax=axis, label="Macro SMAPE [%]")
    figure.savefig(output / "test_family_smape.svg", format="svg")
    figure.savefig(output / "test_family_smape.png", dpi=180)
    plt.close(figure)


def _target_bars(table: pd.DataFrame, output: Path) -> None:
    figure, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True, constrained_layout=True)
    width = 0.19
    x = np.arange(len(TARGETS))
    for axis, split in zip(axes, ("test", "exemplar")):
        for offset, model in enumerate(RUNS):
            selected = table[(table["split"] == split) & (table["model"] == model)].set_index("target").loc[TARGETS]
            axis.bar(x + (offset - 1.5) * width, selected["smape"], width, label=model)
        axis.set_title(f"{split.title()} target-wise SMAPE")
        axis.set_ylabel("SMAPE [%]")
        axis.grid(axis="y", alpha=0.25)
    axes[0].legend(ncol=2)
    axes[-1].set_xticks(x, TARGET_LABELS)
    figure.savefig(output / "target_smape.svg", format="svg")
    figure.savefig(output / "target_smape.png", dpi=180)
    plt.close(figure)


def _fusion_delta_figure(families: pd.DataFrame, targets: pd.DataFrame, output: Path) -> None:
    family = families.pivot(index="kernel_family", columns="model", values="smape")
    family_delta = family["Fusion"] - family["Hierarchy"]
    target = targets[targets["split"] == "test"].pivot(index="target", columns="model", values="smape")
    target_delta = (target["Fusion"] - target["Hierarchy"]).loc[TARGETS]
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.6), constrained_layout=True)
    for axis, values, title, labels in (
        (axes[0], family_delta, "Fusion − hierarchy by family", [x.replace("_", "\n") for x in family_delta.index]),
        (axes[1], target_delta, "Fusion − hierarchy by target", TARGET_LABELS),
    ):
        colors = ["#2b8cbe" if value < 0 else "#d95f0e" for value in values]
        axis.bar(range(len(values)), values, color=colors)
        axis.axhline(0, color="black", linewidth=0.8)
        axis.set_xticks(range(len(values)), labels)
        axis.set_ylabel("SMAPE-point difference")
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.2)
    figure.savefig(output / "fusion_deltas.svg", format="svg")
    figure.savefig(output / "fusion_deltas.png", dpi=180)
    plt.close(figure)


def _exemplar_heatmap(root: Path, output: Path) -> None:
    comparison = pd.read_csv(root / "wahls4ml_comparison/wahls4ml_comparison.csv")
    selected = comparison[
        (comparison["split"] == "exemplar")
        & (comparison["metric"] == "smape")
    ]
    pivot = selected.groupby(["model", "cohort"])["value"].mean().unstack()
    order = ["MLP", "GNN", "Transformer", *RUNS]
    pivot = pivot.loc[order]
    figure, axis = plt.subplots(figsize=(11, 6.5), constrained_layout=True)
    image = axis.imshow(pivot.to_numpy(), cmap="YlOrRd", aspect="auto", vmin=40, vmax=150)
    axis.set_xticks(range(len(pivot.columns)), [name.replace(" ", "\n") for name in pivot.columns])
    axis.set_yticks(range(len(pivot.index)), pivot.index)
    for row in range(len(pivot.index)):
        for column in range(len(pivot.columns)):
            value = pivot.iloc[row, column]
            axis.text(column, row, f"{value:.1f}", ha="center", va="center", color="white" if value > 105 else "black")
    axis.set_title("Exemplar architecture macro SMAPE: paper context and scale-600 runs")
    figure.colorbar(image, ax=axis, label="Macro SMAPE [%]")
    figure.savefig(output / "exemplar_architecture_smape.svg", format="svg")
    figure.savefig(output / "exemplar_architecture_smape.png", dpi=180)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path("artifacts/results/ll_hls4ml_hierarchy_fusion_scale600_results"),
    )
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    root = args.results_root.resolve()
    output = (args.output_dir or root / "comprehensive_analysis").resolve()
    output.mkdir(parents=True, exist_ok=True)
    tables = _current_tables(root, output)
    _paired_current(tables["predictions"], output)
    _matched_scaling(repo_root, root, output)
    _heatmap(tables["families"], output)
    _target_bars(tables["targets"], output)
    _fusion_delta_figure(tables["families"], tables["targets"], output)
    _exemplar_heatmap(root, output)
    print(f"Wrote scale-600 analysis to {output}")


if __name__ == "__main__":
    main()
