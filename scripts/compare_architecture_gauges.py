#!/usr/bin/env python3
"""Compare architecture probes at matched epoch and matched training wall time."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", "/tmp/hls-surrogate-lab-matplotlib")


def _read_csv(path: Path) -> list[dict]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _history_row(history: list[dict], key: str, limit: float) -> dict:
    eligible = [row for row in history if float(row[key]) <= limit]
    if not eligible:
        return history[0]
    return eligible[-1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    runs = []
    for run_dir in args.runs:
        config = json.loads((run_dir / "resolved_config.json").read_text())
        history = _read_csv(run_dir / "learning_curves.csv")
        runs.append(
            {
                "run_dir": run_dir.resolve(),
                "config": config,
                "history": history,
                "macro": _read_csv(run_dir / "macro_metrics.csv"),
            }
        )
    common_epoch = min(int(float(run["history"][-1]["epoch"])) for run in runs)
    common_wall = min(
        float(run["history"][-1]["cumulative_training_seconds"])
        for run in runs
    )
    rows = []
    for run in runs:
        config = run["config"]
        history = run["history"]
        selected = {
            "equal_epoch": _history_row(history, "epoch", common_epoch),
            "equal_wall_time": _history_row(
                history, "cumulative_training_seconds", common_wall
            ),
            "best_validation": min(
                history, key=lambda row: float(row["val_smape"])
            ),
        }
        test_macro = {
            row["scope"]: row
            for row in run["macro"]
            if row["split"] == "test" and row["kernel_family"] == "all"
        }
        for comparison, point in selected.items():
            rows.append(
                {
                    "experiment_name": config["experiment_name"],
                    "model": config["model"],
                    "comparison": comparison,
                    "epoch": int(float(point["epoch"])),
                    "cumulative_training_seconds": float(
                        point["cumulative_training_seconds"]
                    ),
                    "val_loss": float(point["val_loss"]),
                    "val_smape": float(point["val_smape"]),
                    "val_resource_smape": float(point["val_resource_smape"]),
                    "val_timing_smape": float(point["val_timing_smape"]),
                    "parameters": config.get("parameter_count"),
                    "peak_gpu_memory_mb": config.get("peak_gpu_memory_mb"),
                    "test_smape": test_macro.get("overall", {}).get("smape"),
                    "test_resource_smape": test_macro.get("resource", {}).get(
                        "smape"
                    ),
                    "test_timing_smape": test_macro.get("timing", {}).get(
                        "smape"
                    ),
                    "run_dir": str(run["run_dir"]),
                }
            )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(args.output_dir / "architecture_gauge_comparison.csv", rows)
    dispersion = []
    for model, comparison in sorted(
        {(row["model"], row["comparison"]) for row in rows}
    ):
        selected = [
            row
            for row in rows
            if row["model"] == model and row["comparison"] == comparison
        ]
        record = {
            "model": model,
            "comparison": comparison,
            "n_seeds": len(selected),
        }
        for metric in (
            "val_smape",
            "val_resource_smape",
            "val_timing_smape",
            "test_smape",
            "test_resource_smape",
            "test_timing_smape",
        ):
            values = [
                float(row[metric])
                for row in selected
                if row.get(metric) not in {None, ""}
            ]
            record[f"{metric}_mean"] = float(np.mean(values)) if values else None
            record[f"{metric}_std"] = float(np.std(values)) if values else None
        dispersion.append(record)
    _write_csv(args.output_dir / "seed_dispersion.csv", dispersion)

    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.0))
    for run in runs:
        label = run["config"]["model"]
        history = run["history"]
        axes[0].plot(
            [float(row["epoch"]) for row in history],
            [float(row["val_smape"]) for row in history],
            label=label,
        )
        axes[1].plot(
            [float(row["cumulative_training_seconds"]) for row in history],
            [float(row["val_smape"]) for row in history],
            label=label,
        )
    axes[0].set_xlabel("Epoch")
    axes[1].set_xlabel("Cumulative training wall time (s)")
    for axis in axes:
        axis.set_ylabel("Validation SMAPE (%)")
        axis.grid(True, linestyle=":", alpha=0.45)
        axis.legend()
    figure.tight_layout()
    figure.savefig(
        args.output_dir / "architecture_gauge_learning_curves.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(figure)
    print(
        f"Compared {len(runs)} runs at epoch {common_epoch} and "
        f"{common_wall:.1f} training seconds"
    )


if __name__ == "__main__":
    main()
