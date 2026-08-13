#!/usr/bin/env python3
"""Refresh reporting/accounting files from an existing result bundle."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys


_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))
sys.path.insert(0, str(_REPO_ROOT / "scripts"))

from ll_hls4ml.reporting.accounting import (
    cohort_membership,
    hurdle_confusion_rows,
    split_sha256,
)
from train import _write_csv, _write_report


def _read_csv(path: Path) -> list[dict]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def refresh(run_dir: Path) -> None:
    manifest = json.loads((run_dir / "split_manifest.json").read_text())
    summary = json.loads((run_dir / "summary.json").read_text())
    config = json.loads((run_dir / "resolved_config.json").read_text())
    signature_path = run_dir / "notebook_resume_signature.json"
    if signature_path.exists():
        signature = json.loads(signature_path.read_text())
        for field in (
            "scale_percent",
            "tensor_source_revision",
            "distributed_world_size",
            "epochs",
            "patience",
            "learning_rate",
            "weight_decay",
        ):
            if config.get(field) is None and signature.get(field) is not None:
                config[field] = signature[field]
    metric_rows = _read_csv(run_dir / "metrics.csv")
    prediction_rows = _read_csv(run_dir / "predictions.csv")
    numeric_fields = {
        "r2", "smape", "rmse", "rpe_median", "rpe_mean", "rpe_q25",
        "rpe_q75", "inference_seconds_per_sample",
    }
    for row in metric_rows:
        for field in numeric_fields:
            row[field] = float(row[field])
        row["n_samples"] = int(row["n_samples"])
    membership = cohort_membership(manifest)
    manifest_hash = split_sha256(manifest)
    hurdle_rows = hurdle_confusion_rows(prediction_rows)
    previous = {}
    accounting_path = run_dir / "experiment_accounting.json"
    if accounting_path.exists():
        previous = json.loads(accounting_path.read_text())
    accounting = {
        "split_sha256": manifest_hash,
        "cohort_membership": membership,
        "hurdle_confusion": hurdle_rows,
        "validation_cadence_epochs": config.get("validation_cadence_epochs", 1),
        "checkpoint_cadence_epochs": config.get("checkpoint_cadence_epochs", 5),
        "cumulative_training_seconds": previous.get(
            "cumulative_training_seconds",
            config.get("cumulative_training_seconds"),
        ),
        "resumed_from_epoch": previous.get("resumed_from_epoch"),
    }
    config["split_sha256"] = manifest_hash
    config["cohort_membership"] = membership
    config.setdefault("validation_cadence_epochs", 1)
    config.setdefault("checkpoint_cadence_epochs", 5)
    config["cumulative_training_seconds"] = accounting[
        "cumulative_training_seconds"
    ]
    (run_dir / "resolved_config.json").write_text(json.dumps(config, indent=2))
    (run_dir / "experiment_accounting.json").write_text(
        json.dumps(accounting, indent=2)
    )
    _write_csv(run_dir / "hurdle_confusion.csv", hurdle_rows)
    _write_report(
        run_dir,
        config.get("experiment_name", run_dir.name),
        config,
        summary["sizes"],
        metric_rows,
        accounting,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dirs", nargs="+", type=Path)
    args = parser.parse_args()
    for run_dir in args.run_dirs:
        refresh(run_dir.resolve())
        print(f"Refreshed {run_dir.resolve()}")


if __name__ == "__main__":
    main()
