#!/usr/bin/env python3
"""Train matched wa-hls4ml layer/config GNN baselines on saved CDFG splits."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import os
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch

os.environ.setdefault("MPLCONFIGDIR", "/tmp/ll-hls4ml-matplotlib")
_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))
sys.path.insert(0, str(_REPO_ROOT / "scripts"))

from ll_hls4ml.data.high_level import (
    HighLevelLayerDataset,
    PROCESSED_FEATURE_DIM,
    build_high_level_cache,
    feature_statistics,
)
from ll_hls4ml.io.schema import LABEL_KEYS
from ll_hls4ml.models.high_level import HighLevelLayerGNN
from ll_hls4ml.reporting.accounting import (
    cohort_membership,
    hurdle_confusion_rows,
    split_sha256,
)
from ll_hls4ml.training import compute_target_z_stats, make_loader
from ll_hls4ml.training.loops import fit
from ll_hls4ml.training.targets import LogHuberHurdleLoss
from train import (
    _git_state,
    _json_converter,
    _metric_rows,
    _predict,
    _prediction_rows,
    _write_csv,
    _write_report,
)


SCALE_TAGS = {25: "025", 50: "050", 100: "100", 200: "200"}


def _source_manifest(results_root: Path, scale: int) -> Path:
    matches = sorted(results_root.glob(f"*scale{SCALE_TAGS[scale]}_seed42"))
    if len(matches) != 1:
        raise ValueError(f"Expected one CDFG scale{scale} run, found {matches}")
    return matches[0] / "split_manifest.json"


def _paths(manifest: dict, split: str) -> list[str]:
    return [row["tensor_path"] for row in manifest[split]]


def _seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _evaluate(model, datasets, manifests, device, batch_size, num_workers):
    metric_rows = []
    prediction_rows = []
    for split in ("test", "exemplar"):
        loader = make_loader(
            datasets[split],
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        )
        predictions, targets, latency = _predict(model, loader, device)
        metric_rows.extend(_metric_rows(split, predictions, targets, latency))
        families = np.asarray(datasets[split].families)
        for family in sorted(set(families)):
            mask = families == family
            metric_rows.extend(
                _metric_rows(
                    split,
                    predictions[mask],
                    targets[mask],
                    latency,
                    kernel_family=family,
                )
            )
        prediction_rows.extend(
            _prediction_rows(split, manifests[split], predictions, targets)
        )
    return metric_rows, prediction_rows


def _run_one(args, cache, scale: int, seed: int, device: torch.device):
    experiment = args.experiment_name or f"high_level_{args.encoder}_scale{SCALE_TAGS[scale]}_seed{seed}"
    run_dir = args.output_root / experiment
    complete_path = run_dir / "summary.json"
    if complete_path.exists() and not args.force:
        print(f"Skipping complete run {experiment}", flush=True)
        return json.loads(complete_path.read_text())
    run_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = args.manifest or _source_manifest(args.cdfg_results_root, scale)
    manifest = json.loads(manifest_path.read_text())
    membership = cohort_membership(manifest)
    manifest_hash = split_sha256(manifest)
    train_paths = _paths(manifest, "train")
    means, stds = feature_statistics(cache, train_paths)
    datasets = {
        split: HighLevelLayerDataset(cache, _paths(manifest, split), means, stds)
        for split in ("train", "validation", "test", "exemplar")
    }
    y_means, y_stds = compute_target_z_stats(datasets["train"])
    model = HighLevelLayerGNN(
        input_dim=PROCESSED_FEATURE_DIM,
        y_means=y_means,
        y_stds=y_stds,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        heads=args.heads,
        dropout=args.dropout,
        encoder=args.encoder,
        hurdle_heads=True,
        hurdle_prediction_mode="threshold",
    )
    criterion = LogHuberHurdleLoss(
        y_means,
        y_stds,
        delta=args.log_huber_delta,
        classification_weight=args.hurdle_classification_weight,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    train_loader = make_loader(
        datasets["train"],
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    validation_loader = make_loader(
        datasets["validation"],
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    checkpoint_dir = run_dir / "checkpoints"
    backup = checkpoint_dir / f"{experiment}_backup.pt"
    resolved_config = {
        "experiment_name": experiment,
        "model": "high_level_gatv2" if args.encoder == "gatv2" else "high_level_sage",
        "representation": "wa-hls4ml layer/config graph",
        "encoder": args.encoder,
        "scale_percent": scale,
        "seed": seed,
        "hidden_dim": args.hidden_dim,
        "num_layers": args.num_layers,
        "heads": args.heads,
        "dropout": args.dropout,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "epochs": args.epochs,
        "patience": args.patience,
        "precision": args.precision,
        "validation_cadence_epochs": 1,
        "checkpoint_cadence_epochs": args.backup_interval,
        "loss": "log_huber_hurdle",
        "hurdle_prediction_mode": "threshold",
        "source_manifest": str(manifest_path),
        "high_level_cache": str(args.cache),
        "target_contract": "ll-hls4ml post-logic tensor targets",
        "device": str(device),
        "torch_version": torch.__version__,
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "split_sha256": manifest_hash,
        "cohort_membership": membership,
        "ll_hls4ml_git": _git_state(_REPO_ROOT),
    }
    (run_dir / "resolved_config.json").write_text(
        json.dumps(resolved_config, indent=2)
    )
    (run_dir / "split_manifest.json").write_text(json.dumps(manifest, indent=2))
    started = time.perf_counter()
    model = fit(
        model,
        train_loader,
        validation_loader,
        epochs=args.epochs,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=None,
        device=device,
        patience=args.patience,
        mode="min",
        restore_best_weights=True,
        verbose=args.verbose,
        experiment_name=experiment,
        checkpoint_dir=checkpoint_dir,
        resume_from_backup=backup if backup.exists() else None,
        early_stopping_metric="smape",
        checkpoint_interval=args.backup_interval,
        precision=args.precision,
    )
    wall_seconds = time.perf_counter() - started
    metric_rows, prediction_rows = _evaluate(
        model,
        datasets,
        manifest,
        device,
        args.batch_size,
        args.num_workers,
    )
    _write_csv(run_dir / "metrics.csv", metric_rows)
    _write_csv(run_dir / "predictions.csv", prediction_rows)
    base_model = model
    hurdle_rows = hurdle_confusion_rows(prediction_rows)
    _write_csv(run_dir / "hurdle_confusion.csv", hurdle_rows)
    accounting = {
        "split_sha256": manifest_hash,
        "cohort_membership": membership,
        "hurdle_confusion": hurdle_rows,
        "validation_cadence_epochs": 1,
        "checkpoint_cadence_epochs": args.backup_interval,
        "cumulative_training_seconds": getattr(
            base_model, "cumulative_training_seconds", wall_seconds
        ),
        "resumed_from_epoch": getattr(base_model, "resumed_from_epoch", 0),
    }
    (run_dir / "experiment_accounting.json").write_text(
        json.dumps(accounting, indent=2)
    )
    resolved_config["wall_seconds"] = wall_seconds
    resolved_config["cumulative_training_seconds"] = accounting[
        "cumulative_training_seconds"
    ]
    summary = {
        "resolved_config": resolved_config,
        "sizes": {split: len(dataset) for split, dataset in datasets.items()},
        "best_epoch": base_model.best_epoch,
        "best_metric": base_model.best_metric,
        "training_history": base_model.training_history,
        "metrics": metric_rows,
    }
    complete_path.write_text(json.dumps(summary, indent=2, default=_json_converter))
    (run_dir / "resolved_config.json").write_text(
        json.dumps(resolved_config, indent=2)
    )
    _write_report(
        run_dir,
        experiment,
        resolved_config,
        summary["sizes"],
        metric_rows,
        accounting,
    )
    print(
        f"Completed {experiment}: best epoch {base_model.best_epoch}, "
        f"{wall_seconds / 60:.2f} min",
        flush=True,
    )
    model.cpu()
    del model, optimizer, criterion, train_loader, validation_loader, datasets
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return summary


def _write_scaling_summary(output_root: Path):
    rows = []
    for path in sorted(output_root.glob("high_level_*_scale*_seed*/summary.json")):
        result = json.loads(path.read_text())
        config = result["resolved_config"]
        for split in ("test", "exemplar"):
            values = [
                row["smape"]
                for row in result["metrics"]
                if row["split"] == split and row["kernel_family"] == "all"
            ]
            rows.append(
                {
                    "encoder": config["encoder"],
                    "scale_percent": config["scale_percent"],
                    "seed": config["seed"],
                    "split": split,
                    "macro_smape": float(np.mean(values)),
                    "best_epoch": result["best_epoch"],
                    "best_validation_smape": result["best_metric"],
                    "wall_seconds": config["wall_seconds"],
                }
            )
    if rows:
        _write_csv(output_root / "scaling_summary.csv", rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tensor-root", type=Path, default=Path("../data/tensors"))
    parser.add_argument("--label-root", type=Path, default=Path("../data/labels/wa-hls4ml"))
    parser.add_argument("--wa-gnn-dir", type=Path, default=Path("../wa_hls4ml_models/GNN"))
    parser.add_argument(
        "--cdfg-results-root",
        type=Path,
        default=Path("artifacts/results/ll_hls4ml_gatv2_scaling_results"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("artifacts/results/ll_hls4ml_high_level_scaling_results"),
    )
    parser.add_argument(
        "--cache",
        type=Path,
        default=Path("artifacts/cache/wa_high_level_scale200.pt"),
    )
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--experiment-name")
    parser.add_argument("--scales", nargs="+", type=int, choices=SCALE_TAGS, default=[25, 50, 100, 200])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--encoder", choices=("gatv2", "sage"), default="gatv2")
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--heads", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--log-huber-delta", type=float, default=0.35)
    parser.add_argument("--hurdle-classification-weight", type=float, default=0.25)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--precision", choices=("float32", "bf16"), default="float32")
    parser.add_argument("--verbose", type=int, default=10)
    parser.add_argument("--backup-interval", type=int, default=25)
    parser.add_argument("--build-cache-only", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    for name in ("tensor_root", "label_root", "wa_gnn_dir", "cdfg_results_root", "output_root", "cache"):
        setattr(args, name, getattr(args, name).resolve())
    if args.manifest is not None:
        args.manifest = args.manifest.resolve()
    args.output_root.mkdir(parents=True, exist_ok=True)
    if args.cache.exists():
        cache = torch.load(args.cache, weights_only=False)
        print(f"Loaded {len(cache['samples'])} samples from {args.cache}", flush=True)
    else:
        cache = build_high_level_cache(
            args.manifest or _source_manifest(args.cdfg_results_root, 200),
            args.label_root,
            args.tensor_root,
            args.wa_gnn_dir,
            args.cache,
        )
    if args.build_cache_only:
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}", flush=True)
    for scale in args.scales:
        for seed in args.seeds:
            _seed_everything(seed)
            _run_one(args, cache, scale, seed, device)
            _write_scaling_summary(args.output_root)


if __name__ == "__main__":
    main()
