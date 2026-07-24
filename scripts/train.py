#!/usr/bin/env python3
"""Config-driven training entry point for portable local/Colab/Kaggle runs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

os.environ.setdefault("MPLCONFIGDIR", "/tmp/ll-hls4ml-matplotlib")

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))

from ll_hls4ml.data.dataset import HeteroGraphDataset
from ll_hls4ml.data.splits import random_train_val_test_split
from ll_hls4ml.data.vocab import load_vocab
from ll_hls4ml.models.registry import build
from ll_hls4ml.training import compute_target_z_stats, make_loader
from ll_hls4ml.io.schema import LABEL_KEYS
from ll_hls4ml.training.distributed import (
    cleanup_ddp,
    is_main_process,
    setup_from_env,
    unwrap_model,
)
from ll_hls4ml.training.loops import _json_converter, fit
from ll_hls4ml.training.targets import (
    denormalize_target,
    relative_percentage_error,
    wahls4ml_metrics_raw,
)
from ll_hls4ml.viz.training import prediction_scatter_plots, rpe_box_plots


DISPLAY_LABELS = ["LUT", "FF", "DSP", "BRAM", "Cycles", "II"]
PAPER_ORDER = ["BRAM", "DSP", "FF", "LUT", "Cycles", "II"]


def _config_path(value: str, config_dir: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (config_dir / path).resolve()


def _model_from_config(config: dict, vocab_size: int, max_pos: int, train_ds):
    y_means, y_stds = compute_target_z_stats(train_ds)
    common = {
        "instruction_vocab_size": vocab_size,
        "y_means": y_means,
        "y_stds": y_stds,
        "hidden_dim": config.get("hidden_dim", 128),
        "num_layers": config.get("num_layers", 3),
        "dropout": config.get("dropout", 0.1),
        "pool": config.get("pool", "mean"),
    }
    model_name = config.get("model", "rgcn")
    if model_name == "rgcn":
        common["edge_pos_vocab_size"] = max_pos
        common["aggr"] = config.get("aggr", "mean")
    elif model_name == "mlp":
        common["num_var_embed_layers"] = config.get("num_var_embed_layers", 2)
        common["node_aggr"] = config.get("node_aggr", "concat")
    return build(model_name, **common)


def _git_state(repository: Path) -> dict[str, object]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=repository,
            capture_output=True,
            text=True,
            check=True,
        ).stdout
    )
    return {"commit": commit, "dirty": dirty}


def _tensor_snapshot_id(paths: list[Path], root: Path) -> str:
    digest = hashlib.sha256()
    for path in paths:
        stat = path.stat()
        digest.update(
            f"{path.relative_to(root)}|{stat.st_size}|{stat.st_mtime_ns}\n".encode()
        )
    return digest.hexdigest()[:16]


def _split_manifest(dataset, splits: dict[str, object], tensor_dir: Path) -> dict:
    manifest = {}
    for split_name, subset in splits.items():
        manifest[split_name] = [
            {
                "dataset_index": int(index),
                "kernel_family": dataset.type_of(index),
                "tensor_path": str(dataset.paths[index].relative_to(tensor_dir)),
            }
            for index in subset.indices
        ]
    return manifest


def _predict(model, loader, device) -> tuple[np.ndarray, np.ndarray, float]:
    base_model = unwrap_model(model)
    base_model.eval()
    predictions = []
    targets = []
    if device.type == "cuda":
        torch.cuda.synchronize()
    started = time.perf_counter()
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            prediction = denormalize_target(
                base_model(batch),
                base_model.y_means,
                base_model.y_stds,
            )
            predictions.append(prediction.cpu())
            targets.append(batch.y.view(-1, len(LABEL_KEYS)).cpu())
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    prediction_array = torch.cat(predictions).numpy()
    target_array = torch.cat(targets).numpy()
    return (
        np.clip(prediction_array, 0, None),
        target_array,
        elapsed / len(target_array),
    )


def _metric_rows(
    split_name: str,
    predictions: np.ndarray,
    targets: np.ndarray,
    inference_seconds_per_sample: float,
) -> list[dict]:
    metrics = wahls4ml_metrics_raw(
        torch.from_numpy(np.array(predictions, copy=True)),
        torch.from_numpy(np.array(targets, copy=True)),
    )
    rpe = relative_percentage_error(
        torch.from_numpy(np.array(predictions, copy=True)),
        torch.from_numpy(np.array(targets, copy=True)),
    ).numpy()
    rows = []
    for index, target_name in enumerate(LABEL_KEYS):
        rows.append(
            {
                "split": split_name,
                "target": target_name,
                "r2": float(metrics["r2"][index]),
                "smape": float(metrics["smape"][index]),
                "rmse": float(metrics["rmse"][index]),
                "rpe_median": float(np.median(rpe[:, index])),
                "rpe_mean": float(np.mean(rpe[:, index])),
                "rpe_q25": float(np.percentile(rpe[:, index], 25)),
                "rpe_q75": float(np.percentile(rpe[:, index], 75)),
                "inference_seconds_per_sample": inference_seconds_per_sample,
                "n_samples": len(targets),
            }
        )
    return rows


def _prediction_rows(
    split_name: str,
    manifest: list[dict],
    predictions: np.ndarray,
    targets: np.ndarray,
) -> list[dict]:
    rows = []
    for row_index, sample in enumerate(manifest):
        row = {"split": split_name, **sample}
        for target_index, target_name in enumerate(LABEL_KEYS):
            row[f"target_{target_name}"] = float(targets[row_index, target_index])
            row[f"prediction_{target_name}"] = float(
                predictions[row_index, target_index]
            )
        rows.append(row)
    return rows


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _save_plots(
    run_dir: Path,
    experiment_name: str,
    split_name: str,
    predictions: np.ndarray,
    targets: np.ndarray,
) -> None:
    import matplotlib.pyplot as plt

    figures_dir = run_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    title = f"{experiment_name}: {split_name}"
    figure, _ = rpe_box_plots(
        predictions,
        targets,
        DISPLAY_LABELS,
        ordering=PAPER_ORDER,
        title=title,
        show=False,
    )
    figure.savefig(
        figures_dir / f"{split_name}__rpe.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(figure)
    figure, _ = prediction_scatter_plots(
        predictions,
        targets,
        DISPLAY_LABELS,
        title=title,
        show=False,
    )
    figure.savefig(
        figures_dir / f"{split_name}__scatter.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(figure)


def _write_report(
    run_dir: Path,
    experiment_name: str,
    resolved_config: dict,
    sizes: dict[str, int],
    metric_rows: list[dict],
) -> None:
    test_rows = [row for row in metric_rows if row["split"] == "test"]
    table = [
        "| target | R² | SMAPE (%) | RMSE | median RPE (%) |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for row in test_rows:
        table.append(
            f"| {row['target']} | {row['r2']:.3f} | {row['smape']:.2f} | "
            f"{row['rmse']:.2f} | {row['rpe_median']:.2f} |"
        )
    report = f"""# {experiment_name}

Single-model wa-hls4ml-style evaluation generated by `scripts/train.py`.

- Model: `{resolved_config["model"]}`
- Device: `{resolved_config["device"]}`
- Tensor snapshot: `{resolved_config["tensor_snapshot_id"]}`
- Seed: {resolved_config["seed"]}
- Total wall time: {resolved_config["wall_seconds"]:.1f} seconds
- Split sizes: `{json.dumps(sizes, sort_keys=True)}`
- ll-hls4ml state: `{json.dumps(resolved_config["ll_hls4ml_git"])}`

## Test metrics

{chr(10).join(table)}

Per-target validation and test metrics are in `metrics.csv`. Exact split
membership is in `split_manifest.json`, per-sample predictions are in
`predictions.csv`, and RPE/scatter figures are in `figures/`.

This is only directly comparable with wa-hls4ml when dataset membership,
compiler/graph provenance, targets, and evaluation splits are aligned.
"""
    (run_dir / "REPORT.md").write_text(report)


def main() -> None:
    run_started = time.perf_counter()
    parser = argparse.ArgumentParser(description="Train an HLS surrogate model")
    parser.add_argument("--config", required=True, help="Path to JSON training config")
    args = parser.parse_args()

    config_file = Path(args.config).resolve()
    with config_file.open() as handle:
        config = json.load(handle)
    config_dir = config_file.parent

    _rank, world_size, local_rank = setup_from_env()
    distributed = world_size > 1
    main_process = is_main_process()

    seed = config.get("seed", 42)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{local_rank}" if distributed else "cuda")
    else:
        device = torch.device("cpu")

    tensor_dir = _config_path(config["tensor_dir"], config_dir)
    vocab_path = _config_path(
        config.get("vocab_path", str(tensor_dir / "vocab.json")),
        config_dir,
    )
    vocab, max_pos, _counts = load_vocab(vocab_path)

    max_per_type = config.get("max_per_kernel_type")
    kernel_types = list(max_per_type) if max_per_type else config.get("kernel_types")
    dataset = HeteroGraphDataset(
        tensor_dir,
        types=kernel_types,
        max_per_type=max_per_type,
        silent=not main_process,
    )
    train_ds, val_ds, test_ds = random_train_val_test_split(
        dataset,
        val_fraction=config.get("val_fraction", 0.15),
        test_fraction=config.get("test_fraction", 0.15),
        seed=seed,
    )
    if not train_ds or not val_ds or not test_ds:
        raise ValueError(
            "Train/validation/test split is empty; increase the dataset or fractions"
        )

    batch_size = config.get("batch_size", 4)
    num_workers = config.get("num_workers")
    train_loader = make_loader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        distributed=distributed,
    )
    val_loader = make_loader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    test_loader = make_loader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    model = _model_from_config(config, len(vocab), max_pos, train_ds)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.get("learning_rate", 1e-3),
        weight_decay=config.get("weight_decay", 0.0),
    )
    criterion = nn.HuberLoss(delta=config.get("huber_delta", 1.0))

    checkpoint_dir = _config_path(
        config.get("checkpoint_dir", "../artifacts/checkpoints"),
        config_dir,
    )
    results_dir = _config_path(
        config.get("results_dir", "../artifacts/results"),
        config_dir,
    )
    experiment_name = config.get("experiment_name", "baseline")
    run_dir = results_dir / experiment_name
    if main_process:
        run_dir.mkdir(parents=True, exist_ok=True)
    split_manifest = _split_manifest(
        dataset,
        {"train": train_ds, "validation": val_ds, "test": test_ds},
        tensor_dir,
    )
    resolved_config = {
        **config,
        "model": config.get("model", "rgcn"),
        "tensor_dir": str(tensor_dir),
        "vocab_path": str(vocab_path),
        "checkpoint_dir": str(checkpoint_dir),
        "checkpoint_path": str(
            checkpoint_dir / f"{experiment_name}_checkpoint.pt"
        ),
        "backup_checkpoint_path": str(
            checkpoint_dir / f"{experiment_name}_backup.pt"
        ),
        "results_dir": str(results_dir),
        "run_dir": str(run_dir),
        "device": str(device),
        "torch_version": torch.__version__,
        "seed": seed,
        "tensor_snapshot_id": _tensor_snapshot_id(dataset.paths, tensor_dir),
        "ll_hls4ml_git": _git_state(_REPO_ROOT),
    }
    if main_process:
        (run_dir / "resolved_config.json").write_text(
            json.dumps(resolved_config, indent=2)
        )
        (run_dir / "split_manifest.json").write_text(
            json.dumps(split_manifest, indent=2)
        )

    try:
        model = fit(
            model,
            train_loader,
            val_loader,
            epochs=config.get("epochs", 200),
            criterion=criterion,
            optimizer=optimizer,
            scheduler=None,
            device=device,
            patience=config.get("patience", 30),
            mode="min",
            restore_best_weights=True,
            verbose=config.get("verbose", 5),
            experiment_name=experiment_name,
            checkpoint_dir=checkpoint_dir,
            distributed=distributed,
        )

        if main_process:
            base_model = unwrap_model(model)
            sizes = {
                "train": len(train_ds),
                "validation": len(val_ds),
                "test": len(test_ds),
            }
            metric_rows = []
            prediction_rows = []
            for split_name, loader in (
                ("validation", val_loader),
                ("test", test_loader),
            ):
                predictions, targets, inference_latency = _predict(
                    model, loader, device
                )
                metric_rows.extend(
                    _metric_rows(
                        split_name,
                        predictions,
                        targets,
                        inference_latency,
                    )
                )
                prediction_rows.extend(
                    _prediction_rows(
                        split_name,
                        split_manifest[split_name],
                        predictions,
                        targets,
                    )
                )
                _save_plots(
                    run_dir,
                    experiment_name,
                    split_name,
                    predictions,
                    targets,
                )
            _write_csv(run_dir / "metrics.csv", metric_rows)
            _write_csv(run_dir / "predictions.csv", prediction_rows)
            result = {
                "resolved_config": resolved_config,
                "sizes": sizes,
                "target_log_mean": base_model.y_means.detach().cpu(),
                "target_log_std": base_model.y_stds.detach().cpu(),
                "metrics": metric_rows,
            }
            resolved_config["wall_seconds"] = time.perf_counter() - run_started
            result["resolved_config"] = resolved_config
            (run_dir / "resolved_config.json").write_text(
                json.dumps(resolved_config, indent=2)
            )
            output = run_dir / "summary.json"
            with output.open("w") as handle:
                json.dump(result, handle, indent=2, default=_json_converter)
            _write_report(
                run_dir,
                experiment_name,
                resolved_config,
                sizes,
                metric_rows,
            )
            print(f"Wrote result bundle to {run_dir}")
    finally:
        cleanup_ddp()


if __name__ == "__main__":
    main()
