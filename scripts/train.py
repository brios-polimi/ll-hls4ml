#!/usr/bin/env python3
"""Config-driven training entry point for portable local/Colab/Kaggle runs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))

from ll_hls4ml.data.dataset import HeteroGraphDataset
from ll_hls4ml.data.splits import random_train_val_test_split
from ll_hls4ml.data.vocab import load_vocab
from ll_hls4ml.models.registry import build
from ll_hls4ml.training import compute_target_z_stats, make_loader
from ll_hls4ml.training.distributed import (
    cleanup_ddp,
    is_main_process,
    setup_from_env,
    unwrap_model,
)
from ll_hls4ml.training.loops import _json_converter, fit, validate_one_epoch


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


def main() -> None:
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
            test_metrics = validate_one_epoch(model, test_loader, criterion, device)
            base_model = unwrap_model(model)
            results_dir.mkdir(parents=True, exist_ok=True)
            result = {
                "config": config,
                "sizes": {
                    "train": len(train_ds),
                    "validation": len(val_ds),
                    "test": len(test_ds),
                },
                "target_log_mean": base_model.y_means.detach().cpu(),
                "target_log_std": base_model.y_stds.detach().cpu(),
                "test_metrics": test_metrics,
            }
            output = results_dir / f"{experiment_name}.json"
            with output.open("w") as handle:
                json.dump(result, handle, indent=2, default=_json_converter)
            print(f"Wrote results to {output}")
    finally:
        cleanup_ddp()


if __name__ == "__main__":
    main()
