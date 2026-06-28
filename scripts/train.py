#!/usr/bin/env python3
"""Torchrun entry point for R-GCN training."""

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
from ll_hls4ml.data.splits import random_train_val_split
from ll_hls4ml.data.vocab import load_vocab
from ll_hls4ml.models.registry import build
from ll_hls4ml.training import make_loader
from ll_hls4ml.training.distributed import cleanup_ddp, is_main_process, setup_from_env
from ll_hls4ml.training.loops import _json_converter, fit, _persist_on_cpu, validate_one_epoch


def main() -> None:
    parser = argparse.ArgumentParser(description="R-GCN distributed training")
    parser.add_argument("--config", required=True, help="Path to JSON training config")
    args = parser.parse_args()

    with open(args.config) as f:
        config = json.load(f)

    _rank, world_size, local_rank = setup_from_env()
    distributed = world_size > 1

    seed = config.get("seed", 42)
    torch.manual_seed(seed)

    use_cuda = torch.cuda.is_available() and torch.cuda.device_count() > 0
    if distributed and use_cuda:
        device = torch.device(f"cuda:{local_rank}")
    elif use_cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    max_per_kernel_type = config["max_per_kernel_type"]
    kernel_types = list[str](max_per_kernel_type.keys())

    tensor_dir = Path(config["tensor_dir"])
    vocab, max_pos, _ = load_vocab(tensor_dir / "vocab.json")
    vocab_sizes = {k: len(v) for k, v in vocab.items()}

    val_frac = config.get("val_frac", 0.2)
    test_frac = config.get("test_frac", 0.1)
    ds = HeteroGraphDataset(tensor_dir, types=kernel_types, max_per_type=max_per_kernel_type, silent=False)
    train_val_ds, test_ds = random_train_val_split(ds, val_fraction=test_frac, seed=seed)
    train_ds, val_ds = random_train_val_split(train_val_ds, val_fraction=val_frac, seed=seed)

    batch_size = config["batch_size"]
    train_loader = make_loader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = make_loader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = make_loader(test_ds, batch_size=batch_size, shuffle=False)

    model = build(
        "rgcn",
        node_vocab_sizes=vocab_sizes,
        edge_pos_vocab_size=max_pos,
        hidden_dim=config["hidden_dim"],
        num_layers=config["num_layers"],
        dropout=config["dropout"],
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=config.get("lr", 1e-3))
    criterion = nn.HuberLoss(reduction="mean", delta=1.0)

    checkpoint_dir = Path(config["checkpoint_dir"])
    results_dir = Path(config.get("results_dir", checkpoint_dir.parent))

    try:
        model, y_means, y_stds, training_history, best_val_loss, best_epoch = fit(
            model, train_loader, val_loader,
            epochs=config["epochs"],
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            patience=config.get("patience", 50),
            evaluation_metric=config.get("evaluation_metric", "val_loss"),
            mode=config.get("mode", "min"),
            restore_best_weights=config.get("restore_best_weights", True),
            verbose=config.get("verbose", 2),
            experiment_name=config.get("experiment_name", "fit"),
            checkpoint_dir=checkpoint_dir,
            distributed=distributed
        )

        print("Evaluating on test set...")
        test_loss, test_preds, test_targets, _, _ = validate_one_epoch(
            model, test_loader, criterion, device, y_means, y_stds
        )

        training_history = _persist_on_cpu(
            {
                **training_history,
                "y_means": y_means,
                "y_stds": y_stds,
                "best_val_loss": best_val_loss,
                "best_epoch": best_epoch,
                "test_loss": test_loss,
                "test_preds": test_preds,
                "test_targets": test_targets,
            }
        )

        th_path = results_dir / f"training_histories.json"
        with open(th_path, "w") as f:
            json.dump(training_history, f, default=_json_converter)

        print(
            f"Test loss: {test_loss:.4f}, "
            f"Best validation loss: {best_val_loss:.4f}, "
            f"Inductive-Transductive gap: {(test_loss - best_val_loss):.4f}"
        )

    finally:
        cleanup_ddp()

    if is_main_process() and training_history:
        combined_path = results_dir / "training_histories.json"
        with open(combined_path, "w") as f:
            json.dump(training_history, f, default=_json_converter)
        print(f"Wrote combined training histories to {combined_path}")


if __name__ == "__main__":
    main()
