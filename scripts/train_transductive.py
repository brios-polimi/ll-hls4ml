#!/usr/bin/env python3
"""Torchrun entry point for transductive vs inductive heterogeneous GAT training."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))

from ll_hls4ml.data.vocab import load_vocab
from ll_hls4ml.io.schema import LABEL_KEYS
from ll_hls4ml.models.registry import build
from ll_hls4ml.training.distributed import cleanup_ddp, is_main_process, setup_from_env
from ll_hls4ml.training.loops import _json_converter, transductive_vs_inductive_fit


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Transductive vs inductive heterogeneous GAT training"
    )
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

    tensor_dir = Path(config["tensor_dir"])
    vocab, max_pos, _ = load_vocab(tensor_dir / "vocab.json")
    model = build(
        "hetero_gat",
        instruction_vocab_size=len(vocab),
        edge_pos_vocab_size=max_pos,
        y_means=torch.zeros(len(LABEL_KEYS)),
        y_stds=torch.ones(len(LABEL_KEYS)),
        hidden_dim=config["hidden_dim"],
        num_layers=config["num_layers"],
        dropout=config["dropout"],
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=config.get("lr", 1e-3))

    checkpoint_dir = Path(config["checkpoint_dir"])
    results_dir = Path(config.get("results_dir", checkpoint_dir.parent))

    try:
        histories = transductive_vs_inductive_fit(
            original_model=model,
            max_per_kernel_type=config["max_per_kernel_type"],
            tensor_dir=tensor_dir,
            epochs=config["epochs"],
            batch_size=config["batch_size"],
            criterion=nn.HuberLoss(reduction="mean", delta=1.0),
            optimizer=optimizer,
            device=device,
            patience=config.get("patience", 50),
            evaluation_metric=config.get("evaluation_metric", "val_loss"),
            mode=config.get("mode", "min"),
            restore_best_weights=config.get("restore_best_weights", True),
            verbose=config.get("verbose", 2),
            experiment_name=config.get("experiment_name", "trans_vs_ind_fit"),
            checkpoint_dir=checkpoint_dir,
            results_dir=results_dir,
            distributed=distributed,
        )
    finally:
        cleanup_ddp()

    if is_main_process() and histories:
        combined_path = results_dir / "training_histories.json"
        with open(combined_path, "w") as f:
            json.dump(histories, f, default=_json_converter)
        print(f"Wrote combined training histories to {combined_path}")


if __name__ == "__main__":
    main()
