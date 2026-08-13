#!/usr/bin/env python3
"""Profile one forward/backward architecture step on selected tensors."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import yaml
from torch_geometric.data import Batch

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

from ll_hls4ml.data.vocab import load_vocab
from ll_hls4ml.models.registry import build


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("tensors", nargs="+", type=Path)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    config_dir = args.config.resolve().parent
    vocab_path = Path(config["vocab_path"])
    if not vocab_path.is_absolute():
        vocab_path = (config_dir / vocab_path).resolve()
    vocab, max_position, _counts = load_vocab(vocab_path)
    common = {
        "instruction_vocab_size": len(vocab),
        "edge_pos_vocab_size": max_position,
        "y_means": torch.zeros(6),
        "y_stds": torch.ones(6),
        "hidden_dim": config.get("hidden_dim", 64),
        "num_layers": config.get("num_layers", 1),
        "instruction_num_layers": config.get("instruction_num_layers"),
        "block_num_layers": config.get("block_num_layers"),
        "dropout": config.get("dropout", 0.15),
        "use_global_features": config.get("use_global_features", True),
        "use_context": config.get("use_context", True),
        "context_mode": config.get("context_mode", "core"),
        "split_heads": True,
        "hurdle_heads": True,
        "attention_heads": config.get("attention_heads", 4),
        "attention_layers": config.get("attention_layers", 2),
        "cfg_recurrent_steps": config.get("cfg_recurrent_steps", 8),
        "sequence_token_budget": config.get("sequence_token_budget", 16_384),
        "attention_pair_budget": config.get("attention_pair_budget", 131_072),
    }
    model = build(config["model"], **common)
    data = Batch.from_data_list(
        [torch.load(path.resolve(), weights_only=False) for path in args.tensors]
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    data = data.to(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize()
    started = time.perf_counter()
    with torch.autocast(
        device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"
    ):
        prediction = model(data)
        loss = prediction.square().mean()
    loss.backward()
    if device.type == "cuda":
        torch.cuda.synchronize()
    result = {
        "model": config["model"],
        "device": str(device),
        "graphs": data.num_graphs,
        "instructions": data["instruction"].num_nodes,
        "blocks": data["block"].num_nodes,
        "seconds": time.perf_counter() - started,
        "peak_gpu_memory_mb": (
            torch.cuda.max_memory_allocated(device) / (1024**2)
            if device.type == "cuda"
            else 0.0
        ),
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
