"""Model registry for experiment notebooks."""

from __future__ import annotations

import torch.nn as nn

from ll_hls4ml.models.rgcn import CDFGRGCN
from ll_hls4ml.models.mlp import MLP

MODELS: dict[str, type[nn.Module]] = {
    "rgcn": CDFGRGCN,
    "mlp": MLP,
}


def list_models() -> list[str]:
    return sorted(MODELS.keys())


def build(name: str, **kwargs) -> nn.Module:
    if name not in MODELS:
        raise KeyError(f"Unknown model '{name}'. Available: {list_models()}")
    return MODELS[name](**kwargs)
