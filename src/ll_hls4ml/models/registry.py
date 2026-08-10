"""Model registry for experiment notebooks."""

from __future__ import annotations

import torch.nn as nn

from ll_hls4ml.models.hetero_gat import CDFGHeteroGAT
from ll_hls4ml.models.hetero_relational import CDFGHeteroRelational
from ll_hls4ml.models.hierarchical import CDFGHierarchical
from ll_hls4ml.models.mlp import MLP

MODELS: dict[str, type[nn.Module]] = {
    "hetero_gat": CDFGHeteroGAT,
    "hetero_relational": CDFGHeteroRelational,
    "hierarchical": CDFGHierarchical,
    # Compatibility alias for existing configs and recorded experiments.
    "rgcn": CDFGHeteroGAT,
    "mlp": MLP,
}


def list_models() -> list[str]:
    return sorted(MODELS.keys())


def build(name: str, **kwargs) -> nn.Module:
    if name not in MODELS:
        raise KeyError(f"Unknown model '{name}'. Available: {list_models()}")
    return MODELS[name](**kwargs)
