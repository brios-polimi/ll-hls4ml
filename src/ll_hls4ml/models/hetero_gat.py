"""Heterogeneous GATv2 model for LLVM CDFG graphs."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, HeteroConv

from ll_hls4ml.io.schema import (
    EDGE_TYPES,
    EDGE_TYPES_WITH_ATTR,
    NODE_TYPES,
)
from ll_hls4ml.models.hetero_base import CDFGHeteroGraphRegressor
from ll_hls4ml.models.input_projection import CDFGInputProjection


class CDFGConvLayer(nn.Module):
    """One heterogeneous message-passing step over all edge types."""

    def __init__(
        self,
        hidden_dim: int,
        heads: int = 1,
        aggr: str = "sum",
        dropout: float = 0.0,
    ):
        super().__init__()
        self.conv = HeteroConv(
            {
                et: GATv2Conv(
                    in_channels=(hidden_dim, hidden_dim),
                    out_channels=hidden_dim,
                    heads=heads,
                    concat=False,
                    edge_dim=hidden_dim if et in EDGE_TYPES_WITH_ATTR else None,
                    add_self_loops=False,
                    # Attention already normalizes over the neighborhood.
                    aggr="add",
                )
                for et in EDGE_TYPES
            },
            aggr=aggr,
        )
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.ModuleDict(
            {nt: nn.LayerNorm(hidden_dim) for nt in NODE_TYPES}
        )

    def forward(self, h_dict, edge_index_dict, edge_emb_dict):
        h = {nt: h_dict[nt] for nt in NODE_TYPES}
        out = self.conv(h, edge_index_dict, edge_attr_dict=edge_emb_dict)
        updated = {}
        for nt in NODE_TYPES:
            if nt in out:
                message = self.dropout(F.relu(out[nt]))
                updated[nt] = self.norm[nt](h_dict[nt] + message)
            else:
                updated[nt] = self.norm[nt](h_dict[nt])
        return updated


class CDFGHeteroGAT(CDFGHeteroGraphRegressor):
    """Relation-specific heterogeneous GATv2 graph regressor."""

    def __init__(
        self,
        edge_pos_vocab_size: int,
        y_means: torch.Tensor,
        y_stds: torch.Tensor,
        instruction_vocab_size: int | None = None,
        hidden_dim: int = 128,
        num_layers: int = 3,
        dropout: float = 0.1,
        pool: str = "mean",
        aggr: str = "sum",
        heads: int = 1,
        node_vocab_sizes: dict[str, int] | None = None,
        use_global_features: bool = False,
        use_context: bool = False,
        split_heads: bool = False,
        context_mode: str = "core",
        hurdle_heads: bool = False,
        hurdle_prediction_mode: str = "expected",
    ):
        super().__init__(
            edge_pos_vocab_size=edge_pos_vocab_size,
            y_means=y_means,
            y_stds=y_stds,
            instruction_vocab_size=instruction_vocab_size,
            hidden_dim=hidden_dim,
            pool=pool,
            node_vocab_sizes=node_vocab_sizes,
            use_global_features=use_global_features,
            use_context=use_context,
            split_heads=split_heads,
            context_mode=context_mode,
            hurdle_heads=hurdle_heads,
            hurdle_prediction_mode=hurdle_prediction_mode,
            dropout=dropout,
        )
        self.num_layers = num_layers
        self.layers = nn.ModuleList(
            [
                CDFGConvLayer(
                    hidden_dim,
                    heads=heads,
                    aggr=aggr,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )
