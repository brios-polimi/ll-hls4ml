"""Non-attention relational encoder for heterogeneous LLVM CDFGs."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, MessagePassing

from ll_hls4ml.io.schema import EDGE_TYPES, EDGE_TYPES_WITH_ATTR, NODE_TYPES
from ll_hls4ml.models.hetero_base import CDFGHeteroGraphRegressor


class EdgeAwareRelationConv(MessagePassing):
    """Efficient relation-specific neighborhood aggregation without attention."""

    def __init__(
        self,
        hidden_dim: int,
        aggr: str = "mean",
        use_edge_features: bool = False,
    ):
        super().__init__(aggr=aggr)
        self.source_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.edge_proj = (
            nn.Linear(hidden_dim, hidden_dim, bias=False)
            if use_edge_features
            else None
        )

    def forward(self, x, edge_index, edge_attr=None):
        source, target = x if isinstance(x, tuple) else (x, x)
        source = self.source_proj(source)
        edge_features = (
            self.edge_proj(edge_attr)
            if self.edge_proj is not None and edge_attr is not None
            else None
        )
        return self.propagate(
            edge_index,
            x=source,
            edge_attr=edge_features,
            size=(source.size(0), target.size(0)),
        )

    def message(self, x_j, edge_attr):
        return x_j if edge_attr is None else x_j + edge_attr


class CDFGRelationalConvLayer(nn.Module):
    """One residual heterogeneous relation-aware message-passing step."""

    def __init__(
        self,
        hidden_dim: int,
        aggr: str = "sum",
        message_aggr: str = "mean",
        dropout: float = 0.0,
    ):
        super().__init__()
        self.conv = HeteroConv(
            {
                edge_type: EdgeAwareRelationConv(
                    hidden_dim,
                    aggr=message_aggr,
                    use_edge_features=edge_type in EDGE_TYPES_WITH_ATTR,
                )
                for edge_type in EDGE_TYPES
            },
            aggr=aggr,
        )
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.ModuleDict(
            {node_type: nn.LayerNorm(hidden_dim) for node_type in NODE_TYPES}
        )

    def forward(self, h_dict, edge_index_dict, edge_emb_dict):
        out = self.conv(
            h_dict,
            edge_index_dict,
            edge_attr_dict=edge_emb_dict,
        )
        updated = {}
        for node_type in NODE_TYPES:
            if node_type in out:
                message = self.dropout(F.relu(out[node_type]))
                updated[node_type] = self.norm[node_type](
                    h_dict[node_type] + message
                )
            else:
                updated[node_type] = self.norm[node_type](h_dict[node_type])
        return updated


class CDFGHeteroRelational(CDFGHeteroGraphRegressor):
    """Relation-specific heterogeneous graph regressor without attention."""

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
        message_aggr: str = "mean",
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
        self.message_aggr = message_aggr
        self.layers = nn.ModuleList(
            [
                CDFGRelationalConvLayer(
                    hidden_dim,
                    aggr=aggr,
                    message_aggr=message_aggr,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )
