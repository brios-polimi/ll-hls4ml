"""Heterogeneous R-GCN for LLVM CDFG graphs."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.nn import GATv2Conv, HeteroConv
from torch_geometric.nn import global_add_pool, global_max_pool, global_mean_pool

from ll_hls4ml.io.schema import (
    EDGE_TYPES,
    EDGE_TYPES_WITH_ATTR,
    LABEL_KEYS,
    NODE_TYPES,
    PRAGMA_ARGUMENT_SIZE,
    PRAGMA_VOCAB_SIZE,
)
from ll_hls4ml.data.tensorize import EMBED_SIZE


def _dense_projection(in_dim: int, hidden_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, hidden_dim),
    )


class CDFGInputProjection(nn.Module):
    """
    Embed categorical instruction/pragma nodes and project numeric type features.
    """

    def __init__(
        self,
        instruction_vocab_size: int,
        edge_pos_vocab_size: int,
        hidden_dim: int,
    ):
        super().__init__()
        self.instruction_emb = nn.Embedding(
            instruction_vocab_size, hidden_dim, padding_idx=0
        )
        self.pragma_emb = nn.Embedding(
            PRAGMA_VOCAB_SIZE, hidden_dim, padding_idx=0
        )
        self.pragma_arg_proj = _dense_projection(PRAGMA_ARGUMENT_SIZE, hidden_dim)
        self.variable_proj = _dense_projection(EMBED_SIZE, hidden_dim)
        self.constant_proj = _dense_projection(EMBED_SIZE, hidden_dim)
        self.edge_pos_emb = nn.Embedding(edge_pos_vocab_size + 1, hidden_dim)

    def forward(self, x_dict, edge_attr_dict):
        h_dict = {
            "instruction": self.instruction_emb(x_dict["instruction"][:, 0].long()),
            "variable": self.variable_proj(x_dict["variable"].float()),
            "constant": self.constant_proj(x_dict["constant"].float()),
            "pragma": (
                self.pragma_emb(x_dict["pragma"][:, 0].long())
                + self.pragma_arg_proj(x_dict["pragma"][:, 1:].float())
            ),
        }
        edge_emb_dict = {
            et: self.edge_pos_emb(attr[:, 0])
            for et, attr in edge_attr_dict.items()
        }
        return h_dict, edge_emb_dict


class CDFGConvLayer(nn.Module):
    """One heterogeneous message-passing step over all edge types."""

    def __init__(
        self,
        hidden_dim: int,
        aggr: str = "mean",
        dropout: float = 0.0,
    ):
        super().__init__()
        self.conv = HeteroConv(
            {
                et: GATv2Conv(
                    in_channels=(hidden_dim, hidden_dim),
                    out_channels=hidden_dim,
                    heads=4,
                    concat=False,
                    edge_dim=hidden_dim if et in EDGE_TYPES_WITH_ATTR else None,
                    add_self_loops=False,
                    aggr=aggr,
                )
                for et in EDGE_TYPES
            },
            aggr="mean",
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


class CDFGRGCN(nn.Module):
    """Heterogeneous R-GCN for LLVM CDFG graphs with graph-level regression head."""

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
        aggr: str = "mean",
        node_vocab_sizes: dict[str, int] | None = None,
    ):
        super().__init__()
        if instruction_vocab_size is None:
            if not node_vocab_sizes or "instruction" not in node_vocab_sizes:
                raise ValueError("instruction_vocab_size is required")
            instruction_vocab_size = node_vocab_sizes["instruction"]
        self.register_buffer("y_means", y_means.clone())
        self.register_buffer("y_stds", y_stds.clone())
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.pool = pool
        self.output_dim = len(LABEL_KEYS)

        self.input_proj = CDFGInputProjection(
            instruction_vocab_size, edge_pos_vocab_size, hidden_dim
        )
        self.layers = nn.ModuleList([
            CDFGConvLayer(hidden_dim, aggr=aggr, dropout=dropout)
            for _ in range(num_layers)
        ])
        graph_stats_dim = len(NODE_TYPES)
        self.classifier = nn.Sequential(
            nn.Linear(
                hidden_dim * len(NODE_TYPES) + graph_stats_dim,
                hidden_dim,
            ),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, self.output_dim),
        )

    def forward(self, data: HeteroData):
        x_dict = {nt: data[nt].x for nt in NODE_TYPES}
        edge_index_dict = {et: data[et].edge_index for et in EDGE_TYPES}
        edge_attr_dict = {
            et: data[et].edge_attr.long()
            for et in EDGE_TYPES_WITH_ATTR
            if hasattr(data[et], "edge_attr") and data[et].edge_attr is not None
        }

        # # Create self loops for each node type
        # for nt in NODE_TYPES:
        #     num_nodes = x_dict[nt].size(0)
        #     self_loop_idx = torch.arange(num_nodes, device=x_dict[nt].device)
        #     edge_index_dict[(nt, 'self', nt)] = torch.stack([self_loop_idx, self_loop_idx])

        h_dict, edge_emb_dict = self.input_proj(x_dict, edge_attr_dict)

        for layer in self.layers:
            h_dict = layer(h_dict, edge_index_dict, edge_emb_dict)

        pool_fn = {
            "mean": global_mean_pool,
            "sum": global_add_pool,
            "max": global_max_pool,
        }[self.pool]

        pooled = torch.cat(
            [
                pool_fn(h_dict[nt], data[nt].batch, size=data.num_graphs)
                for nt in NODE_TYPES
            ],
            dim=-1,
        )
        # Mean pooling captures composition but not scale. Log counts restore a
        # bounded graph-size signal without letting large graphs dominate.
        node_counts = torch.stack(
            [
                torch.bincount(
                    data[nt].batch,
                    minlength=data.num_graphs,
                )
                for nt in NODE_TYPES
            ],
            dim=-1,
        ).to(dtype=pooled.dtype)
        graph_features = torch.cat(
            [pooled, torch.log1p(node_counts)],
            dim=-1,
        )
        return self.classifier(graph_features)
