"""Shared graph-level structure for heterogeneous CDFG encoders."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch_geometric.data import HeteroData
from torch_geometric.nn import global_add_pool, global_max_pool, global_mean_pool
from torch_geometric.utils import degree

from ll_hls4ml.io.schema import (
    EDGE_TYPES,
    EDGE_TYPES_WITH_ATTR,
    LABEL_KEYS,
    NODE_TYPES,
)
from ll_hls4ml.models.input_projection import (
    CDFGInputProjection,
    _dense_projection,
)
from ll_hls4ml.models.readout import (
    GlobalFeatureEncoder,
    GraphContextEncoder,
    SplitRegressionHead,
    multi_pool,
)


class CDFGHeteroGraphRegressor(nn.Module):
    """Common input, readout, shortcut, and regression-head implementation."""

    def __init__(
        self,
        edge_pos_vocab_size: int,
        y_means: torch.Tensor,
        y_stds: torch.Tensor,
        instruction_vocab_size: int | None = None,
        hidden_dim: int = 128,
        pool: str = "mean",
        node_vocab_sizes: dict[str, int] | None = None,
        use_global_features: bool = False,
        use_context: bool = False,
        split_heads: bool = False,
        context_mode: str = "core",
        hurdle_heads: bool = False,
        hurdle_prediction_mode: str = "expected",
        dropout: float = 0.1,
    ):
        super().__init__()
        if hurdle_heads and not split_heads:
            raise ValueError("hurdle_heads requires split_heads=True")
        if instruction_vocab_size is None:
            if not node_vocab_sizes or "instruction" not in node_vocab_sizes:
                raise ValueError("instruction_vocab_size is required")
            instruction_vocab_size = node_vocab_sizes["instruction"]

        self.register_buffer("y_means", y_means.clone())
        self.register_buffer("y_stds", y_stds.clone())
        self.hidden_dim = hidden_dim
        self.pool = pool
        self.use_global_features = use_global_features
        self.use_context = use_context
        self.output_dim = len(LABEL_KEYS)
        self.hurdle_heads = hurdle_heads
        self.hurdle_prediction_mode = hurdle_prediction_mode

        self.input_proj = CDFGInputProjection(
            instruction_vocab_size, edge_pos_vocab_size, hidden_dim
        )
        if pool == "multi":
            self.readout_proj = nn.ModuleDict(
                {
                    node_type: _dense_projection(4 * hidden_dim + 1, hidden_dim)
                    for node_type in NODE_TYPES
                }
            )
            classifier_in = hidden_dim * len(NODE_TYPES)
        else:
            self.readout_proj = None
            classifier_in = hidden_dim * len(NODE_TYPES) + len(NODE_TYPES)
        if use_global_features:
            self.global_features = GlobalFeatureEncoder(
                instruction_vocab_size, hidden_dim
            )
            classifier_in += hidden_dim
        if use_context:
            self.context_encoder = GraphContextEncoder(hidden_dim, context_mode)
            classifier_in += hidden_dim
        self.classifier = (
            SplitRegressionHead(
                classifier_in,
                hidden_dim,
                dropout,
                hurdle_heads=hurdle_heads,
            )
            if split_heads
            else nn.Sequential(
                nn.Linear(classifier_in, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, self.output_dim),
            )
        )

    def forward(self, data: HeteroData):
        x_dict = {node_type: data[node_type].x for node_type in NODE_TYPES}
        edge_index_dict = {
            edge_type: data[edge_type].edge_index for edge_type in EDGE_TYPES
        }
        edge_attr_dict = {
            edge_type: data[edge_type].edge_attr.long()
            for edge_type in EDGE_TYPES_WITH_ATTR
            if (
                hasattr(data[edge_type], "edge_attr")
                and data[edge_type].edge_attr is not None
            )
        }
        h_dict, edge_emb_dict = self.input_proj(x_dict, edge_attr_dict)

        for layer in self.layers:
            h_dict = layer(h_dict, edge_index_dict, edge_emb_dict)

        if self.pool == "multi":
            graph_features = torch.cat(
                [
                    self.readout_proj[node_type](
                        multi_pool(
                            h_dict[node_type],
                            data[node_type].batch,
                            data.num_graphs,
                        )
                    )
                    for node_type in NODE_TYPES
                ],
                dim=-1,
            )
        else:
            pool_fn = {
                "mean": global_mean_pool,
                "sum": global_add_pool,
                "max": global_max_pool,
            }[self.pool]
            pooled = torch.cat(
                [
                    pool_fn(
                        h_dict[node_type],
                        data[node_type].batch,
                        size=data.num_graphs,
                    )
                    for node_type in NODE_TYPES
                ],
                dim=-1,
            )
            node_counts = torch.stack(
                [
                    degree(
                        data[node_type].batch,
                        num_nodes=data.num_graphs,
                        dtype=pooled.dtype,
                    )
                    for node_type in NODE_TYPES
                ],
                dim=-1,
            )
            graph_features = torch.cat(
                [pooled, torch.log1p(node_counts)],
                dim=-1,
            )

        shortcuts = [graph_features]
        if self.use_global_features:
            shortcuts.append(self.global_features(data))
        if self.use_context:
            shortcuts.append(self.context_encoder(data))
        return self.classifier(torch.cat(shortcuts, dim=-1))
