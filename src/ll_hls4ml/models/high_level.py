"""Matched GNN over wa-hls4ml layer/config graphs."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, SAGEConv

from ll_hls4ml.io.schema import LABEL_KEYS
from ll_hls4ml.models.readout import SplitRegressionHead, multi_pool


class HighLevelLayerGNN(nn.Module):
    def __init__(
        self,
        input_dim: int,
        y_means: torch.Tensor,
        y_stds: torch.Tensor,
        hidden_dim: int = 64,
        num_layers: int = 3,
        heads: int = 1,
        dropout: float = 0.15,
        encoder: str = "gatv2",
        hurdle_heads: bool = True,
        hurdle_prediction_mode: str = "threshold",
    ):
        super().__init__()
        if encoder not in {"gatv2", "sage"}:
            raise ValueError("encoder must be 'gatv2' or 'sage'")
        self.register_buffer("y_means", y_means.clone())
        self.register_buffer("y_stds", y_stds.clone())
        self.hurdle_prediction_mode = hurdle_prediction_mode
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
        )
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(num_layers):
            if encoder == "gatv2":
                layer = GATv2Conv(
                    hidden_dim,
                    hidden_dim,
                    heads=heads,
                    concat=False,
                    dropout=dropout,
                )
            else:
                layer = SAGEConv(hidden_dim, hidden_dim, aggr="sum")
            self.layers.append(layer)
            self.norms.append(nn.LayerNorm(hidden_dim))
        self.dropout = nn.Dropout(dropout)
        self.readout = nn.Sequential(
            nn.Linear(4 * hidden_dim + 1, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
        )
        self.classifier = SplitRegressionHead(
            hidden_dim + 4,
            hidden_dim,
            dropout,
            hurdle_heads=hurdle_heads,
        )

    def forward(self, data):
        x = self.input_projection(data.x)
        for layer, norm in zip(self.layers, self.norms):
            message = self.dropout(F.relu(layer(x, data.edge_index)))
            x = norm(x + message)
        graph = self.readout(multi_pool(x, data.batch, data.num_graphs))
        global_features = torch.cat([data.strategy, data.io_type], dim=-1)
        return self.classifier(torch.cat([graph, global_features], dim=-1))
