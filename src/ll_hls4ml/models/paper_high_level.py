"""Faithful wa-hls4ml paper surrogate architectures.

These models intentionally preserve the published high-level GATv2 and
transformer architectures. They are separate from the compact encoder used by
the hierarchical fusion experiment.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, global_add_pool, global_max_pool, global_mean_pool


class PaperHighLevelGATv2(nn.Module):
    """Five-layer, five-head GATv2 model from the wa-hls4ml paper."""

    def __init__(
        self,
        input_dim: int,
        y_means: torch.Tensor,
        y_stds: torch.Tensor,
        hidden_dim: int = 512,
        num_layers: int = 5,
        heads: int = 5,
        mlp_hidden_dim: int = 512,
        dropout: float = 0.3,
        target_log_shift: float = 1e-8,
    ):
        super().__init__()
        self.register_buffer("y_means", y_means.clone())
        self.register_buffer("y_stds", y_stds.clone())
        self.target_log_shift = target_log_shift
        width = hidden_dim * heads
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.residual_projections = nn.ModuleList()
        for layer_index in range(num_layers):
            layer_input_dim = input_dim if layer_index == 0 else width
            self.convs.append(
                GATv2Conv(
                    layer_input_dim,
                    hidden_dim,
                    heads=heads,
                    concat=True,
                    dropout=dropout,
                )
            )
            self.norms.append(nn.LayerNorm(width))
            self.residual_projections.append(
                nn.Linear(input_dim, width) if layer_index == 0 else nn.Identity()
            )
        self.final_projection = nn.Linear(width, hidden_dim)
        self.pool_weight = nn.Parameter(torch.ones(3) / 3)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim + 4, mlp_hidden_dim),
            nn.LayerNorm(mlp_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, mlp_hidden_dim // 2),
            nn.LayerNorm(mlp_hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim // 2, 6),
        )

    def forward(self, data):
        x = data.x
        for conv, norm, residual_projection in zip(
            self.convs, self.norms, self.residual_projections
        ):
            residual = x
            x = conv(x, data.edge_index)
            x = norm(x)
            x = F.elu(x)
            x = F.dropout(x, p=self.mlp[3].p, training=self.training)
            x = x + residual_projection(residual)
        x = self.final_projection(x)
        weights = F.softmax(self.pool_weight, dim=0)
        graph = (
            weights[0] * global_add_pool(x, data.batch)
            + weights[1] * global_mean_pool(x, data.batch)
            + weights[2] * global_max_pool(x, data.batch)
        )
        return self.mlp(torch.cat([graph, data.strategy, data.io_type], dim=-1))


class PaperTransformerRegressor(nn.Module):
    """Two-block, eight-head transformer from the wa-hls4ml paper."""

    def __init__(
        self,
        y_means: torch.Tensor,
        y_stds: torch.Tensor,
        feature_dim: int = 33,
        embed_dim: int = 512,
        num_heads: int = 8,
        ff_dim: int = 512,
        num_layers: int = 2,
        max_layers: int = 51,
        dropout: float = 0.1,
        target_log_shift: float = 1e-8,
    ):
        super().__init__()
        self.register_buffer("y_means", y_means.clone())
        self.register_buffer("y_stds", y_stds.clone())
        self.target_log_shift = target_log_shift
        self.input_embedding = nn.Linear(feature_dim, embed_dim)
        self.positional_embedding = nn.Parameter(torch.randn(max_layers, embed_dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=False,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers)
        self.head = nn.Linear(embed_dim, 6)

    def forward(self, data):
        features = data.x
        batch_size, layer_count, _ = features.shape
        tokens = self.input_embedding(features)
        tokens = tokens + self.positional_embedding[:layer_count].unsqueeze(0)
        cls = self.cls_token.expand(batch_size, -1, -1)
        tokens = torch.cat([cls, tokens], dim=1).transpose(0, 1)
        cls_padding = torch.zeros(
            batch_size, 1, dtype=torch.bool, device=features.device
        )
        padding_mask = torch.cat([cls_padding, data.pad_mask], dim=1)
        encoded = self.transformer(tokens, src_key_padding_mask=padding_mask)
        return self.head(encoded[0])
