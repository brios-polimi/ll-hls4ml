"""Graph readout and global shortcut features shared by baseline models."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch_geometric.nn import global_add_pool, global_max_pool, global_mean_pool
from torch_geometric.utils import degree

from ll_hls4ml.data.tensorize import EMBED_SIZE
from ll_hls4ml.io.schema import (
    EDGE_TYPES,
    GRAPH_CONTEXT_CATEGORICAL_VOCABS,
    GRAPH_CONTEXT_NUMERIC_KEYS,
    NODE_TYPES,
    PRAGMA_ARGUMENT_SIZE,
    PRAGMA_VOCAB_SIZE,
)


def multi_pool(x, batch, size: int) -> torch.Tensor:
    """Stable additive, mean, max, dispersion, and count statistics."""
    summed = global_add_pool(x, batch, size=size)
    mean = global_mean_pool(x, batch, size=size)
    maximum = global_max_pool(x, batch, size=size)
    mean_square = global_mean_pool(x.square(), batch, size=size)
    std = (mean_square - mean.square()).clamp_min(0).add(1e-6).sqrt()
    counts = degree(batch, num_nodes=size, dtype=x.dtype).unsqueeze(-1)
    signed_log_sum = torch.sign(summed) * torch.log1p(summed.abs())
    return torch.cat(
        [signed_log_sum, mean, maximum, std, torch.log1p(counts)],
        dim=-1,
    )


def _histogram(ids, batch, bins: int, size: int) -> torch.Tensor:
    flat = batch.long() * bins + ids.long().clamp(0, bins - 1)
    return torch.bincount(flat, minlength=size * bins).reshape(size, bins).float()


class GraphContextEncoder(nn.Module):
    def __init__(self, hidden_dim: int, mode: str = "core"):
        super().__init__()
        if mode not in {"core", "all"}:
            raise ValueError("context mode must be 'core' or 'all'")
        self.category_indices = (
            range(len(GRAPH_CONTEXT_CATEGORICAL_VOCABS))
            if mode == "all"
            else range(2)
        )
        embedding_dim = max(4, hidden_dim // 8)
        self.embeddings = nn.ModuleList(
            [
                nn.Embedding(
                    max(vocabulary.values(), default=0) + 1,
                    embedding_dim,
                    padding_idx=0,
                )
                for index, vocabulary in enumerate(
                    GRAPH_CONTEXT_CATEGORICAL_VOCABS.values()
                )
                if index in self.category_indices
            ]
        )
        # Schema-known categories may still be absent from the training split.
        # Zero initialization keeps those unseen rows neutral at inference;
        # rows that occur during training receive ordinary embedding gradients.
        for embedding in self.embeddings:
            nn.init.zeros_(embedding.weight)
        input_dim = embedding_dim * len(self.embeddings) + len(
            GRAPH_CONTEXT_NUMERIC_KEYS
        )
        self.projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
        )

    def forward(self, data) -> torch.Tensor:
        categorical = data.graph_context_categorical.long()
        numeric = data.graph_context_numeric.float()
        embedded = [
            embedding(categorical[:, category_index])
            for category_index, embedding in zip(
                self.category_indices, self.embeddings
            )
        ]
        return self.projection(torch.cat([*embedded, numeric], dim=-1))


class GlobalFeatureEncoder(nn.Module):
    """Encode deterministic graph-wide statistics as a shortcut branch."""

    def __init__(self, instruction_vocab_size: int, hidden_dim: int):
        super().__init__()
        self.instruction_vocab_size = instruction_vocab_size
        raw_dim = (
            2 * instruction_vocab_size
            + 2 * PRAGMA_VOCAB_SIZE
            + len(NODE_TYPES)
            + len(EDGE_TYPES)
            + 6 * EMBED_SIZE
            + 2 * PRAGMA_ARGUMENT_SIZE
        )
        self.projection = nn.Sequential(
            nn.Linear(raw_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
        )

    def forward(self, data) -> torch.Tensor:
        size = data.num_graphs
        instruction_hist = _histogram(
            data["instruction"].x[:, 0],
            data["instruction"].batch,
            self.instruction_vocab_size,
            size,
        )
        pragma_hist = _histogram(
            data["pragma"].x[:, 0],
            data["pragma"].batch,
            PRAGMA_VOCAB_SIZE,
            size,
        )
        histograms = []
        for histogram in (instruction_hist, pragma_hist):
            total = histogram.sum(dim=-1, keepdim=True).clamp_min(1)
            histograms.extend([torch.log1p(histogram), histogram / total])

        node_counts = torch.stack(
            [
                degree(data[node_type].batch, num_nodes=size, dtype=torch.float)
                for node_type in NODE_TYPES
            ],
            dim=-1,
        )
        edge_counts = []
        for edge_type in EDGE_TYPES:
            edge_index = data[edge_type].edge_index
            source_batch = data[edge_type[0]].batch[edge_index[0]]
            edge_counts.append(
                degree(source_batch, num_nodes=size, dtype=torch.float)
            )

        type_statistics = []
        for node_type in ("variable", "constant"):
            values = data[node_type].x.float()
            batch = data[node_type].batch
            mean = global_mean_pool(values, batch, size=size)
            maximum = global_max_pool(values, batch, size=size)
            mean_square = global_mean_pool(values.square(), batch, size=size)
            std = (
                (mean_square - mean.square()).clamp_min(0).add(1e-6).sqrt()
            )
            type_statistics.extend([mean, maximum, std])

        pragma_args = data["pragma"].x[:, 1:].float()
        pragma_batch = data["pragma"].batch
        pragma_mean = global_mean_pool(pragma_args, pragma_batch, size=size)
        pragma_max = global_max_pool(pragma_args, pragma_batch, size=size)
        raw = torch.cat(
            [
                *histograms,
                torch.log1p(node_counts),
                torch.log1p(torch.stack(edge_counts, dim=-1)),
                *type_statistics,
                pragma_mean,
                pragma_max,
            ],
            dim=-1,
        )
        return self.projection(raw)


class SplitRegressionHead(nn.Module):
    """Separate resource and timing towers over a shared graph representation."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        dropout: float,
        hurdle_heads: bool = False,
    ):
        super().__init__()
        self.hurdle_heads = hurdle_heads

        def tower(outputs: int):
            return nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, outputs),
            )

        self.resource = tower(6 if hurdle_heads else 4)
        self.timing = tower(2)

    def forward(self, features):
        resources = self.resource(features)
        timing = self.timing(features)
        if not self.hurdle_heads:
            return torch.cat([resources, timing], dim=-1)
        return torch.cat(
            [resources[:, :4], timing, resources[:, 4:]],
            dim=-1,
        )
