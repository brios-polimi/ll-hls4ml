"""Baseline pooled MLP for LLVM CDFG graphs."""

import torch
import torch.nn as nn
from torch_geometric.data import HeteroData
from torch_geometric.nn import (
    global_add_pool,
    global_max_pool,
    global_mean_pool,
)

from ll_hls4ml.io.schema import NODE_TYPES, LABEL_KEYS

class CDFGInputProjection(nn.Module):
    """
    Embeds opcode vocab per node type and positional arg encoding for edges.

    Node feature layout: [vocab_id]
    """

    def __init__(
        self,
        node_vocab_sizes: dict[str, int],
        hidden_dim: int,
    ):
        super().__init__()
        self.node_emb = nn.ModuleDict({
            nt: nn.Embedding(vocab_size, hidden_dim, padding_idx=0)
            for nt, vocab_size in node_vocab_sizes.items()
        })

    def forward(self, x_dict):
        h_dict = {nt: self.node_emb[nt](x[:, 0]) for nt, x in x_dict.items()}
        return h_dict

class MLP(nn.Module):
    """Graph-level MLP baseline without message passing."""

    def __init__(
        self,
        node_vocab_sizes: dict[str, int],
        y_means: torch.Tensor,
        y_stds: torch.Tensor,
        hidden_dim: int = 128,
        num_layers: int = 3,
        dropout: float = 0.1,
        pool: str = "mean",
        node_aggr: str = "concat",  # concat | sum | mean
    ):
        super().__init__()

        self.register_buffer("y_means", y_means.clone())
        self.register_buffer("y_stds", y_stds.clone())
        self.node_vocab_sizes = node_vocab_sizes

        self.pool_fn = {
            "mean": global_mean_pool,
            "sum": global_add_pool,
            "max": global_max_pool,
        }[pool]
        self.node_aggr = node_aggr
        self.output_dim = len(LABEL_KEYS)

        self.node_encoder = CDFGInputProjection(node_vocab_sizes, hidden_dim)

        layers = []
        for _ in range(num_layers - 1):
            layers.extend([
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
        self.mlp = nn.Sequential(*layers)

        if node_aggr == "concat":
            classifier_in = hidden_dim * len(NODE_TYPES)
        else:
            classifier_in = hidden_dim

        self.classifier = nn.Sequential(
            nn.Linear(classifier_in, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, self.output_dim),
        )

    def forward(self, data: HeteroData):
        x_dict = {nt: data[nt].x.long() for nt in NODE_TYPES}

        # for nt, x in x_dict.items():
        #     ids = x[:, 0]
        #     if ids.min() < 0 or ids.max() >= self.node_vocab_sizes[nt].num_embeddings:
        #         print("BAD NODE TYPE:", nt)
        #         print("min id:", ids.min().item())
        #         print("max id:", ids.max().item())
        #         print("allowed:", self.node_emb[nt].num_embeddings)
        #         raise ValueError("Invalid vocab id detected")


        h_dict = self.node_encoder(x_dict)

        pooled = []
        for nt in NODE_TYPES:
            h = self.mlp(h_dict[nt])
            pooled.append(self.pool_fn(h, data[nt].batch))

        if self.node_aggr == "concat":
            graph_emb = torch.cat(pooled, dim=-1)
        elif self.node_aggr == "sum": ############################################### REPRESENTATION BLOWS UP
            graph_emb = torch.stack(pooled).sum(dim=0)
        elif self.node_aggr == "mean":
            graph_emb = torch.stack(pooled).mean(dim=0)
        else:
            raise ValueError(f"Unknown node_aggr: {self.node_aggr}")

        return self.classifier(graph_emb)