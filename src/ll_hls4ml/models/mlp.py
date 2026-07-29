"""Baseline pooled MLP for LLVM CDFG graphs."""

import torch
import torch.nn as nn
from torch_geometric.data import HeteroData
from torch_geometric.nn import (
    global_add_pool,
    global_max_pool,
    global_mean_pool,
)

from ll_hls4ml.io.schema import (
    BLOCK_FEATURE_SIZE,
    LABEL_KEYS,
    NODE_TYPES,
    PRAGMA_ARGUMENT_SIZE,
    PRAGMA_VOCAB_SIZE,
)
from ll_hls4ml.data.tensorize import EMBED_SIZE
from ll_hls4ml.models.readout import (
    GlobalFeatureEncoder,
    GraphContextEncoder,
    SplitRegressionHead,
    multi_pool,
)

class MultilayerDense(nn.Module):
    def __init__(self, in_dim, out_dim, n_layers):
        super().__init__()
        layers = []

        dims = [in_dim] + [out_dim] * n_layers
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            if i < len(dims) - 2:
                layers.append(nn.ReLU())

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

class CDFGInputProjection(nn.Module):
    """
    Embeds opcode vocab per node type and feature vectors for variable/constant nodes.

    Expected node feature layout:
        instruction: LongTensor of shape (N, 1)
        variable:    FloatTensor of shape (N, EMBED_SIZE)
        constant:    FloatTensor of shape (N, EMBED_SIZE)
        pragma:      FloatTensor of shape (N, 1 + PRAGMA_ARGUMENT_SIZE)
        block:       FloatTensor of shape (N, BLOCK_FEATURE_SIZE)
    """

    def __init__(
        self,
        instruction_vocab_size: int,
        variable_constant_size: int,
        hidden_dim: int,
        n_layers: int,
    ):
        super().__init__()

        self.instruction_emb = nn.Embedding(
            instruction_vocab_size, hidden_dim, padding_idx=0
        )
        self.variable_emb = MultilayerDense(variable_constant_size, hidden_dim, n_layers)
        self.constant_emb = MultilayerDense(variable_constant_size, hidden_dim, n_layers)
        self.block_emb = MultilayerDense(BLOCK_FEATURE_SIZE, hidden_dim, n_layers)
        self.pragma_emb = nn.Embedding(PRAGMA_VOCAB_SIZE, hidden_dim, padding_idx=0)
        self.pragma_arg_proj = MultilayerDense(
            PRAGMA_ARGUMENT_SIZE, hidden_dim, n_layers
        )
        self.pragma_proj = MultilayerDense(2 * hidden_dim, hidden_dim, n_layers)

    def forward(self, x_dict):
        h_dict = {}

        # instruction: (N, 1) -> (N,)
        instr = x_dict["instruction"]
        h_dict["instruction"] = self.instruction_emb(instr.squeeze(-1))

        # variable / constant: (N, D)
        h_dict["variable"] = self.variable_emb(x_dict["variable"])
        h_dict["constant"] = self.constant_emb(x_dict["constant"])
        h_dict["block"] = self.block_emb(x_dict["block"])
        pragma = x_dict["pragma"]
        h_dict["pragma"] = self.pragma_proj(
            torch.cat(
                [
                    self.pragma_emb(pragma[:, 0].long()),
                    self.pragma_arg_proj(pragma[:, 1:].float()),
                ],
                dim=-1,
            )
        )

        return h_dict

class MLP(nn.Module):
    """Graph-level MLP baseline without message passing."""

    def __init__(
        self,
        instruction_vocab_size: int,
        y_means: torch.Tensor,
        y_stds: torch.Tensor,
        hidden_dim: int = 128,
        num_layers: int = 3,
        num_var_embed_layers: int = 3,
        dropout: float = 0.1,
        pool: str = "mean",
        node_aggr: str = "concat",  # concat | sum | mean
        use_global_features: bool = False,
        use_context: bool = False,
        split_heads: bool = False,
        context_mode: str = "core",
        hurdle_heads: bool = False,
        hurdle_prediction_mode: str = "expected",
    ):
        super().__init__()
        if hurdle_heads and not split_heads:
            raise ValueError("hurdle_heads requires split_heads=True")

        self.register_buffer("y_means", y_means.clone())
        self.register_buffer("y_stds", y_stds.clone())
        self.instruction_vocab_size = instruction_vocab_size

        self.pool = pool
        self.pool_fn = (
            {
                "mean": global_mean_pool,
                "sum": global_add_pool,
                "max": global_max_pool,
            }.get(pool)
        )
        self.node_aggr = node_aggr
        self.use_global_features = use_global_features
        self.use_context = use_context
        self.output_dim = len(LABEL_KEYS)
        self.hurdle_heads = hurdle_heads
        self.hurdle_prediction_mode = hurdle_prediction_mode

        self.node_encoder = CDFGInputProjection(instruction_vocab_size, EMBED_SIZE, hidden_dim, num_var_embed_layers)

        layers = []
        for _ in range(num_layers - 1):
            layers.extend([
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
        self.mlp = nn.Sequential(*layers)

        if pool == "multi":
            if node_aggr != "concat":
                raise ValueError("multi pooling requires node_aggr='concat'")
            self.readout_proj = nn.ModuleDict(
                {
                    node_type: MultilayerDense(
                        4 * hidden_dim + 1, hidden_dim, 2
                    )
                    for node_type in NODE_TYPES
                }
            )
        else:
            self.readout_proj = None

        if node_aggr == "concat":
            classifier_in = hidden_dim * len(NODE_TYPES)
        else:
            classifier_in = hidden_dim

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
        x_dict = {nt: data[nt].x for nt in NODE_TYPES}

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
            if self.pool == "multi":
                pooled.append(
                    self.readout_proj[nt](
                        multi_pool(h, data[nt].batch, data.num_graphs)
                    )
                )
            else:
                pooled.append(
                    self.pool_fn(h, data[nt].batch, size=data.num_graphs)
                )

        if self.node_aggr == "concat":
            graph_emb = torch.cat(pooled, dim=-1)
        elif self.node_aggr == "sum": ############################################### REPRESENTATION BLOWS UP
            graph_emb = torch.stack(pooled).sum(dim=0)
        elif self.node_aggr == "mean":
            graph_emb = torch.stack(pooled).mean(dim=0)
        else:
            raise ValueError(f"Unknown node_aggr: {self.node_aggr}")

        shortcuts = [graph_emb]
        if self.use_global_features:
            shortcuts.append(self.global_features(data))
        if self.use_context:
            shortcuts.append(self.context_encoder(data))
        return self.classifier(torch.cat(shortcuts, dim=-1))
