"""Shared input projections for heterogeneous LLVM CDFG models."""

from __future__ import annotations

import torch
import torch.nn as nn

from ll_hls4ml.data.tensorize import EMBED_SIZE
from ll_hls4ml.io.schema import (
    BLOCK_FEATURE_SIZE,
    FUNCTION_FEATURE_SIZE,
    LOOP_FEATURE_SIZE,
    PRAGMA_ARGUMENT_SIZE,
    PRAGMA_VOCAB_SIZE,
)


def _dense_projection(in_dim: int, hidden_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, hidden_dim),
    )


class CDFGInputProjection(nn.Module):
    """Project tensor-v2 CDFG node and positional-edge features."""

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
        self.pragma_proj = _dense_projection(2 * hidden_dim, hidden_dim)
        self.variable_proj = _dense_projection(EMBED_SIZE, hidden_dim)
        self.constant_proj = _dense_projection(EMBED_SIZE, hidden_dim)
        self.block_proj = _dense_projection(BLOCK_FEATURE_SIZE, hidden_dim)
        self.function_proj = _dense_projection(FUNCTION_FEATURE_SIZE, hidden_dim)
        self.edge_pos_emb = nn.Embedding(edge_pos_vocab_size + 1, hidden_dim)

    def forward(self, x_dict, edge_attr_dict):
        h_dict = {
            "instruction": self.instruction_emb(
                x_dict["instruction"][:, 0].long()
            ),
            "variable": self.variable_proj(x_dict["variable"].float()),
            "constant": self.constant_proj(x_dict["constant"].float()),
            "block": self.block_proj(x_dict["block"].float()),
            "function": self.function_proj(x_dict["function"].float()),
            "pragma": self.pragma_proj(
                torch.cat(
                    [
                        self.pragma_emb(x_dict["pragma"][:, 0].long()),
                        self.pragma_arg_proj(x_dict["pragma"][:, 1:].float()),
                    ],
                    dim=-1,
                )
            ),
        }
        edge_emb_dict = {
            edge_type: self.edge_pos_emb(attributes[:, 0])
            for edge_type, attributes in edge_attr_dict.items()
        }
        return h_dict, edge_emb_dict


class CDFGRegionInputProjection(CDFGInputProjection):
    """Add schema-v3 natural-loop features without changing the H0 projector."""

    def __init__(
        self,
        instruction_vocab_size: int,
        edge_pos_vocab_size: int,
        hidden_dim: int,
    ):
        super().__init__(instruction_vocab_size, edge_pos_vocab_size, hidden_dim)
        self.loop_proj = _dense_projection(LOOP_FEATURE_SIZE, hidden_dim)

    def forward(self, x_dict, edge_attr_dict):
        h_dict, edge_emb_dict = super().forward(x_dict, edge_attr_dict)
        h_dict["loop"] = self.loop_proj(x_dict["loop"].float())
        return h_dict, edge_emb_dict
