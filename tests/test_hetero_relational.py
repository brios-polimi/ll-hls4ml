import unittest

import torch
from torch_geometric.data import HeteroData
from torch_geometric.nn import GATv2Conv

from ll_hls4ml.data.tensorize import EMBED_SIZE
from ll_hls4ml.io.schema import (
    BLOCK_FEATURE_SIZE,
    EDGE_TYPES,
    EDGE_TYPES_WITH_ATTR,
    LABEL_KEYS,
    NODE_TYPES,
    PRAGMA_FEATURE_SIZE,
)
from ll_hls4ml.models.registry import build


def _tiny_graph(edge_position: int = 0) -> HeteroData:
    data = HeteroData()
    data["instruction"].x = torch.tensor([[1], [2]], dtype=torch.long)
    data["variable"].x = torch.zeros((1, EMBED_SIZE))
    data["constant"].x = torch.zeros((1, EMBED_SIZE))
    data["pragma"].x = torch.zeros((1, PRAGMA_FEATURE_SIZE))
    data["block"].x = torch.zeros((1, BLOCK_FEATURE_SIZE))

    indices = {
        ("instruction", "control", "instruction"): [[0], [1]],
        ("instruction", "data", "variable"): [[0], [0]],
        ("variable", "data", "instruction"): [[0], [1]],
        ("constant", "data", "instruction"): [[0], [0]],
        ("instruction", "call", "instruction"): [[0], [1]],
        ("pragma", "applies_to", "instruction"): [[0], [0]],
        ("pragma", "applies_to", "variable"): [[0], [0]],
        ("pragma", "applies_to", "constant"): [[0], [0]],
        ("pragma", "applies_to", "block"): [[0], [0]],
        ("block", "control", "block"): [[0], [0]],
        ("block", "contains", "instruction"): [[0], [0]],
        ("instruction", "in_block", "block"): [[0], [0]],
    }
    for edge_type in EDGE_TYPES:
        data[edge_type].edge_index = torch.tensor(
            indices[edge_type], dtype=torch.long
        )
        if edge_type in EDGE_TYPES_WITH_ATTR:
            data[edge_type].edge_attr = torch.tensor(
                [[edge_position]], dtype=torch.long
            )
    for node_type in NODE_TYPES:
        data[node_type].batch = torch.zeros(
            data[node_type].num_nodes, dtype=torch.long
        )
    data.graph_context_categorical = torch.zeros((1, 4), dtype=torch.long)
    data.graph_context_numeric = torch.zeros((1, 1))
    data.num_graphs = 1
    return data


class HeteroRelationalTests(unittest.TestCase):
    def test_forward_backward_and_edge_positions(self):
        model = build(
            "hetero_relational",
            instruction_vocab_size=3,
            edge_pos_vocab_size=2,
            y_means=torch.zeros(len(LABEL_KEYS)),
            y_stds=torch.ones(len(LABEL_KEYS)),
            hidden_dim=8,
            num_layers=2,
            dropout=0.0,
            pool="multi",
            aggr="sum",
            message_aggr="mean",
            use_global_features=True,
            use_context=True,
            split_heads=True,
        )
        self.assertFalse(
            any(isinstance(module, GATv2Conv) for module in model.modules())
        )

        model.eval()
        first = model(_tiny_graph(edge_position=0))
        second = model(_tiny_graph(edge_position=1))
        self.assertEqual(first.shape, (1, len(LABEL_KEYS)))
        self.assertTrue(torch.isfinite(first).all())
        self.assertFalse(torch.allclose(first, second))

        first.sum().backward()
        self.assertTrue(
            any(
                parameter.grad is not None
                for parameter in model.parameters()
                if parameter.requires_grad
            )
        )


if __name__ == "__main__":
    unittest.main()
