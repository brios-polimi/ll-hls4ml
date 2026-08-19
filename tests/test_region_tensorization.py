import json
import unittest

import numpy as np
import torch
from torch_geometric.data import Batch

from ll_hls4ml.data.tensorize import _json_to_hetero


def _loop_node(node_id, function, depth, block_count, labels=""):
    return {
        "id": node_id,
        "type": 6,
        "text": "llvm.natural_loop",
        "function": function,
        "block": 0,
        "features": {
            "schema_version": ["3"],
            "header": [f"loop.{node_id}"],
            "depth": [str(depth)],
            "block_count": [str(block_count)],
            "latch_count": ["1"],
            "exit_count": ["1"],
            "trip_count": ["0"],
            "trip_count_known": ["false"],
            "source_loop_labels": [labels],
        },
    }


def region_graph():
    nodes = []
    for node_id, block in enumerate(range(3)):
        nodes.append(
            {
                "id": node_id,
                "type": 0,
                "text": "add",
                "function": 0,
                "block": block,
            }
        )
    for block in range(3):
        nodes.append(
            {
                "id": 3 + block,
                "type": 4,
                "text": "llvm.basic_block",
                "function": 0,
                "block": block,
                "features": {
                    "name": ["entry" if block == 2 else f"body.{block}"],
                    "is_source_loop": ["false"],
                },
            }
        )
    nodes.append(
        {
            "id": 6,
            "type": 5,
            "text": "llvm.function",
            "function": 0,
            "block": -1,
            "features": {"name": ["kernel"], "is_defined": ["true"]},
        }
    )
    nodes.extend(
        [
            _loop_node(7, 0, 0, 2, "OuterLoop"),
            _loop_node(8, 0, 1, 1, "InnerLoop"),
            {
                "id": 9,
                "type": 3,
                "text": "pragma.pipeline",
                "features": {
                    "schema_version": ["2"],
                    "arguments_json": [json.dumps({"ii": ["1"]})],
                },
            },
        ]
    )
    links = []
    for block in range(3):
        links.extend(
            [
                {
                    "source": 3 + block,
                    "target": block,
                    "relation": "contains",
                    "position": 0,
                },
                {
                    "source": 6,
                    "target": 3 + block,
                    "relation": "contains",
                    "position": 0,
                },
            ]
        )
    links.extend(
        [
            {"source": 6, "target": 7, "relation": "contains", "position": 0},
            {"source": 7, "target": 8, "relation": "contains", "position": 0},
            {"source": 7, "target": 3, "relation": "contains", "position": 0},
            {"source": 8, "target": 4, "relation": "contains", "position": 0},
            {
                "source": 9,
                "target": 7,
                "relation": "applies_to",
                "position": 0,
            },
        ]
    )
    return {
        "nodes": nodes,
        "links": links,
        "hierarchy_enrichment": {"schema_version": 3},
    }


class RegionTensorizationTests(unittest.TestCase):
    def test_tensorizes_nested_loops_and_schedule(self):
        data, _ = _json_to_hetero(
            region_graph(), {"add": 1}, inference_mode=True
        )

        self.assertEqual(data.hierarchy_schema_version, 3)
        self.assertEqual(tuple(data["loop"].x.shape), (2, 7))
        np.testing.assert_array_equal(
            data[("loop", "contains", "loop")].edge_index.numpy(),
            np.array([[0], [1]]),
        )
        np.testing.assert_array_equal(
            data[("loop", "contains", "block")].edge_index.numpy(),
            np.array([[0, 1], [0, 1]]),
        )
        np.testing.assert_array_equal(
            data["loop"].nesting_depth.numpy(), np.array([0, 1])
        )
        np.testing.assert_array_equal(
            data["loop"].call_depth.numpy(), np.array([0, 0])
        )
        self.assertEqual(data["loop"].x[:, -1].tolist(), [1.0, 1.0])

    def test_rejects_duplicate_direct_block_ownership(self):
        graph = region_graph()
        graph["links"].append(
            {"source": 8, "target": 3, "relation": "contains", "position": 0}
        )
        with self.assertRaisesRegex(ValueError, "multiple direct loop owners"):
            _json_to_hetero(graph, {"add": 1}, inference_mode=True)

    def test_batching_offsets_loop_and_block_indices(self):
        first, _ = _json_to_hetero(
            region_graph(), {"add": 1}, inference_mode=True
        )
        second, _ = _json_to_hetero(
            region_graph(), {"add": 1}, inference_mode=True
        )
        batch = Batch.from_data_list([first, second])

        np.testing.assert_array_equal(
            batch[("loop", "contains", "block")].edge_index.numpy(),
            np.array([[0, 1, 2, 3], [0, 1, 3, 4]]),
        )
        np.testing.assert_array_equal(
            batch[("loop", "contains", "loop")].edge_index.numpy(),
            np.array([[0, 2], [1, 3]]),
        )


class RegionModelTests(unittest.TestCase):
    def _model(self, name, **extra):
        from ll_hls4ml.models.registry import build

        return build(
            name,
            instruction_vocab_size=2,
            edge_pos_vocab_size=2,
            y_means=torch.zeros(6),
            y_stds=torch.ones(6),
            hidden_dim=16,
            num_layers=1,
            dropout=0.0,
            **extra,
        )

    def test_region_forward_backward_with_both_message_ablations(self):
        data, _ = _json_to_hetero(
            region_graph(), {"add": 1}, inference_mode=True
        )
        batch = Batch.from_data_list([data, data])

        for cardinality_messages, composition in (
            (False, "generic"),
            (True, "generic"),
            (True, "hardware_aligned"),
        ):
            with self.subTest(
                cardinality_messages=cardinality_messages,
                composition=composition,
            ):
                model = self._model(
                    "hierarchical_region",
                    cardinality_messages=cardinality_messages,
                    composition=composition,
                )
                prediction = model(batch)
                self.assertEqual(tuple(prediction.shape), (2, 6))
                prediction.square().mean().backward()
                self.assertIsNotNone(model.loop_input[0].weight.grad)
                self.assertGreater(
                    torch.count_nonzero(model.loop_input[0].weight.grad), 0
                )
                if composition == "hardware_aligned":
                    self.assertIsNotNone(
                        model.composition_pool.resource.weight.grad
                    )
                    self.assertIsNotNone(model.instruction_path.raw_decay.grad)

    def test_h0_remains_usable_on_schema_v2_tensor(self):
        graph = region_graph()
        graph["nodes"] = [node for node in graph["nodes"] if node["type"] != 6]
        graph["links"] = [
            link
            for link in graph["links"]
            if link["source"] not in {7, 8, 9} and link["target"] not in {7, 8}
        ]
        graph["hierarchy_enrichment"]["schema_version"] = 2
        data, _ = _json_to_hetero(graph, {"add": 1}, inference_mode=True)

        prediction = self._model("hierarchical")(Batch.from_data_list([data]))

        self.assertEqual(tuple(prediction.shape), (1, 6))

    def test_schema_v3_graph_without_loops_keeps_zero_gradient_paths(self):
        graph = region_graph()
        graph["nodes"] = [node for node in graph["nodes"] if node["type"] != 6]
        graph["links"] = [
            link
            for link in graph["links"]
            if link["source"] not in {7, 8, 9} and link["target"] not in {7, 8}
        ]
        data, _ = _json_to_hetero(graph, {"add": 1}, inference_mode=True)
        model = self._model(
            "hierarchical_region", composition="hardware_aligned"
        )

        model(Batch.from_data_list([data])).sum().backward()

        gradient = model.loop_input[0].weight.grad
        self.assertIsNotNone(gradient)
        self.assertEqual(torch.count_nonzero(gradient), 0)


if __name__ == "__main__":
    unittest.main()
