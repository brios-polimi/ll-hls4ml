import json
import unittest

import numpy as np

from ll_hls4ml.data.tensorize import _json_to_hetero, pragma_embedding
from ll_hls4ml.io.schema import (
    PRAGMA_CATEGORICAL_ARGUMENTS,
    PRAGMA_FEATURE_SIZE,
    PRAGMA_NUMERIC_ARGUMENTS,
    PRAGMA_VOCAB,
)


def pragma_node(arguments):
    return {
        "id": 3,
        "type": 3,
        "text": "pragma.pipeline",
        "features": {
            "schema_version": ["2"],
            "arguments_json": [json.dumps(arguments)],
        },
    }


class PragmaTensorizationTests(unittest.TestCase):
    def test_encodes_directive_numeric_mask_and_categorical_arguments(self):
        embedding = pragma_embedding(
            pragma_node(
                {
                    "ii": ["2"],
                    "rewind": ["true"],
                    "style": ["flp"],
                    "variable": ["unstable_source_name"],
                }
            )
        )

        ii_index = PRAGMA_NUMERIC_ARGUMENTS.index("ii")
        mask_offset = 1 + len(PRAGMA_NUMERIC_ARGUMENTS)
        categorical_offset = 1 + 2 * len(PRAGMA_NUMERIC_ARGUMENTS)
        self.assertEqual(embedding.shape, (PRAGMA_FEATURE_SIZE,))
        self.assertEqual(embedding[0], PRAGMA_VOCAB["pragma.pipeline"])
        self.assertAlmostEqual(embedding[1 + ii_index], np.log1p(2))
        self.assertEqual(embedding[mask_offset + ii_index], 1)
        self.assertGreater(embedding[categorical_offset:].sum(), 0)

    def test_target_names_do_not_change_features_because_edges_encode_targets(self):
        first = pragma_embedding(
            pragma_node({"ii": ["1"], "variable": ["weights_a"]})
        )
        second = pragma_embedding(
            pragma_node({"ii": ["1"], "variable": ["weights_b"]})
        )

        np.testing.assert_array_equal(first, second)

    def test_rejects_unstructured_old_schema(self):
        with self.assertRaisesRegex(ValueError, "schema_version"):
            pragma_embedding({"id": 3, "type": 3, "text": "pragma.pipeline"})

    def test_unknown_directive_maps_to_unknown_without_argument_features(self):
        node = pragma_node({})
        node["text"] = "pragma.future_vitis_directive"
        node["features"]["arguments_json"] = [json.dumps({"factor": ["8"]})]

        embedding = pragma_embedding(node)

        self.assertEqual(embedding[0], PRAGMA_VOCAB["UNK"])
        self.assertEqual(embedding[1:].sum(), 0)

    def test_unlisted_arguments_are_retained_in_json_but_not_tensorized(self):
        baseline = pragma_embedding(pragma_node({"ii": ["2"]}))
        with_unknown = pragma_embedding(
            pragma_node({"ii": ["2"], "experimental_knob": ["surprise"]})
        )

        np.testing.assert_array_equal(baseline, with_unknown)

    def test_rejects_symbolic_values_for_numeric_arguments(self):
        node = pragma_node({"ii": ["CONFIG_T::reuse_factor"]})

        with self.assertRaisesRegex(
            ValueError,
            "Numeric pragma argument 'ii' received non-number "
            "'CONFIG_T::reuse_factor' on node 3",
        ):
            pragma_embedding(node)

    def test_categorical_feature_count_is_explicit(self):
        self.assertEqual(
            PRAGMA_FEATURE_SIZE,
            1 + 2 * len(PRAGMA_NUMERIC_ARGUMENTS)
            + len(PRAGMA_CATEGORICAL_ARGUMENTS),
        )

    def test_preserves_pragma_edges_to_llvm_global_constants(self):
        graph = {
            "nodes": [
                {
                    "id": 0,
                    "type": 2,
                    "text": "i32*",
                    "features": {"full_text": ["@global"]},
                },
                {
                    "id": 1,
                    "type": 3,
                    "text": "pragma.array_partition",
                    "features": {
                        "schema_version": ["2"],
                        "arguments_json": [
                            json.dumps({"variable": ["global"]})
                        ],
                    },
                },
            ],
            "links": [
                {
                    "source": 1,
                    "target": 0,
                    "flow": 3,
                    "position": 0,
                }
            ],
        }

        data, _ = _json_to_hetero(graph, {}, inference_mode=True)

        edge_index = data[
            ("pragma", "applies_to", "constant")
        ].edge_index
        np.testing.assert_array_equal(
            edge_index.numpy(),
            np.array([[0], [0]]),
        )

    def test_tensorizes_named_blocks_and_loop_pragma_edges(self):
        graph = {
            "nodes": [
                {"id": 0, "type": 0, "text": "br"},
                {
                    "id": 1,
                    "type": 4,
                    "text": "llvm.basic_block",
                    "features": {
                        "name": ["ReuseLoop"],
                        "is_source_loop": ["true"],
                    },
                },
                {
                    "id": 2,
                    "type": 3,
                    "text": "pragma.pipeline",
                    "features": {
                        "schema_version": ["2"],
                        "arguments_json": [json.dumps({"ii": ["1"]})],
                    },
                },
            ],
            "links": [
                {"source": 1, "target": 0, "flow": 4, "position": 0},
                {"source": 0, "target": 1, "flow": 4, "position": 0},
                {"source": 2, "target": 1, "flow": 3, "position": 0},
            ],
        }

        data, _ = _json_to_hetero(graph, {"br": 1}, inference_mode=True)

        np.testing.assert_array_equal(
            data["block"].x.numpy(),
            np.array([[0, 1, 1]], dtype=np.float32),
        )
        np.testing.assert_array_equal(
            data[("pragma", "applies_to", "block")].edge_index.numpy(),
            np.array([[0], [0]]),
        )
        np.testing.assert_array_equal(
            data[("block", "contains", "instruction")].edge_index.numpy(),
            np.array([[0], [0]]),
        )


if __name__ == "__main__":
    unittest.main()
