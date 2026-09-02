import json
import unittest

import numpy as np

from ll_hls4ml.data.tensorize import (
    SPATIAL_LEN_OFF,
    TEMPORAL_LEN_OFF,
    LITERAL_OFF,
    IS_AC_OFF,
    BITS_OFF,
    FRAC_OFF,
    SIGNED_OFF,
    OVERFLOW_OFF,
    QUANT_OFF,
    _json_to_hetero,
    pragma_embedding,
    type_embedding,
)
from ll_hls4ml.features.graph_stats import semantic_type_stats
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
    def test_ap_fixed_shorthand_uses_truncate_and_wrap_defaults(self):
        embedding = type_embedding('%"struct.ap_fixed<8, 1>"')
        self.assertEqual(embedding[QUANT_OFF + 5], 1)
        self.assertEqual(embedding[OVERFLOW_OFF + 3], 1)
        self.assertEqual(embedding[QUANT_OFF], 0)
        self.assertEqual(embedding[OVERFLOW_OFF], 0)

    def test_bambu_quoted_ac_fixed_type_is_embedded(self):
        embedding = type_embedding(
            '%"ac_fixed<16, 6, true, (ac_q_mode)0, (ac_o_mode)0>"*'
        )
        self.assertEqual(embedding[4], 1)
        self.assertEqual(embedding[IS_AC_OFF], 1)
        self.assertEqual(embedding[BITS_OFF], 16)
        self.assertAlmostEqual(embedding[FRAC_OFF], 10 / 16)
        self.assertEqual(embedding[SIGNED_OFF], 1)

    def test_shift_register_length_and_constant_literal_are_retained(self):
        shift = type_embedding(
            '%"class.ap_shift_reg<ap_ufixed<4, 1, AP_RND_CONV, AP_SAT>, 9>"'
        )
        self.assertAlmostEqual(shift[TEMPORAL_LEN_OFF], np.log2(9))

        graph = {
            "nodes": [
                {
                    "id": 0,
                    "type": 2,
                    "text": "i64",
                    "features": {"full_text": ["i64 64"]},
                }
            ],
            "links": [],
        }
        data, _ = _json_to_hetero(graph, {}, inference_mode=True)
        literal = data["constant"].x[0, LITERAL_OFF:].numpy()
        self.assertEqual(literal[0], 1)
        self.assertAlmostEqual(literal[1], np.log1p(64), places=6)
        self.assertEqual(literal[-1], 1)

    def test_spatial_and_temporal_lengths_are_aggregated_separately(self):
        stats = semantic_type_stats(
            [
                {
                    "type": 1,
                    "text": (
                        '%"struct.nnet::array<'
                        'ap_fixed<20, 10, AP_TRN, AP_WRAP, 0>, 32>"'
                    ),
                },
                {
                    "type": 1,
                    "text": (
                        '%"class.ap_shift_reg<'
                        'ap_ufixed<4, 1, AP_RND_CONV, AP_SAT>, 9>"'
                    ),
                },
            ]
        )
        self.assertAlmostEqual(stats["type_spatial_log_length_mean"], 5.0)
        self.assertAlmostEqual(
            stats["type_temporal_log_length_mean"], np.log2(9)
        )

    def test_graph_context_is_tensorized(self):
        graph = {
            "nodes": [],
            "links": [],
            "synthesis_metadata": {
                "backend": "vitis",
                "target_part": "xcu250-figd2104-2L-e",
                "vivado_version": "2023.2",
                "hls4ml_version": "0.8.1",
                "target_clock": "5.0",
            },
        }
        data, _ = _json_to_hetero(graph, {}, inference_mode=True)
        np.testing.assert_array_equal(
            data.graph_context_categorical.numpy(),
            np.array([[1, 2, 3, 5]]),
        )
        self.assertAlmostEqual(
            data.graph_context_numeric.item(),
            np.log1p(5),
        )

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
                    "relation": "applies_to",
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
                {
                    "id": 0,
                    "type": 0,
                    "text": "br",
                    "function": 0,
                    "block": 0,
                },
                {
                    "id": 1,
                    "type": 4,
                    "text": "llvm.basic_block",
                    "function": 0,
                    "block": 0,
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
                {
                    "id": 3,
                    "type": 5,
                    "text": "llvm.function",
                    "function": 0,
                    "block": -1,
                    "features": {
                        "name": ["kernel"],
                        "is_defined": ["true"],
                    },
                },
            ],
            "links": [
                {"source": 2, "target": 1, "relation": "applies_to", "position": 0},
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
        np.testing.assert_array_equal(
            data[("function", "contains", "block")].edge_index.numpy(),
            np.array([[0], [0]]),
        )

    def test_rejects_serialized_containment(self):
        graph = {
            "nodes": [
                {"id": 0, "type": 0, "text": "ret", "function": 0, "block": 0},
                {"id": 1, "type": 4, "text": "llvm.basic_block", "function": 0, "block": 0},
                {"id": 2, "type": 5, "text": "llvm.function", "function": 0, "block": -1},
            ],
            "links": [
                {"source": 1, "target": 0, "relation": "contains"},
            ],
        }

        with self.assertRaisesRegex(ValueError, "outside the canonical schema"):
            _json_to_hetero(graph, {"ret": 1}, inference_mode=True)


if __name__ == "__main__":
    unittest.main()
