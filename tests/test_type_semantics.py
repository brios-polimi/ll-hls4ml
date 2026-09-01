"""Cross-library semantic equivalence of the fixed-point node features."""

import unittest

import numpy as np

from ll_hls4ml.data.tensorize import (
    IS_AP_OFF,
    IS_AC_OFF,
    QUANT_OFF,
    QUANT_SIZE,
    OVERFLOW_OFF,
    OVERFLOW_SIZE,
    BITS_OFF,
    SIGNED_OFF,
    FRAC_OFF,
    SPATIAL_LEN_OFF,
    _json_to_hetero,
    type_embedding,
)


class TypeSemanticsTests(unittest.TestCase):
    def test_equivalent_ap_ac_modes_share_features_except_library_identity(self):
        quant_pairs = [
            ("TRN", 0), ("RND", 1), ("TRN_ZERO", 2), ("RND_ZERO", 3),
            ("RND_INF", 4), ("RND_MIN_INF", 5), ("RND_CONV", 6),
        ]
        overflow_pairs = [("WRAP", 0), ("SAT", 1), ("SAT_ZERO", 2), ("SAT_SYM", 3)]
        for ap_quant, ac_quant in quant_pairs:
            for ap_overflow, ac_overflow in overflow_pairs:
                for signed in (True, False):
                    with self.subTest(q=ap_quant, o=ap_overflow, signed=signed):
                        ap_kind = "ap_fixed" if signed else "ap_ufixed"
                        ap = type_embedding(
                            f'%"class.{ap_kind}<16, 6, AP_{ap_quant}, AP_{ap_overflow}, 0>"*'
                        ).copy()
                        ac = type_embedding(
                            f'%"class.ac_fixed<16, 6, {str(signed).lower()}, '
                            f'(ac_q_mode){ac_quant}, (ac_o_mode){ac_overflow}>"*'
                        ).copy()
                        self.assertEqual(ap[IS_AP_OFF], 1)
                        self.assertEqual(ap[IS_AC_OFF], 0)
                        self.assertEqual(ac[IS_AP_OFF], 0)
                        self.assertEqual(ac[IS_AC_OFF], 1)
                        ap[IS_AP_OFF] = ac[IS_AC_OFF] = 0
                        np.testing.assert_array_equal(ap, ac)

    def test_library_specific_modes_keep_distinct_slots(self):
        odd = type_embedding('ac_fixed<16, 6, true, (ac_q_mode)7, (ac_o_mode)0>')
        even = type_embedding('ap_fixed<16, 6, AP_RND_CONV, AP_WRAP>')
        wrap_sm = type_embedding('ap_fixed<16, 6, AP_TRN, AP_WRAP_SM>')
        np.testing.assert_array_equal(
            odd[QUANT_OFF:QUANT_OFF + QUANT_SIZE], [0, 0, 0, 0, 0, 0, 0, 1]
        )
        self.assertEqual(even[QUANT_OFF + 4], 1)
        np.testing.assert_array_equal(
            odd[OVERFLOW_OFF:OVERFLOW_OFF + OVERFLOW_SIZE], [0, 0, 0, 1, 0]
        )
        np.testing.assert_array_equal(
            wrap_sm[OVERFLOW_OFF:OVERFLOW_OFF + OVERFLOW_SIZE], [0, 0, 0, 0, 1]
        )

    def test_ap_modes_and_range_helpers_match_whole_tokens(self):
        quant_modes = ["RND", "RND_ZERO", "RND_MIN_INF", "RND_INF", "RND_CONV", "TRN", "TRN_ZERO"]
        overflow_modes = ["SAT", "SAT_ZERO", "SAT_SYM", "WRAP", "WRAP_SM"]
        for q_slot, q in enumerate(quant_modes):
            for o_slot, o in enumerate(overflow_modes):
                with self.subTest(q=q, o=o):
                    fixed = type_embedding(f'ap_fixed<16, 6, AP_{q}, AP_{o}>')
                    helper = type_embedding(f'af_range_ref<16, 6, true, AP_{q}, AP_{o}, 0>')
                    np.testing.assert_array_equal(
                        fixed[QUANT_OFF:QUANT_OFF + QUANT_SIZE], np.eye(QUANT_SIZE)[q_slot]
                    )
                    np.testing.assert_array_equal(
                        fixed[OVERFLOW_OFF:OVERFLOW_OFF + OVERFLOW_SIZE], np.eye(OVERFLOW_SIZE)[o_slot]
                    )
                    np.testing.assert_array_equal(fixed, helper)

    def test_unknown_ac_modes_cannot_overwrite_adjacent_features(self):
        for q, o in ((8, 0), (0, 4), (99, 99)):
            with self.subTest(q=q, o=o):
                with self.assertRaisesRegex(ValueError, "Unsupported AC"):
                    type_embedding(f'ac_fixed<16, 6, true, (ac_q_mode){q}, (ac_o_mode){o}>')

    def test_tensorization_records_semantic_schema_and_aligned_features(self):
        graph = {
            "nodes": [{
                "id": 0, "type": 1,
                "text": '%"class.ac_fixed<16, 6, true, (ac_q_mode)0, (ac_o_mode)0>"',
            }],
            "links": [],
        }
        data, unknown = _json_to_hetero(graph, {}, inference_mode=True)
        self.assertEqual(data.type_embedding_schema_version, 3)
        self.assertFalse(unknown)
        self.assertEqual(data["variable"].x[0, QUANT_OFF + 5].item(), 1)
        self.assertEqual(data["variable"].x[0, OVERFLOW_OFF + 3].item(), 1)

    def test_debug_ac_integer_types_use_semantic_width_and_signedness(self):
        for spelling in ('ac_int<9, false>', 'ac_private::iv<1, true, 9, false>',
                         'ac_private::iv_base<1, true, 9, false>',
                         'ac_private::iv_conv<1, false, true, true, 9>'):
            with self.subTest(spelling=spelling):
                emb = type_embedding(f'%"{spelling}"*')
                self.assertEqual(emb[BITS_OFF], 9)
                self.assertEqual(emb[SIGNED_OFF], 0)
                self.assertEqual(emb[IS_AC_OFF], 1)
        self.assertEqual(type_embedding('ac_private::iv<1, false, 9, true>')[SIGNED_OFF], 1)

    def test_debug_channels_arrays_and_negative_integer_bits(self):
        payload = 'nnet::array<ac_fixed<16, -2, true, (ac_q_mode)0, (ac_o_mode)0>, 32U>'
        for spelling in (f'ac_channel<{payload} >', f'ac_channel<{payload} >::fifo',
                         f'hls::stream<{payload}, 0>'):
            with self.subTest(spelling=spelling):
                emb = type_embedding(f'%"{spelling}"*')
                self.assertEqual(emb[7], 1)
                self.assertEqual(emb[8], 1)
                self.assertEqual(emb[BITS_OFF], 16)
                self.assertEqual(emb[FRAC_OFF], 18 / 16)
                self.assertEqual(emb[SPATIAL_LEN_OFF], 5)


if __name__ == "__main__":
    unittest.main()
