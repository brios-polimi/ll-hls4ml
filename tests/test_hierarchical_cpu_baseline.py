import importlib.util
from pathlib import Path
import unittest

import pandas as pd
import numpy as np


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "hierarchical_cpu_baseline.py"
SPEC = importlib.util.spec_from_file_location("hierarchical_cpu_baseline", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class HierarchicalCpuBaselineTest(unittest.TestCase):
    def test_archive_number(self):
        self.assertEqual(MODULE.archive_number("conv1d/archive_15/sample.pt"), 15)
        with self.assertRaises(ValueError):
            MODULE.archive_number("conv1d/sample.pt")

    def test_feature_sets_keep_core_context_only(self):
        frame = pd.DataFrame(
            columns=[
                "dataset_index", "kernel_family", "tensor_path", "archive",
                *MODULE.LABEL_KEYS,
                "total_nodes", "edge_a__b__c_count", "instruction_id_1_ratio",
                "context_categorical_0_1", "context_categorical_2_3",
                "context_numeric_0",
            ]
        )
        sets = MODULE.feature_sets(frame)
        self.assertIn("context_categorical_0_1", sets["core_context"])
        self.assertIn("context_numeric_0", sets["core_context"])
        self.assertNotIn("context_categorical_2_3", sets["core_context"])
        self.assertNotIn("edge_a__b__c_count", sets["graph_no_edge_summaries"])
        self.assertIn("instruction_id_1_ratio", sets["graph_opcodes_size"])

    def test_bootstrap_delta_uses_stabilized_smape(self):
        target = np.zeros((1, 6), dtype=np.float32)
        cpu = np.ones((1, 6), dtype=np.float32)
        neural = np.zeros((1, 6), dtype=np.float32)
        delta, low, high, win_fraction = MODULE.bootstrap_delta(
            cpu, neural, target, seed=42, replicates=10
        )
        self.assertEqual(delta, 100.0)
        self.assertEqual(low, 100.0)
        self.assertEqual(high, 100.0)
        self.assertEqual(win_fraction, 0.0)


if __name__ == "__main__":
    unittest.main()
