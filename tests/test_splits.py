import unittest

from ll_hls4ml.data.splits import benchmark_train_val_test_split


class _Dataset:
    def __init__(self, metadata):
        self.metadata = metadata
        self.paths = [type("Path", (), {"stem": str(i)})() for i in range(len(metadata))]

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, index):
        return index

    def metadata_of(self, index):
        return self.metadata[index]

    def type_of(self, index):
        return self.metadata[index].get("family", "dense")


class BenchmarkSplitTests(unittest.TestCase):
    def test_prefers_complete_official_membership(self):
        dataset = _Dataset(
            [
                {"dataset_split": "train"},
                {"dataset_split": "train"},
                {"dataset_split": "val"},
                {"dataset_split": "test"},
            ]
        )
        train, validation, test = benchmark_train_val_test_split(dataset)
        self.assertEqual(train.indices, [0, 1])
        self.assertEqual(validation.indices, [2])
        self.assertEqual(test.indices, [3])

    def test_group_fallback_keeps_group_members_together(self):
        metadata = []
        for family in ("dense", "conv"):
            for group in range(10):
                metadata.extend(
                    [
                        {"family": family, "group_id": str(group)},
                        {"family": family, "group_id": str(group)},
                    ]
                )
        dataset = _Dataset(metadata)
        subsets = benchmark_train_val_test_split(dataset, seed=7)
        destinations = {}
        for split_index, subset in enumerate(subsets):
            for index in subset.indices:
                key = (dataset.type_of(index), dataset.metadata_of(index)["group_id"])
                destinations.setdefault(key, set()).add(split_index)
        self.assertTrue(all(len(value) == 1 for value in destinations.values()))


if __name__ == "__main__":
    unittest.main()
