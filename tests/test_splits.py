import unittest
from pathlib import Path

from torch.utils.data import Subset

from ll_hls4ml.data.splits import (
    benchmark_train_val_test_split,
    limit_subset_archives,
    nested_group_train_subset,
)


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

    def test_nested_group_scaling_uses_fixed_baseline_and_archive_expansion(self):
        metadata = []
        paths = []
        for family in ("dense", "conv"):
            for archive in range(1, 5):
                for group in range(4):
                    metadata.append(
                        {
                            "family": family,
                            "group_id": f"{family}-{archive}-{group}",
                        }
                    )
                    paths.append(
                        Path(
                            f"/tensors/{family}/archive_{archive}/"
                            f"{group}.pt"
                        )
                    )
        dataset = _Dataset(metadata)
        dataset.paths = paths
        full = Subset(dataset, range(len(dataset)))

        subsets = {}
        for scale, expected_per_family in (
            (0.25, 2),
            (0.5, 4),
            (1.0, 8),
            (2.0, 16),
        ):
            subset, report = nested_group_train_subset(
                dataset,
                full,
                scale,
                baseline_archives_per_family=2,
                seed=11,
            )
            subsets[scale] = set(subset.indices)
            self.assertEqual(
                {
                    family: values["selected_samples"]
                    for family, values in report["families"].items()
                },
                {"conv": expected_per_family, "dense": expected_per_family},
            )

        self.assertTrue(subsets[0.25] < subsets[0.5])
        self.assertTrue(subsets[0.5] < subsets[1.0])
        self.assertTrue(subsets[1.0] < subsets[2.0])

        limited, archives = limit_subset_archives(dataset, full, 2)
        self.assertEqual(
            set(archives["dense"]),
            {"archive_1", "archive_2"},
        )
        self.assertEqual(len(limited), 16)


if __name__ == "__main__":
    unittest.main()
