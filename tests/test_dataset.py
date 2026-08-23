import json
from pathlib import Path
import tempfile
import unittest

import torch

from ll_hls4ml.data.dataset import HeteroGraphDataset
from ll_hls4ml.io.schema import LABEL_KEYS


class HeteroGraphDatasetTests(unittest.TestCase):
    def test_indexes_manifest_paths_without_tensor_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            relative_paths = [
                "conv2d/archive_1/first.pt",
                "exemplar/archive_1/second.pt",
            ]
            labels = {
                "label_keys": LABEL_KEYS,
                "labels": {
                    path: [float(index)] * len(LABEL_KEYS)
                    for index, path in enumerate(relative_paths)
                },
                "metadata": {
                    path: {"graph_id": Path(path).stem}
                    for path in relative_paths
                },
            }
            (root / "labels.json").write_text(json.dumps(labels))

            dataset = HeteroGraphDataset(root, relative_paths=relative_paths)

            self.assertEqual(len(dataset), 2)
            self.assertFalse(any(path.exists() for path in dataset.paths))
            self.assertEqual(dataset.type_of(0), "conv2d")
            self.assertEqual(dataset.metadata[1]["graph_id"], "second")
            torch.testing.assert_close(
                dataset.targets[1], torch.ones(len(LABEL_KEYS))
            )

    def test_deduplicates_graph_ids_across_archives(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "conv2d" / "archive_1" / "same-id.pt"
            duplicate = root / "conv2d" / "archive_2" / "same-id.pt"
            unique = root / "conv2d" / "archive_2" / "unique-id.pt"
            for path in (first, duplicate, unique):
                path.parent.mkdir(parents=True, exist_ok=True)
                torch.save({"path": path.as_posix()}, path)

            labels = {
                "label_keys": LABEL_KEYS,
                "labels": {
                    path.relative_to(root).as_posix(): [0.0] * len(LABEL_KEYS)
                    for path in (first, duplicate, unique)
                },
                "metadata": {},
            }
            (root / "labels.json").write_text(json.dumps(labels))

            dataset = HeteroGraphDataset(root)

            self.assertEqual([path.stem for path in dataset.paths], ["same-id", "unique-id"])
            self.assertEqual(dataset.duplicate_paths, [duplicate])


if __name__ == "__main__":
    unittest.main()
