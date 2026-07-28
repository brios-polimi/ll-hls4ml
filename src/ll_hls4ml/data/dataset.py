"""PyG HeteroData dataset over preprocessed .pt files."""
from collections import defaultdict
import json
from pathlib import Path

import torch
from torch.utils.data import Dataset
from torch_geometric.data import HeteroData

from ll_hls4ml.io.schema import LABEL_KEYS


class HeteroGraphDataset(Dataset):
    """
    Lazy-loading dataset for HeteroData .pt graphs.
    Indexes the filesystem at init time, loads graphs on demand.
    """

    def __init__(
        self,
        root: str | Path,
        types: list[str] | None = None,
        max_per_type: dict[str, int] | int | None = None,
        transform=None,
        silent: bool = True,
    ):
        self.root = Path(root)
        self.transform = transform
        self.paths = self._index(types, max_per_type, silent)
        self.targets = self._load_targets()

    def _load_targets(self) -> torch.Tensor | None:
        """Load per-tensor labels without deserializing graph tensors, if available."""
        index_path = self.root / "labels.json"
        if not index_path.exists():
            return None
        try:
            with index_path.open() as handle:
                index = json.load(handle)
            if index["label_keys"] != LABEL_KEYS:
                return None
            labels = index["labels"]
            values = [labels[path.relative_to(self.root).as_posix()] for path in self.paths]
        except (KeyError, OSError, ValueError, json.JSONDecodeError):
            return None
        return torch.tensor(values, dtype=torch.float)

    def _index(self, types: list[str] | None, max_per_type: dict[str, int] | int | None, silent: bool) -> list[Path]:
        paths = []
        root = self.root

        type_dirs = (
            [root / t for t in types]
            if types
            else [p for p in root.iterdir() if p.is_dir()]
        )

        type_counts = defaultdict(int)
        type_sizes = defaultdict(int)
        for type_dir in sorted(type_dirs):
            if not type_dir.exists():
                raise FileNotFoundError(f"Type directory not found: {type_dir}")
            for archive_dir in sorted(type_dir.iterdir()):
                if not archive_dir.is_dir():
                    continue

                if isinstance(max_per_type, dict):
                    max_this_kernel = max_per_type.get(type_dir.name, None)
                elif isinstance(max_per_type, int):
                    max_this_kernel = max_per_type
                else:
                    max_this_kernel = None

                for pt_file in sorted(archive_dir.glob("*.pt")):
                    if max_this_kernel is not None and type_counts[type_dir.name] >= max_this_kernel:
                        break
                    type_counts[type_dir.name] += 1
                    paths.append(pt_file)
                    type_sizes[type_dir.name] += pt_file.stat().st_size

        if not silent:
            print(f"Indexed {len(paths)} graphs across {len(type_dirs)} type(s)")
            for type_name, count in type_counts.items():
                print(f"  {type_name}: {count}")
                print(f"    avg size: {type_sizes[type_name] / count / 1024 / 1024 :.2f} MB")
                print(f"    total size: {type_sizes[type_name] / 1024 / 1024 :.2f} MB")
        return paths

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> HeteroData:
        data = torch.load(self.paths[idx], weights_only=False)
        if self.transform:
            data = self.transform(data)
        return data

    def type_of(self, idx: int) -> str:
        """Return kernel type (e.g. 'exemplar') for a given index."""
        return self.paths[idx].parts[-3]
