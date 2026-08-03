"""wa-hls4ml layer/config graphs aligned to ll-hls4ml split manifests."""

from __future__ import annotations

import contextlib
import importlib.util
import io
import json
from pathlib import Path
import sys

import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch_geometric.data import Data


FEATURE_COLUMNS = (
    "d_in1", "d_in2", "d_in3", "d_out1", "d_out2", "d_out3",
    "prec", "rf", "strategy", "layer_type", "activation_type",
    "filters", "kernel_size", "stride", "padding", "pooling",
    "batchnorm", "io_type",
)
NUMERICAL_INDICES = torch.tensor(
    [0, 1, 2, 3, 4, 5, 6, 7, 11, 12, 13, 15], dtype=torch.long
)
LAYER_TYPE_CLASSES = 12
ACTIVATION_CLASSES = 6
PADDING_CLASSES = 3
PROCESSED_FEATURE_DIM = (
    len(NUMERICAL_INDICES)
    + LAYER_TYPE_CLASSES
    + ACTIVATION_CLASSES
    + PADDING_CLASSES
)

FAMILY_LABEL_STEMS = {
    "2layer": "2layer",
    "3layer": "3layer",
    "conv1d": "conv1d",
    "conv2d": "conv2d",
    "dense_latency": "latency",
    "dense_resource": "resource",
    "rule4ml": "2_20",
}


def _load_wa_processor(wa_gnn_dir: Path):
    module_path = wa_gnn_dir / "Dataset_to_csvs6_with_ii.py"
    if not module_path.exists():
        raise FileNotFoundError(f"wa-hls4ml converter not found: {module_path}")
    spec = importlib.util.spec_from_file_location("wa_hls4ml_gnn_converter", module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.ModelProcessor()


def _record_id(record: dict) -> str | None:
    metadata = record.get("meta_data", {})
    value = (
        metadata.get("uuid")
        or metadata.get("model_id")
        or metadata.get("artifacts_file")
    )
    return str(value).removesuffix(".tar.gz") if value else None


def _iter_selected_records(path: Path, wanted: set[str]):
    """Read a pretty-printed top-level JSON array without materializing the file."""
    in_record = False
    keep = False
    prefix: list[str] = []
    lines: list[str] = []
    identifier = None
    with path.open() as handle:
        for line in handle:
            if not in_record:
                if line == "    {\n":
                    in_record = True
                    prefix = [line]
                continue

            if identifier is None:
                prefix.append(line)
                stripped = line.strip()
                if stripped.startswith('"artifacts_file"'):
                    value = json.loads(
                        stripped.split(":", 1)[1].rstrip(",").strip()
                    )
                    identifier = value.removesuffix(".tar.gz")
                    keep = identifier in wanted
                    if keep:
                        lines = prefix
                    prefix = []
            elif keep:
                lines.append(line)

            if line in {"    },\n", "    }\n"}:
                if keep:
                    text = "".join(lines).rstrip()
                    if text.endswith(","):
                        text = text[:-1]
                    yield json.loads(text)
                in_record = False
                keep = False
                prefix = []
                lines = []
                identifier = None


def _processor_frame(processor, record: dict) -> tuple[pd.DataFrame, bool]:
    names, indices = processor.get_layers(record)
    rows = [
        processor.get_layer_info(
            record,
            indices[index],
            indices[index + 1],
            index == len(names) - 1,
        )
        for index in range(len(names))
    ]
    frame = pd.DataFrame(rows, columns=processor.feature_columns)
    with contextlib.redirect_stdout(io.StringIO()):
        processor._process_hls_config(record, frame)

    patched = bool(frame.isna().any().any())
    if patched:
        # The upstream converter drops Add nodes in BiPC, then returns early when
        # its HLS-layer count no longer matches. These model-level values are the
        # same fields it uses for its no-LayerName fallback.
        model = record.get("hls_config", {}).get("Model", {})
        precision = model.get("Precision")
        if isinstance(precision, dict):
            precision = precision.get("weight") or precision.get("default")
        precision = processor.parse_weight_string(precision)
        frame["prec"] = frame["prec"].fillna(precision)
        frame["rf"] = frame["rf"].fillna(model.get("ReuseFactor"))
        strategy = str(model.get("Strategy", "latency")).lower()
        frame["strategy"] = frame["strategy"].fillna(
            processor.STRATEGY_MAPPING.get(strategy, 0)
            if hasattr(processor, "STRATEGY_MAPPING")
            else (1 if strategy == "resource" else 0)
        )
    if frame.empty or frame.isna().any().any():
        missing = frame.columns[frame.isna().any()].tolist()
        raise ValueError(f"Incomplete high-level features: {missing}")
    return frame.loc[:, FEATURE_COLUMNS], patched


def build_high_level_cache(
    manifest_path: Path,
    label_root: Path,
    tensor_root: Path,
    wa_gnn_dir: Path,
    cache_path: Path,
) -> dict:
    """Extract only manifest members and cache compact raw layer features."""
    manifest = json.loads(manifest_path.read_text())
    rows = [
        row
        for split in ("train", "validation", "test", "exemplar")
        for row in manifest[split]
    ]
    path_by_id = {Path(row["tensor_path"]).stem: row for row in rows}
    wanted_by_family: dict[str, set[str]] = {}
    for identifier, row in path_by_id.items():
        wanted_by_family.setdefault(row["kernel_family"], set()).add(identifier)

    processor = _load_wa_processor(wa_gnn_dir)
    records: dict[str, tuple[dict, str]] = {}
    for family, stem in FAMILY_LABEL_STEMS.items():
        wanted = wanted_by_family.get(family, set())
        if not wanted:
            continue
        for directory, prefix in (("train", "train"), ("test", "test"), ("val", "val")):
            path = label_root / directory / f"{prefix}_{stem}_merged.json"
            found = 0
            for record in _iter_selected_records(path, wanted - records.keys()):
                identifier = _record_id(record)
                records[identifier] = (record, str(path.relative_to(label_root)))
                found += 1
            print(f"{path.name}: retained {found}", flush=True)

    exemplar_path = label_root / "exemplar" / "exemplar_models.json"
    exemplar_wanted = wanted_by_family.get("exemplar", set())
    for record in json.loads(exemplar_path.read_text()):
        identifier = _record_id(record)
        if identifier in exemplar_wanted:
            records[identifier] = (
                record,
                str(exemplar_path.relative_to(label_root)),
            )

    missing = sorted(set(path_by_id) - set(records))
    if missing:
        raise ValueError(f"Missing {len(missing)} label records; first: {missing[:5]}")

    tensor_index = json.loads((tensor_root / "labels.json").read_text())
    labels = tensor_index["labels"]
    samples = {}
    patched_ids = []
    for position, (identifier, row) in enumerate(path_by_id.items(), start=1):
        record, source_file = records[identifier]
        frame, patched = _processor_frame(processor, record)
        tensor_path = row["tensor_path"]
        samples[tensor_path] = {
            "features": torch.tensor(frame.to_numpy(dtype="float32")),
            "target": torch.tensor(labels[tensor_path], dtype=torch.float32),
            "kernel_family": row["kernel_family"],
            "label_source": source_file,
            "bi_pc_fallback": patched,
        }
        if patched:
            patched_ids.append(identifier)
        if position % 500 == 0:
            print(f"Converted {position}/{len(path_by_id)}", flush=True)

    cache = {
        "format_version": 1,
        "manifest_path": str(manifest_path),
        "feature_columns": FEATURE_COLUMNS,
        "processed_feature_dim": PROCESSED_FEATURE_DIM,
        "samples": samples,
        "bi_pc_fallback_ids": patched_ids,
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(cache, cache_path)
    print(
        f"Saved {len(samples)} high-level samples to {cache_path}; "
        f"BiPC fallbacks: {len(patched_ids)}",
        flush=True,
    )
    return cache


def feature_statistics(cache: dict, tensor_paths: list[str]):
    values = torch.cat(
        [cache["samples"][path]["features"][:, NUMERICAL_INDICES] for path in tensor_paths]
    )
    means = values.mean(dim=0)
    stds = values.std(dim=0).clamp_min(1e-5)
    return means, stds


class HighLevelLayerDataset(Dataset):
    """Preprocessed homogeneous layer graphs for one saved split."""

    def __init__(self, cache: dict, tensor_paths: list[str], means, stds):
        self.tensor_paths = list(tensor_paths)
        self.targets = torch.stack(
            [cache["samples"][path]["target"] for path in self.tensor_paths]
        )
        self.families = [
            cache["samples"][path]["kernel_family"] for path in self.tensor_paths
        ]
        self.graphs = []
        for index, path in enumerate(self.tensor_paths):
            raw = cache["samples"][path]["features"]
            numerical = (raw[:, NUMERICAL_INDICES] - means) / stds
            layer_type = F.one_hot(
                raw[:, 9].long().clamp(0, LAYER_TYPE_CLASSES - 1),
                LAYER_TYPE_CLASSES,
            ).float()
            activation = F.one_hot(
                raw[:, 10].long().clamp(0, ACTIVATION_CLASSES - 1),
                ACTIVATION_CLASSES,
            ).float()
            padding = F.one_hot(
                raw[:, 14].long().clamp(0, PADDING_CLASSES - 1),
                PADDING_CLASSES,
            ).float()
            x = torch.cat([numerical, layer_type, activation, padding], dim=-1)
            node_count = len(x)
            edge_index = (
                torch.stack([torch.arange(node_count - 1), torch.arange(1, node_count)])
                if node_count > 1
                else torch.empty((2, 0), dtype=torch.long)
            )
            graph = Data(x=x, edge_index=edge_index, y=self.targets[index])
            graph.strategy = F.one_hot(raw[0, 8].long(), 2).float().unsqueeze(0)
            graph.io_type = F.one_hot(raw[0, 17].long(), 2).float().unsqueeze(0)
            self.graphs.append(graph)

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, index):
        return self.graphs[index]
