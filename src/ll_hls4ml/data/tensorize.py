"""Convert CDFG JSON graphs to PyG HeteroData tensors."""

from __future__ import annotations

from pathlib import Path
import shutil
import torch
import tqdm
from torch_geometric.data import HeteroData

from ll_hls4ml.io.discovery import iter_graph_paths
from ll_hls4ml.io.load_json import load_graph_json
from ll_hls4ml.io.schema import (
    FLOW_CALL,
    FLOW_CONTROL,
    FLOW_DATA,
    NODE_CONSTANT,
    NODE_INSTRUCTION,
    NODE_VARIABLE,
    EDGE_TYPES,
    EDGE_TYPES_WITH_ATTR,
    LABEL_KEYS,
    safe_int,
)

from concurrent.futures import ProcessPoolExecutor
from functools import partial
import os

def _process_one(vocab: dict, inference_mode: bool, paths: tuple[Path, Path]) -> None:
    graph_path, out_path = paths
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        graph_data = load_graph_json(graph_path)
        data = _json_to_hetero(graph_data, vocab, inference_mode)
        torch.save(data, out_path)
    except Exception as e:
        raise RuntimeError(f"Error processing graph {graph_path}: {e}") from e


def create_graph_tensors(
    graph_dir: str | Path,
    pt_dir: str | Path,
    vocab: dict,
    kernel_subset: str | list[str] | None = None,
    max_archives: int | None = None,
    inference_mode: bool = False,
    n_workers: int | None = None,
):
    """
    Walk graph_dir and convert JSON files to PyG HeteroData, mirroring structure in pt_dir.
    Fully deletes the pt_dir before creating new tensors, so no vocab mismatch can occur.

    Example: graphs/exemplar/archive_1/*.json → tensors/exemplar/archive_1/*.pt
    """
    graph_dir = Path(graph_dir)
    pt_dir = Path(pt_dir)
    if pt_dir.exists():
        print(f"Deleting existing pt_dir {pt_dir}")
        shutil.rmtree(pt_dir)
    pt_dir.mkdir(parents=True, exist_ok=True)

    work = [
        (graph_path, pt_dir / graph_path.relative_to(graph_dir).parent / (graph_path.stem + ".pt"))
        for ks in ([kernel_subset] if kernel_subset else [None])
        for _, graph_path in iter_graph_paths(graph_dir, ks, max_archives)
    ]

    worker = partial(_process_one, vocab, inference_mode)
    with ProcessPoolExecutor(max_workers=n_workers or os.cpu_count()) as pool:
        list(tqdm.tqdm(pool.map(worker, work), total=len(work), desc="Processing graph files into PyTorch tensors"))


def _json_to_hetero(graph_data: dict, vocab: dict, inference_mode: bool) -> HeteroData:
    data = HeteroData()
    inst_map = {}
    var_map = {}
    const_map = {}
    node_type_map: dict[int, int] = {}

    features = {
        "instruction": [],
        "variable": [],
        "constant": [],
    }
    nodes = graph_data.get("nodes") or []
    for n in nodes:
        try:
            node_id = int(n["id"])
        except KeyError:
            raise ValueError(f"Missing node id in node {n}")

        try:
            node_type = int(n["type"])
        except KeyError:
            raise ValueError(f"Missing node type in node {n}")

        # Convenience for edges mapping source/target node IDs to node types
        node_type_map[node_id] = node_type

        # Map node text to vocabulary index (0 is unknown token)
        # Create global index map for each node type
        node_term = n.get("text", None)
        if node_term is None:
            raise ValueError(f"Missing text field in node {n}")
        if node_type == NODE_INSTRUCTION:
            text_idx = vocab["instruction"].get(node_term, -1) + 1
            features["instruction"].append([text_idx])
            inst_map[node_id] = len(inst_map)
        elif node_type == NODE_VARIABLE:
            text_idx = vocab["variable"].get(node_term, -1) + 1
            features["variable"].append([text_idx])
            var_map[node_id] = len(var_map)
        elif node_type == NODE_CONSTANT:
            text_idx = vocab["constant"].get(node_term, -1) + 1
            features["constant"].append([text_idx])
            const_map[node_id] = len(const_map)
        else:
            raise ValueError(f"Invalid node type: {node_type} in node {n}")

    for k, v in features.items():
        if v:
            data[k].x = torch.tensor(v, dtype=torch.long)


    edge_index = { k: [] for k in EDGE_TYPES }
    edge_attrs = { k: [] for k in EDGE_TYPES_WITH_ATTR }
    edges = graph_data.get("links") or []
    for edge in edges:
        flow = safe_int(edge.get("flow", -1))
        source = safe_int(edge.get("source", -1))
        target = safe_int(edge.get("target", -1))
        if source < 0 or source >= len(nodes) or target < 0 or target >= len(nodes) or flow not in [FLOW_CONTROL, FLOW_DATA, FLOW_CALL]:
            raise ValueError(f"Invalid edge with invalid source/target/flow: {edge}")
            
        position = safe_int(edge.get("position", 0))
        local_idx_source = None
        local_idx_target = None

        if flow == FLOW_CONTROL:
            local_idx_source = inst_map.get(source)
            local_idx_target = inst_map.get(target)
            edge_index[("instruction", "control", "instruction")].append([local_idx_source, local_idx_target])
            edge_attrs[("instruction", "control", "instruction")].append([position])
        elif flow == FLOW_DATA:
            src_type = node_type_map[source]
            if src_type == NODE_INSTRUCTION:
                local_idx_source = inst_map.get(source)
                local_idx_target = var_map.get(target)
                edge_index[("instruction", "data", "variable")].append([local_idx_source, local_idx_target])
            elif src_type == NODE_VARIABLE:
                local_idx_source = var_map.get(source)
                local_idx_target = inst_map.get(target)
                edge_index[("variable", "data", "instruction")].append([local_idx_source, local_idx_target])
                edge_attrs[("variable", "data", "instruction")].append([position])
            elif src_type == NODE_CONSTANT:
                local_idx_source = const_map.get(source)
                local_idx_target = inst_map.get(target)
                edge_index[("constant", "data", "instruction")].append([local_idx_source, local_idx_target])
                edge_attrs[("constant", "data", "instruction")].append([position])
        elif flow == FLOW_CALL:
            local_idx_source = inst_map.get(source)
            local_idx_target = inst_map.get(target)
            edge_index[("instruction", "call", "instruction")].append([local_idx_source, local_idx_target])

        if local_idx_source is None or local_idx_target is None:
            raise ValueError(
                f"Invalid edge indices: {local_idx_source=}, {local_idx_target=}, "
                f"original source={source}, target={target}"
            )

    for et, v in edge_index.items():
        if v:
            data[et].edge_index = torch.tensor(v, dtype=torch.long).t().contiguous()

    for et, v in edge_attrs.items():
        if v:
            data[et].edge_attr = torch.tensor(v, dtype=torch.long)

    # Always add all expected labels unless it is an inference-mode graph
    if not inference_mode:
        try:
            labels = graph_data["labels"]
            data.y = torch.tensor(
                [labels[k] for k in LABEL_KEYS],
                dtype=torch.float,
            )
        except KeyError as e:
            raise ValueError(f"Missing labels in graph data: {e}") from e

    return data
