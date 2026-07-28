"""Convert CDFG JSON graphs to PyG HeteroData tensors."""

from __future__ import annotations

from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache, partial
import json
import math
import os
from pathlib import Path
import re
import tempfile

import numpy as np
import tqdm

from ll_hls4ml.io.schema import (
    EDGE_TYPES,
    EDGE_TYPES_WITH_ATTR,
    BLOCK_FEATURE_SIZE,
    FLOW_BLOCK,
    FLOW_CALL,
    FLOW_CONTROL,
    FLOW_DATA,
    FLOW_PRAGMA,
    NODE_CONSTANT,
    NODE_BLOCK,
    NODE_INSTRUCTION,
    NODE_PRAGMA,
    NODE_VARIABLE,
    LABEL_KEYS,
    PRAGMA_ARGUMENT_SIZE,
    PRAGMA_ARGUMENT_DIRECTIVES,
    PRAGMA_CATEGORICAL_ARGUMENTS,
    PRAGMA_FEATURE_SIZE,
    PRAGMA_NUMERIC_ARGUMENTS,
    PRAGMA_SCHEMA_VERSION,
    PRAGMA_TARGET_ARGUMENTS,
    pragma_directive_id,
    safe_int,
)
from ll_hls4ml.io.discovery import iter_graph_paths
from ll_hls4ml.io.load_json import load_graph_json


_LABEL_INDEX_NAME = "labels.json"
_PRAGMA_NUMERIC_INDICES = {
    name: index for index, name in enumerate(PRAGMA_NUMERIC_ARGUMENTS)
}
_PRAGMA_CATEGORICAL_INDICES = {
    token: index for index, token in enumerate(PRAGMA_CATEGORICAL_ARGUMENTS)
}
_PRAGMA_NUMERIC_OFFSET = 1
_PRAGMA_NUMERIC_MASK_OFFSET = _PRAGMA_NUMERIC_OFFSET + len(PRAGMA_NUMERIC_ARGUMENTS)
_PRAGMA_CATEGORICAL_OFFSET = _PRAGMA_NUMERIC_MASK_OFFSET + len(PRAGMA_NUMERIC_ARGUMENTS)


def _update_label_index(
    pt_dir: Path,
    records: list[tuple[Path, list[float] | None]],
) -> None:
    """Merge labels from this tensorization run into the tensor sidecar index."""
    index_path = pt_dir / _LABEL_INDEX_NAME
    index = {"label_keys": LABEL_KEYS, "labels": {}}
    if index_path.exists():
        with index_path.open() as handle:
            existing = json.load(handle)
        if existing.get("label_keys") == LABEL_KEYS:
            index["labels"].update(existing.get("labels", {}))

    for out_path, labels in records:
        relative_path = out_path.relative_to(pt_dir).as_posix()
        if labels is None:
            index["labels"].pop(relative_path, None)
        else:
            index["labels"][relative_path] = labels

    with tempfile.NamedTemporaryFile(
        mode="w", dir=pt_dir, prefix=".labels-", suffix=".json", delete=False
    ) as handle:
        json.dump(index, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temp_path = Path(handle.name)
    temp_path.replace(index_path)


def _single_feature(node: dict, name: str) -> str:
    values = node.get("features", {}).get(name)
    if not isinstance(values, list) or len(values) != 1:
        raise ValueError(
            f"Pragma node {node.get('id')} requires one features.{name} value"
        )
    return str(values[0])


def _scaled_number(value: str) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return math.copysign(math.log1p(abs(number)), number)


def pragma_embedding(node: dict) -> np.ndarray:
    """Encode the explicit, inspectable v2 pragma feature schema."""

    version = int(_single_feature(node, "schema_version"))
    if version != PRAGMA_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported pragma schema version {version}; "
            f"expected {PRAGMA_SCHEMA_VERSION}"
        )
    try:
        arguments = json.loads(_single_feature(node, "arguments_json"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid pragma arguments_json on node {node.get('id')}") from exc
    if not isinstance(arguments, dict):
        raise ValueError(f"Pragma arguments_json must be an object on node {node.get('id')}")

    embedding = np.zeros(PRAGMA_FEATURE_SIZE, dtype=np.float32)
    embedding[0] = pragma_directive_id(node["text"])
    if node["text"] not in PRAGMA_ARGUMENT_DIRECTIVES:
        return embedding

    for raw_key, raw_values in arguments.items():
        key = str(raw_key)
        if not isinstance(raw_values, list):
            raise ValueError(
                f"Pragma argument {key!r} must contain a list on node {node.get('id')}"
            )
        if key in PRAGMA_TARGET_ARGUMENTS:
            # Targets are represented structurally by FLOW_PRAGMA edges.
            continue
        for raw_value in raw_values:
            value = str(raw_value)
            numeric_index = _PRAGMA_NUMERIC_INDICES.get(key)
            if numeric_index is not None:
                number = _scaled_number(value)
                if number is None:
                    raise ValueError(
                        f"Numeric pragma argument {key!r} received non-number "
                        f"{value!r} on node {node.get('id')} "
                        f"({node.get('text', 'pragma.unknown')})"
                    )
                embedding[_PRAGMA_NUMERIC_OFFSET + numeric_index] = number
                embedding[_PRAGMA_NUMERIC_MASK_OFFSET + numeric_index] = 1.0
                continue
            categorical_index = _PRAGMA_CATEGORICAL_INDICES.get((key, value.lower()))
            if categorical_index is not None:
                embedding[_PRAGMA_CATEGORICAL_OFFSET + categorical_index] = 1.0

    if embedding.shape != (1 + PRAGMA_ARGUMENT_SIZE,):
        raise AssertionError("Pragma feature schema size is inconsistent")
    return embedding


def block_embedding(node: dict) -> np.ndarray:
    """Encode stable block roles while retaining the exact name in graph JSON."""

    features = node.get("features", {})
    names = features.get("name", [])
    name = str(names[0]) if len(names) == 1 else ""
    loop_values = features.get("is_source_loop", [])
    is_source_loop = (
        len(loop_values) == 1 and str(loop_values[0]).lower() == "true"
    )
    return np.asarray(
        [name == "entry", bool(name), is_source_loop],
        dtype=np.float32,
    )


def _process_one(
    vocab: dict,
    max_pos: int,
    inference_mode: bool,
    paths: tuple[Path, Path],
) -> tuple[set[str], list[float] | None]:
    import torch

    graph_path, out_path = paths
    try:
        graph_data = load_graph_json(graph_path)
        data, unknown_types = _json_to_hetero(graph_data, vocab, max_pos, inference_mode)
        torch.save(data, out_path)
        labels = None if inference_mode else data.y.tolist()
        return unknown_types, labels
    except Exception as e:
        raise RuntimeError(f"Error processing graph {graph_path}: {e}") from e


def create_graph_tensors(
    graph_dir: str | Path,
    pt_dir: str | Path,
    instruction_vocab: dict,
    max_pos: int | None = None,
    kernel_subset: str | list[str] | None = None,
    archive_subset: str | None = None,
    max_archives: int | None = None,
    inference_mode: bool = False,
    n_workers: int | None = None,
):
    """
    Walk graph_dir and convert JSON files to PyG HeteroData, mirroring structure in pt_dir.

    Example: graphs/exemplar/archive_1/*.json → tensors/exemplar/archive_1/*.pt
    """
    graph_dir = Path(graph_dir)
    pt_dir = Path(pt_dir)
    pt_dir.mkdir(parents=True, exist_ok=True)

    graph_paths = [
        graph_path
        for _, graph_path in iter_graph_paths(
            graph_dir, kernel_subset, max_archives
        )
    ]
    if archive_subset is not None:
        if kernel_subset is None or not isinstance(kernel_subset, str):
            raise ValueError("archive_subset requires one explicit kernel_subset")
        graph_paths = [
            path
            for path in graph_paths
            if path.relative_to(graph_dir / kernel_subset).parts[0] == archive_subset
        ]
        if not graph_paths:
            raise ValueError(
                f"No graphs found for {kernel_subset}/{archive_subset}"
            )
    work = [
        (
            graph_path,
            pt_dir
            / graph_path.relative_to(graph_dir).parent
            / (graph_path.stem + ".pt"),
        )
        for graph_path in graph_paths
    ]
    for output_dir in {out_path.parent for _, out_path in work}:
        output_dir.mkdir(parents=True, exist_ok=True)

    worker = partial(_process_one, instruction_vocab, max_pos, inference_mode)
    temp_on_mounted_filesystem = (
        Path(tempfile.gettempdir()).as_posix().startswith("/mnt/")
    )
    default_workers = (
        1 if temp_on_mounted_filesystem else min(8, os.cpu_count() or 1)
    )
    workers = n_workers or default_workers
    unknown_types = Counter()

    if workers == 1:
        results = map(worker, work)
        pool = None
    else:
        pool = ProcessPoolExecutor(max_workers=workers)
        results = pool.map(worker, work, chunksize=32)

    try:
        label_records = []
        for (_graph_path, out_path), (result, labels) in zip(
            work,
            tqdm.tqdm(
                results,
                total=len(work),
                desc="Processing graph files into PyTorch tensors",
            ),
        ):
            normalized = {
                re.sub(r'\.\d+(?="|\b)', '', type_name)
                for type_name in result
            }
            unknown_types.update(normalized)
            label_records.append((out_path, labels))
    finally:
        if pool is not None:
            pool.shutdown()

    _update_label_index(pt_dir, label_records)

    if unknown_types:
        print("Types not parsed by embedder:")
        for t, count in unknown_types.most_common():
            print(f"{count:>6}  {t}")


def _json_to_hetero(
    graph_data: dict,
    instruction_vocab: dict,
    max_pos: int | None = None,
    inference_mode: bool = False,
):
    import torch
    from torch_geometric.data import HeteroData

    data = HeteroData()
    inst_map = {}
    var_map = {}
    const_map = {}
    pragma_map = {}
    block_map = {}
    node_type_map: dict[int, int] = {}

    unknown_types = set()

    features = {
        "instruction": [],
        "variable": [],
        "constant": [],
        "pragma": [],
        "block": [],
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

        try:
            node_text = n["text"]
        except KeyError:
            raise ValueError(f"Missing text field in node {n}")

        # Convenience for edges mapping source/target node IDs to node types
        node_type_map[node_id] = node_type

        # Instruction: Map node text to vocabulary index (0 is unknown token)
        # Variable/Constant: Translate node text to type tensor schema
        # Create global index map for each node type
        if node_type == NODE_INSTRUCTION:
            text_idx = instruction_vocab.get(node_text, 0)
            features["instruction"].append([text_idx])
            inst_map[node_id] = len(inst_map)
        elif node_type == NODE_VARIABLE:
            type_emb = type_embedding(node_text)
            if type_emb[TYPE_SIZE - 1]:
                unknown_types.add(node_text)
            features["variable"].append(type_emb)
            var_map[node_id] = len(var_map)
        elif node_type == NODE_CONSTANT:
            type_emb = type_embedding(node_text)
            if type_emb[TYPE_SIZE - 1]:
                unknown_types.add(node_text)
            features["constant"].append(type_emb)
            const_map[node_id] = len(const_map)
        elif node_type == NODE_PRAGMA:
            features["pragma"].append(pragma_embedding(n))
            pragma_map[node_id] = len(pragma_map)
        elif node_type == NODE_BLOCK:
            features["block"].append(block_embedding(n))
            block_map[node_id] = len(block_map)
        else:
            raise ValueError(f"Invalid node type: {node_type} in node {n}")

    data["instruction"].x = torch.tensor(
        features["instruction"], dtype=torch.long
    ).reshape(-1, 1)
    data["variable"].x = torch.from_numpy(
        np.stack(features["variable"])
        if features["variable"]
        else np.empty((0, EMBED_SIZE), dtype=np.float32)
    )
    data["constant"].x = torch.from_numpy(
        np.stack(features["constant"])
        if features["constant"]
        else np.empty((0, EMBED_SIZE), dtype=np.float32)
    )
    data["pragma"].x = torch.tensor(
        np.stack(features["pragma"])
        if features["pragma"]
        else np.empty((0, PRAGMA_FEATURE_SIZE), dtype=np.float32),
        dtype=torch.float,
    )
    data["block"].x = torch.tensor(
        np.stack(features["block"])
        if features["block"]
        else np.empty((0, BLOCK_FEATURE_SIZE), dtype=np.float32),
        dtype=torch.float,
    )


    edge_index = { k: [] for k in EDGE_TYPES }
    edge_attrs = { k: [] for k in EDGE_TYPES_WITH_ATTR }
    edges = graph_data.get("links") or []
    for edge in edges:
        flow = safe_int(edge.get("flow", -1))
        source = safe_int(edge.get("source", -1))
        target = safe_int(edge.get("target", -1))
        if source < 0 or source >= len(nodes) or target < 0 or target >= len(nodes) or flow not in [FLOW_CONTROL, FLOW_DATA, FLOW_CALL, FLOW_PRAGMA, FLOW_BLOCK]:
            raise ValueError(f"Invalid edge with invalid source/target/flow: {edge}")
            
        position = safe_int(edge.get("position", 0))
        if max_pos is not None and position > max_pos:
            raise ValueError(f"Invalid edge with max pos too high: {position} > {max_pos}")
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
        elif flow == FLOW_PRAGMA:
            local_idx_source = pragma_map.get(source)
            target_type = node_type_map[target]
            if target_type == NODE_INSTRUCTION:
                local_idx_target = inst_map.get(target)
                edge_index[("pragma", "applies_to", "instruction")].append(
                    [local_idx_source, local_idx_target]
                )
            elif target_type == NODE_VARIABLE:
                local_idx_target = var_map.get(target)
                edge_index[("pragma", "applies_to", "variable")].append(
                    [local_idx_source, local_idx_target]
                )
            elif target_type == NODE_CONSTANT:
                local_idx_target = const_map.get(target)
                edge_index[("pragma", "applies_to", "constant")].append(
                    [local_idx_source, local_idx_target]
                )
            elif target_type == NODE_BLOCK:
                local_idx_target = block_map.get(target)
                edge_index[("pragma", "applies_to", "block")].append(
                    [local_idx_source, local_idx_target]
                )
        elif flow == FLOW_BLOCK:
            source_type = node_type_map[source]
            target_type = node_type_map[target]
            if source_type == NODE_BLOCK and target_type == NODE_BLOCK:
                local_idx_source = block_map.get(source)
                local_idx_target = block_map.get(target)
                edge_index[("block", "control", "block")].append(
                    [local_idx_source, local_idx_target]
                )
            elif source_type == NODE_BLOCK and target_type == NODE_INSTRUCTION:
                local_idx_source = block_map.get(source)
                local_idx_target = inst_map.get(target)
                edge_index[("block", "contains", "instruction")].append(
                    [local_idx_source, local_idx_target]
                )
            elif source_type == NODE_INSTRUCTION and target_type == NODE_BLOCK:
                local_idx_source = inst_map.get(source)
                local_idx_target = block_map.get(target)
                edge_index[("instruction", "in_block", "block")].append(
                    [local_idx_source, local_idx_target]
                )

        if local_idx_source is None or local_idx_target is None:
            raise ValueError(
                f"Invalid edge indices: {local_idx_source=}, {local_idx_target=}, "
                f"original source={source}, target={target}"
            )

    for et, v in edge_index.items():
        data[et].edge_index = (
            torch.tensor(v, dtype=torch.long).t().contiguous()
            if v
            else torch.empty((2, 0), dtype=torch.long)
        )

    for et, v in edge_attrs.items():
        data[et].edge_attr = (
            torch.tensor(v, dtype=torch.long)
            if v
            else torch.empty((0, 1), dtype=torch.long)
        )

    # Always add all expected labels unless it is an inference-mode graph
    if not inference_mode:
        try:
            labels = graph_data["labels"]
            data.y = torch.tensor(
                [labels[k] for k in LABEL_KEYS],
                dtype=torch.float,
            )
        except KeyError as e:
            raise ValueError(f"Missing labels in graph data") from e

    return data, unknown_types



# ---- regexes compiled once ----
# %"class.hls::stream<nnet::array<ap_fixed<37, 8, AP_TRN, AP_WRAP, 0>, 32>, 0>"
STREAM_RE = re.compile(
    r'^(?:%"class\.)?hls::stream<\s*(.*)(?:,\s*(\d+))?>"?'
)

# nnet::array<ap_fixed<20, 10, AP_TRN, AP_WRAP, 0>
NNET_ARRAY_RE = re.compile(
    r'^(?:%"struct\.)?nnet::array<\s*(.*),\s*(\d+)>"?'
)

# %"class.ap_shift_reg<ap_ufixed<4, 1, AP_RND_CONV, AP_SAT>, 9>"*
SHIFT_REG_RE = re.compile(
    r'^(?:%"class\.)?ap_shift_reg<\s*(.*)(?:,\s*(\d+))?>"?'
)

# i32, i57, float, double
INT_RE = re.compile(r"^i(\d+)$")
FLOAT_RE = re.compile(r"^(float|double)$")

# ap_int<55>
# ap_int_base<64, true>
# ap_private<65, true, false>
# ssdm_int_sim<6, true>
AP_INT_RE = re.compile(
    r'^(?:%"struct\.)?(?:ap|ssdm)_(u?)(?:int|private)(?:_base|_sim)?<(\d+)(?:,\s*(true|false))?'
)

# ap_fixed_base<8, 1, true, AP_TRN, AP_WRAP, 0>
# ap_fixed<14, 1, AP_TRN, AP_WRAP, 0>
# ap_fixed_base<8, 1, true> # DEFAULTS implied
AP_FIXED_RE = re.compile(
    r'^(?:%"struct\.)?ap_(u)?fixed(?:_base)?<'
    r'(\d+),\s*(\d+)(?:,\s*(true|false))?'
    r'(?:,\s*AP_(RND|RND_ZERO|RND_MIN_INF|RND_INF|RND_CONV|TRN|TRN_ZERO))?'
    r'(?:,\s*AP_(SAT|SAT_ZERO|SAT_SYM|WRAP|WRAP_SM))?'
)

# iv<1, false, 32, true>
# iv_base<1, false, 32, true>
AC_INT_RE = re.compile(
    r"iv(?:_base)?<\d+,\s*(true|false),\s*(\d+)"
)

# ac_fixed<34, 14, true, (ac_q_mode)0, (ac_o_mode)0>
AC_FIXED_RE = re.compile(
    r"ac_fixed<(\d+),\s*(\d+),\s*(true|false),\s*\(ac_q_mode\)(\d+),\s*\(ac_o_mode\)(\d+)"
)


AP_QUANT_MAP = {
    "RND": 0,
    "RND_ZERO": 1,
    "RND_MIN_INF": 2,
    "RND_INF": 3,
    "RND_CONV": 4,
    "TRN": 5,
    "TRN_ZERO": 6,
}

AP_OVERFLOW_MAP = {
    "SAT": 0,
    "SAT_ZERO": 1,
    "SAT_SYM": 2,
    "WRAP": 3,
    "WRAP_SM": 4,
}


# offsets in embedding
TYPE_SIZE      = 11
IS_AP_SIZE     = 1
IS_AC_SIZE     = 1
BITS_SIZE      = 1
FRAC_SIZE      = 1
SIGNED_SIZE    = 1
QUANT_SIZE     = 8
OVERFLOW_SIZE  = 5
ARRAY_LEN_SIZE = 1
PTR_DEPTH_SIZE = 1

TYPE_OFF      = 0
IS_AP_OFF     = TYPE_OFF      + TYPE_SIZE
IS_AC_OFF     = IS_AP_OFF     + IS_AP_SIZE
BITS_OFF      = IS_AC_OFF     + IS_AC_SIZE
FRAC_OFF      = BITS_OFF      + BITS_SIZE
SIGNED_OFF    = FRAC_OFF      + FRAC_SIZE
QUANT_OFF     = SIGNED_OFF    + SIGNED_SIZE
OVERFLOW_OFF  = QUANT_OFF     + QUANT_SIZE
ARRAY_LEN_OFF = OVERFLOW_OFF  + OVERFLOW_SIZE
PTR_DEPTH_OFF = ARRAY_LEN_OFF + ARRAY_LEN_SIZE

EMBED_SIZE = sum([
    TYPE_SIZE, IS_AP_SIZE, IS_AC_SIZE, BITS_SIZE, FRAC_SIZE,
    SIGNED_SIZE, QUANT_SIZE, OVERFLOW_SIZE, ARRAY_LEN_SIZE, PTR_DEPTH_SIZE
])


@lru_cache(maxsize=10000)
def type_embedding(type_str):
    """
    Embeds any LLVM, ap_types, or ac_types into a
    vector with the given schema. Unknown/not parsed types are assigned
    as a general "struct"

    return np.array:
     type:          multi-hot (integer, float, double, arb_int, arb_fixed, arr, ptr, stream, nnet_array, shift_reg, struct)
     is_ap:         boolean
     is_ac:         boolean
     n_bits:        integer
     frac_ratio:    float   (fractional bits / total bits)
     signed:        boolean
     quantize_m:    one-hot AP: RND, RND_ZERO, RND_MIN_INF, RND_INF, RND_CONV, TRN, TRN_ZERO
                             AC: TRN, RND, TRN_ZERO, RND_ZERO, RND_INF, RND_MIN_INF, RND_CONV, RND_CONV_ODD) TODO: Semantically line-up mappings between ap/ac
     overflow_m:    one-hot AP: SAT, SAT_ZERO, SAT_SYM, WRAP, WRAP_SM
                             AC: WRAP, SAT, SAT_ZERO, SAT_SYM)
     array_length:  float (log2 of array length: e.g. [16 x %class.ac_fixed] -> 4.0) (accumulates length additively)
     ptr_depth:     integer (*** -> 3)
    """

    emb = np.zeros(EMBED_SIZE, dtype=np.float32)

    # -----------------
    # pointer handling
    # -----------------

    ptr_depth = len(type_str) - len(type_str.rstrip("*"))
    if ptr_depth:
        emb[6] = 1          # ptr type
        emb[PTR_DEPTH_OFF] = ptr_depth
        type_str = type_str[:-ptr_depth]


    # -----------------
    # array handling
    # -----------------

    while type_str.startswith("[") and type_str.endswith("]"):
        separator = type_str.find(" x ", 1)
        if separator == -1:
            break

        try:
            arr_len = int(type_str[1:separator])
        except ValueError:
            break

        emb[5] = 1  # arr type
        emb[ARRAY_LEN_OFF] += np.log2(arr_len)

        type_str = type_str[separator + 3:-1]


    stream = STREAM_RE.match(type_str)
    if stream:
        emb[7] = 1   
        type_str = stream.group(1)      


    nnet_array = NNET_ARRAY_RE.match(type_str)
    if nnet_array:
        emb[8] = 1
        emb[ARRAY_LEN_OFF] += np.log2(int(nnet_array.group(2)))
        type_str = nnet_array.group(1) 


    shift_reg = SHIFT_REG_RE.match(type_str)
    if shift_reg:
        emb[9] = 1
        if shift_reg.group(2):
            emb[ARRAY_LEN_OFF] += np.log2(int(shift_reg.group(2)))
        type_str = shift_reg.group(1)


    # -----------------
    # primitive LLVM
    # -----------------

    m = INT_RE.match(type_str)
    if m:
        emb[0] = 1       # integer
        emb[BITS_OFF] = int(m.group(1))
        return emb


    m = FLOAT_RE.match(type_str)
    if m:
        if m.group(1) == "float":
            emb[1] = 1
        else:
            emb[2] = 1
        emb[BITS_OFF] = 32 if m.group(1) == "float" else 64
        return emb


    # -----------------
    # arbitrary precision
    # -----------------

    m = AP_INT_RE.match(type_str)
    if m:
        emb[3] = 1       # arb_int
        emb[IS_AP_OFF] = 1
        emb[BITS_OFF] = int(m.group(2))
        emb[SIGNED_OFF] = (m.group(1) != "u" or m.group(3) == "true")

        return emb


    m = AP_FIXED_RE.match(type_str)
    if m:
        emb[4] = 1       # arb_fixed
        emb[IS_AP_OFF] = 1

        width = int(m.group(2))
        int_bits = int(m.group(3))

        emb[BITS_OFF] = width
        emb[FRAC_OFF] = (width - int_bits) / width
        emb[SIGNED_OFF] = not (m.group(1) == "u" or m.group(4) == "false")

        # defaults are AP_TRN and AP_WRAP per https://docs.amd.com/r/2023.1-English/ug1399-vitis-hls/Fixed-Point-Identifier-Summary
        quant = AP_QUANT_MAP.get(m.group(5), "TRN")
        overflow = AP_OVERFLOW_MAP.get(m.group(6), "WRAP")
        

        # one-hot assignments here
        emb[QUANT_OFF + quant] = 1
        emb[OVERFLOW_OFF + overflow] = 1

        return emb


    m = AC_INT_RE.match(type_str)
    if m:
        emb[3] = 1
        emb[IS_AC_OFF] = 1
        emb[BITS_OFF] = int(m.group(2))
        emb[SIGNED_OFF] = (m.group(1) == "true")

        return emb


    m = AC_FIXED_RE.match(type_str)
    if m:
        emb[4] = 1
        emb[IS_AC_OFF] = 1

        width = int(m.group(1))
        int_bits = int(m.group(2))

        emb[BITS_OFF] = width
        emb[FRAC_OFF] = (width - int_bits) / width
        emb[SIGNED_OFF] = (m.group(3) == "true")

        quant = int(m.group(4))
        overflow = int(m.group(5))

        # one-hot assignments here
        emb[QUANT_OFF + quant] = 1
        emb[OVERFLOW_OFF + overflow] = 1

        return emb


    # fallback
    emb[TYPE_SIZE - 1] = 1   # struct/unknown
    return emb
