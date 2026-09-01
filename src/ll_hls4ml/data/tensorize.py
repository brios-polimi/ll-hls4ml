"""Convert CDFG JSON graphs to PyG HeteroData tensors."""

from __future__ import annotations

from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache, partial
import json
import math
import multiprocessing
import os
from pathlib import Path
import re
import tempfile

import numpy as np
import tqdm

from ll_hls4ml.io.schema import (
    DERIVED_DEF_USE_EDGE,
    EDGE_TYPES,
    EDGE_TYPE_SET,
    EDGE_TYPES_WITH_ATTR,
    BLOCK_FEATURE_SIZE,
    FUNCTION_FEATURE_SIZE,
    GRAPH_CONTEXT_CATEGORICAL_VOCABS,
    GRAPH_CONTEXT_NUMERIC_KEYS,
    NODE_CONSTANT,
    NODE_BLOCK,
    NODE_FUNCTION,
    NODE_INSTRUCTION,
    NODE_PRAGMA,
    NODE_TYPE_NAMES,
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
from ll_hls4ml.data.hierarchy import function_schedule
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
    records: list[tuple[Path, list[float] | None, dict]],
) -> None:
    """Merge labels from this tensorization run into the tensor sidecar index."""
    index_path = pt_dir / _LABEL_INDEX_NAME
    index = {"label_keys": LABEL_KEYS, "labels": {}, "metadata": {}}
    if index_path.exists():
        with index_path.open() as handle:
            existing = json.load(handle)
        if existing.get("label_keys") == LABEL_KEYS:
            index["labels"].update(existing.get("labels", {}))
            index["metadata"].update(existing.get("metadata", {}))

    for out_path, labels, metadata in records:
        relative_path = out_path.relative_to(pt_dir).as_posix()
        if labels is None:
            index["labels"].pop(relative_path, None)
        else:
            index["labels"][relative_path] = labels
        if metadata:
            index["metadata"][relative_path] = metadata
        else:
            index["metadata"].pop(relative_path, None)

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
            # Targets are represented structurally by applies_to relations.
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


def function_embedding(node: dict) -> np.ndarray:
    """Encode whether the function has a body while retaining its symbol in JSON."""

    features = node.get("features", {})
    names = features.get("name", [])
    defined_values = features.get("is_defined", [])
    return np.asarray(
        [
            len(names) == 1 and bool(str(names[0])),
            len(defined_values) == 1 and str(defined_values[0]).lower() == "true",
        ],
        dtype=np.float32,
    )


def constant_literal_embedding(node: dict) -> np.ndarray:
    """Encode scalar LLVM literals while leaving symbolic constants masked."""
    full_text = node.get("features", {}).get("full_text", [])
    if not isinstance(full_text, list) or len(full_text) != 1:
        return np.zeros(CONSTANT_LITERAL_SIZE, dtype=np.float32)
    match = SCALAR_LITERAL_RE.match(str(full_text[0]).strip())
    if not match:
        return np.zeros(CONSTANT_LITERAL_SIZE, dtype=np.float32)
    literal = match.group(1)
    if literal == "true":
        value = 1.0
    elif literal == "false":
        value = 0.0
    else:
        try:
            value = float(literal)
        except ValueError:
            return np.zeros(CONSTANT_LITERAL_SIZE, dtype=np.float32)
    if not math.isfinite(value):
        return np.zeros(CONSTANT_LITERAL_SIZE, dtype=np.float32)
    integer_value = int(value)
    absolute_integer = abs(integer_value)
    is_power_of_two = (
        value == integer_value
        and absolute_integer > 0
        and absolute_integer & (absolute_integer - 1) == 0
    )
    return np.asarray(
        [
            1.0,
            math.copysign(math.log1p(abs(value)), value),
            value == 0,
            value == 1,
            value == -1,
            is_power_of_two,
        ],
        dtype=np.float32,
    )


def synthesis_metadata(graph_data: dict) -> dict:
    """Return non-target synthesis context from either graph metadata location."""
    metadata = dict(graph_data.get("synthesis_metadata") or {})
    labels = graph_data.get("labels") or {}
    for key in (
        *GRAPH_CONTEXT_CATEGORICAL_VOCABS,
        *GRAPH_CONTEXT_NUMERIC_KEYS,
        "dataset_split",
    ):
        if key not in metadata and key in labels:
            metadata[key] = labels[key]
    return metadata


def graph_context_embedding(metadata: dict) -> tuple[np.ndarray, np.ndarray]:
    categorical = np.asarray(
        [
            vocabulary.get(str(metadata.get(key, "")), 0)
            for key, vocabulary in GRAPH_CONTEXT_CATEGORICAL_VOCABS.items()
        ],
        dtype=np.int64,
    )
    numeric = []
    for key in GRAPH_CONTEXT_NUMERIC_KEYS:
        value = _scaled_number(str(metadata.get(key, "")))
        numeric.append(0.0 if value is None else value)
    return categorical, np.asarray(numeric, dtype=np.float32)


def _process_one(
    vocab: dict,
    max_pos: int,
    inference_mode: bool,
    metadata_by_graph_id: dict[str, dict] | None,
    paths: tuple[Path, Path],
) -> tuple[set[str], list[float] | None, dict]:
    import torch

    graph_path, out_path = paths
    try:
        graph_data = load_graph_json(graph_path)
        if metadata_by_graph_id and graph_path.stem in metadata_by_graph_id:
            graph_data["synthesis_metadata"] = metadata_by_graph_id[graph_path.stem]
        data, unknown_types = _json_to_hetero(graph_data, vocab, max_pos, inference_mode)
        torch.save(data, out_path)
        labels = None if inference_mode else data.y.tolist()
        return unknown_types, labels, synthesis_metadata(graph_data)
    except Exception as e:
        raise RuntimeError(f"Error processing graph {graph_path}: {e}") from e


def create_graph_tensors(
    graph_dir: str | Path,
    pt_dir: str | Path,
    instruction_vocab: dict,
    max_pos: int | None = None,
    kernel_subset: str | list[str] | None = None,
    archive_subset: str | list[str] | None = None,
    max_archives: int | None = None,
    inference_mode: bool = False,
    n_workers: int | None = None,
    metadata_by_graph_id: dict[str, dict] | None = None,
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
            graph_dir,
            kernel_subset,
            None if archive_subset is not None else max_archives,
        )
    ]
    if archive_subset is not None:
        archive_names = (
            [archive_subset] if isinstance(archive_subset, str) else archive_subset
        )
        graph_paths = [
            path
            for path in graph_paths
            if path.relative_to(graph_dir).parts[1] in archive_names
        ]
        if not graph_paths:
            raise ValueError(
                f"No graphs found for kernels={kernel_subset}, archives={archive_names}"
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

    worker = partial(
        _process_one,
        instruction_vocab,
        max_pos,
        inference_mode,
        metadata_by_graph_id,
    )
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
        multiprocessing_context = (
            multiprocessing.get_context("spawn")
            if os.name != "nt"
            else None
        )
        pool = ProcessPoolExecutor(
            max_workers=workers,
            mp_context=multiprocessing_context,
        )
        results = pool.map(worker, work, chunksize=32)

    try:
        label_records = []
        for (_graph_path, out_path), (result, labels, metadata) in zip(
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
            label_records.append((out_path, labels, metadata))
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
    data.type_embedding_schema_version = TYPE_EMBEDDING_SCHEMA_VERSION
    inst_map = {}
    var_map = {}
    const_map = {}
    pragma_map = {}
    block_map = {}
    function_map = {}
    function_id_map = {}
    node_type_map: dict[int, int] = {}
    instruction_function_ids = []
    block_function_ids = []

    unknown_types = set()

    features = {
        "instruction": [],
        "variable": [],
        "constant": [],
        "pragma": [],
        "block": [],
        "function": [],
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
            instruction_function_ids.append(safe_int(n.get("function", -1)))
        elif node_type == NODE_VARIABLE:
            type_emb = type_embedding(node_text)
            if type_emb[TYPE_SIZE - 1]:
                unknown_types.add(node_text)
            features["variable"].append(type_emb)
            var_map[node_id] = len(var_map)
        elif node_type == NODE_CONSTANT:
            type_emb = type_embedding(node_text).copy()
            type_emb[LITERAL_OFF:] = constant_literal_embedding(n)
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
            block_function_ids.append(safe_int(n.get("function", -1)))
        elif node_type == NODE_FUNCTION:
            features["function"].append(function_embedding(n))
            function_map[node_id] = len(function_map)
            function_id = safe_int(n.get("function", -1))
            if function_id in function_id_map:
                raise ValueError(f"Duplicate function hierarchy ID {function_id}")
            function_id_map[function_id] = function_map[node_id]
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
    data["function"].x = torch.tensor(
        np.stack(features["function"])
        if features["function"]
        else np.empty((0, FUNCTION_FEATURE_SIZE), dtype=np.float32),
        dtype=torch.float,
    )

    metadata = synthesis_metadata(graph_data)
    categorical_context, numeric_context = graph_context_embedding(metadata)
    data.graph_context_categorical = torch.from_numpy(
        categorical_context.reshape(1, -1)
    )
    data.graph_context_numeric = torch.from_numpy(
        numeric_context.reshape(1, -1)
    )


    local_maps = {
        NODE_INSTRUCTION: inst_map,
        NODE_VARIABLE: var_map,
        NODE_CONSTANT: const_map,
        NODE_PRAGMA: pragma_map,
        NODE_BLOCK: block_map,
        NODE_FUNCTION: function_map,
    }
    edge_index = {edge_type: [] for edge_type in EDGE_TYPES}
    edge_attrs = {edge_type: [] for edge_type in EDGE_TYPES_WITH_ATTR}
    edges = graph_data.get("links") or []
    for edge in edges:
        source = safe_int(edge.get("source", -1))
        target = safe_int(edge.get("target", -1))
        relation = str(edge.get("relation", ""))
        if source not in node_type_map or target not in node_type_map:
            raise ValueError(f"Invalid edge source/target: {edge}")
        source_type_id = node_type_map[source]
        target_type_id = node_type_map[target]
        edge_type = (
            NODE_TYPE_NAMES[source_type_id],
            relation,
            NODE_TYPE_NAMES[target_type_id],
        )
        if edge_type not in EDGE_TYPE_SET:
            raise ValueError(f"Edge is outside the canonical schema: {edge_type}")
        local_source = local_maps[source_type_id].get(source)
        local_target = local_maps[target_type_id].get(target)
        if local_source is None or local_target is None:
            raise ValueError(f"Could not localize canonical edge: {edge}")
        edge_index[edge_type].append([local_source, local_target])

        if edge_type in edge_attrs:
            position = safe_int(edge.get("position", 0))
            if position < 0 or (max_pos is not None and position > max_pos):
                raise ValueError(f"Invalid edge position {position}: {edge}")
            edge_attrs[edge_type].append([position])

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

    # Materialize the expensive relational join once, not during every epoch:
    # instruction --defines--> variable --operand--> instruction.
    producers_by_variable: dict[int, list[int]] = {}
    for producer, variable in edge_index[("instruction", "defines", "variable")]:
        producers_by_variable.setdefault(variable, []).append(producer)
    def_use = {
        (producer, consumer)
        for variable, consumer in edge_index[("variable", "operand", "instruction")]
        for producer in producers_by_variable.get(variable, ())
    }
    data[DERIVED_DEF_USE_EDGE].edge_index = (
        torch.tensor(sorted(def_use), dtype=torch.long).t().contiguous()
        if def_use
        else torch.empty((2, 0), dtype=torch.long)
    )

    instruction_owner = [
        function_id_map.get(function_id, -1)
        for function_id in instruction_function_ids
    ]
    block_owner = [
        function_id_map.get(function_id, -1)
        for function_id in block_function_ids
    ]
    if any(owner < 0 for owner in instruction_owner + block_owner):
        raise ValueError("Every instruction and block must belong to a function node")

    block_by_instruction = {}
    for block, instruction in edge_index[("block", "contains", "instruction")]:
        if instruction in block_by_instruction:
            raise ValueError(f"Instruction {instruction} has multiple block owners")
        block_by_instruction[instruction] = block
    function_by_block = {}
    for function, block in edge_index[("function", "contains", "block")]:
        if block in function_by_block:
            raise ValueError(f"Block {block} has multiple function owners")
        function_by_block[block] = function
    if len(block_by_instruction) != len(inst_map):
        raise ValueError("Every instruction requires exactly one contains edge")
    if len(function_by_block) != len(block_map):
        raise ValueError("Every block requires exactly one contains edge")
    for instruction, expected_function in enumerate(instruction_owner):
        block = block_by_instruction[instruction]
        if function_by_block[block] != expected_function:
            raise ValueError("Instruction metadata disagrees with containment edges")
    for block, expected_function in enumerate(block_owner):
        if function_by_block[block] != expected_function:
            raise ValueError("Block metadata disagrees with containment edges")

    for relation_edges in (
        edge_index[("instruction", "control", "instruction")],
        def_use,
    ):
        for source, target in relation_edges:
            if instruction_owner[source] != instruction_owner[target]:
                raise ValueError("Intraprocedural instruction edge crosses functions")
    for source, target in edge_index[("block", "control", "block")]:
        if block_owner[source] != block_owner[target]:
            raise ValueError("Basic-block CFG edge crosses functions")

    call_pairs = [
        (instruction_owner[callsite], callee)
        for callsite, callee in edge_index[("instruction", "calls", "function")]
    ]
    instruction_counts = [0] * len(function_map)
    for owner in instruction_owner:
        instruction_counts[owner] += 1
    schedule = function_schedule(
        len(function_map), call_pairs, instruction_counts
    )
    data["function"].call_depth = torch.tensor(schedule.depth, dtype=torch.long)
    data["function"].is_root = torch.tensor(schedule.roots, dtype=torch.bool)
    data["function"].is_entry = torch.tensor(schedule.entry, dtype=torch.bool)
    data["function"].is_reachable = torch.tensor(
        schedule.reachable, dtype=torch.bool
    )
    data["instruction"].call_depth = torch.tensor(
        [
            schedule.depth[owner] if schedule.reachable[owner] else -1
            for owner in instruction_owner
        ],
        dtype=torch.long,
    )
    data["block"].call_depth = torch.tensor(
        [
            schedule.depth[owner] if schedule.reachable[owner] else -1
            for owner in block_owner
        ],
        dtype=torch.long,
    )
    data.hierarchy_schema_version = 2

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
    r'^(?:%"class\.)?hls::stream<\s*(.+),\s*(\d+)\s*>"?$'
)
# Bambu's nested fifo is the channel's payload holder, not a distinct numeric
# format. Neither spelling specifies a realized FIFO depth.
AC_CHANNEL_RE = re.compile(r'^ac_channel<\s*(.+)\s*>\s*(?:::fifo)?$')

# nnet::array<ap_fixed<20, 10, AP_TRN, AP_WRAP, 0>
NNET_ARRAY_RE = re.compile(
    r'^(?:%"struct\.)?nnet::array<\s*(.*),\s*(\d+)[uUlL]*\s*>"?'
)

# %"class.ap_shift_reg<ap_ufixed<4, 1, AP_RND_CONV, AP_SAT>, 9>"*
SHIFT_REG_RE = re.compile(
    r'^(?:%"class\.)?ap_shift_reg<\s*(.+?)(?:,\s*(\d+)\s*)?>"?$'
)


SCALAR_LITERAL_RE = re.compile(
    r"^(?:i\d+|half|float|double)\s+"
    r"(true|false|[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)$"
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
# Require whole enum tokens: with optional trailing groups, RND must not match
# the prefix of RND_CONV, nor SAT the prefix of SAT_ZERO.
AP_FIXED_RE = re.compile(
    r'^(?:%"struct\.)?ap_(u)?fixed(?:_base)?<'
    r'(\d+),\s*(-?\d+)(?:,\s*(true|false))?'
    r'(?:,\s*AP_(RND|RND_ZERO|RND_MIN_INF|RND_INF|RND_CONV|TRN|TRN_ZERO)\b)?'
    r'(?:,\s*AP_(SAT|SAT_ZERO|SAT_SYM|WRAP|WRAP_SM)\b)?'
)

# iv/iv_base<N, C, W, S>: C is a storage implementation flag, NOT signedness.
# iv_conv<N, S, LTE64, C, W> uses a different parameter order. Preserve the
# semantic W/S from each declared template, not storage bytes or limb counts.
AC_INT_RE = re.compile(
    r"^(?:ac_private::)?iv(?:_base)?<\d+,\s*(?:true|false),\s*(?P<bits>\d+),\s*(?P<signed>true|false)>"
)
AC_INT_CONV_RE = re.compile(
    r"^(?:ac_private::)?iv_conv<\d+,\s*(?P<signed>true|false),\s*(?:true|false),\s*(?:true|false),\s*(?P<bits>\d+)>"
)
AC_PUBLIC_INT_RE = re.compile(
    r"^ac_int<(?P<bits>\d+),\s*(?P<signed>true|false)>"
)

# ac_fixed<34, 14, true, (ac_q_mode)0, (ac_o_mode)0>
AC_FIXED_RE = re.compile(
    r"ac_fixed<(\d+),\s*(-?\d+),\s*(true|false),\s*\(ac_q_mode\)(\d+),\s*\(ac_o_mode\)(\d+)"
)

# af_range_ref<18, 8, true, AP_RND, AP_SAT, 0>
# This helper type carries the same fixed-point properties relevant to HLS
# arithmetic as ap_fixed, but is emitted by some hls4ml code paths.
AF_RANGE_REF_RE = re.compile(
    r'^(?:%"struct\.)?af_range_ref<'
    r'(\d+),\s*(\d+),\s*(true|false),\s*'
    r'AP_(RND|RND_ZERO|RND_MIN_INF|RND_INF|RND_CONV|TRN|TRN_ZERO),\s*'
    r'AP_(SAT|SAT_ZERO|SAT_SYM|WRAP|WRAP_SM)\b'
)


# Feature positions encode behavior, not the libraries' enum ordinals. Keep the
# existing AP positions for compatibility: RND rounds to nearest with ties
# toward +infinity; RND_ZERO/MIN_INF/INF break ties toward zero/-infinity/away
# from zero; RND_CONV breaks ties to even. TRN truncates toward -infinity and
# TRN_ZERO toward zero. The corresponding AC modes implement the same rules.
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

# AC spells modes numerically in Clang type names, but its enum order differs
# from AP's feature order. Direct indexing would conflate AC_TRN with AP_RND
# and AC_WRAP with AP_SAT. Normalize both libraries to the same semantic slots;
# their separate is_ap/is_ac features still retain library identity.
AC_QUANT_MAP = {
    0: AP_QUANT_MAP["TRN"],          # AC_TRN
    1: AP_QUANT_MAP["RND"],          # AC_RND
    2: AP_QUANT_MAP["TRN_ZERO"],     # AC_TRN_ZERO
    3: AP_QUANT_MAP["RND_ZERO"],     # AC_RND_ZERO
    4: AP_QUANT_MAP["RND_INF"],      # AC_RND_INF
    5: AP_QUANT_MAP["RND_MIN_INF"],  # AC_RND_MIN_INF
    6: AP_QUANT_MAP["RND_CONV"],     # AC_RND_CONV: ties to even
    7: 7,                          # AC_RND_CONV_ODD: ties to odd; no AP equivalent
}
AC_OVERFLOW_MAP = {
    0: AP_OVERFLOW_MAP["WRAP"],      # AC_WRAP: wrap modulo the destination width
    1: AP_OVERFLOW_MAP["SAT"],       # AC_SAT: clamp to representable bounds
    2: AP_OVERFLOW_MAP["SAT_ZERO"],  # AC_SAT_ZERO: replace overflow with zero
    3: AP_OVERFLOW_MAP["SAT_SYM"],   # AC_SAT_SYM: symmetric saturation
}
# AP_WRAP_SM remains a distinct AP-only mode, not an alias for AC_WRAP.
# Version 1 (including tensors without a version) used raw AC ordinals and
# could parse AP enum prefixes instead of whole tokens. The shape is unchanged,
# so record these semantic corrections to make migration visible.
# Version 3 additionally recognizes debug-qualified AC storage/channel types
# and corrects iv signedness. The vector dimensions remain unchanged.
TYPE_EMBEDDING_SCHEMA_VERSION = 3


# offsets in embedding
TYPE_SIZE               = 11
IS_AP_SIZE              = 1
IS_AC_SIZE              = 1
BITS_SIZE               = 1
FRAC_SIZE               = 1
SIGNED_SIZE             = 1
QUANT_SIZE              = 8
OVERFLOW_SIZE           = 5
SPATIAL_LEN_SIZE        = 1
TEMPORAL_LEN_SIZE       = 1
PTR_DEPTH_SIZE          = 1
CONSTANT_LITERAL_SIZE   = 6

TYPE_OFF      = 0
IS_AP_OFF     = TYPE_OFF      + TYPE_SIZE
IS_AC_OFF     = IS_AP_OFF     + IS_AP_SIZE
BITS_OFF      = IS_AC_OFF     + IS_AC_SIZE
FRAC_OFF      = BITS_OFF      + BITS_SIZE
SIGNED_OFF    = FRAC_OFF      + FRAC_SIZE
QUANT_OFF     = SIGNED_OFF    + SIGNED_SIZE
OVERFLOW_OFF  = QUANT_OFF     + QUANT_SIZE
SPATIAL_LEN_OFF = OVERFLOW_OFF + OVERFLOW_SIZE
TEMPORAL_LEN_OFF = SPATIAL_LEN_OFF + SPATIAL_LEN_SIZE
PTR_DEPTH_OFF = TEMPORAL_LEN_OFF + TEMPORAL_LEN_SIZE
LITERAL_OFF   = PTR_DEPTH_OFF + PTR_DEPTH_SIZE

EMBED_SIZE = sum([
    TYPE_SIZE, IS_AP_SIZE, IS_AC_SIZE, BITS_SIZE, FRAC_SIZE,
    SIGNED_SIZE, QUANT_SIZE, OVERFLOW_SIZE, SPATIAL_LEN_SIZE, TEMPORAL_LEN_SIZE, PTR_DEPTH_SIZE,
    CONSTANT_LITERAL_SIZE,
])


def _unwrap_llvm_named_type(type_str):
    """Remove LLVM's spelling wrappers from a named semantic type.

    Bambu's debug metadata is recovered as a quoted LLVM named type,
    for example ``%"ac_fixed<16, 6, true, (ac_q_mode)0, (ac_o_mode)0>"``.
    ProGraML preserves those wrappers in node text, while the semantic regexes
    operate on the source-level type spelling.
    """

    type_str = type_str.strip()
    if type_str.startswith("%"):
        type_str = type_str[1:]
    if len(type_str) >= 2 and type_str.startswith('"') and type_str.endswith('"'):
        type_str = type_str[1:-1]
    for prefix in ("struct.", "class."):
        if type_str.startswith(prefix):
            type_str = type_str[len(prefix):]
            break
    return re.sub(r'\.debug\.\d+$', '', type_str)


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
     quantize_m:    shared one-hot: RND, RND_ZERO, RND_MIN_INF, RND_INF,
                   RND_CONV, TRN, TRN_ZERO, RND_CONV_ODD (AC-only)
     overflow_m:    shared one-hot: SAT, SAT_ZERO, SAT_SYM, WRAP, WRAP_SM (AP-only)
     spatial_length:  float (log2 of spatial array lengths: e.g. [16 x %class.ac_fixed] -> 4.0) (accumulates length additively)
     temporal_length: float (log2 of temporal array lengths)
     ptr_depth:     integer (*** -> 3)
    """

    emb = np.zeros(EMBED_SIZE, dtype=np.float32)
    type_str = str(type_str).strip()

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
        emb[SPATIAL_LEN_OFF] += np.log2(arr_len)

        type_str = type_str[separator + 3:-1]

    type_str = _unwrap_llvm_named_type(type_str)


    stream = STREAM_RE.match(type_str)
    if stream:
        emb[7] = 1   
        type_str = stream.group(1).strip()

    channel = AC_CHANNEL_RE.match(type_str)
    if channel:
        emb[7] = 1
        emb[IS_AC_OFF] = 1
        type_str = channel.group(1).strip()


    nnet_array = NNET_ARRAY_RE.match(type_str)
    if nnet_array:
        emb[8] = 1
        emb[SPATIAL_LEN_OFF] += np.log2(int(nnet_array.group(2)))
        type_str = nnet_array.group(1) 


    shift_reg = SHIFT_REG_RE.match(type_str)
    if shift_reg:
        emb[9] = 1
        depth = int(shift_reg.group(2) or 32)
        emb[TEMPORAL_LEN_OFF] = np.log2(depth)
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
        quant = AP_QUANT_MAP[m.group(5) or "TRN"]
        overflow = AP_OVERFLOW_MAP[m.group(6) or "WRAP"]
        

        # one-hot assignments here
        emb[QUANT_OFF + quant] = 1
        emb[OVERFLOW_OFF + overflow] = 1

        return emb


    m = AF_RANGE_REF_RE.match(type_str)
    if m:
        emb[4] = 1       # arb_fixed
        emb[IS_AP_OFF] = 1

        width = int(m.group(1))
        int_bits = int(m.group(2))
        emb[BITS_OFF] = width
        emb[FRAC_OFF] = (width - int_bits) / width
        emb[SIGNED_OFF] = (m.group(3) == "true")
        emb[QUANT_OFF + AP_QUANT_MAP[m.group(4)]] = 1
        emb[OVERFLOW_OFF + AP_OVERFLOW_MAP[m.group(5)]] = 1
        return emb


    m = AC_INT_RE.match(type_str) or AC_INT_CONV_RE.match(type_str) or AC_PUBLIC_INT_RE.match(type_str)
    if m:
        emb[3] = 1
        emb[IS_AC_OFF] = 1
        emb[BITS_OFF] = int(m.group('bits'))
        emb[SIGNED_OFF] = (m.group('signed') == "true")

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

        try:
            quant = AC_QUANT_MAP[int(m.group(4))]
            overflow = AC_OVERFLOW_MAP[int(m.group(5))]
        except KeyError as exc:
            # Never let unknown enum ordinals spill into adjacent features or
            # silently acquire another mode's meaning.
            raise ValueError(f"Unsupported AC rounding/overflow mode in {type_str!r}") from exc

        # one-hot assignments here
        emb[QUANT_OFF + quant] = 1
        emb[OVERFLOW_OFF + overflow] = 1

        return emb


    # fallback
    emb[TYPE_SIZE - 1] = 1   # struct/unknown
    return emb
