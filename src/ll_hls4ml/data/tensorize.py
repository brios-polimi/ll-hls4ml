"""Convert CDFG JSON graphs to PyG HeteroData tensors."""

from __future__ import annotations

from pathlib import Path
import shutil
import math
import tqdm
import re
import numpy as np
from functools import lru_cache
from collections import Counter

from ll_hls4ml.io.discovery import iter_graph_paths
from ll_hls4ml.io.load_json import load_graph_json
from ll_hls4ml.io.schema import (
    FLOW_CALL,
    FLOW_CONTROL,
    FLOW_DATA,
    FLOW_PRAGMA,
    NODE_CONSTANT,
    NODE_INSTRUCTION,
    NODE_PRAGMA,
    NODE_VARIABLE,
    PRAGMA_VOCAB,
    EDGE_TYPES,
    EDGE_TYPES_WITH_ATTR,
    LABEL_KEYS,
    safe_int,
)

from concurrent.futures import ProcessPoolExecutor
from functools import partial
import os

def _process_one(vocab: dict, inference_mode: bool, paths: tuple[Path, Path]) -> None:
    import torch

    graph_path, out_path = paths
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        graph_data = load_graph_json(graph_path)
        data, unknown_types = _json_to_hetero(graph_data, vocab, inference_mode)
        torch.save(data, out_path)
        return unknown_types
    except Exception as e:
        raise RuntimeError(f"Error processing graph {graph_path}: {e}") from e


def create_graph_tensors(
    graph_dir: str | Path,
    pt_dir: str | Path,
    instruction_vocab: dict,
    kernel_subset: str | list[str] | None = None,
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

    work = [
        (graph_path, pt_dir / graph_path.relative_to(graph_dir).parent / (graph_path.stem + ".pt"))
        for ks in ([kernel_subset] if kernel_subset else [None])
        for _, graph_path in iter_graph_paths(graph_dir, ks, max_archives)
    ]

    worker = partial(_process_one, instruction_vocab, inference_mode)
    with ProcessPoolExecutor(max_workers=n_workers or os.cpu_count()) as pool:
        unknown_types = Counter()

        for result in tqdm.tqdm(
            pool.map(worker, work),
            total=len(work),
            desc="Processing graph files into PyTorch tensors",
        ):
            result = {
                re.sub(r'\.\d+(?="|\b)', '', t)
                for t in result
            }
            unknown_types.update(result)

    if unknown_types:
        print("Types not parsed by embedder:")
        for t, count in unknown_types.most_common():
            print(f"{count:>6}  {t}")


def _json_to_hetero(graph_data: dict, instruction_vocab: dict, inference_mode: bool) -> HeteroData:
    import torch
    from torch_geometric.data import HeteroData

    data = HeteroData()
    inst_map = {}
    var_map = {}
    const_map = {}
    pragma_map = {}
    node_type_map: dict[int, int] = {}

    unknown_types = set()

    features = {
        "instruction": [],
        "variable": [],
        "constant": [],
        "pragma": [],
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
            features["pragma"].append([PRAGMA_VOCAB.get(node_text, 0)])
            pragma_map[node_id] = len(pragma_map)
        else:
            raise ValueError(f"Invalid node type: {node_type} in node {n}")

    data["instruction"].x = torch.tensor(features["instruction"], dtype=torch.long)
    data["variable"].x    = torch.from_numpy(np.stack(features["variable"]))
    data["constant"].x    = torch.from_numpy(np.stack(features["constant"]))
    data["pragma"].x      = torch.tensor(features["pragma"], dtype=torch.long).reshape(-1, 1)


    edge_index = { k: [] for k in EDGE_TYPES }
    edge_attrs = { k: [] for k in EDGE_TYPES_WITH_ATTR }
    edges = graph_data.get("links") or []
    for edge in edges:
        flow = safe_int(edge.get("flow", -1))
        source = safe_int(edge.get("source", -1))
        target = safe_int(edge.get("target", -1))
        if source < 0 or source >= len(nodes) or target < 0 or target >= len(nodes) or flow not in [FLOW_CONTROL, FLOW_DATA, FLOW_CALL, FLOW_PRAGMA]:
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
            raise ValueError(f"Missing labels in graph data") from e

    return data, unknown_types



# ---- regexes compiled once ----
# examples:
# ssdm_int_sim<6, true>**
# i64***
PTR_RE = re.compile(
    r"(\*+)$"
)

# examples:
# [1152 x ap_fixed<10, 1, (ap_q_mode)5, (ap_o_mode)3, 0>]**
# [12 x i32]
ARRAY_RE = re.compile(
    r"\[(\d+) x (.+)\]"
)

# examples
# stream<nnet::array<ap_fixed<20, 10, (ap_q_mode)5, (ap_o_mode)3, 0>, 4U> >
# stream<nnet::array<ap_fixed<16, 6, (ap_q_mode)5, (ap_o_mode)3, 0>, 4U> >
STREAM_RE = re.compile(
    r"^stream<\s*(.*?)\s*>$"
)

# examples:
# nnet::array<ap_fixed<20, 10, (ap_q_mode)5, (ap_o_mode)3, 0>
# array<ap_fixed<4, 1, (ap_q_mode)4, (ap_o_mode)0, 0>, 4U>
NNET_ARRAY_RE = re.compile(
    r"^(?:nnet::)?array<\s*(.*),\s*(\d+)U\s*>$"
)

# examples:
# i32, i57, float, double
INT_RE = re.compile(r"^i(\d+)$")
FLOAT_RE = re.compile(r"^(float|double)$")

# examples:
# ap_int<55>
# ap_int_base<64, true>
# ap_private<65, true, false>
# ssdm_int_sim<6, true>
AP_INT_RE = re.compile(
    r"(?:ap|ssdm)_(u?)(?:int|private)(?:_base|_sim)?<(\d+)(?:,\s*(true|false))?"
)

# examples:
# ap_fixed_base<8, 1, true, (ap_q_mode)5, (ap_o_mode)3, 0>
# ap_fixed<14, 1, (ap_q_mode)5, (ap_o_mode)3, 0>
AP_FIXED_RE = re.compile(
    r"ap_(u?)fixed(?:_base)?<"
    r"(\d+),\s*(\d+)(?:,\s*(true|false))?,\s*"
    r"\(ap_q_mode\)(\d+),\s*"
    r"\(ap_o_mode\)(\d+)"
)

# examples:
# iv<1, false, 32, true>
# iv_base<1, false, 32, true>
AC_INT_RE = re.compile(
    r"iv(?:_base)?<\d+,\s*(true|false),\s*(\d+)"
)

# examples:
# ac_fixed<34, 14, true, (ac_q_mode)0, (ac_o_mode)0>
AC_FIXED_RE = re.compile(
    r"ac_fixed<(\d+),\s*(\d+),\s*(true|false),\s*\(ac_q_mode\)(\d+),\s*\(ac_o_mode\)(\d+)"
)


# offsets in embedding
TYPE_SIZE      = 10
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
     type:          multi-hot (integer, float, double, arb_int, arb_fixed, arr, ptr, stream, nnet_array, struct)
     is_ap:         boolean
     is_ac:         boolean
     n_bits:        integer 
     frac_ratio:    float   (fractional bits / total bits)
     signed:        boolean
     quantize_m:    one-hot (ap: RND, RND_ZERO, RND_MIN_INF, RND_INF, RND_CONV, TRN, TRN_ZERO
                             ac: TRN, RND, TRN_ZERO, RND_ZERO, RND_INF, RND_MIN_INF, RND_CONV, RND_CONV_ODD)
     overflow_m:    one-hot (ap: SAT, SAT_ZERO, SAT_SYM, WRAP, WRAP_SM 
                             ac: WRAP, SAT, SAT_ZERO, SAT_SYM) 
     array_length:  float (log2 of array length: e.g. [16 x %class.ac_fixed] -> 4.0) (accumulates length additively)
     ptr_depth:     integer (*** -> 3)
    """

    emb = np.zeros(EMBED_SIZE, dtype=np.float32)

    # -----------------
    # pointer handling
    # -----------------

    ptr = PTR_RE.search(type_str)
    if ptr:
        emb[6] = 1           # ptr type
        emb[PTR_DEPTH_OFF] = len(ptr.group(1))
        type_str = type_str.rstrip("*")


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
        emb[SIGNED_OFF] = (m.group(1) != "u" or m.group(4) == "true")

        quant = int(m.group(5))
        overflow = int(m.group(6))
        

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
    emb[9] = 1   # struct/unknown
    return emb
