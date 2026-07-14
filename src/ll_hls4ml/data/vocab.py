"""Build and persist instruction/variable/constant vocabularies from CDFG JSON."""

from __future__ import annotations

import json
from pathlib import Path

import tqdm
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from collections import defaultdict
from networkx.readwrite import json_graph

from ll_hls4ml.config import load_config
from ll_hls4ml.io.discovery import iter_graph_paths
from ll_hls4ml.io.load_json import load_graph_json
from ll_hls4ml.io.schema import NODE_CONSTANT, NODE_INSTRUCTION, NODE_VARIABLE

import re


STRUCT_RE = re.compile(
    r'^(%'
    r'"?(?:'
        r'struct\.(?:ap_fixed(?:_base)?|ap_int(?:_base)?|ssdm_int_sim)'
        r'|class\.(?:ap_private|ac_fixed|anon|ac_private::iv(?:_base)?)'
    r')(?:\.\d+)?'
    r'"?)\s*=\s*type\s*(.*)$'
)
# call void @llvm.dbg.declare(metadata %class.ac_fixed** %3, metadata !4030, metadata !DIExpression()), !dbg !4031
DBG_RE = re.compile(
    r'call void @llvm\.dbg\.declare\(metadata '
    r'(%'
    r'"?(?:'
        r'struct\.(?:ap_fixed(?:_base)?|ap_int(?:_base)?|ssdm_int_sim)'
        r'|class\.(?:ap_private|ac_fixed|anon|ac_private::iv(?:_base)?)'
    r')(?:\.\d+)?'
    r'"?)(\**)'
    r'.*metadata !(\d+)'
)
META_RE = re.compile(r'^!(\d+)\s*=\s*(.*)$')

def find_base_type(metadata, line_num):
    """
    Recursively searches DWARF debug info (Clang compiler flag `-g`)
    to find base type semantic of LLVM locally-defined type.
    Returns None if not found.
    """
    line = metadata[line_num]

    type_match = re.search(r'(?:type|baseType): !(\d+)', line)
    if type_match:
        return find_base_type(metadata, int(type_match.group(1)))

    name_match = re.search(r'name: "(.+?)"', line)
    if name_match:
        return name_match.group(1)

    return None

def get_llvm_type_converter(path: str | Path) -> dict(str, str):
    """ 
    Returns dictionary mapping each local ap/ac_type string
    to a tuple containing base ap/ac_type
    """
    structs = {}      # "%struct.ap_fixed.1" -> definition line
    dbg_links = {}    # "%struct.ap_fixed.1" -> metadata ID
    metadata = {}     # metadata ID -> metadata text

    with open(Path(path)) as f:
        for line in f:
            # structs
            m = STRUCT_RE.match(line)
            if m:
                structs[m.group(1)] = m.group(2)

            # debug
            m = DBG_RE.search(line)
            if m:
                llvm_type = m.group(1)
                dbg_id = int(m.group(3))
                dbg_links[llvm_type] = dbg_id

            # metadata
            m = META_RE.match(line)
            if m:
                metadata[int(m.group(1))] = m.group(2)

    if set(structs.keys()) != set(dbg_links.keys()):
        raise RuntimeError(
            f"LLVM type declarations and usages do not match up in {path}:"
            f"{structs.keys()=}"
            f"{dbg_links.keys()=}"
        )

    final_types = {}
    for k, v in dbg_links.items():
        base_type = find_base_type(metadata, v)
        if base_type:
            final_types[k] = base_type
        else:
            final_types[k] = k

    return final_types

ARRAY_RE = re.compile(
    r"\[(\d+) x (.+)\]"
)

PTR_RE = re.compile(
    r"(\*+)$"
)

def make_local_types_global(json_path: str | Path, ll_path: str | Path, new_json_path: str | Path):
    type_converter = get_llvm_type_converter(ll_path)
    
    with open(json_path) as f:
        data = json.load(f)
    G = json_graph.node_link_graph(data, edges="links")
    
    for node, data in list(G.nodes().data()):
        typ = data['type']
        txt = data['text']
        if typ == 1 or typ == 2: # 1: Variable, 2: Constant
            ptr_depth = 0
            arr_length = 0
            
            ptr = PTR_RE.search(txt)
            if ptr:
                ptr_depth = len(ptr.group(1))
                txt = txt.rstrip("*")
                
            arr = ARRAY_RE.match(txt)
            if arr:
                arr_length = int(arr.group(1))
                txt = arr.group(2)
            
            new_txt = type_converter.get(txt, txt)
            if arr_length > 0:
                new_txt = f"[{arr_length} x {new_txt}]"
            if ptr_depth > 0:
                new_txt = new_txt + ("*" * ptr_depth)
                
            data['text'] = new_txt

    data = json_graph.node_link_data(G, edges="links")
    with open(new_json_path, 'w') as f:
        json.dump(data, f)


def vocab_scan(
    graph_dir: str | Path,
    kernel_subset: str | list[str] | None = None,
    max_archives: int | None = None,
    first_n: int | None = None,
):
    """
    Walk graph_dir and collect vocabularies of instructions, variables, and constants.

    Returns vocab dict, max edge position, and per-token counts.
    """
    vocab_sets = {
        "instruction": set(),
        "variable": set(),
        "constant": set(),
    }
    vocab_counts = {
        "instruction": {},
        "variable": {},
        "constant": {},
    }
    max_pos = 0

    paths = list(iter_graph_paths(graph_dir, kernel_subset, max_archives, first_n))
    i = 0
    path_number = defaultdict(int)
    credit = defaultdict(lambda: defaultdict(list))
    for _ks, graph_path in tqdm.tqdm(paths, desc="Parsing graph files for building vocab"):
        try:
            graph_data = load_graph_json(graph_path)
        except Exception as e:
            print(f"Error loading JSON: {graph_path}: {e}")
            continue
        nodes = graph_data.get("nodes") or []
        i += 1
        for n in nodes:
            node_type = n.get("type", -1)
            term = n.get("text", "")
            if node_type == NODE_INSTRUCTION:
                if term not in vocab_sets["instruction"]:
                    credit[graph_path]["instruction"].append(term)
                    path_number[graph_path] = i
                vocab_sets["instruction"].add(term)
                vocab_counts["instruction"][term] = vocab_counts["instruction"].get(term, 0) + 1
            elif node_type == NODE_VARIABLE:
                if term not in vocab_sets["variable"]:
                    credit[graph_path]["variable"].append(term)
                    path_number[graph_path] = i
                vocab_sets["variable"].add(term)
                vocab_counts["variable"][term] = vocab_counts["variable"].get(term, 0) + 1
            elif node_type == NODE_CONSTANT:
                if term not in vocab_sets["constant"]:
                    credit[graph_path]["constant"].append(term)
                    path_number[graph_path] = i
                vocab_sets["constant"].add(term)
                vocab_counts["constant"][term] = vocab_counts["constant"].get(term, 0) + 1

        for link in graph_data.get("links") or []:
            position = link.get("position", 0)
            if position > max_pos:
                max_pos = position

    for graph_path, credit in credit.items():
        print(f"Credit for {graph_path} (number {path_number[graph_path]})")
        for k, v in credit.items():
            print(f"  {k}: {v}")

    vocab = {}
    for k, v in vocab_sets.items():
        vocab[k] = {"UNK": 0}
        vocab[k].update({t: i + 1 for i, t in enumerate(sorted(v))})
    return vocab, max_pos, vocab_counts, credit, path_number


def save_vocab(vocab: dict, path: str | Path, max_pos: int | None = None, vocab_counts: dict | None = None) -> None:
    """Persist vocab mapping to JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"vocab": vocab}
    if max_pos is not None:
        payload["max_pos"] = max_pos
    if vocab_counts is not None:
        payload["vocab_counts"] = vocab_counts
    with path.open("w") as f:
        json.dump(payload, f, indent=2)


def load_vocab(path: str | Path) -> tuple[dict, int, dict]:
    """Load vocab from JSON. Returns (vocab, max_pos, vocab_counts)."""
    with Path(path).open() as f:
        payload = json.load(f)
    return payload["vocab"], payload.get("max_pos", 0), payload.get("vocab_counts", {})
