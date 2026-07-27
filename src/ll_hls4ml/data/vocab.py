"""Build and persist instruction/variable/constant vocabularies from CDFG JSON."""

from __future__ import annotations

import json
from pathlib import Path

import tqdm

from ll_hls4ml.io.discovery import iter_graph_paths
from ll_hls4ml.io.load_json import load_graph_json
from ll_hls4ml.io.schema import (
    NODE_CONSTANT,
    NODE_BLOCK,
    NODE_INSTRUCTION,
    NODE_PRAGMA,
    NODE_VARIABLE,
)

import re

TYPE_RE = re.compile(r'(?:type|baseType): !(\d+)')
NAME_RE = re.compile(r'name: "(.+?)"')

STRUCT_RE = re.compile(
    r'^(%'
    r'"?(?:'
        r'struct\.(?:ap_(?:u)?fixed(?:_base)?|ap_(?:u)?int(?:_base)?|ssdm_int_sim|nnet::array)'
        r'|class\.(?:ap_private|ac_fixed|anon|ac_private::iv(?:_base)?|hls::stream)'
    r')(?:\.\d+)?'
    r'"?)\s*=\s*type\s*(.*)$'
)
# ... call void @llvm.dbg.declare(metadata %class.ac_fixed** %3, metadata !4030, ...
DBG_RE = re.compile(
    r'call void @llvm\.dbg\.declare\(metadata '
    r'(%'
    r'"?(?:'
        r'struct\.(?:ap_(?:u)?fixed(?:_base)?|ap_(?:u)?int(?:_base)?|ssdm_int_sim|nnet::array)'
        r'|class\.(?:ap_private|ac_fixed|anon|ac_private::iv(?:_base)?|hls::stream)'
    r')(?:\.\d+)?'
    r'"?)(\**)'
    r'.*metadata !(\d+)'
)
META_RE = re.compile(r'^!(\d+)\s*=\s*(.*)$')

def _get_llvm_type_converter(path: str | Path) -> dict[str, str]:
    """ 
    Returns dictionary mapping each local ap/ac_type string
    to a tuple containing base ap/ac_type
    """
    structs = set()   # "%struct.ap_fixed.1" -> definition line
    dbg_links = {}         # "%struct.ap_fixed.1" -> metadata ID
    metadata = {}          # metadata ID -> metadata text

    with open(Path(path)) as f:
        for line in f:
            c = line[0]

            # structs
            if c == "%":
                if m := STRUCT_RE.match(line):
                    structs.add(m.group(1))

            # debug
            elif "llvm.dbg.declare" in line:
                if m := DBG_RE.search(line):
                    llvm_type = m.group(1)
                    dbg_id = int(m.group(3))
                    dbg_links[llvm_type] = dbg_id

            # metadata
            elif c == "!":
                if m := META_RE.match(line):
                    metadata[int(m.group(1))] = m.group(2)

    if structs != set(dbg_links.keys()):
        raise RuntimeError(
            f"LLVM type declarations and usages do not match up in {path}:"
            f"{structs.keys()=}"
            f"{dbg_links.keys()=}"
        )

    memo = {}
    def find_base_type(line_num):
        """
        Recursively searches DWARF debug info (Clang compiler flag `-g`)
        to find base type semantic of LLVM locally-defined type.
        Returns None if not found.
        """
        if line_num in memo:
            return memo[line_num]

        line = metadata[line_num]

        if m := TYPE_RE.search(line):
            result = find_base_type(int(m.group(1)))
        elif m := NAME_RE.search(line):
            result = m.group(1)
        else:
            result = None

        memo[line_num] = result
        return result

    final_types = {}
    for k, v in dbg_links.items():
        base_type = find_base_type(v)
        if base_type:
            final_types[k] = base_type
        else:
            final_types[k] = k

    return final_types


def make_local_types_global(json_path: str | Path, ll_path: str | Path):
    type_converter = _get_llvm_type_converter(ll_path)

    with open(json_path) as f:
        data = json.load(f)

    for node in data["nodes"]:
        if node["type"] not in (1, 2):
            continue

        txt = node["text"]

        # Pointer parsing
        ptr_depth = len(txt) - len(txt.rstrip("*"))
        if ptr_depth:
            txt = txt[:-ptr_depth]

        # Nested array parsing
        array_lengths = []
        while txt.startswith("[") and txt.endswith("]"):
            separator = txt.find(" x ", 1)
            if separator == -1:
                break

            try:
                length = int(txt[1:separator])
            except ValueError:
                break

            array_lengths.append(length)
            txt = txt[separator + 3:-1]

        # Convert base type
        new_txt = type_converter.get(txt, txt)

        # Rebuild arrays from innermost to outermost
        for length in reversed(array_lengths):
            new_txt = f"[{length} x {new_txt}]"

        if ptr_depth:
            new_txt += "*" * ptr_depth

        node["text"] = new_txt

    with open(json_path, "w") as f:
        json.dump(data, f, separators=(",", ":"))


def vocab_scan(
    graph_dir: str | Path,
    kernel_subset: str | list[str] | None = None,
    max_archives: int | None = None,
    first_n: int | None = None,
) -> tuple(dict, int, dict):
    """
    Walk graph_dir and collect instruction vocabulary mappings,
    max edge position for positional arguments, and vocab counts
    for every type of node.

    Returns instruction vocab mapping, max edge position, and vocab counts.
    """
    instruction_set = set()
    vocab_counts = {
        "instruction": {},
        "variable": {},
        "constant": {},
        "pragma": {},
        "block": {},
    }
    max_pos = 0

    paths = list(iter_graph_paths(graph_dir, kernel_subset, max_archives, first_n))
    for _ks, graph_path in tqdm.tqdm(paths, desc="Parsing graph files for building vocab"):
        try:
            graph_data = load_graph_json(graph_path)
        except Exception as e:
            print(f"Error loading JSON: {graph_path}: {e}")
            continue
        nodes = graph_data.get("nodes") or []
        for n in nodes:
            node_type = n.get("type", -1)
            term = n.get("text", "")
            if node_type == NODE_INSTRUCTION:
                instruction_set.add(term)
                vocab_counts["instruction"][term] = vocab_counts["instruction"].get(term, 0) + 1
            elif node_type == NODE_VARIABLE:
                vocab_counts["variable"][term] = vocab_counts["variable"].get(term, 0) + 1
            elif node_type == NODE_CONSTANT:
                vocab_counts["constant"][term] = vocab_counts["constant"].get(term, 0) + 1
            elif node_type == NODE_PRAGMA:
                vocab_counts["pragma"][term] = vocab_counts["pragma"].get(term, 0) + 1
            elif node_type == NODE_BLOCK:
                vocab_counts["block"][term] = vocab_counts["block"].get(term, 0) + 1

        for link in graph_data.get("links") or []:
            position = link.get("position", 0)
            if position > max_pos:
                max_pos = position

    instruction_vocab = {"UNK": 0}
    instruction_vocab.update({t: i + 1 for i, t in enumerate(sorted(instruction_set))})
    return instruction_vocab, max_pos, vocab_counts


def save_vocab(vocab: dict, max_pos: int, path: str | Path, vocab_counts: dict | None = None) -> None:
    """Persist vocab mapping to JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"vocab": vocab, "max_pos": max_pos}
    if vocab_counts is not None:
        payload["vocab_counts"] = vocab_counts
    with path.open("w") as f:
        json.dump(payload, f, indent=2)


def load_vocab(path: str | Path) -> tuple[dict, int, dict]:
    """Load vocab from JSON. Returns (vocab, max_pos, vocab_counts)."""
    with Path(path).open() as f:
        payload = json.load(f)
    return payload["vocab"], payload["max_pos"], payload.get("vocab_counts", {})
