"""Build and persist instruction/variable/constant vocabularies from CDFG JSON."""

from __future__ import annotations

import json
from pathlib import Path

import tqdm
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from collections import defaultdict

from ll_hls4ml.config import load_config
from ll_hls4ml.io.discovery import iter_graph_paths
from ll_hls4ml.io.load_json import load_graph_json
from ll_hls4ml.io.schema import NODE_CONSTANT, NODE_INSTRUCTION, NODE_VARIABLE


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


def _compile_and_graph(hls4ml_dir: Path, clang_binary: str, programl_binary: str, programl_to_json_binary: str) -> Path | None:
    """Compile one HLS4ML project to LLVM IR, run ProGraML, and write JSON. Returns json_path or None on failure."""
    try:
        if any(p.name == "firmware" for p in hls4ml_dir.iterdir() if p.is_dir()):
            types_path = "ac_types" if "Bambu" in hls4ml_dir.name else "ap_types"
            compile_cmd = (
                f"{clang_binary} -S -emit-llvm {hls4ml_dir}/firmware/myproject.cpp "
                f"-I{hls4ml_dir}/firmware/{types_path} -o {hls4ml_dir}/myproject.ll"
            )
            result = subprocess.run(compile_cmd, shell=True, capture_output=True)
            if result.returncode != 0:
                #print(f"[SKIP] Compilation failed for {hls4ml_dir}: {result.stderr.decode()[:200]}")
                return None

        ll_path = hls4ml_dir / "myproject.ll"
        if not ll_path.exists():
            #print(f"[SKIP] No myproject.ll found in {hls4ml_dir}")
            return None

        ir_result = subprocess.run(
            [programl_binary, str(ll_path)],
            capture_output=True,
            timeout=60,
        )
        if ir_result.returncode != 0:
            #print(f"[SKIP] ProGraML IR failed for {hls4ml_dir}: {ir_result.stderr.decode()[:200]}")
            return None

        json_result = subprocess.run(
            [programl_to_json_binary],
            input=ir_result.stdout,
            capture_output=True,
            timeout=60,
        )
        if json_result.returncode != 0:
            #print(f"[SKIP] ProGraML JSON failed for {hls4ml_dir}: {json_result.stderr.decode()[:200]}")
            return None

        json_path = hls4ml_dir / "myproject.json"
        json_path.write_bytes(json_result.stdout)
        return json_path

    except Exception as e:
        #print(f"[SKIP] Unexpected error for {hls4ml_dir}: {e}")
        return None


def vocab_scan_from_hls4ml_projects(
    hls4ml_dir: str | Path,
    programl_binary: str,
    programl_to_json_binary: str,
    clang_binary: str,
    max_workers: int = 8,
):
    """
    Scan directories of HLS4ML projects, compile them, graph them, and parse CDFG JSONs.
    Walk graph_dir and collect vocabularies of instructions, variables, and constants.

    Returns vocab dict, max edge position, and per-token counts.
    """
    hls4ml_dir = Path(hls4ml_dir)
    hls4ml_dirs = [d for d in hls4ml_dir.iterdir() if d.is_dir()]

    # Multithreaded compilation and graph generation
    # ThreadPoolExecutor is appropriate here: the bottleneck is subprocess I/O, not the GIL.
    json_paths = []
    futures_map = {}

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for d in hls4ml_dirs:
            future = executor.submit(
                _compile_and_graph, d, clang_binary, programl_binary, programl_to_json_binary
            )
            futures_map[future] = d

        with tqdm.tqdm(total=len(futures_map), desc="Compiling HLS4ML kernels") as pbar:
            for future in as_completed(futures_map):
                result = future.result()   # exceptions are re-raised here if _compile_and_graph let any through
                if result is not None:
                    json_paths.append(result)
                pbar.update(1)

    # Build vocab from CDFG JSONs
    vocab_sets: dict[str, set] = {"instruction": set(), "variable": set(), "constant": set()}
    vocab_counts: dict[str, dict] = {"instruction": {}, "variable": {}, "constant": {}}
    max_pos = 0

    for json_path in tqdm.tqdm(json_paths, desc="Parsing CDFG JSONs for building vocab"):
        try:
            graph_data = load_graph_json(json_path)
        except Exception as e:
            print(f"Error loading JSON: {json_path}: {e}")
            continue
        nodes = graph_data.get("nodes") or []
        for n in nodes:
            node_type = n.get("type", -1)
            term = n.get("text", "")
            if node_type == NODE_INSTRUCTION:
                vocab_sets["instruction"].add(term)
                vocab_counts["instruction"][term] = vocab_counts["instruction"].get(term, 0) + 1
            elif node_type == NODE_VARIABLE:
                vocab_sets["variable"].add(term)
                vocab_counts["variable"][term] = vocab_counts["variable"].get(term, 0) + 1
            elif node_type == NODE_CONSTANT:
                vocab_sets["constant"].add(term)
                vocab_counts["constant"][term] = vocab_counts["constant"].get(term, 0) + 1

        for link in graph_data.get("links") or []:
            position = link.get("position", 0)
            if position > max_pos:
                max_pos = position

    vocab = {}
    for k, v in vocab_sets.items():
        vocab[k] = {"UNK": 0}
        vocab[k].update({t: i + 1 for i, t in enumerate(sorted(v))})
    return vocab, max_pos, vocab_counts


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
