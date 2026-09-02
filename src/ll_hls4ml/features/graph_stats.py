"""Handcrafted graph-level features from CDFG JSON."""

from collections import Counter, deque

import networkx as nx
import numpy as np

from ll_hls4ml.data.tensorize import (
    SPATIAL_LEN_OFF,
    TEMPORAL_LEN_OFF,
    BITS_OFF,
    FRAC_OFF,
    IS_AC_OFF,
    IS_AP_OFF,
    PTR_DEPTH_OFF,
    SIGNED_OFF,
    type_embedding,
)
from ll_hls4ml.io.schema import (
    SERIALIZED_EDGE_TYPE_SET,
    NODE_BLOCK,
    NODE_CONSTANT,
    NODE_PRAGMA,
    NODE_FUNCTION,
    NODE_INSTRUCTION,
    NODE_TYPE_NAMES,
    NODE_VARIABLE,
)


TYPE_FLAGS = {
    "integer": 0,
    "float": 1,
    "double": 2,
    "arb_int": 3,
    "arb_fixed": 4,
    "array": 5,
    "pointer": 6,
    "stream": 7,
    "nnet_array": 8,
    "shift_reg": 9,
    "unknown": 10,
}


def semantic_type_stats(nodes):
    """Aggregate variable/constant type vectors into compact graph features."""
    typed_nodes = [
        n for n in nodes
        if n.get("type") in (NODE_VARIABLE, NODE_CONSTANT)
    ]
    embeddings = (
        np.stack([type_embedding(n.get("text", "")) for n in typed_nodes])
        if typed_nodes
        else np.empty((0, PTR_DEPTH_OFF + 1), dtype=np.float32)
    )

    stats = {
        f"type_{name}_ratio": float(embeddings[:, index].mean()) if len(embeddings) else 0.0
        for name, index in TYPE_FLAGS.items()
    }
    stats["type_ap_ratio"] = float(embeddings[:, IS_AP_OFF].mean()) if len(embeddings) else 0.0
    stats["type_ac_ratio"] = float(embeddings[:, IS_AC_OFF].mean()) if len(embeddings) else 0.0

    parsed_numeric = embeddings[:, BITS_OFF] > 0 if len(embeddings) else np.array([], dtype=bool)
    fixed = embeddings[:, TYPE_FLAGS["arb_fixed"]] > 0 if len(embeddings) else np.array([], dtype=bool)
    spatial_containers = (
        (embeddings[:, TYPE_FLAGS["array"]] > 0)
        | (embeddings[:, TYPE_FLAGS["nnet_array"]] > 0)
        if len(embeddings)
        else np.array([], dtype=bool)
    )
    temporal_containers = (
        embeddings[:, TYPE_FLAGS["shift_reg"]] > 0
        if len(embeddings)
        else np.array([], dtype=bool)
    )
    pointers = embeddings[:, TYPE_FLAGS["pointer"]] > 0 if len(embeddings) else np.array([], dtype=bool)

    stats.update({
        "type_bits_mean": float(embeddings[parsed_numeric, BITS_OFF].mean()) if parsed_numeric.any() else 0.0,
        "type_bits_max": float(embeddings[parsed_numeric, BITS_OFF].max()) if parsed_numeric.any() else 0.0,
        "type_signed_ratio": float(embeddings[parsed_numeric, SIGNED_OFF].mean()) if parsed_numeric.any() else 0.0,
        "type_fractional_ratio_mean": float(embeddings[fixed, FRAC_OFF].mean()) if fixed.any() else 0.0,
        "type_spatial_log_length_mean": float(embeddings[spatial_containers, SPATIAL_LEN_OFF].mean()) if spatial_containers.any() else 0.0,
        "type_temporal_log_length_mean": float(embeddings[temporal_containers, TEMPORAL_LEN_OFF].mean()) if temporal_containers.any() else 0.0,
        "type_pointer_depth_mean": float(embeddings[pointers, PTR_DEPTH_OFF].mean()) if pointers.any() else 0.0,
    })

    for node_type, name in [(NODE_VARIABLE, "variable"), (NODE_CONSTANT, "constant")]:
        mask = np.array([n.get("type") == node_type for n in typed_nodes], dtype=bool)
        stats[f"type_{name}_unknown_ratio"] = (
            float(embeddings[mask, TYPE_FLAGS["unknown"]].mean()) if mask.any() else 0.0
        )

    return stats


def dag_level_stats(G):
    in_deg = dict(G.in_degree())
    level = {n: 0 for n in G.nodes()}
    queue = deque([n for n, d in in_deg.items() if d == 0])

    while queue:
        node = queue.popleft()
        for succ in G.successors(node):
            level[succ] = max(level[succ], level[node] + 1)
            in_deg[succ] -= 1
            if in_deg[succ] == 0:
                queue.append(succ)

    level_counts = Counter(level.values())
    widths = list(level_counts.values())

    return {
        "critical_path_depth": max(level.values()) if level else 0,
        "max_width": max(widths) if widths else 0,
        "mean_width": np.mean(widths) if widths else 0.0,
        "total_levels": len(widths),
    }


def extract_graph_features(graph_data):
    """Return a flat dict of handcrafted graph features."""
    nodes = graph_data.get("nodes", [])
    links = graph_data.get("links", [])

    G = nx.DiGraph()
    num_nodes = len(nodes)
    num_edges = len(links)

    node_type_counts = Counter()

    for n in nodes:
        nid = n.get("id")
        G.add_node(nid)
        node_type = n.get("type", -1)
        node_type_counts[node_type] += 1

    num_instruction_nodes = node_type_counts[NODE_INSTRUCTION]
    num_variable_nodes = node_type_counts[NODE_VARIABLE]
    num_constant_nodes = node_type_counts[NODE_CONSTANT]
    num_pragma_nodes = node_type_counts[NODE_PRAGMA]
    num_block_nodes = node_type_counts[NODE_BLOCK]
    num_function_nodes = node_type_counts[NODE_FUNCTION]

    instruction_ratio = num_instruction_nodes / num_nodes if num_nodes > 0 else 0.0
    variable_ratio = num_variable_nodes / num_nodes if num_nodes > 0 else 0.0
    constant_ratio = num_constant_nodes / num_nodes if num_nodes > 0 else 0.0
    pragma_ratio = num_pragma_nodes / num_nodes if num_nodes > 0 else 0.0
    block_ratio = num_block_nodes / num_nodes if num_nodes > 0 else 0.0
    function_ratio = num_function_nodes / num_nodes if num_nodes > 0 else 0.0

    edge_type_counts = Counter()
    in_degree = Counter()
    out_degree = Counter()

    for e in links:
        source = e.get("source", -1)
        target = e.get("target", -1)

        out_degree[source] += 1
        in_degree[target] += 1
        G.add_edge(source, target)

        edge_type = (
            NODE_TYPE_NAMES.get(nodes[source].get("type", -1)),
            str(e.get("relation", "")),
            NODE_TYPE_NAMES.get(nodes[target].get("type", -1)),
        )
        if edge_type not in SERIALIZED_EDGE_TYPE_SET:
            raise ValueError(f"Unknown canonical edge type: {edge_type}")
        edge_type_counts[edge_type] += 1

    edge_ratios = {
        f"edge_{source}_{relation}_{target}_ratio": (
            edge_type_counts[(source, relation, target)] / num_edges
            if num_edges
            else 0.0
        )
        for source, relation, target in SERIALIZED_EDGE_TYPE_SET
    }

    density = num_edges / (num_nodes * (num_nodes - 1)) if num_nodes > 1 else 0.0
    condensed = nx.condensation(G)
    geometry_features = dag_level_stats(condensed)

    all_node_ids = [n["id"] for n in nodes]
    in_degs = [in_degree[nid] for nid in all_node_ids]
    out_degs = [out_degree[nid] for nid in all_node_ids]

    mean_in_degree = np.mean(in_degs) if in_degs else 0.0
    max_in_degree = np.max(in_degs) if in_degs else 0.0
    std_in_degree = np.std(in_degs) if in_degs else 0.0
    mean_out_degree = np.mean(out_degs) if out_degs else 0.0
    max_out_degree = np.max(out_degs) if out_degs else 0.0
    std_out_degree = np.std(out_degs) if out_degs else 0.0

    features = {
        "num_nodes": num_nodes,
        "num_edges": num_edges,
        "density": density,
        "instruction_ratio": instruction_ratio,
        "variable_ratio": variable_ratio,
        "constant_ratio": constant_ratio,
        "pragma_ratio": pragma_ratio,
        "block_ratio": block_ratio,
        "function_ratio": function_ratio,
        "mean_in_degree": mean_in_degree,
        "max_in_degree": max_in_degree,
        "std_in_degree": std_in_degree,
        "mean_out_degree": mean_out_degree,
        "max_out_degree": max_out_degree,
        "std_out_degree": std_out_degree,
    }
    features.update(edge_ratios)
    features.update(geometry_features)
    features.update(semantic_type_stats(nodes))

    labels = graph_data.get("labels", {})
    for k, v in labels.items():
        features[k] = v

    return features
