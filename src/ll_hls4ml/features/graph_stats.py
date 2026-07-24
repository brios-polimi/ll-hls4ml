"""Handcrafted graph-level features from CDFG JSON."""

from collections import Counter, deque

import networkx as nx
import numpy as np

from ll_hls4ml.data.tensorize import (
    ARRAY_LEN_OFF,
    BITS_OFF,
    FRAC_OFF,
    IS_AC_OFF,
    IS_AP_OFF,
    PTR_DEPTH_OFF,
    SIGNED_OFF,
    type_embedding,
)
from ll_hls4ml.io.schema import NODE_CONSTANT, NODE_PRAGMA, NODE_VARIABLE


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
    arrays = embeddings[:, TYPE_FLAGS["array"]] > 0 if len(embeddings) else np.array([], dtype=bool)
    pointers = embeddings[:, TYPE_FLAGS["pointer"]] > 0 if len(embeddings) else np.array([], dtype=bool)

    stats.update({
        "type_bits_mean": float(embeddings[parsed_numeric, BITS_OFF].mean()) if parsed_numeric.any() else 0.0,
        "type_bits_max": float(embeddings[parsed_numeric, BITS_OFF].max()) if parsed_numeric.any() else 0.0,
        "type_signed_ratio": float(embeddings[parsed_numeric, SIGNED_OFF].mean()) if parsed_numeric.any() else 0.0,
        "type_fractional_ratio_mean": float(embeddings[fixed, FRAC_OFF].mean()) if fixed.any() else 0.0,
        "type_array_log_length_mean": float(embeddings[arrays, ARRAY_LEN_OFF].mean()) if arrays.any() else 0.0,
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

    num_instruction_nodes = node_type_counts[0]
    num_variable_nodes = node_type_counts[1]
    num_constant_nodes = node_type_counts[2]
    num_pragma_nodes = node_type_counts[NODE_PRAGMA]

    instruction_ratio = num_instruction_nodes / num_nodes if num_nodes > 0 else 0.0
    variable_ratio = num_variable_nodes / num_nodes if num_nodes > 0 else 0.0
    constant_ratio = num_constant_nodes / num_nodes if num_nodes > 0 else 0.0
    pragma_ratio = num_pragma_nodes / num_nodes if num_nodes > 0 else 0.0

    flow_counts = Counter()
    in_degree = Counter()
    out_degree = Counter()

    flow_types = [
        ("instruction", "control", "instruction"),
        ("instruction", "data", "variable"),
        ("variable", "data", "instruction"),
        ("constant", "data", "instruction"),
        ("instruction", "call", "instruction"),
        ("pragma", "applies_to", "instruction"),
        ("pragma", "applies_to", "variable"),
    ]

    for e in links:
        flow = e.get("flow", -1)
        source = e.get("source", -1)
        target = e.get("target", -1)

        out_degree[source] += 1
        in_degree[target] += 1
        G.add_edge(source, target)

        source_type = nodes[source].get("type", -1)
        target_type = nodes[target].get("type", -1)

        known_edge_type = False
        if flow == 0:
            if source_type == 0 and target_type == 0:
                flow_counts[flow_types[0]] += 1
                known_edge_type = True
        elif flow == 1:
            if source_type == 0 and target_type == 1:
                flow_counts[flow_types[1]] += 1
                known_edge_type = True
            elif source_type == 1 and target_type == 0:
                flow_counts[flow_types[2]] += 1
                known_edge_type = True
            elif source_type == 2 and target_type == 0:
                flow_counts[flow_types[3]] += 1
                known_edge_type = True
        elif flow == 2:
            if source_type == 0 and target_type == 0:
                flow_counts[flow_types[4]] += 1
                known_edge_type = True
        elif flow == 3:
            if source_type == NODE_PRAGMA and target_type == 0:
                flow_counts[flow_types[5]] += 1
                known_edge_type = True
            elif source_type == NODE_PRAGMA and target_type == NODE_VARIABLE:
                flow_counts[flow_types[6]] += 1
                known_edge_type = True

        if not known_edge_type:
            raise ValueError(
                f"Unknown edge type with flow={flow}, "
                f"source_type={source_type}, target_type={target_type}"
            )

    num_inst_control_inst_edges = flow_counts[flow_types[0]]
    num_inst_data_var_edges = flow_counts[flow_types[1]]
    num_var_data_inst_edges = flow_counts[flow_types[2]]
    num_const_data_inst_edges = flow_counts[flow_types[3]]
    num_inst_call_inst_edges = flow_counts[flow_types[4]]
    num_pragma_inst_edges = flow_counts[flow_types[5]]
    num_pragma_var_edges = flow_counts[flow_types[6]]

    inst_control_inst_ratio = num_inst_control_inst_edges / num_edges if num_edges > 0 else 0.0
    inst_data_var_ratio = num_inst_data_var_edges / num_edges if num_edges > 0 else 0.0
    var_data_inst_ratio = num_var_data_inst_edges / num_edges if num_edges > 0 else 0.0
    const_data_inst_ratio = num_const_data_inst_edges / num_edges if num_edges > 0 else 0.0
    inst_call_inst_ratio = num_inst_call_inst_edges / num_edges if num_edges > 0 else 0.0
    pragma_inst_ratio = num_pragma_inst_edges / num_edges if num_edges > 0 else 0.0
    pragma_var_ratio = num_pragma_var_edges / num_edges if num_edges > 0 else 0.0

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
        "inst_control_inst_ratio": inst_control_inst_ratio,
        "inst_data_var_ratio": inst_data_var_ratio,
        "var_data_inst_ratio": var_data_inst_ratio,
        "const_data_inst_ratio": const_data_inst_ratio,
        "inst_call_inst_ratio": inst_call_inst_ratio,
        "pragma_inst_ratio": pragma_inst_ratio,
        "pragma_var_ratio": pragma_var_ratio,
        "mean_in_degree": mean_in_degree,
        "max_in_degree": max_in_degree,
        "std_in_degree": std_in_degree,
        "mean_out_degree": mean_out_degree,
        "max_out_degree": max_out_degree,
        "std_out_degree": std_out_degree,
    }
    features.update(geometry_features)
    features.update(semantic_type_stats(nodes))

    labels = graph_data.get("labels", {})
    for k, v in labels.items():
        features[k] = v

    return features
