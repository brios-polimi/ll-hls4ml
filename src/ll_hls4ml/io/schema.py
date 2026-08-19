"""CDFG JSON schema constants."""

NODE_INSTRUCTION = 0
NODE_VARIABLE = 1
NODE_CONSTANT = 2
NODE_PRAGMA = 3
NODE_BLOCK = 4
NODE_FUNCTION = 5
NODE_LOOP = 6

NODE_TYPE_NAMES = {
    NODE_INSTRUCTION: "instruction",
    NODE_VARIABLE: "variable",
    NODE_CONSTANT: "constant",
    NODE_PRAGMA: "pragma",
    NODE_BLOCK: "block",
    NODE_FUNCTION: "function",
    NODE_LOOP: "loop",
}
# Canonical JSON relations. Endpoint types carry most semantics; the relation
# remains explicit for validation and for the few same-endpoint alternatives.
EDGE_TYPES = [
    ("instruction", "control", "instruction"),
    ("instruction", "defines", "variable"),
    ("variable", "operand", "instruction"),
    ("constant", "operand", "instruction"),
    ("instruction", "calls", "function"),
    ("pragma", "applies_to", "instruction"),
    ("pragma", "applies_to", "variable"),
    ("pragma", "applies_to", "constant"),
    ("pragma", "applies_to", "block"),
    ("pragma", "applies_to", "function"),
    ("block", "control", "block"),
    ("block", "contains", "instruction"),
    ("function", "contains", "block"),
]
EDGE_TYPE_SET = frozenset(EDGE_TYPES)
REGION_EDGE_TYPES = [
    *EDGE_TYPES,
    ("pragma", "applies_to", "loop"),
    ("loop", "contains", "block"),
    ("loop", "contains", "loop"),
    ("function", "contains", "loop"),
]
REGION_EDGE_TYPE_SET = frozenset(REGION_EDGE_TYPES)
RELATION_NAMES = tuple(dict.fromkeys(relation for _, relation, _ in EDGE_TYPES))

# Tensor-only adjacency derived once during preprocessing. It is intentionally
# absent from graph JSON because it is a lossless join of defines + operand.
DERIVED_DEF_USE_EDGE = ("instruction", "def_use", "instruction")

# Hierarchy projection keys used by schema-driven visualization. Add a level
# here when the graph producer introduces another structural node type.
HIERARCHY_LEVEL_KEYS = {
    "function": ("function",),
    "block": ("function", "block"),
    "loop": ("function", "loop"),
}
# The six-node list remains the schema-v2/H0 contract. Region-aware models opt
# into the seventh type explicitly so old model behavior and parameterization
# do not change merely because a schema-v3 tensor is loaded.
NODE_TYPES = [NODE_TYPE_NAMES[node_type] for node_type in range(NODE_LOOP)]
REGION_NODE_TYPES = [*NODE_TYPES, "loop"]
BLOCK_FEATURE_SIZE = 3
FUNCTION_FEATURE_SIZE = 2
LOOP_FEATURE_SIZE = 7
LOOP_HIERARCHY_SCHEMA_VERSION = 3

PRAGMA_VOCAB = {
    "UNK": 0,
    "pragma.allocation": 1,
    "pragma.array_partition": 2,
    "pragma.array_reshape": 3,
    "pragma.dataflow": 4,
    "pragma.function_instantiate": 5,
    "pragma.inline": 6,
    "pragma.interface": 7,
    "pragma.pipeline": 8,
    "pragma.unroll": 9,
    "pragma.dependence": 10,
    "pragma.latency": 11,
    "pragma.loop_flatten": 12,
    "pragma.loop_merge": 13,
    "pragma.loop_tripcount": 14,
    "pragma.occurrence": 15,
    "pragma.protocol": 16,
    "pragma.reset": 17,
    "pragma.resource": 18,
    "pragma.stable": 19,
    "pragma.stream": 20,
    "pragma.bind_op": 21,
    "pragma.bind_storage": 22,
    "pragma.reqd_pipe_depth": 23,
    "pragma.aggregate": 24,
}

PRAGMA_SCHEMA_VERSION = 2
PRAGMA_VOCAB_SIZE = max(PRAGMA_VOCAB.values()) + 1


def pragma_directive_id(text: str) -> int:
    return PRAGMA_VOCAB.get(text, PRAGMA_VOCAB["UNK"])


# Only these directives receive learned argument features. Other directives
# retain their categorical directive ID and their full arguments in graph JSON.
PRAGMA_ARGUMENT_DIRECTIVES = frozenset(
    {
        "pragma.allocation",
        "pragma.array_partition",
        "pragma.array_reshape",
        "pragma.bind_op",
        "pragma.bind_storage",
        "pragma.dataflow",
        "pragma.inline",
        "pragma.interface",
        "pragma.latency",
        "pragma.loop_flatten",
        "pragma.loop_merge",
        "pragma.loop_tripcount",
        "pragma.pipeline",
        "pragma.resource",
        "pragma.stream",
        "pragma.unroll",
    }
)

# Common numeric HLS arguments receive stable, interpretable value/mask slots.
# Other arguments remain in graph JSON but are not learned features.
PRAGMA_NUMERIC_ARGUMENTS = (
    "ii",
    "factor",
    "dim",
    "depth",
    "min",
    "max",
    "avg",
    "latency",
    "interval",
    "num",
    "max_read_burst_length",
    "max_write_burst_length",
    "num_read_outstanding",
    "num_write_outstanding",
    "max_widen_bitwidth",
    "limit",
)
PRAGMA_TARGET_ARGUMENTS = frozenset({"variable", "port"})
PRAGMA_CATEGORICAL_ARGUMENTS = (
    # Array partitioning and reshaping.
    ("complete", "true"),
    ("block", "true"),
    ("cyclic", "true"),
    # Scheduling and loop controls.
    ("rewind", "true"),
    ("enable_flush", "true"),
    ("disable_start_propagation", "true"),
    ("off", "true"),
    ("recursive", "true"),
    ("skip_exit_check", "true"),
    ("force", "true"),
    ("style", "flp"),
    ("style", "stp"),
    ("style", "frp"),
    # Interface modes and controls.
    ("ap_none", "true"),
    ("ap_stable", "true"),
    ("ap_vld", "true"),
    ("ap_ack", "true"),
    ("ap_hs", "true"),
    ("ap_memory", "true"),
    ("bram", "true"),
    ("ap_fifo", "true"),
    ("axis", "true"),
    ("m_axi", "true"),
    ("s_axilite", "true"),
    ("register", "true"),
    ("direct_io", "true"),
    ("offset", "direct"),
    ("offset", "slave"),
    ("offset", "off"),
    # Storage binding choices with clear resource meaning.
    ("type", "fifo"),
    ("type", "ram_1p"),
    ("type", "ram_2p"),
    ("type", "ram_s2p"),
    ("type", "ram_t2p"),
    ("type", "rom_1p"),
    ("type", "rom_2p"),
    ("type", "rom_np"),
    ("impl", "auto"),
    ("impl", "bram"),
    ("impl", "lutram"),
    ("impl", "uram"),
    ("impl", "memory"),
    ("impl", "dsp"),
    ("impl", "fabric"),
    # Operation/resource kinds that materially affect allocation and binding.
    ("instances", "mul"),
    ("instances", "add"),
    ("instances", "sub"),
    ("instances", "div"),
    ("op", "mul"),
    ("op", "add"),
    ("op", "sub"),
    ("op", "fadd"),
    ("op", "fsub"),
    ("op", "fmul"),
    ("op", "udiv"),
    ("op", "sdiv"),
    ("op", "fdiv"),
    ("op", "sqrt"),
)
PRAGMA_ARGUMENT_SIZE = (
    2 * len(PRAGMA_NUMERIC_ARGUMENTS)
    + len(PRAGMA_CATEGORICAL_ARGUMENTS)
)
PRAGMA_FEATURE_SIZE = 1 + PRAGMA_ARGUMENT_SIZE

SELF_LOOP_EDGE_TYPES = [
    (nt, 'self', nt) for nt in NODE_TYPES
]
ALL_EDGE_TYPES = [*EDGE_TYPES, *SELF_LOOP_EDGE_TYPES]

EDGE_TYPES_WITH_ATTR = [
    ("instruction", "control", "instruction"),
    ("variable", "operand", "instruction"),
    ("constant", "operand", "instruction"),
]

LABEL_KEYS = [
    "lut",
    "ff",
    "dsp",
    "bram",
    #"uram",
    "cycles_max",
    #"cycles_min",
    #"estimated_clock",
    "interval_max",
    #"interval_min",
    #"target_clock",
]

# Graph-level causal synthesis context. Index zero is always unknown so new
# tool/device versions remain loadable without pretending they are known.
GRAPH_CONTEXT_CATEGORICAL_VOCABS = {
    "backend": {"vitis": 1, "bambu": 2, "VivadoAccelerator": 3},
    "target_part": {
        "xcu200-fsgd2104-2-e": 1,
        "xcu250-figd2104-2L-e": 2,
        "xczu9eg-ffvb1156-2-e": 3,
        "xc7z020clg400-1": 4,
    },
    "vivado_version": {
        "2019.1": 1,
        "2020.1": 2,
        "2023.2": 3,
        "2024.2": 4,
    },
    "hls4ml_version": {
        "0.5.0": 1,
        "0.6.0": 2,
        "0.7.0": 3,
        "0.8.0": 4,
        "0.8.1": 5,
        "0.9.0": 6,
        "1.0.0": 7,
        "1.1.0": 8,
    },
}
GRAPH_CONTEXT_NUMERIC_KEYS = ("target_clock",)


def safe_int(x):
    try:
        return int(x)
    except (ValueError, TypeError):
        return -1
