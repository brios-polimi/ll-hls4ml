"""CDFG JSON schema constants."""

NODE_INSTRUCTION = 0
NODE_VARIABLE = 1
NODE_CONSTANT = 2
NODE_PRAGMA = 3

FLOW_CONTROL = 0
FLOW_DATA = 1
FLOW_CALL = 2
FLOW_PRAGMA = 3

NODE_TYPES = ["instruction", "variable", "constant", "pragma"]

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
}

EDGE_TYPES = [
    ("instruction", "control", "instruction"),
    ("instruction", "data", "variable"),
    ("variable", "data", "instruction"),
    ("constant", "data", "instruction"),
    ("instruction", "call", "instruction"),
    ("pragma", "applies_to", "instruction"),
    ("pragma", "applies_to", "variable"),
]

SELF_LOOP_EDGE_TYPES = [
    (nt, 'self', nt) for nt in NODE_TYPES
]
ALL_EDGE_TYPES = [*EDGE_TYPES, *SELF_LOOP_EDGE_TYPES]

EDGE_TYPES_WITH_ATTR = [
    ("instruction", "control", "instruction"),
    ("variable", "data", "instruction"),
    ("constant", "data", "instruction"),
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


def safe_int(x):
    try:
        return int(x)
    except (ValueError, TypeError):
        return -1
