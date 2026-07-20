"""Inject Vitis HLS pragma nodes into a ProGraML JSON graph.

Vitis exposes pragmas in two complementary compiler outputs:

* ``-Wdump-hls-pragmas`` reports every recognized source pragma with its
  directive, function, and options.
* the synthesis LLVM IR materializes some variable directives as
  ``llvm.sideeffect`` calls carrying an ``xlx_*`` bundle.

This module uses those compiler-native signals only.  It deliberately does
not read source files or DWARF metadata, which are not reliable after the
Vitis LLVM 7 -> LLVM 16 compatibility conversion.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import json
from pathlib import Path
import re

from ll_hls4ml.io.schema import (
    NODE_INSTRUCTION, NODE_VARIABLE, NODE_CONSTANT, NODE_PRAGMA,
    FLOW_CALL, FLOW_CONTROL, FLOW_DATA, FLOW_PRAGMA
)

_DUMP_RE = re.compile(
    r"^(?P<path>.+?):(?P<line>\d+):(?P<column>\d+): warning: "
    r"HLS pragma dump (?P<fields>.+?) \[-Wdump-hls-pragmas\]$"
)
_FIELD_RE = re.compile(r"(?P<key>Pragma\w+)=(?P<value>.*)")
_CARRIER_RE = re.compile(
    r'@llvm\.sideeffect\(\).*?\[\s*"xlx_(?P<directive>[A-Za-z0-9_]+)"'
    r"\((?P<arguments>.*?)\)\s*\]",
    re.DOTALL,
)
_TARGET_RE = re.compile(r"\b(?:variable|port)\s*=\s*([^\s]+)", re.IGNORECASE)
_SSA_NAME_RE = re.compile(r"%([A-Za-z_$.-][\w$.-]*)")


@dataclass(frozen=True)
class VitisPragma:
    """A pragma record emitted by ``-Wdump-hls-pragmas``."""

    path: str
    line: int
    column: int
    directive: str
    function: str
    options: str
    raw_fields: str
    targets: tuple[str, ...]

    @property
    def text(self) -> str:
        return f"pragma.{self.directive}"


def _normalise_directive(value: str) -> str:
    return re.sub(r"[^a-z0-9_]+", "_", value.lower()).strip("_") or "unknown"


def _targets(options: str) -> tuple[str, ...]:
    match = _TARGET_RE.search(options)
    if not match:
        return ()
    return tuple(name.strip() for name in match.group(1).split(",") if name.strip())


def read_vitis_pragma_dump(path: str | Path) -> list[VitisPragma]:
    """Parse Vitis ``-Wdump-hls-pragmas`` diagnostics from ``path``.

    Unknown directives are retained. They become ``pragma.<directive>`` graph
    nodes and therefore remain available to later vocabulary/feature work.
    """

    records: list[VitisPragma] = []
    for line in Path(path).read_text(errors="replace").splitlines():
        match = _DUMP_RE.match(line)
        if not match:
            continue
        fields: dict[str, str] = {}
        for piece in match.group("fields").split("_XLX_SEP_"):
            field = _FIELD_RE.fullmatch(piece.strip())
            if field:
                fields[field.group("key")] = field.group("value").strip()
        directive = _normalise_directive(fields.get("PragmaType", "unknown"))
        options = fields.get("PragmaOptions", "")
        records.append(
            VitisPragma(
                path=match.group("path"),
                line=int(match.group("line")),
                column=int(match.group("column")),
                directive=directive,
                function=fields.get("PragmaFunction", ""),
                options=options,
                raw_fields=match.group("fields"),
                targets=_targets(options),
            )
        )
    return records


def _full_text(node: dict) -> str:
    value = node.get("features", {}).get("full_text", [])
    return "\n".join(value) if isinstance(value, list) else str(value)


def _remove_previous_injection(graph: dict) -> None:
    removed = {
        int(node["id"])
        for node in graph.get("nodes", [])
        if node.get("features", {}).get("injector") == ["ll_hls4ml.vitis_pragmas"]
    }
    if not removed:
        return

    nodes = [node for node in graph["nodes"] if int(node["id"]) not in removed]
    id_map = {int(node["id"]): index for index, node in enumerate(nodes)}
    for index, node in enumerate(nodes):
        node["id"] = index
    graph["nodes"] = nodes
    graph["links"] = [
        {**link, "source": id_map[int(link["source"])], "target": id_map[int(link["target"])]}
        for link in graph.get("links", [])
        if int(link["source"]) not in removed and int(link["target"]) not in removed
    ]


def _carrier(node: dict) -> tuple[str, tuple[str, ...]] | None:
    if int(node.get("type", -1)) != NODE_INSTRUCTION:
        return None
    match = _CARRIER_RE.search(_full_text(node))
    if not match:
        return None
    return _normalise_directive(match.group("directive")), tuple(
        dict.fromkeys(_SSA_NAME_RE.findall(match.group("arguments")))
    )


def _function_ids(graph: dict, function: str) -> list[int]:
    if not function:
        return []
    return [
        index
        for index, item in enumerate(graph.get("graph", {}).get("function", []))
        if function in str(item.get("name", ""))
    ]


def _entry(nodes: list[dict], function: int) -> dict | None:
    candidates = [
        node
        for node in nodes
        if int(node.get("type", -1)) == NODE_INSTRUCTION
        and int(node.get("function", -1)) == function
        and node.get("text") != "[external]"
    ]
    return min(candidates, key=lambda node: int(node["id"]), default=None)


def _named_nodes(nodes: list[dict], function: int, name: str, node_type: int) -> list[dict]:
    pattern = re.compile(rf"%{re.escape(name)}\b")
    return [
        node
        for node in nodes
        if int(node.get("type", -1)) == node_type
        and int(node.get("function", -1)) == function
        and pattern.search(_full_text(node))
    ]


def _add_pragma_node(
    nodes: list[dict],
    links: list[dict],
    pragma: VitisPragma,
    anchors: list[dict],
    anchor_reason: str,
) -> bool:
    if not anchors:
        return False
    node_id = len(nodes)
    first = anchors[0]
    nodes.append(
        {
            "block": int(first.get("block", 0)),
            "features": {
                "full_text": [pragma.raw_fields],
                "injector": ["ll_hls4ml.vitis_pragmas"],
                "origin": ["vitis_pragma_dump"],
                "options": [pragma.options],
                "source_file": [pragma.path],
                "source_line": [str(pragma.line)],
                "target_names": [",".join(pragma.targets)],
                "anchor_reason": [anchor_reason],
            },
            "function": int(first.get("function", 0)),
            "id": node_id,
            "text": pragma.text,
            "type": NODE_PRAGMA,
        }
    )
    for anchor in {int(node["id"]): node for node in anchors}.values():
        links.append(
            {
                "flow": FLOW_PRAGMA,
                "key": 0,
                "position": 0,
                "source": node_id,
                "target": int(anchor["id"]),
            }
        )
    return True


def inject_vitis_pragmas(
    graph_path: str | Path,
    pragma_dump_path: str | Path,
) -> dict[str, int]:
    """Inject compiler-reported Vitis pragma nodes into ``graph_path`` in place.

    Carrier calls receive a pragma node plus links to both the carrier
    instruction and any graph variables feeding it. Other dump records attach
    to named variables/instructions in their reported function, or to that
    function's entry instruction when no more precise IR anchor exists.
    """

    graph_path = Path(graph_path)
    records = read_vitis_pragma_dump(pragma_dump_path)
    with graph_path.open() as handle:
        graph = json.load(handle)
    _remove_previous_injection(graph)
    nodes = graph.setdefault("nodes", [])
    links = graph.setdefault("links", [])

    data_inputs: dict[int, list[dict]] = defaultdict(list)
    for link in links:
        if int(link.get("flow", -1)) != FLOW_DATA:
            continue
        source = int(link["source"])
        target = int(link["target"])
        if 0 <= source < len(nodes) and int(nodes[source].get("type", -1)) == NODE_VARIABLE:
            data_inputs[target].append(nodes[source])

    injected = 0
    carrier_injected = 0
    represented: set[tuple[str, str]] = set()
    for node in list(nodes):
        parsed = _carrier(node)
        if parsed is None:
            continue
        directive, operand_names = parsed
        targets = tuple(name for name in operand_names if name)
        pragma = VitisPragma(
            path="",
            line=0,
            column=0,
            directive=directive,
            function="",
            options="",
            raw_fields=f"PragmaType={directive}_XLX_SEP_ PragmaOptions=ir-carrier",
            targets=targets,
        )
        anchors = [node, *data_inputs.get(int(node["id"]), [])]
        if _add_pragma_node(nodes, links, pragma, anchors, "llvm.sideeffect"):
            injected += 1
            carrier_injected += 1
            for target in targets:
                represented.add((directive, target))

    dump_injected = 0
    unmatched = 0
    for pragma in records:
        if pragma.targets and all((pragma.directive, target) in represented for target in pragma.targets):
            continue
        functions = _function_ids(graph, pragma.function)
        anchors: list[dict] = []
        reason = "function_entry"
        for function in functions:
            if pragma.targets:
                for target in pragma.targets:
                    anchors.extend(_named_nodes(nodes, function, target, NODE_VARIABLE))
                    anchors.extend(_named_nodes(nodes, function, target, NODE_INSTRUCTION))
                reason = "named_target"
            if not anchors:
                entry = _entry(nodes, function)
                if entry is not None:
                    anchors.append(entry)
        if _add_pragma_node(nodes, links, pragma, anchors, reason):
            injected += 1
            dump_injected += 1
        else:
            unmatched += 1

    with graph_path.open("w") as handle:
        json.dump(graph, handle, separators=(",", ":"))
    return {
        "pragma_dump_records": len(records),
        "carrier_pragmas_injected": carrier_injected,
        "dump_pragmas_injected": dump_injected,
        "pragma_nodes_injected": injected,
        "pragmas_unmatched": unmatched,
    }
