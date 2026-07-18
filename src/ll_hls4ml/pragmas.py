"""Inject source-level HLS pragma nodes into a ProGraML JSON graph.

The injector uses DWARF metadata in debug-enabled textual LLVM IR to connect
source locations to ProGraML instruction nodes. It intentionally scans only
project files named by ``DIFile`` metadata, rather than walking the project.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from difflib import SequenceMatcher
import json
from pathlib import Path
import re


NODE_INSTRUCTION = 0
NODE_PRAGMA = 3
FLOW_PRAGMA = 3

_META_RE = re.compile(r"^!(\d+)\s*=\s*(.*)$")
_DBG_RE = re.compile(r"!dbg !(\d+)")
_ID_FIELD_RE = re.compile(r"\b({}): !(\d+)")
_INT_FIELD_RE = re.compile(r"\b({}): (\d+)")
_STRING_FIELD_RE = re.compile(r'\b({}): "((?:\\.|[^"\\])*)"')
_PRAGMA_RE = re.compile(r"^\s*#\s*pragma\s+HLS\s+(.+?)\s*$", re.IGNORECASE)
_TARGET_RE = re.compile(r"\b(?:variable|port)\s*=\s*([A-Za-z_]\w*)", re.IGNORECASE)
_FUNCTION_RE = re.compile(r'^define\b.*@((?:"(?:\\.|[^"\\])*")|[^\s(]+)\(')

_FUNCTION_DIRECTIVES = {"dataflow", "inline", "interface"}
_LOOP_DIRECTIVES = {"pipeline", "unroll"}
_VARIABLE_DIRECTIVES = {"array_partition", "array_reshape"}
_SUPPORTED_DIRECTIVES = (
    _FUNCTION_DIRECTIVES | _LOOP_DIRECTIVES | _VARIABLE_DIRECTIVES
)


@dataclass(frozen=True)
class Pragma:
    path: Path
    line: int
    directive: str
    clause: str
    raw: str
    target_name: str | None


@dataclass(frozen=True)
class SourceLocation:
    path: Path
    line: int


def _id_field(text: str, field: str) -> int | None:
    match = re.search(_ID_FIELD_RE.pattern.format(re.escape(field)), text)
    return int(match.group(2)) if match else None


def _int_field(text: str, field: str) -> int | None:
    match = re.search(_INT_FIELD_RE.pattern.format(re.escape(field)), text)
    return int(match.group(2)) if match else None


def _string_field(text: str, field: str) -> str | None:
    match = re.search(_STRING_FIELD_RE.pattern.format(re.escape(field)), text)
    if not match:
        return None
    return bytes(match.group(2), "utf-8").decode("unicode_escape")


def _read_metadata(ll_path: Path) -> dict[int, str]:
    metadata: dict[int, str] = {}
    with ll_path.open(errors="replace") as handle:
        for line in handle:
            if line.startswith("!") and (match := _META_RE.match(line)):
                metadata[int(match.group(1))] = match.group(2).rstrip()
    return metadata


class DebugInfo:
    def __init__(self, ll_path: Path, project_path: Path):
        self.metadata = _read_metadata(ll_path)
        self.project_path = project_path.resolve()
        self.files: dict[int, Path] = {}
        self.locations: dict[int, tuple[int, int]] = {}
        self.variables: dict[int, tuple[str, int | None, int | None]] = {}
        self._scope_file_memo: dict[int, Path | None] = {}

        for meta_id, text in self.metadata.items():
            if "!DIFile(" in text:
                filename = _string_field(text, "filename")
                directory = _string_field(text, "directory") or ""
                if filename:
                    path = Path(filename)
                    if not path.is_absolute():
                        path = Path(directory) / path
                    self.files[meta_id] = path.resolve()
            elif "!DILocation(" in text:
                line = _int_field(text, "line")
                scope = _id_field(text, "scope")
                if line is not None and scope is not None:
                    self.locations[meta_id] = (line, scope)
            elif "!DILocalVariable(" in text:
                name = _string_field(text, "name")
                if name:
                    self.variables[meta_id] = (
                        name,
                        _id_field(text, "scope"),
                        _int_field(text, "line"),
                    )

    def scope_file(self, scope_id: int, seen: set[int] | None = None) -> Path | None:
        if scope_id in self._scope_file_memo:
            return self._scope_file_memo[scope_id]
        if scope_id in self.files:
            return self.files[scope_id]

        seen = set() if seen is None else seen
        if scope_id in seen:
            return None
        seen.add(scope_id)

        text = self.metadata.get(scope_id, "")
        file_id = _id_field(text, "file")
        if file_id in self.files:
            result = self.files[file_id]
        else:
            parent = _id_field(text, "scope")
            result = self.scope_file(parent, seen) if parent is not None else None
        self._scope_file_memo[scope_id] = result
        return result

    def source_location(self, dbg_id: int) -> SourceLocation | None:
        location = self.locations.get(dbg_id)
        if location is None:
            return None
        line, scope = location
        path = self.scope_file(scope)
        if path is None:
            return None
        return SourceLocation(path=path, line=line)

    def project_source_files(self) -> list[Path]:
        result = set()
        for path in self.files.values():
            try:
                path.relative_to(self.project_path)
            except ValueError:
                continue
            if path.suffix.lower() in {".c", ".cc", ".cpp", ".h", ".hh", ".hpp"}:
                result.add(path)
        return sorted(result)


def _read_pragmas(paths: list[Path]) -> tuple[list[Pragma], int]:
    pragmas: list[Pragma] = []
    unsupported = 0
    for path in paths:
        if not path.is_file():
            continue
        lines = path.read_text(errors="replace").splitlines()
        for line_number, line in enumerate(lines, start=1):
            match = _PRAGMA_RE.match(line)
            if not match:
                continue
            clause = " ".join(match.group(1).split())
            directive = clause.split(None, 1)[0].lower()
            if directive not in _SUPPORTED_DIRECTIVES:
                unsupported += 1
                continue
            target = _TARGET_RE.search(clause)
            pragmas.append(
                Pragma(
                    path=path.resolve(),
                    line=line_number,
                    directive=directive,
                    clause=clause,
                    raw=line.strip(),
                    target_name=target.group(1) if target else None,
                )
            )
    return pragmas, unsupported


def _full_text(node: dict) -> str:
    values = node.get("features", {}).get("full_text", [])
    return "\n".join(values) if isinstance(values, list) else str(values)


def _llvm_opcode(line: str) -> str:
    text = line.strip()
    if " = " in text:
        text = text.split(" = ", 1)[1]
    tokens = text.split()
    while tokens and tokens[0] in {"musttail", "notail", "tail"}:
        tokens.pop(0)
    return tokens[0] if tokens else ""


def _llvm_instruction_locations(ll_path: Path) -> dict[str, list[tuple[str, int | None]]]:
    """Return ordered ``(opcode, dbg_id)`` instructions for each definition."""

    functions: dict[str, list[tuple[str, int | None]]] = defaultdict(list)
    current_name: str | None = None
    current_lines: list[str] = []

    def flush() -> None:
        if current_name is None or not current_lines:
            return
        text = "\n".join(current_lines)
        dbg_ids = _DBG_RE.findall(text)
        functions[current_name].append(
            (_llvm_opcode(current_lines[0]), int(dbg_ids[-1]) if dbg_ids else None)
        )

    with ll_path.open(errors="replace") as handle:
        for line in handle:
            if current_name is None:
                match = _FUNCTION_RE.match(line)
                if match:
                    name = match.group(1)
                    current_name = name[1:-1] if name.startswith('"') else name
                continue
            if line.startswith("}"):
                flush()
                current_name = None
                current_lines = []
                continue
            # LLVM instruction starts have exactly two spaces. Labels and
            # multiline continuations use different indentation.
            if re.match(r"^  \S", line):
                flush()
                current_lines = [line.rstrip()]
            elif current_lines:
                current_lines.append(line.rstrip())
    return functions


def _remove_previous_injection(graph: dict) -> None:
    old_nodes = graph.get("nodes") or []
    removed = {
        int(node["id"])
        for node in old_nodes
        if int(node.get("type", -1)) == NODE_PRAGMA
        and node.get("features", {}).get("injector") == ["ll_hls4ml.pragmas"]
    }
    if not removed:
        return

    kept_nodes = [node for node in old_nodes if int(node["id"]) not in removed]
    id_map = {int(node["id"]): new_id for new_id, node in enumerate(kept_nodes)}
    for new_id, node in enumerate(kept_nodes):
        node["id"] = new_id

    kept_links = []
    for link in graph.get("links") or []:
        source = int(link["source"])
        target = int(link["target"])
        if source in removed or target in removed:
            continue
        link["source"] = id_map[source]
        link["target"] = id_map[target]
        kept_links.append(link)

    graph["nodes"] = kept_nodes
    graph["links"] = kept_links


def _node_locations(
    graph: dict,
    debug: DebugInfo,
    ll_path: Path,
) -> tuple[dict[int, SourceLocation], int, int]:
    result = {}
    for node in graph.get("nodes") or []:
        if int(node.get("type", -1)) != NODE_INSTRUCTION:
            continue
        dbg_ids = _DBG_RE.findall(_full_text(node))
        if dbg_ids and (location := debug.source_location(int(dbg_ids[-1]))):
            result[int(node["id"])] = location

    direct_count = len(result)
    if direct_count:
        return result, direct_count, 0

    llvm_functions = _llvm_instruction_locations(ll_path)
    graph_functions = graph.get("graph", {}).get("function", [])
    nodes_by_function: dict[int, list[dict]] = defaultdict(list)
    for node in graph.get("nodes") or []:
        if (
            int(node.get("type", -1)) == NODE_INSTRUCTION
            and node.get("text") not in {"[external]", "; undefined function"}
        ):
            nodes_by_function[int(node.get("function", 0))].append(node)

    aligned = 0
    for function, graph_nodes in nodes_by_function.items():
        if function >= len(graph_functions):
            continue
        name = graph_functions[function].get("name", "")
        llvm_instructions = llvm_functions.get(name)
        if not llvm_instructions:
            continue
        graph_nodes.sort(key=lambda node: int(node["id"]))
        graph_opcodes = [str(node.get("text", "")) for node in graph_nodes]
        llvm_opcodes = [opcode for opcode, _dbg_id in llvm_instructions]
        matcher = SequenceMatcher(a=llvm_opcodes, b=graph_opcodes, autojunk=False)
        for block in matcher.get_matching_blocks():
            for offset in range(block.size):
                _opcode, dbg_id = llvm_instructions[block.a + offset]
                node = graph_nodes[block.b + offset]
                if dbg_id is not None and (location := debug.source_location(dbg_id)):
                    result[int(node["id"])] = location
                    aligned += 1
    return result, direct_count, aligned


def _function_candidates(
    pragma: Pragma,
    nodes: list[dict],
    locations: dict[int, SourceLocation],
) -> list[int]:
    lines_by_function: dict[int, list[int]] = defaultdict(list)
    for node in nodes:
        location = locations.get(int(node["id"]))
        if location and location.path == pragma.path:
            lines_by_function[int(node.get("function", 0))].append(location.line)

    candidates = []
    for function, lines in lines_by_function.items():
        # A small margin covers directives placed before the first executable
        # statement or immediately after a loop header.
        if min(lines) - 8 <= pragma.line <= max(lines) + 3:
            candidates.append(function)
    return sorted(candidates)


def _entry_anchor(nodes: list[dict], function: int) -> dict | None:
    candidates = [
        node
        for node in nodes
        if int(node.get("type", -1)) == NODE_INSTRUCTION
        and int(node.get("function", 0)) == function
        and node.get("text") != "[external]"
    ]
    return min(candidates, key=lambda node: int(node["id"]), default=None)


def _nearest_anchor(
    pragma: Pragma,
    nodes: list[dict],
    locations: dict[int, SourceLocation],
    function: int,
) -> dict | None:
    candidates = []
    for node in nodes:
        node_id = int(node["id"])
        location = locations.get(node_id)
        if (
            int(node.get("type", -1)) == NODE_INSTRUCTION
            and int(node.get("function", 0)) == function
            and location is not None
            and location.path == pragma.path
        ):
            candidates.append(node)

    def score(node: dict) -> tuple[int, int, int, int]:
        location = locations[int(node["id"])]
        return (
            abs(location.line - pragma.line),
            0 if location.line <= pragma.line else 1,
            0 if node.get("text") in {"br", "icmp", "phi"} else 1,
            int(node["id"]),
        )

    return min(candidates, key=score, default=None)


def _variable_anchor(
    pragma: Pragma,
    nodes: list[dict],
    debug: DebugInfo,
    locations: dict[int, SourceLocation],
    function: int,
) -> dict | None:
    if pragma.target_name is None:
        return None
    variable_ids = {
        meta_id
        for meta_id, (name, _scope, _line) in debug.variables.items()
        if name == pragma.target_name
    }
    if not variable_ids:
        return None

    candidates = []
    for node in nodes:
        if (
            int(node.get("type", -1)) != NODE_INSTRUCTION
            or int(node.get("function", 0)) != function
        ):
            continue
        text = _full_text(node)
        if "llvm.dbg.declare" in text and any(
            re.search(rf"metadata !{meta_id}(?:\D|$)", text)
            for meta_id in variable_ids
        ):
            candidates.append(node)
    direct = min(candidates, key=lambda node: int(node["id"]), default=None)
    if direct is not None:
        return direct

    # Some ProGraML backends remove llvm.dbg.declare instructions entirely.
    # Anchor those variables to the closest surviving instruction to their
    # declaration line in the same specialized function.
    variable_lines = []
    for meta_id in variable_ids:
        _name, scope, line = debug.variables[meta_id]
        if scope is not None and line is not None and debug.scope_file(scope) == pragma.path:
            variable_lines.append(line)
    if not variable_lines:
        return None
    declaration = Pragma(
        path=pragma.path,
        line=min(variable_lines, key=lambda line: abs(line - pragma.line)),
        directive=pragma.directive,
        clause=pragma.clause,
        raw=pragma.raw,
        target_name=pragma.target_name,
    )
    return _nearest_anchor(declaration, nodes, locations, function) or _entry_anchor(
        nodes, function
    )


def inject_pragmas(
    project_path: str | Path,
    ll_path: str | Path,
    json_path: str | Path,
) -> dict[str, int]:
    """Add supported HLS pragma nodes and application edges to ``json_path``.

    The JSON file is updated in place. Each generated pragma node has type 3 and
    one flow-3 edge directed from the pragma to an instruction anchor.
    """

    project_path = Path(project_path).resolve()
    ll_path = Path(ll_path).resolve()
    json_path = Path(json_path).resolve()
    if not project_path.is_dir():
        raise FileNotFoundError(f"Project directory does not exist: {project_path}")

    debug = DebugInfo(ll_path, project_path)
    pragmas, unsupported = _read_pragmas(debug.project_source_files())
    with json_path.open() as handle:
        graph = json.load(handle)

    _remove_previous_injection(graph)
    nodes = graph.get("nodes") or []
    links = graph.get("links") or []
    locations, direct_locations, aligned_locations = _node_locations(
        graph, debug, ll_path
    )

    injected = 0
    unmatched = 0
    seen = set()
    for pragma in pragmas:
        functions = _function_candidates(pragma, nodes, locations)
        matched = False
        for function in functions:
            if pragma.directive in _FUNCTION_DIRECTIVES:
                target_kind = "function"
                anchor = _entry_anchor(nodes, function)
            elif pragma.directive in _LOOP_DIRECTIVES:
                target_kind = "loop"
                anchor = _nearest_anchor(pragma, nodes, locations, function)
            else:
                target_kind = "variable"
                anchor = _variable_anchor(
                    pragma, nodes, debug, locations, function
                )

            if anchor is None:
                continue
            key = (
                str(pragma.path),
                pragma.line,
                pragma.clause.lower(),
                function,
                int(anchor["id"]),
            )
            if key in seen:
                continue
            seen.add(key)

            node_id = len(nodes)
            try:
                source_file = str(pragma.path.relative_to(project_path))
            except ValueError:
                source_file = str(pragma.path)
            node = {
                "block": int(anchor.get("block", 0)),
                "features": {
                    "full_text": [pragma.raw],
                    "injector": ["ll_hls4ml.pragmas"],
                    "source_file": [source_file],
                    "source_line": [str(pragma.line)],
                    "target_kind": [target_kind],
                    "target_name": [pragma.target_name or ""],
                },
                "function": function,
                "id": node_id,
                "text": f"pragma.{pragma.directive}",
                "type": NODE_PRAGMA,
            }
            nodes.append(node)
            links.append(
                {
                    "flow": FLOW_PRAGMA,
                    "key": 0,
                    "position": 0,
                    "source": node_id,
                    "target": int(anchor["id"]),
                }
            )
            injected += 1
            matched = True
        if not matched:
            unmatched += 1

    graph["nodes"] = nodes
    graph["links"] = links
    with json_path.open("w") as handle:
        json.dump(graph, handle, separators=(",", ":"))

    return {
        "source_files": len(debug.project_source_files()),
        "direct_debug_locations": direct_locations,
        "aligned_debug_locations": aligned_locations,
        "pragmas_found": len(pragmas),
        "pragma_nodes_injected": injected,
        "pragmas_unmatched": unmatched,
        "unsupported_pragmas": unsupported,
    }
