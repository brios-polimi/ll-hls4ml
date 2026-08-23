"""Bounded, schema-driven exploration of very large CDFG JSON graphs."""

from __future__ import annotations

from collections import Counter, defaultdict, deque
from dataclasses import dataclass
import json
from pathlib import Path
import shutil
import subprocess
from typing import Iterable

from ll_hls4ml.io import schema

try:
    import orjson
except ImportError:
    orjson = None


@dataclass(frozen=True)
class ProjectedEdge:
    source: int
    target: int
    relations: tuple[str, ...]
    count: int


@dataclass(frozen=True)
class GraphSlice:
    node_ids: tuple[int, ...]
    edges: tuple[ProjectedEdge, ...]
    truncated: bool
    candidate_nodes: int
    center: int | None
    level: str


def _demangle(symbols: list[str]) -> list[str]:
    binary = shutil.which("c++filt")
    if not binary or not symbols:
        return symbols
    try:
        result = subprocess.run(
            [binary, *symbols],
            capture_output=True,
            check=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return symbols
    values = result.stdout.splitlines()
    return values if len(values) == len(symbols) else symbols


class GraphExplorer:
    """Index one JSON graph and cheaply extract bounded hierarchy/radius slices."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        if orjson is not None:
            self.graph = orjson.loads(self.path.read_bytes())
        else:
            with self.path.open() as handle:
                self.graph = json.load(handle)

        self.nodes = {int(node["id"]): node for node in self.graph.get("nodes", [])}
        raw_links = tuple(self.graph.get("links", []))
        if set(self.nodes) != set(range(len(self.nodes))):
            raise ValueError("Graph node IDs must be contiguous from zero")

        self.node_type_names = dict(schema.NODE_TYPE_NAMES)
        self.node_type_ids = {name: value for value, name in self.node_type_names.items()}
        self.level_keys = dict(schema.HIERARCHY_LEVEL_KEYS)
        raw_function_names = [
            str(record.get("name", f"function_{index}"))
            for index, record in enumerate(self.graph.get("graph", {}).get("function", []))
        ]
        self.function_symbols = dict(enumerate(raw_function_names))
        self.function_names = dict(enumerate(_demangle(raw_function_names)))
        self._projection_lookups = self._build_projection_lookups()
        self._hierarchy_levels = tuple(self.level_keys)
        self.links = self._normalize_links(raw_links)
        self.relation_names = tuple(
            sorted({str(link["relation"]) for link in self.links})
        )
        self._owner_candidates = self._build_owner_candidates()

    def _normalize_links(self, links: tuple[dict, ...]) -> tuple[dict, ...]:
        """Validate and deduplicate canonical relation links."""

        normalized: list[dict] = []
        seen: set[tuple[int, str, int, int]] = set()
        for link in links:
            source = int(link["source"])
            target = int(link["target"])
            relation = link.get("relation")
            if relation is None:
                raise ValueError("Graph links must use canonical relation names")
            edge_type = (
                self.node_type_names.get(int(self.nodes[source].get("type", -1))),
                str(relation),
                self.node_type_names.get(int(self.nodes[target].get("type", -1))),
            )
            if edge_type not in schema.EDGE_TYPE_SET:
                raise ValueError(f"Edge is outside the canonical schema: {edge_type}")
            position = int(link.get("position", 0))
            key = (source, str(relation), target, position)
            if key in seen:
                continue
            seen.add(key)
            normalized.append(
                {
                    "source": source,
                    "target": target,
                    "relation": str(relation),
                    "position": position,
                }
            )
        return tuple(normalized)

    def _build_projection_lookups(self) -> dict[str, dict[tuple[int, ...], int]]:
        lookups: dict[str, dict[tuple[int, ...], int]] = {}
        for level, keys in self.level_keys.items():
            type_id = self.node_type_ids.get(level)
            lookups[level] = {
                tuple(int(node.get(key, -1)) for key in keys): node_id
                for node_id, node in self.nodes.items()
                if int(node.get("type", -1)) == type_id
            }
        return lookups

    def _build_owner_candidates(self) -> dict[str, dict[int, set[int]]]:
        """Infer hierarchy owners from instruction adjacency, not data-node metadata."""

        owners = {
            level: defaultdict(set) for level in self._hierarchy_levels
        }
        instruction_type = self.node_type_ids.get("instruction")
        for node_id, node in self.nodes.items():
            type_name = self.node_type_names.get(int(node.get("type", -1)))
            for level_index, level in enumerate(self._hierarchy_levels):
                if type_name in self._hierarchy_levels:
                    node_level = self._hierarchy_levels.index(type_name)
                    if node_level <= level_index:
                        owners[level][node_id].add(node_id)
                        continue
                    keys = self.level_keys[level]
                    key = tuple(int(node.get(field, -1)) for field in keys)
                    owner = self._projection_lookups[level].get(key)
                    if owner is not None:
                        owners[level][node_id].add(owner)
                    continue
                if int(node.get("type", -1)) != instruction_type:
                    continue
                keys = self.level_keys[level]
                key = tuple(int(node.get(field, -1)) for field in keys)
                owner = self._projection_lookups[level].get(key)
                if owner is not None:
                    owners[level][node_id].add(owner)

        # Data nodes inherit possible owners only from actual data-adjacent
        # instructions. Control/call adjacency must never imply ownership.
        for edge in self.links:
            if edge["relation"] not in {"defines", "operand"}:
                continue
            source = int(edge["source"])
            target = int(edge["target"])
            source_is_instruction = int(self.nodes[source].get("type", -1)) == instruction_type
            target_is_instruction = int(self.nodes[target].get("type", -1)) == instruction_type
            for level in self._hierarchy_levels:
                if source_is_instruction and not target_is_instruction:
                    owners[level][target].update(owners[level].get(source, ()))
                if target_is_instruction and not source_is_instruction:
                    owners[level][source].update(owners[level].get(target, ()))
        # Pragma ownership follows its explicit target after data ownership has
        # been established, independent of input edge ordering.
        for edge in self.links:
            if edge["relation"] != "applies_to":
                continue
            source = int(edge["source"])
            target = int(edge["target"])
            for level in self._hierarchy_levels:
                owners[level][source].update(owners[level].get(target, ()))
        return owners

    def visible_node_types(self, level: str) -> tuple[str, ...]:
        if level == "node":
            return tuple(self.node_type_ids)
        if level not in self._hierarchy_levels:
            return ()
        index = self._hierarchy_levels.index(level)
        return (*self._hierarchy_levels[: index + 1], "pragma")

    def selectable_node_types(self, level: str) -> tuple[str, ...]:
        """Return only types which can produce a node in this view."""

        present = {
            self.node_type_names.get(int(self.nodes[node_id].get("type", -1)))
            for node_id in self._level_nodes(level)
        }
        return tuple(
            name for name in self.visible_node_types(level) if name in present
        )

    @property
    def available_levels(self) -> tuple[str, ...]:
        return (
            "node",
            *(
                level
                for level in reversed(self._hierarchy_levels)
                if self._projection_lookups.get(level)
            ),
        )

    def available_relations(self, level: str) -> tuple[str, ...]:
        if level == "node":
            return self.relation_names
        level_nodes = set(self._level_nodes(level))
        available: list[str] = []
        for relation in self.relation_names:
            if relation in {"defines", "operand"}:
                continue
            visible = False
            for edge in self.links:
                if edge["relation"] != relation:
                    continue
                source = int(edge["source"])
                if relation == "control" and self.node_type_names.get(
                    int(self.nodes[source].get("type", -1))
                ) == "instruction":
                    continue
                sources = self.project_all(source, level) & level_nodes
                targets = self.project_all(int(edge["target"]), level) & level_nodes
                if any(source_id != target_id for source_id in sources for target_id in targets):
                    visible = True
                    break
            if visible:
                available.append(relation)
        if self._has_cross_owner_data(level):
            available.append("data_dependency")
        return tuple(available)

    def default_relations(self, level: str) -> tuple[str, ...]:
        if level == "function":
            preferred = ("calls", "applies_to")
        elif level == "block":
            preferred = ("control", "contains", "applies_to")
        else:
            return self.available_relations(level)
        return tuple(
            relation for relation in preferred if relation in self.available_relations(level)
        )

    def _has_cross_owner_data(self, level: str) -> bool:
        producers, consumers = self._data_paths()
        for data_node, source_instructions in producers.items():
            for source_instruction in source_instructions:
                sources = self.project_all(source_instruction, level)
                for target_instruction in consumers.get(data_node, ()):
                    targets = self.project_all(target_instruction, level)
                    if any(source != target for source in sources for target in targets):
                        return True
        return False

    @property
    def summary(self) -> dict:
        return {
            "path": str(self.path),
            "nodes": len(self.nodes),
            "edges": len(self.links),
            "node_types": dict(
                Counter(self.node_type_names.get(int(n.get("type", -1)), str(n.get("type"))) for n in self.nodes.values())
            ),
            "relations": dict(
                Counter(str(edge["relation"]) for edge in self.links)
            ),
            "hierarchy_enrichment": self.graph.get("hierarchy_enrichment"),
            "intrinsic_pruning": self.graph.get("intrinsic_pruning"),
        }

    def project(self, node_id: int, level: str) -> int | None:
        return min(self.project_all(node_id, level), default=None)

    def project_all(self, node_id: int, level: str) -> set[int]:
        if level == "node":
            return {node_id} if node_id in self.nodes else set()
        node = self.nodes.get(node_id)
        if level not in self.level_keys or node is None:
            return set()
        if self.node_type_names.get(int(node.get("type", -1))) == "pragma":
            return {node_id}
        return set(self._owner_candidates[level].get(node_id, ()))

    def label(self, node_id: int) -> str:
        node = self.nodes[node_id]
        type_name = self.node_type_names.get(int(node.get("type", -1)), str(node.get("type")))
        features = node.get("features", {})
        names = features.get("name", [])
        name = str(names[0]) if isinstance(names, list) and names else str(node.get("text", ""))
        function_id = int(node.get("function", -1))
        function_name = self.function_names.get(function_id, "")
        if type_name == "function" and function_name:
            name = function_name
        return f"{node_id} | {type_name} | {name}"

    def details(self, node_id: int) -> dict:
        node = dict(self.nodes[node_id])
        node["type_name"] = self.node_type_names.get(int(node.get("type", -1)))
        node["function_name"] = self.function_names.get(int(node.get("function", -1)))
        node["function_symbol"] = self.function_symbols.get(int(node.get("function", -1)))
        return node

    def search(self, query: str, level: str = "node", limit: int = 50) -> list[tuple[str, int]]:
        query = query.strip().lower()
        candidates = self._level_nodes(level)
        if not query:
            candidates = candidates[:limit]
        else:
            candidates = [
                node_id
                for node_id in candidates
                if query
                in "\n".join(
                    (
                        self.label(node_id),
                        self.function_symbols.get(
                            int(self.nodes[node_id].get("function", -1)), ""
                        ),
                        _full_text(self.nodes[node_id]),
                    )
                ).lower()
            ][:limit]
        return [(self.label(node_id), node_id) for node_id in candidates]

    def _level_nodes(self, level: str) -> list[int]:
        if level == "node":
            return list(self.nodes)
        hierarchy_nodes = [
            node_id
            for visible_level in self.visible_node_types(level)
            if visible_level != "pragma"
            for node_id in self._projection_lookups.get(visible_level, {}).values()
        ]
        pragma_type = self.node_type_ids.get("pragma")
        return hierarchy_nodes + [
            node_id
            for node_id, node in self.nodes.items()
            if int(node.get("type", -1)) == pragma_type
        ]

    def _data_paths(self) -> tuple[dict[int, list[int]], dict[int, list[int]]]:
        instruction_type = self.node_type_ids.get("instruction")
        producers: dict[int, list[int]] = defaultdict(list)
        consumers: dict[int, list[int]] = defaultdict(list)
        for edge in self.links:
            source = int(edge["source"])
            target = int(edge["target"])
            if edge["relation"] == "defines":
                producers[target].append(source)
            elif edge["relation"] == "operand":
                if int(self.nodes[target].get("type", -1)) == instruction_type:
                    consumers[source].append(target)
        return producers, consumers

    def _add_projected_data_dependencies(
        self,
        projected_edges: dict[tuple[int, int], list[str]],
        level: str,
        candidates: set[int],
        include_internal: bool,
    ) -> None:
        """Collapse instruction→variable→instruction paths into owner dependencies."""

        producers, consumers = self._data_paths()

        for data_node, source_instructions in producers.items():
            for source_instruction in source_instructions:
                sources = self.project_all(source_instruction, level)
                for target_instruction in consumers.get(data_node, ()):
                    targets = self.project_all(target_instruction, level)
                    for source in sources:
                        for target in targets:
                            if source not in candidates or target not in candidates:
                                continue
                            if source == target and not include_internal:
                                continue
                            projected_edges[(source, target)].append("data_dependency")

    def slice(
        self,
        *,
        level: str = "node",
        center: int | None = None,
        radius: int = 2,
        max_nodes: int = 200,
        node_types: Iterable[str] | None = None,
        edge_relations: Iterable[str] | None = None,
        direction: str = "both",
        include_internal: bool = False,
    ) -> GraphSlice:
        allowed_type_ids = {
            self.node_type_ids[name] for name in (node_types or self.node_type_ids) if name in self.node_type_ids
        }
        allowed_relations = set(edge_relations or self.available_relations(level))

        projected_edges: dict[tuple[int, int], list[str]] = defaultdict(list)
        candidates = {
            node_id
            for node_id in self._level_nodes(level)
            if int(self.nodes[node_id].get("type", -1)) in allowed_type_ids
        }
        for edge in self.links:
            relation = str(edge["relation"])
            if relation not in allowed_relations:
                continue
            if level != "node" and relation in {"defines", "operand"}:
                continue
            # The explicit block CFG already represents collapsed instruction
            # control; aggregating both creates duplicate dense edges.
            if level != "node" and relation == "control":
                source_type = self.node_type_names.get(
                    int(self.nodes[int(edge["source"])].get("type", -1))
                )
                if source_type == "instruction":
                    continue
            sources = self.project_all(int(edge["source"]), level)
            targets = self.project_all(int(edge["target"]), level)
            for source in sources:
                for target in targets:
                    if source not in candidates or target not in candidates:
                        continue
                    if source == target and not include_internal:
                        continue
                    projected_edges[(source, target)].append(relation)
        if level != "node" and "data_dependency" in allowed_relations:
            self._add_projected_data_dependencies(
                projected_edges,
                level,
                candidates,
                include_internal,
            )

        adjacency: dict[int, set[int]] = defaultdict(set)
        for source, target in projected_edges:
            if direction in ("both", "out"):
                adjacency[source].add(target)
            if direction in ("both", "in"):
                adjacency[target].add(source)

        projected_center = self.project(center, level) if center is not None else None
        if projected_center not in candidates:
            projected_center = None
        if projected_center is None:
            ranked = sorted(candidates, key=lambda n: (-len(adjacency[n]), n))
            selected = set(ranked[:max_nodes])
            truncated = len(candidates) > max_nodes
        else:
            selected = {projected_center}
            queue = deque([(projected_center, 0)])
            truncated = False
            constant_type = self.node_type_ids.get("constant")
            while queue:
                current, depth = queue.popleft()
                if depth >= radius:
                    continue
                # Shared LLVM constants are often graph-wide hubs. Show the
                # constant, but do not let an incidental zero/one operand turn
                # a local radius query into an arbitrary sample of the graph.
                if (
                    current != projected_center
                    and int(self.nodes[current].get("type", -1)) == constant_type
                    and len(adjacency[current]) > 24
                ):
                    continue
                for neighbor in sorted(adjacency[current]):
                    if neighbor in selected:
                        continue
                    if len(selected) >= max_nodes:
                        truncated = True
                        queue.clear()
                        break
                    selected.add(neighbor)
                    queue.append((neighbor, depth + 1))

        edges = tuple(
            ProjectedEdge(source, target, tuple(sorted(set(relations))), len(relations))
            for (source, target), relations in projected_edges.items()
            if source in selected and target in selected
        )
        return GraphSlice(
            tuple(sorted(selected)),
            edges,
            truncated,
            len(candidates),
            projected_center,
            level,
        )


def _full_text(node: dict) -> str:
    values = node.get("features", {}).get("full_text", [])
    return "\n".join(map(str, values)) if isinstance(values, list) else str(values)


def _ranked_positions(
    node_ids: Iterable[int],
    edges: Iterable[tuple[int, int]],
) -> dict[int, tuple[float, float]]:
    """Return stable left-to-right ranks, condensing loop SCCs first."""

    import networkx as nx

    node_ids = tuple(sorted(set(node_ids)))
    graph = nx.DiGraph()
    graph.add_nodes_from(node_ids)
    graph.add_edges_from(
        (source, target)
        for source, target in edges
        if source in graph and target in graph and source != target
    )
    if not graph:
        return {}
    condensed = nx.condensation(graph)
    rank: dict[int, int] = {}
    for component in nx.topological_sort(condensed):
        predecessors = tuple(condensed.predecessors(component))
        rank[component] = 1 + max((rank[parent] for parent in predecessors), default=-1)
    layers: dict[int, list[int]] = defaultdict(list)
    for component, depth in rank.items():
        layers[depth].extend(sorted(condensed.nodes[component]["members"]))
    positions: dict[int, tuple[float, float]] = {}
    for depth, layer in sorted(layers.items()):
        for row, node_id in enumerate(layer):
            positions[node_id] = (float(depth), row - (len(layer) - 1) / 2)
    return positions


def _node_type(explorer: GraphExplorer, node_id: int) -> str:
    return explorer.node_type_names.get(
        int(explorer.nodes[node_id].get("type", -1)), "unknown"
    )


def _compact_label(explorer: GraphExplorer, node_id: int, limit: int = 26) -> str:
    node = explorer.nodes[node_id]
    type_name = _node_type(explorer, node_id)
    features = node.get("features", {})
    names = features.get("name", [])
    name = str(names[0]) if isinstance(names, list) and names else str(node.get("text", ""))
    if type_name == "function":
        name = explorer.function_names.get(int(node.get("function", -1)), name)
    name = " ".join(name.split()) or type_name
    if type_name in {"variable", "constant"}:
        limit = min(limit, 12)
    if len(name) > limit:
        name = name[: limit - 1] + "…"
    prefixes = {
        "instruction": "i",
        "variable": "v",
        "constant": "c",
        "pragma": "p",
        "block": "bb",
        "function": "fn",
    }
    return f"{prefixes.get(type_name, type_name)}{node_id}  {name}"


def _function_positions(
    explorer: GraphExplorer,
    graph_slice: GraphSlice,
    function_nodes: list[int],
) -> dict[int, tuple[float, float]]:
    call_edges = [
        (edge.source, edge.target)
        for edge in graph_slice.edges
        if "calls" in edge.relations
        and edge.source in function_nodes
        and edge.target in function_nodes
    ]
    ranked = _ranked_positions(function_nodes, call_edges)
    return {
        node_id: (position[1] * 5.2, -position[0] * 4.0)
        for node_id, position in ranked.items()
    }


def _block_positions(
    explorer: GraphExplorer,
    graph_slice: GraphSlice,
    block_nodes: list[int],
    function_nodes: list[int],
) -> dict[int, tuple[float, float]]:
    """Lay out each function CFG separately, with the function as its header."""

    function_of = {
        node_id: int(explorer.nodes[node_id].get("function", -1))
        for node_id in (*block_nodes, *function_nodes)
    }
    blocks_by_function: dict[int, list[int]] = defaultdict(list)
    for node_id in block_nodes:
        blocks_by_function[function_of[node_id]].append(node_id)
    cfg_edges = [
        (edge.source, edge.target)
        for edge in graph_slice.edges
        if "control" in edge.relations
        and edge.source in block_nodes
        and edge.target in block_nodes
    ]

    function_ids = sorted(
        set(blocks_by_function) | {function_of[node_id] for node_id in function_nodes}
    )
    positions: dict[int, tuple[float, float]] = {}
    x_cursor = 0.0
    for function_id in function_ids:
        owned = blocks_by_function.get(function_id, [])
        local = _ranked_positions(owned, cfg_edges)
        layer_heights = Counter(int(x) for x, _ in local.values())
        width = max((x for x, _ in local.values()), default=0.0) * 5.0 + 5.0
        height = max(layer_heights.values(), default=1) * 2.4
        for node_id, (depth, row) in local.items():
            positions[node_id] = (x_cursor + depth * 5.0, -row * 2.4)
        function_node = next(
            (node_id for node_id in function_nodes if function_of[node_id] == function_id),
            None,
        )
        if function_node is not None:
            positions[function_node] = (x_cursor + max(width - 5.0, 0.0) / 2, height / 2 + 2.2)
        x_cursor += max(width, 6.0) + 4.0
    return positions


def _node_flow_positions(
    explorer: GraphExplorer,
    graph_slice: GraphSlice,
    node_ids: list[int],
) -> dict[int, tuple[float, float]]:
    """Place SSA producers/values/consumers in block-sized flow regions."""

    selected = set(node_ids)
    structure = {"block", "function"}
    block_groups: dict[int | None, list[int]] = defaultdict(list)
    for node_id in node_ids:
        if _node_type(explorer, node_id) in structure:
            continue
        owners = {
            owner
            for owner in explorer._owner_candidates["block"].get(node_id, set())
            if _node_type(explorer, owner) == "block"
        }
        block_groups[min(owners, default=None)].append(node_id)

    # Derive the block CFG from explicit block nodes even when the block toggle
    # is off. Filters therefore change content, not the spatial grammar.
    relevant_blocks = {owner for owner in block_groups if owner is not None}
    cfg_edges = [
        (int(edge["source"]), int(edge["target"]))
        for edge in explorer.links
        if edge["relation"] == "control"
        and int(edge["source"]) in relevant_blocks
        and int(edge["target"]) in relevant_blocks
        and _node_type(explorer, int(edge["source"])) == "block"
    ]
    block_rank = _ranked_positions(relevant_blocks, cfg_edges)

    local_positions: dict[int | None, dict[int, tuple[float, float]]] = {}
    local_sizes: dict[int | None, tuple[float, float]] = {}
    for owner, members in block_groups.items():
        instructions = [node_id for node_id in members if _node_type(explorer, node_id) == "instruction"]
        variables = [node_id for node_id in members if _node_type(explorer, node_id) == "variable"]
        constants = [node_id for node_id in members if _node_type(explorer, node_id) == "constant"]
        pragmas = [node_id for node_id in members if _node_type(explorer, node_id) == "pragma"]
        dependency_edges: list[tuple[int, int]] = []
        producers: dict[int, list[int]] = defaultdict(list)
        consumers: dict[int, list[int]] = defaultdict(list)
        pragma_targets: dict[int, list[int]] = defaultdict(list)
        for edge in graph_slice.edges:
            if edge.source not in selected or edge.target not in selected:
                continue
            if "control" in edge.relations and edge.source in instructions and edge.target in instructions:
                dependency_edges.append((edge.source, edge.target))
            if "defines" in edge.relations:
                producers[edge.target].append(edge.source)
            if "operand" in edge.relations:
                consumers[edge.source].append(edge.target)
            if "applies_to" in edge.relations:
                pragma_targets[edge.source].append(edge.target)
        for variable in variables:
            dependency_edges.extend(
                (source, target)
                for source in producers.get(variable, ())
                for target in consumers.get(variable, ())
                if source in instructions and target in instructions
            )
        ranked = _ranked_positions(instructions, dependency_edges)
        local: dict[int, tuple[float, float]] = {
            node_id: (depth * 4.2, -row * 2.0)
            for node_id, (depth, row) in ranked.items()
        }
        for index, variable in enumerate(sorted(variables)):
            neighbors = producers.get(variable, []) + consumers.get(variable, [])
            points = [local[node_id] for node_id in neighbors if node_id in local]
            if points:
                x = sum(point[0] for point in points) / len(points)
                y = sum(point[1] for point in points) / len(points) - 1.05 - (index % 2) * 0.55
            else:
                x, y = (0.0, -index * 1.25)
            local[variable] = (x, y)
        instruction_xs = [local[node_id][0] for node_id in instructions if node_id in local]
        constant_x = min(instruction_xs, default=0.0) - 3.2
        used_y: Counter[int] = Counter()
        for constant in sorted(constants):
            target_ys = [local[node_id][1] for node_id in consumers.get(constant, ()) if node_id in local]
            y = sum(target_ys) / len(target_ys) if target_ys else -len(used_y) * 1.2
            bucket = round(y * 2)
            y -= used_y[bucket] * 0.95
            used_y[bucket] += 1
            local[constant] = (constant_x, y)
        for index, pragma in enumerate(sorted(pragmas)):
            targets = [local[node_id] for node_id in pragma_targets.get(pragma, ()) if node_id in local]
            if targets:
                x, y = targets[0][0], targets[0][1] + 1.15
            else:
                x, y = constant_x, 1.4 + index * 1.0
            local[pragma] = (x, y)
        missing = sorted(set(members) - set(local))
        for index, node_id in enumerate(missing):
            local[node_id] = (constant_x, -index * 1.2)
        xs = [point[0] for point in local.values()] or [0.0]
        ys = [point[1] for point in local.values()] or [0.0]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        local_positions[owner] = {
            node_id: (x - min_x, y - (min_y + max_y) / 2)
            for node_id, (x, y) in local.items()
        }
        local_sizes[owner] = (max_x - min_x + 4.0, max_y - min_y + 3.0)

    # Pack variable-sized block regions by CFG rank. This prevents one large
    # basic block from overlapping its siblings.
    owners_by_rank: dict[int, list[int | None]] = defaultdict(list)
    for owner in block_groups:
        rank = int(block_rank.get(owner, (max((p[0] for p in block_rank.values()), default=-1) + 1, 0))[0])
        owners_by_rank[rank].append(owner)
    rank_x: dict[int, float] = {}
    x_cursor = 0.0
    for rank in sorted(owners_by_rank):
        rank_x[rank] = x_cursor
        x_cursor += max(local_sizes[owner][0] for owner in owners_by_rank[rank]) + 4.0
    positions: dict[int, tuple[float, float]] = {}
    for rank, owners in sorted(owners_by_rank.items()):
        y_cursor = 0.0
        total_height = sum(local_sizes[owner][1] + 2.5 for owner in owners) - 2.5
        y_cursor = total_height / 2
        for owner in sorted(owners, key=lambda value: (-1 if value is None else value)):
            width, height = local_sizes[owner]
            center_y = y_cursor - height / 2
            for node_id, (x, y) in local_positions[owner].items():
                positions[node_id] = (rank_x[rank] + x, center_y + y)
            y_cursor -= height + 2.5

    # Structural nodes become headers for their actual regions rather than
    # competing with dataflow nodes for a force-layout position.
    for node_id in node_ids:
        type_name = _node_type(explorer, node_id)
        if type_name == "block":
            members = block_groups.get(node_id, [])
            points = [positions[member] for member in members if member in positions]
            if points:
                positions[node_id] = (
                    sum(x for x, _ in points) / len(points),
                    max(y for _, y in points) + 2.0,
                )
            else:
                positions[node_id] = (x_cursor, 0.0)
                x_cursor += 4.0
        elif type_name == "function":
            function_id = int(explorer.nodes[node_id].get("function", -1))
            owned = [
                member
                for member in node_ids
                if member in positions
                and int(explorer.nodes[member].get("function", -2)) == function_id
            ]
            if owned:
                positions[node_id] = (
                    sum(positions[member][0] for member in owned) / len(owned),
                    max(positions[member][1] for member in owned) + 3.0,
                )
            else:
                positions[node_id] = (x_cursor, 3.0)
                x_cursor += 4.0
    return positions


def hierarchy_layout(
    explorer: GraphExplorer,
    graph_slice: GraphSlice,
) -> dict[int, tuple[float, float]]:
    """Use a different spatial grammar for each LLVM hierarchy level."""

    selected = set(graph_slice.node_ids)
    function_type = explorer.node_type_ids.get("function")
    block_type = explorer.node_type_ids.get("block")
    function_nodes = [
        node_id
        for node_id in selected
        if int(explorer.nodes[node_id].get("type", -1)) == function_type
    ]
    block_nodes = [
        node_id
        for node_id in selected
        if int(explorer.nodes[node_id].get("type", -1)) == block_type
    ]

    if graph_slice.level == "function" and not block_nodes:
        positions = _function_positions(explorer, graph_slice, function_nodes)
    elif graph_slice.level == "block":
        positions = _block_positions(explorer, graph_slice, block_nodes, function_nodes)
    else:
        positions = _node_flow_positions(explorer, graph_slice, list(selected))
    # Keep pragma annotations beside the function/block they describe instead
    # of collecting them into a distant lane full of crossing edges.
    satellites: dict[int, list[int]] = defaultdict(list)
    for edge in graph_slice.edges:
        if (
            "applies_to" in edge.relations
            and edge.source not in positions
            and edge.target in positions
        ):
            satellites[edge.target].append(edge.source)
    for target, node_ids in satellites.items():
        target_x, target_y = positions[target]
        for index, node_id in enumerate(sorted(set(node_ids))):
            positions[node_id] = (
                target_x + 2.6,
                target_y + (index - (len(set(node_ids)) - 1) / 2) * 0.72,
            )
    # Aggressive relation filtering can leave unowned nodes; keep every
    # selected node renderable and draggable.
    next_x = max((x for x, _ in positions.values()), default=0.0) + 4.0
    for index, node_id in enumerate(sorted(selected - set(positions))):
        positions[node_id] = (next_x, -index * 1.5)
    return positions


def _hover_text(
    explorer: GraphExplorer,
    graph_slice: GraphSlice,
    node_id: int,
) -> str:
    node = explorer.nodes[node_id]
    type_name = _node_type(explorer, node_id)
    relations = Counter(
        relation
        for edge in graph_slice.edges
        if node_id in (edge.source, edge.target)
        for relation in edge.relations
    )
    feature_name = node.get("features", {}).get("name", [])
    if isinstance(feature_name, list):
        feature_name = feature_name[0] if feature_name else ""
    full_text = " ".join(_full_text(node).split())
    if len(full_text) > 220:
        full_text = full_text[:219] + "…"
    lines = [f"{type_name}  ·  node {node_id}"]
    if feature_name:
        lines.append(str(feature_name))
    function_name = explorer.function_names.get(int(node.get("function", -1)))
    if function_name:
        lines.append(f"function: {function_name}")
    if int(node.get("block", -1)) >= 0:
        lines.append(f"block: {node.get('block')}")
    if relations:
        lines.append("edges: " + ", ".join(f"{name} ×{count}" for name, count in sorted(relations.items())))
    if full_text and full_text != str(feature_name):
        lines.append(full_text)
    return "\n".join(lines)


class _GraphInteractor:
    """Small Matplotlib interaction layer: hover, inspect, and drag."""

    def __init__(
        self,
        figure,
        axes,
        explorer: GraphExplorer,
        graph_slice: GraphSlice,
        positions: dict[int, tuple[float, float]],
        node_artists: dict[int, object],
        label_artists: dict[int, object],
        edge_artists: list[tuple[ProjectedEdge, object]],
        annotation,
        on_select=None,
    ):
        self.figure = figure
        self.axes = axes
        self.explorer = explorer
        self.graph_slice = graph_slice
        self.positions = positions
        self.node_artists = node_artists
        self.label_artists = label_artists
        self.edge_artists = edge_artists
        self.annotation = annotation
        self.on_select = on_select
        self.dragging: int | None = None
        self.selected = graph_slice.center
        canvas = figure.canvas
        self.connections = (
            canvas.mpl_connect("motion_notify_event", self._motion),
            canvas.mpl_connect("button_press_event", self._press),
            canvas.mpl_connect("button_release_event", self._release),
            canvas.mpl_connect("figure_leave_event", self._leave),
        )
        if self.selected is not None:
            artist = self.node_artists[self.selected]
            artist.set_linewidths([2.4])
            artist.set_edgecolors(["#111827"])

    def _nearest(self, event) -> int | None:
        if event.inaxes is not self.axes or event.x is None or event.y is None:
            return None
        nearest = None
        distance = 22.0**2
        for node_id, point in self.positions.items():
            x, y = self.axes.transData.transform(point)
            candidate = (x - event.x) ** 2 + (y - event.y) ** 2
            if candidate < distance:
                nearest, distance = node_id, candidate
        return nearest

    def _toolbar_active(self) -> bool:
        toolbar = getattr(self.figure.canvas, "toolbar", None)
        return bool(toolbar and getattr(toolbar, "mode", ""))

    def _motion(self, event) -> None:
        if self.dragging is not None and event.inaxes is self.axes and event.xdata is not None:
            node_id = self.dragging
            self.positions[node_id] = (event.xdata, event.ydata)
            self.node_artists[node_id].set_offsets([[event.xdata, event.ydata]])
            self.label_artists[node_id].xy = (event.xdata, event.ydata)
            for edge, artist in self.edge_artists:
                if node_id in (edge.source, edge.target):
                    artist.set_positions(self.positions[edge.source], self.positions[edge.target])
            self.annotation.set_visible(False)
            self.figure.canvas.draw_idle()
            return
        node_id = self._nearest(event)
        if node_id is None:
            if self.annotation.get_visible():
                self.annotation.set_visible(False)
                self.figure.canvas.draw_idle()
            return
        self.annotation.xy = self.positions[node_id]
        self.annotation.set_text(_hover_text(self.explorer, self.graph_slice, node_id))
        self.annotation.set_visible(True)
        self.figure.canvas.draw_idle()

    def _press(self, event) -> None:
        if event.button != 1 or self._toolbar_active():
            return
        node_id = self._nearest(event)
        if node_id is None:
            return
        self.dragging = node_id
        self.selected = node_id
        self._highlight(node_id)
        if self.on_select is not None:
            self.on_select(node_id)

    def _release(self, event) -> None:
        self.dragging = None

    def _leave(self, event) -> None:
        if self.dragging is None and self.annotation.get_visible():
            self.annotation.set_visible(False)
            self.figure.canvas.draw_idle()

    def _highlight(self, node_id: int) -> None:
        neighbors = {
            endpoint
            for edge in self.graph_slice.edges
            if node_id in (edge.source, edge.target)
            for endpoint in (edge.source, edge.target)
        }
        neighbors.add(node_id)
        for candidate, artist in self.node_artists.items():
            artist.set_alpha(1.0 if candidate in neighbors else 0.22)
            artist.set_linewidths([2.4 if candidate == node_id else 0.9])
            artist.set_edgecolors(["#111827" if candidate == node_id else "#ffffff"])
        for edge, artist in self.edge_artists:
            connected = node_id in (edge.source, edge.target)
            artist.set_alpha(0.95 if connected else 0.08)
            artist.set_linewidth(2.0 if connected else 0.7)
        for candidate, artist in self.label_artists.items():
            artist.set_alpha(1.0 if candidate in neighbors else 0.16)
        self.figure.canvas.draw_idle()


def _draw_hierarchy_regions(axes, explorer, graph_slice, positions) -> None:
    from matplotlib.patches import FancyBboxPatch

    if graph_slice.level == "function":
        return
    groups: dict[tuple[str, int], list[int]] = defaultdict(list)
    if graph_slice.level == "block":
        for node_id in graph_slice.node_ids:
            function_id = int(explorer.nodes[node_id].get("function", -1))
            if function_id >= 0 and node_id in positions:
                groups[("function", function_id)].append(node_id)
    else:
        for node_id in graph_slice.node_ids:
            owners = {
                owner
                for owner in explorer._owner_candidates["block"].get(node_id, set())
                if _node_type(explorer, owner) == "block"
            }
            owner = min(owners, default=None)
            if owner is not None and node_id in positions:
                groups[("block", owner)].append(node_id)
    for (kind, owner), members in groups.items():
        points = [positions[node_id] for node_id in members]
        if not points:
            continue
        xs, ys = zip(*points)
        left, bottom = min(xs) - 1.25, min(ys) - 1.15
        width, height = max(xs) - min(xs) + 2.5, max(ys) - min(ys) + 2.3
        patch = FancyBboxPatch(
            (left, bottom), width, height,
            boxstyle="round,pad=0.2,rounding_size=0.35",
            facecolor="#f8fafc", edgecolor="#cbd5e1", linewidth=0.9,
            alpha=0.72, zorder=-3,
        )
        axes.add_patch(patch)
        if kind == "function":
            title = explorer.function_names.get(owner, f"function {owner}")
        else:
            title = f"basic block {explorer.nodes[owner].get('block', owner)}"
        if len(title) > 52:
            title = title[:51] + "…"
        axes.text(left + 0.35, bottom + height - 0.35, title, fontsize=7.5,
                  color="#64748b", va="top", zorder=-2)


def plot_slice(
    explorer: GraphExplorer,
    graph_slice: GraphSlice,
    *,
    figsize=(15, 9),
    seed: int = 7,
    layout: str = "hierarchy",
    on_select=None,
):
    """Render an interactive, draggable, LLVM-aware graph slice."""

    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import FancyArrowPatch
    import networkx as nx

    graph = nx.DiGraph()
    graph.add_nodes_from(graph_slice.node_ids)
    graph.add_edges_from((edge.source, edge.target) for edge in graph_slice.edges)
    if layout == "hierarchy":
        positions = hierarchy_layout(explorer, graph_slice)
    elif layout == "spring":
        raw = nx.spring_layout(graph, seed=seed, iterations=90, k=1.8 / max(len(graph), 1) ** 0.5) if graph else {}
        positions = {node_id: (float(point[0]) * 12, float(point[1]) * 9) for node_id, point in raw.items()}
    else:
        raise ValueError(f"Unknown layout: {layout}")

    type_colors = {
        "instruction": "#2563eb",
        "variable": "#10b981",
        "constant": "#f59e0b",
        "pragma": "#ec4899",
        "block": "#7c3aed",
        "function": "#0f766e",
        "unknown": "#64748b",
    }
    type_markers = {
        "instruction": "s", "variable": "o", "constant": "D",
        "pragma": "P", "block": "h", "function": "8", "unknown": "o",
    }
    relation_colors = {
        "control": "#64748b", "defines": "#059669", "operand": "#d97706",
        "calls": "#7c3aed", "contains": "#94a3b8", "applies_to": "#db2777",
        "data_dependency": "#0891b2",
    }

    fig, ax = plt.subplots(figsize=figsize)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.92, bottom=0.10)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    _draw_hierarchy_regions(ax, explorer, graph_slice, positions)

    reverse_edges = {(edge.target, edge.source) for edge in graph_slice.edges}
    edge_artists: list[tuple[ProjectedEdge, object]] = []
    node_degree = Counter(
        endpoint for edge in graph_slice.edges for endpoint in (edge.source, edge.target)
    )
    for index, edge in enumerate(graph_slice.edges):
        relation = edge.relations[0] if edge.relations else ""
        source_type = _node_type(explorer, edge.source)
        radius = 0.13 if (edge.source, edge.target) in reverse_edges else ((index % 3) - 1) * 0.045
        alpha = 0.2 if source_type == "constant" and node_degree[edge.source] > 8 else 0.58
        patch = FancyArrowPatch(
            positions[edge.source], positions[edge.target],
            arrowstyle="-|>", mutation_scale=9.0,
            connectionstyle=f"arc3,rad={radius}",
            color=relation_colors.get(relation, "#64748b"),
            linewidth=0.8 + min(1.8, 0.18 * (edge.count - 1)),
            alpha=alpha, shrinkA=10, shrinkB=10, zorder=0,
        )
        ax.add_patch(patch)
        edge_artists.append((edge, patch))

    node_artists: dict[int, object] = {}
    label_artists: dict[int, object] = {}
    for node_id in graph_slice.node_ids:
        type_name = _node_type(explorer, node_id)
        x, y = positions[node_id]
        size = 120 if type_name in {"variable", "constant"} else 175
        if type_name in {"block", "function"}:
            size = 235
        artist = ax.scatter(
            [x], [y], s=size, marker=type_markers.get(type_name, "o"),
            c=[type_colors.get(type_name, type_colors["unknown"])],
            edgecolors="white", linewidths=0.9, zorder=3,
        )
        node_artists[node_id] = artist
        label_artists[node_id] = ax.annotate(
            _compact_label(explorer, node_id), (x, y), xytext=(0, -11),
            textcoords="offset points", ha="center", va="top", fontsize=7.2,
            color="#172033", zorder=4,
            bbox={"boxstyle": "round,pad=0.12", "fc": "white", "ec": "none", "alpha": 0.82},
        )

    annotation = ax.annotate(
        "", xy=(0, 0), xytext=(16, 16), textcoords="offset points",
        ha="left", va="bottom", fontsize=8.5, color="#e2e8f0",
        bbox={"boxstyle": "round,pad=0.55", "fc": "#0f172a", "ec": "#334155", "alpha": 0.96},
        arrowprops={"arrowstyle": "->", "color": "#475569"}, zorder=10,
    )
    annotation.set_visible(False)

    visible_types = sorted({_node_type(explorer, node_id) for node_id in graph_slice.node_ids})
    visible_relations = sorted({relation for edge in graph_slice.edges for relation in edge.relations})
    handles = [
        Line2D([0], [0], marker=type_markers[name], color="none",
               markerfacecolor=type_colors[name], markeredgecolor="white",
               markersize=7, label=name)
        for name in visible_types
    ] + [
        Line2D([0], [0], color=relation_colors.get(name, "#64748b"),
               linewidth=2, label=name)
        for name in visible_relations
    ]
    if handles:
        ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.04),
                  ncol=min(7, len(handles)), fontsize=7.5, frameon=False)
    ax.set_title(
        f"{graph_slice.level} · {len(graph_slice.node_ids):,} shown / "
        f"{graph_slice.candidate_nodes:,} matching · hover for LLVM details · drag to untangle"
        + (" · node limit reached" if graph_slice.truncated else ""),
        fontsize=11, loc="left", pad=12,
    )
    ax.autoscale_view()
    ax.margins(0.12)
    ax.axis("off")
    if hasattr(fig.canvas, "toolbar_visible"):
        fig.canvas.toolbar_visible = True
        fig.canvas.header_visible = False
        fig.canvas.footer_visible = False
        fig.canvas.resizable = True
    interactor = _GraphInteractor(
        fig, ax, explorer, graph_slice, positions, node_artists,
        label_artists, edge_artists, annotation, on_select,
    )
    # Keep callbacks alive for as long as the displayed canvas exists.
    fig._graph_interactor = interactor
    return fig
