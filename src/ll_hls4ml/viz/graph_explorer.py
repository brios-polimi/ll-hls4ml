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
            while queue:
                current, depth = queue.popleft()
                if depth >= radius:
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


def _grid_positions(
    node_ids: list[int],
    center_x: float,
    top_y: float,
    *,
    columns: int = 7,
    x_step: float = 0.34,
    y_step: float = 0.24,
) -> dict[int, tuple[float, float]]:
    if not node_ids:
        return {}
    columns = min(columns, len(node_ids))
    return {
        node_id: (
            center_x + (index % columns - (columns - 1) / 2) * x_step,
            top_y - (index // columns) * y_step,
        )
        for index, node_id in enumerate(sorted(node_ids))
    }


def hierarchy_layout(
    explorer: GraphExplorer,
    graph_slice: GraphSlice,
) -> dict[int, tuple[float, float]]:
    """Lay out each function's CFG left-to-right and keep children nearby."""

    selected = set(graph_slice.node_ids)
    function_type = explorer.node_type_ids.get("function")
    block_type = explorer.node_type_ids.get("block")
    instruction_type = explorer.node_type_ids.get("instruction")
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

    function_of = {
        node_id: int(explorer.nodes[node_id].get("function", -1))
        for node_id in selected
    }
    blocks_by_function: dict[int, list[int]] = defaultdict(list)
    for block_node in block_nodes:
        blocks_by_function[function_of[block_node]].append(block_node)
    cfg: dict[int, set[int]] = defaultdict(set)
    indegree: Counter[int] = Counter()
    for edge in graph_slice.edges:
        if "control" not in edge.relations:
            continue
        if edge.source in block_nodes and edge.target in block_nodes:
            cfg[edge.source].add(edge.target)
            indegree[edge.target] += 1

    positions: dict[int, tuple[float, float]] = {}
    block_positions: dict[int, tuple[float, float]] = {}
    x_offset = 0.0
    for function_id, owned_blocks in sorted(blocks_by_function.items()):
        owned = set(owned_blocks)
        roots = sorted(owned - set(indegree)) or [min(owned)]
        rank: dict[int, int] = {}
        queue = deque((root, 0) for root in roots)
        while queue:
            node_id, depth = queue.popleft()
            if node_id in rank and rank[node_id] <= depth:
                continue
            rank[node_id] = depth
            queue.extend(
                (target, depth + 1)
                for target in sorted(cfg[node_id])
                if target in owned
            )
        next_rank = max(rank.values(), default=-1) + 1
        for node_id in sorted(owned - set(rank)):
            rank[node_id] = next_rank
            next_rank += 1
        by_rank: dict[int, list[int]] = defaultdict(list)
        for node_id, depth in rank.items():
            by_rank[depth].append(node_id)
        for depth, ranked_nodes in sorted(by_rank.items()):
            ranked_nodes.sort(key=lambda node_id: int(explorer.nodes[node_id].get("block", -1)))
            for index, node_id in enumerate(ranked_nodes):
                y = 2.8 + (index - (len(ranked_nodes) - 1) / 2) * 1.35
                block_positions[node_id] = (x_offset + depth * 3.2, y)
        function_node = next(
            (node_id for node_id in function_nodes if function_of[node_id] == function_id),
            None,
        )
        if function_node is not None:
            xs = [block_positions[node_id][0] for node_id in owned_blocks]
            ys = [block_positions[node_id][1] for node_id in owned_blocks]
            positions[function_node] = (sum(xs) / len(xs), max(ys) + 1.7)
        x_offset += (max(rank.values(), default=0) + 1) * 3.2 + 4.0
    positions.update(block_positions)

    # Function-only views have no blocks to anchor their function nodes.
    for function_node in sorted(function_nodes, key=lambda node_id: function_of[node_id]):
        if function_node not in positions:
            positions[function_node] = (x_offset, 4.5)
            x_offset += 4.0

    children_by_anchor: dict[int | None, list[int]] = defaultdict(list)
    for node_id in selected - set(function_nodes) - set(block_nodes):
        block_owner = min(
            explorer._owner_candidates["block"].get(node_id, ()) & set(block_nodes),
            default=None,
        )
        function_owner = min(
            explorer._owner_candidates["function"].get(node_id, ())
            & set(function_nodes),
            default=None,
        )
        anchor = block_owner if block_owner is not None else function_owner
        children_by_anchor[anchor].append(node_id)
    for owner, children in children_by_anchor.items():
        center_x, anchor_y = positions.get(owner, (x_offset, 3.0))
        instructions = [
            node_id
            for node_id in children
            if int(explorer.nodes[node_id].get("type", -1)) == instruction_type
        ]
        lower_nodes = [node_id for node_id in children if node_id not in instructions]
        positions.update(_grid_positions(instructions, center_x, anchor_y - 1.0))
        instruction_rows = (len(instructions) + 6) // 7
        lower_top = anchor_y - 1.65 - instruction_rows * 0.24
        positions.update(_grid_positions(lower_nodes, center_x, lower_top))
        if owner is None:
            x_offset += 3.0
    return positions


def plot_slice(
    explorer: GraphExplorer,
    graph_slice: GraphSlice,
    *,
    figsize=(14, 9),
    seed: int = 7,
    layout: str = "hierarchy",
):
    """Render a bounded slice; controls and repeated updates live in the notebook."""

    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    import networkx as nx

    graph = nx.DiGraph()
    graph.add_nodes_from(graph_slice.node_ids)
    graph.add_edges_from((edge.source, edge.target) for edge in graph_slice.edges)
    if layout == "hierarchy":
        positions = hierarchy_layout(explorer, graph_slice)
    elif layout == "spring":
        positions = nx.spring_layout(graph, seed=seed, iterations=60) if graph else {}
    else:
        raise ValueError(f"Unknown layout: {layout}")

    type_ids = sorted(explorer.node_type_names)
    relation_names = sorted(
        {relation for edge in graph_slice.edges for relation in edge.relations}
    )
    type_palette = plt.cm.get_cmap("tab10", max(len(type_ids), 1))
    relation_palette = plt.cm.get_cmap("Set2", max(len(relation_names), 1))
    node_colors = {
        type_id: type_palette(index) for index, type_id in enumerate(type_ids)
    }
    relation_colors = {
        relation: relation_palette(index)
        for index, relation in enumerate(relation_names)
    }

    fig, ax = plt.subplots(figsize=figsize)
    nx.draw_networkx_nodes(
        graph,
        positions,
        node_color=[node_colors.get(int(explorer.nodes[n].get("type", -1)), "grey") for n in graph],
        node_size=650,
        linewidths=0.8,
        edgecolors="white",
        ax=ax,
    )
    for edge in graph_slice.edges:
        primary_relation = edge.relations[0] if edge.relations else ""
        nx.draw_networkx_edges(
            graph,
            positions,
            edgelist=[(edge.source, edge.target)],
            edge_color=[relation_colors.get(primary_relation, "#888888")],
            width=0.8 + min(2.5, 0.25 * (edge.count - 1)),
            alpha=0.8,
            arrows=True,
            arrowsize=12,
            ax=ax,
        )
    if graph_slice.center in graph:
        nx.draw_networkx_nodes(
            graph,
            positions,
            nodelist=[graph_slice.center],
            node_color="none",
            node_size=880,
            linewidths=1.5,
            edgecolors="black",
            ax=ax,
        )
    labels = {node_id: explorer.label(node_id).split(" | ", 2)[-1][:42] for node_id in graph}
    nx.draw_networkx_labels(graph, positions, labels=labels, font_size=7, ax=ax)
    visible_type_ids = sorted(
        {int(explorer.nodes[node_id].get("type", -1)) for node_id in graph}
    )
    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=node_colors.get(type_id, "grey"),
            markeredgecolor="white",
            markersize=9,
            label=f"node: {explorer.node_type_names.get(type_id, type_id)}",
        )
        for type_id in visible_type_ids
    ]
    legend_handles.extend(
        Line2D(
            [0],
            [0],
            color=relation_colors.get(relation, "#888888"),
            linewidth=2,
            label=f"relation: {relation}",
        )
        for relation in relation_names
    )
    if graph_slice.center in graph:
        legend_handles.append(
            Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                markerfacecolor="none",
                markeredgecolor="black",
                markeredgewidth=2.0,
                markersize=10,
                label="selected center",
            )
        )
    if legend_handles:
        ax.legend(
            handles=legend_handles,
            loc="upper left",
            bbox_to_anchor=(1.01, 1),
            borderaxespad=0,
            fontsize=8,
            frameon=True,
        )
    ax.set_title(
        f"{graph_slice.level} view: "
        f"{len(graph_slice.node_ids):,} / {graph_slice.candidate_nodes:,} candidate nodes; "
        f"{len(graph_slice.edges):,} displayed edge pairs"
        + (" (node limit reached)" if graph_slice.truncated else "")
    )
    ax.axis("off")
    fig.tight_layout()
    return fig
