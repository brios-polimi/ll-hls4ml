"""One-panel notebook UI for bounded CDFG exploration."""

from __future__ import annotations

import json
from pathlib import Path
import re

from ll_hls4ml.viz.graph_explorer import GraphExplorer, plot_slice


def _natural_key(value: object) -> list[object]:
    return [
        int(piece) if piece.isdigit() else piece.lower()
        for piece in re.split(r"(\d+)", str(value))
    ]


class GraphViewer:
    """Coordinate graph discovery, slicing, inspection, and rendering."""

    def __init__(self, graph_root: str | Path):
        import ipywidgets as widgets

        self.widgets = widgets
        self.graph_root = Path(graph_root)
        self.explorer: GraphExplorer | None = None
        self._figure = None

        wide = widgets.Layout(width="100%")
        half = widgets.Layout(width="49%")
        self.kernel = widgets.Dropdown(description="Kernel", layout=half)
        self.archive = widgets.Dropdown(description="Archive", layout=half)
        self.graph_file = widgets.Dropdown(description="Graph", layout=wide)
        self.load = widgets.Button(
            description="Load graph", button_style="primary", icon="folder-open"
        )
        self.status = widgets.HTML()

        self.level = widgets.Dropdown(description="Level", options=("node",))
        self.layout = widgets.ToggleButtons(
            description="Layout",
            options=(("Hierarchy", "hierarchy"), ("Spring", "spring")),
            value="hierarchy",
        )
        self.radius = widgets.IntSlider(
            description="Radius", min=0, max=8, value=2, continuous_update=False
        )
        self.max_nodes = widgets.BoundedIntText(
            description="Node limit", min=1, max=1000, value=160
        )
        select_layout = widgets.Layout(width="48%", min_width="320px")
        select_style = {"description_width": "initial"}
        self.node_types = widgets.SelectMultiple(
            description="Visible node types",
            rows=7,
            layout=select_layout,
            style=select_style,
        )
        self.relations = widgets.SelectMultiple(
            description="Visible relations",
            rows=7,
            layout=select_layout,
            style=select_style,
        )
        self.search = widgets.Text(
            description="Search",
            placeholder="ID, instruction, pragma, block, or function",
            layout=wide,
        )
        self.center = widgets.Dropdown(description="Center", options=(), layout=wide)
        self.render = widgets.Button(
            description="Render / update", button_style="success", icon="refresh"
        )
        self.direction = widgets.ToggleButtons(
            description="Direction", options=("both", "out", "in"), value="both"
        )
        self.include_internal = widgets.Checkbox(
            description="Include projected self-edges", value=False
        )
        self.details_output = widgets.Output()
        self.plot_output = widgets.Output()

        source = widgets.VBox(
            [
                widgets.HBox([self.kernel, self.archive]),
                self.graph_file,
                widgets.HBox([self.load, self.status]),
            ]
        )
        filters = widgets.VBox(
            [
                widgets.HBox(
                    [self.level, self.layout, self.radius, self.max_nodes]
                ),
                widgets.HBox([self.node_types, self.relations]),
                self.search,
                self.center,
                self.render,
            ]
        )
        advanced = widgets.Accordion(
            children=(
                widgets.VBox(
                    [self.direction, self.include_internal, self.details_output]
                ),
            ),
            selected_index=None,
        )
        advanced.set_title(0, "Direction and selected-node details")
        self.widget = widgets.VBox([source, filters, advanced, self.plot_output])

        self.kernel.observe(self._refresh_archives, names="value")
        self.archive.observe(self._refresh_graphs, names="value")
        self.level.observe(self._level_changed, names="value")
        self.search.observe(self._refresh_search, names="value")
        self.center.observe(self._show_details, names="value")
        self.load.on_click(self._load_selected)
        self.render.on_click(self._render)
        self._refresh_kernels()

    def _refresh_kernels(self) -> None:
        self.kernel.options = tuple(
            sorted(
                (path.name for path in self.graph_root.iterdir() if path.is_dir()),
                key=_natural_key,
            )
        )
        self._refresh_archives()

    def _refresh_archives(self, change=None) -> None:
        root = self.graph_root / self.kernel.value if self.kernel.value else None
        self.archive.options = tuple(
            sorted(
                (path.name for path in root.iterdir() if path.is_dir()),
                key=_natural_key,
            )
        ) if root else ()
        self._refresh_graphs()

    def _refresh_graphs(self, change=None) -> None:
        root = (
            self.graph_root / self.kernel.value / self.archive.value
            if self.kernel.value and self.archive.value
            else None
        )
        paths = sorted(root.glob("*.json"), key=lambda path: _natural_key(path.name)) if root else []
        self.graph_file.options = tuple((path.name, path) for path in paths)

    def _load_selected(self, change=None) -> None:
        if self.graph_file.value is None:
            self.status.value = "<b>Select a graph first.</b>"
            return
        try:
            self.explorer = GraphExplorer(self.graph_file.value)
        except Exception as error:
            self.status.value = f"<b>Load failed:</b> {error}"
            return
        self.level.options = self.explorer.available_levels
        self.level.value = "node"
        self._level_changed()
        summary = self.explorer.summary
        type_counts = ", ".join(
            f"{name}: {count:,}" for name, count in summary["node_types"].items()
        )
        self.status.value = (
            f"<b>{summary['nodes']:,}</b> nodes · <b>{summary['edges']:,}</b> edges"
            f"<br><small>{type_counts}</small>"
        )
        self._render()

    def _level_changed(self, change=None) -> None:
        if self.explorer is None:
            return
        node_options = self.explorer.selectable_node_types(self.level.value)
        relation_options = self.explorer.available_relations(self.level.value)
        self.node_types.options = node_options
        self.node_types.value = node_options
        self.relations.options = relation_options
        self.relations.value = self.explorer.default_relations(self.level.value)
        self._refresh_search()

    def _refresh_search(self, change=None) -> None:
        if self.explorer is None:
            self.center.options = ()
            return
        self.center.options = self.explorer.search(
            self.search.value, self.level.value, limit=80
        )

    def _show_details(self, change=None) -> None:
        if self.center.value is not None:
            self._show_node_details(self.center.value)

    def _show_node_details(self, node_id: int) -> None:
        from IPython.display import clear_output

        with self.details_output:
            clear_output(wait=True)
            if self.explorer is not None:
                print(json.dumps(self.explorer.details(node_id), indent=2))

    def _render(self, change=None) -> None:
        from IPython.display import clear_output, display
        import matplotlib.pyplot as plt

        with self.plot_output:
            if self._figure is not None:
                plt.close(self._figure)
                self._figure = None
            clear_output(wait=True)
            if self.explorer is None:
                print("Load a graph first.")
                return
            graph_slice = self.explorer.slice(
                level=self.level.value,
                center=self.center.value,
                radius=self.radius.value,
                max_nodes=self.max_nodes.value,
                node_types=self.node_types.value,
                edge_relations=self.relations.value,
                direction=self.direction.value,
                include_internal=self.include_internal.value,
            )
            if not graph_slice.node_ids:
                print("No nodes match these filters.")
                return
            figure = plot_slice(
                self.explorer,
                graph_slice,
                layout=self.layout.value,
                on_select=self._show_node_details,
            )
            self._figure = figure
            display(figure.canvas)
