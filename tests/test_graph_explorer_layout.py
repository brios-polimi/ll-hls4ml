import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from ll_hls4ml.viz.graph_explorer import GraphExplorer, _ranked_positions


class GraphExplorerLayoutTest(unittest.TestCase):
    def test_ranked_positions_condense_loops(self):
        positions = _ranked_positions(
            range(5),
            [(0, 1), (1, 2), (2, 1), (2, 3), (3, 4)],
        )

        self.assertEqual(set(positions), set(range(5)))
        self.assertEqual(positions[1][0], positions[2][0])
        self.assertLess(positions[0][0], positions[1][0])
        self.assertLess(positions[2][0], positions[3][0])

    def test_shared_constant_does_not_expand_local_slice(self):
        graph = {
            "nodes": [
                {"id": 0, "type": 2, "features": {"name": ["0"]}},
                *[
                    {"id": node_id, "type": 0, "features": {"name": [f"i{node_id}"]}}
                    for node_id in range(1, 28)
                ],
            ],
            "links": [
                {"source": 0, "target": node_id, "relation": "operand", "position": 0}
                for node_id in range(1, 28)
            ],
        }
        with TemporaryDirectory() as directory:
            path = Path(directory) / "graph.json"
            path.write_text(json.dumps(graph))
            explorer = GraphExplorer(path)

        graph_slice = explorer.slice(
            level="node",
            center=1,
            radius=2,
            max_nodes=100,
            node_types=("instruction", "constant"),
            edge_relations=("operand",),
        )

        self.assertEqual(set(graph_slice.node_ids), {0, 1})


if __name__ == "__main__":
    unittest.main()
