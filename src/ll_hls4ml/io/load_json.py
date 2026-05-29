"""Load CDFG JSON files."""

from __future__ import annotations

import orjson
from pathlib import Path
from typing import Any


def load_graph_json(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with path.open("rb") as f:
        return orjson.loads(f.read())
