"""Load CDFG JSON files."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

try:
    import orjson
except ImportError:  # Optional acceleration; stdlib JSON keeps fresh installs usable.
    orjson = None


def load_graph_json(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with path.open("rb") as f:
        payload = f.read()
    return orjson.loads(payload) if orjson is not None else json.loads(payload)
