"""Content-stable fingerprints for tensor datasets."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Callable, Iterable


MANIFEST_SCHEMA_VERSION = 1
_HASH_CHUNK_SIZE = 8 * 1024 * 1024


def file_sha256(path: str | Path) -> str:
    """Return the SHA-256 digest of a file's contents."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(_HASH_CHUNK_SIZE):
            digest.update(chunk)
    return digest.hexdigest()


def _snapshot_sha256(files: list[dict[str, object]]) -> str:
    digest = hashlib.sha256()
    for entry in files:
        digest.update(
            (
                f"{entry['path']}\0{entry['size_bytes']}\0"
                f"{entry['sha256']}\n"
            ).encode()
        )
    return digest.hexdigest()


def build_content_manifest(
    paths: Iterable[str | Path],
    root: str | Path,
    progress: Callable[[int, int, Path], None] | None = None,
) -> dict[str, object]:
    """Hash files and return a path-independent, content-stable manifest."""
    root = Path(root).resolve()
    unique_paths = sorted(
        {Path(path).resolve() for path in paths},
        key=lambda path: path.relative_to(root).as_posix(),
    )
    files = []
    total = len(unique_paths)
    for index, path in enumerate(unique_paths, start=1):
        relative = path.relative_to(root).as_posix()
        files.append(
            {
                "path": relative,
                "size_bytes": path.stat().st_size,
                "sha256": file_sha256(path),
            }
        )
        if progress is not None:
            progress(index, total, path)
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "algorithm": "sha256",
        "snapshot_sha256": _snapshot_sha256(files),
        "files": files,
    }


def validate_content_manifest(manifest: dict[str, object]) -> None:
    """Validate manifest structure and its aggregate content digest."""
    if manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise ValueError(
            "Unsupported tensor manifest schema version: "
            f"{manifest.get('schema_version')}"
        )
    if manifest.get("algorithm") != "sha256":
        raise ValueError("Tensor manifest must use SHA-256")
    files = manifest.get("files")
    if not isinstance(files, list):
        raise ValueError("Tensor manifest files must be a list")
    paths = [entry.get("path") for entry in files]
    if len(paths) != len(set(paths)):
        raise ValueError("Tensor manifest contains duplicate paths")
    expected = _snapshot_sha256(files)
    if manifest.get("snapshot_sha256") != expected:
        raise ValueError("Tensor manifest aggregate digest is invalid")


def load_content_manifest(path: str | Path) -> dict[str, object]:
    """Load and validate a content manifest."""
    manifest = json.loads(Path(path).read_text())
    validate_content_manifest(manifest)
    return manifest


def write_content_manifest(
    path: str | Path,
    manifest: dict[str, object],
) -> None:
    """Validate and write a content manifest as deterministic JSON."""
    validate_content_manifest(manifest)
    Path(path).write_text(json.dumps(manifest, indent=2, sort_keys=True))


def assert_manifest_covers(
    manifest: dict[str, object],
    paths: Iterable[str | Path],
    root: str | Path,
) -> None:
    """Fail if any expected file is absent from a loaded manifest."""
    root = Path(root).resolve()
    available = {entry["path"] for entry in manifest["files"]}
    expected = {
        Path(path).resolve().relative_to(root).as_posix()
        for path in paths
    }
    missing = sorted(expected - available)
    if missing:
        preview = ", ".join(missing[:5])
        suffix = " ..." if len(missing) > 5 else ""
        raise ValueError(
            f"Tensor manifest is missing {len(missing)} indexed file(s): "
            f"{preview}{suffix}"
        )
