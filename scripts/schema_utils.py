"""Utility helpers for schema version management across pipeline phases."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Iterable, Sequence

_DEFAULT_SCHEMA_FILES = (
    Path('schema/main-schema-critically-important.md'),
    Path('schema/AI-prompt-very-important.md'),
)


def _resolve_project_root() -> Path:
    """Return the repository root based on this module's location."""
    return Path(__file__).resolve().parent.parent


def resolve_schema_files(
    project_root: Path | None = None,
    schema_files: Sequence[Path] = _DEFAULT_SCHEMA_FILES,
) -> list[Path]:
    """Return absolute schema file paths that participate in the version hash."""
    root = project_root or _resolve_project_root()
    resolved: list[Path] = []
    for relative_path in schema_files:
        path = relative_path if relative_path.is_absolute() else root / relative_path
        if not path.exists():
            continue
        resolved.append(path)
    return resolved


def compute_schema_hash(project_root: Path | None = None) -> str:
    """Compute a SHA256 hash for the current schema snapshot."""
    files = resolve_schema_files(project_root)
    digest = hashlib.sha256()
    for path in sorted(files):
        digest.update(path.name.encode('utf-8'))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def snapshot_path_for_directory(directory: Path, filename: str = 'schema_snapshot.json') -> Path:
    """Return the schema snapshot file path for a given directory."""
    return Path(directory) / filename


def write_schema_snapshot(
    directory: Path,
    schema_hash: str,
    schema_files: Iterable[Path],
    *,
    filename: str = 'schema_snapshot.json',
) -> Path:
    """Persist the schema hash metadata next to validation/repair artefacts."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    snapshot_path = snapshot_path_for_directory(directory, filename)
    payload = {
        'schema_hash': schema_hash,
        'schema_files': [str(Path(path).resolve()) for path in schema_files],
        'generated': datetime.utcnow().isoformat() + 'Z',
    }
    snapshot_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding='utf-8')
    return snapshot_path


def read_schema_snapshot(snapshot_path: Path) -> dict | None:
    """Load a schema snapshot JSON payload if it exists."""
    path = Path(snapshot_path)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except json.JSONDecodeError:
        return None
