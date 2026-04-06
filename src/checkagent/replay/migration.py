"""Cassette migration engine for versioned schema upgrades.

When the cassette format changes between library versions, this module
handles upgrading cassettes from older schema versions to the current one.
Migrations run in sequence (v1->v2->v3) and are non-destructive: the
original file is backed up before modification.

Implements F2.6 (Cassette Versioning) from the PRD.
"""

from __future__ import annotations

import json
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any

from checkagent.replay.cassette import CASSETTE_SCHEMA_VERSION

# Type alias for a migration function: takes a dict, returns a dict.
MigrationFn = Callable[[dict[str, Any]], dict[str, Any]]

# Registry of migrations keyed by source version.
# Each entry maps from_version -> (to_version, migration_fn).
_MIGRATIONS: dict[int, tuple[int, MigrationFn]] = {}


def register_migration(
    from_version: int, to_version: int
) -> Callable[[MigrationFn], MigrationFn]:
    """Decorator to register a cassette migration function."""

    def decorator(fn: MigrationFn) -> MigrationFn:
        _MIGRATIONS[from_version] = (to_version, fn)
        return fn

    return decorator


def get_migration_path(from_version: int) -> list[tuple[int, int, MigrationFn]]:
    """Build the ordered chain of migrations from from_version to current.

    Returns list of (from_ver, to_ver, fn) tuples.
    Raises ValueError if no path exists to the current version.
    """
    path: list[tuple[int, int, MigrationFn]] = []
    current = from_version

    while current < CASSETTE_SCHEMA_VERSION:
        if current not in _MIGRATIONS:
            raise ValueError(
                f"No migration registered from v{current}. "
                f"Cannot upgrade to v{CASSETTE_SCHEMA_VERSION}."
            )
        to_ver, fn = _MIGRATIONS[current]
        path.append((current, to_ver, fn))
        current = to_ver

    return path


def migrate_cassette_data(data: dict[str, Any]) -> dict[str, Any]:
    """Apply all necessary migrations to a cassette dict.

    Returns the migrated data with updated schema_version in meta.
    If already at current version, returns the data unchanged.
    """
    meta = data.get("meta", {})
    version = meta.get("schema_version", 1)

    if version == CASSETTE_SCHEMA_VERSION:
        return data

    if version > CASSETTE_SCHEMA_VERSION:
        raise ValueError(
            f"Cassette schema v{version} is newer than "
            f"current v{CASSETTE_SCHEMA_VERSION}. "
            f"Upgrade checkagent to load this cassette."
        )

    path = get_migration_path(version)
    for _from_ver, _to_ver, fn in path:
        data = fn(data)

    # Ensure meta reflects final version
    data.setdefault("meta", {})["schema_version"] = CASSETTE_SCHEMA_VERSION
    return data


class MigrationResult:
    """Result of migrating a single cassette file."""

    def __init__(
        self,
        path: Path,
        *,
        from_version: int,
        to_version: int,
        success: bool = True,
        error: str = "",
    ) -> None:
        self.path = path
        self.from_version = from_version
        self.to_version = to_version
        self.success = success
        self.error = error

    def __repr__(self) -> str:
        status = "ok" if self.success else f"FAILED: {self.error}"
        return (
            f"MigrationResult({self.path.name}, "
            f"v{self.from_version}->v{self.to_version}, {status})"
        )


def migrate_file(
    path: Path, *, backup: bool = True, dry_run: bool = False
) -> MigrationResult:
    """Migrate a single cassette file to the current schema version.

    Args:
        path: Path to the cassette JSON file.
        backup: If True, create a .bak copy before modifying.
        dry_run: If True, report what would change without writing.

    Returns:
        MigrationResult describing the outcome.
    """
    try:
        raw = path.read_text(encoding="utf-8")
        data = json.loads(raw)
    except (json.JSONDecodeError, OSError) as e:
        return MigrationResult(
            path, from_version=0, to_version=0, success=False, error=str(e)
        )

    meta = data.get("meta", {})
    from_version = meta.get("schema_version", 1)

    if from_version == CASSETTE_SCHEMA_VERSION:
        return MigrationResult(
            path,
            from_version=from_version,
            to_version=from_version,
        )

    try:
        migrated = migrate_cassette_data(data)
    except ValueError as e:
        return MigrationResult(
            path,
            from_version=from_version,
            to_version=from_version,
            success=False,
            error=str(e),
        )

    if dry_run:
        return MigrationResult(
            path,
            from_version=from_version,
            to_version=CASSETTE_SCHEMA_VERSION,
        )

    if backup:
        shutil.copy2(path, path.with_suffix(".json.bak"))

    path.write_text(
        json.dumps(migrated, indent=2, default=str), encoding="utf-8"
    )

    return MigrationResult(
        path,
        from_version=from_version,
        to_version=CASSETTE_SCHEMA_VERSION,
    )


def migrate_directory(
    directory: Path,
    *,
    backup: bool = True,
    dry_run: bool = False,
) -> list[MigrationResult]:
    """Migrate all cassette files in a directory tree.

    Discovers all *.json files recursively, skips non-cassette files.

    Returns:
        List of MigrationResult for each cassette found.
    """
    results: list[MigrationResult] = []

    for json_file in sorted(directory.rglob("*.json")):
        # Quick check: skip files that don't look like cassettes
        try:
            raw = json_file.read_text(encoding="utf-8")
            data = json.loads(raw)
        except (json.JSONDecodeError, OSError):
            continue

        # Must have meta and interactions to be a cassette
        if "meta" not in data or "interactions" not in data:
            continue

        results.append(
            migrate_file(json_file, backup=backup, dry_run=dry_run)
        )

    return results
