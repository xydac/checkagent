"""Tests for cassette migration engine and CLI command."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from checkagent.replay.cassette import CASSETTE_SCHEMA_VERSION
from checkagent.replay.migration import (
    _MIGRATIONS,
    MigrationResult,
    get_migration_path,
    migrate_cassette_data,
    migrate_directory,
    migrate_file,
    register_migration,
)

# --- Fixtures ---


@pytest.fixture(autouse=True)
def _clean_migrations():
    """Save and restore the migration registry around each test."""
    saved = dict(_MIGRATIONS)
    yield
    _MIGRATIONS.clear()
    _MIGRATIONS.update(saved)


def _make_cassette_json(
    version: int = 1,
    interactions: list | None = None,
) -> str:
    """Create a minimal valid cassette JSON string."""
    data = {
        "meta": {
            "schema_version": version,
            "checkagent_version": "0.1.0",
            "recorded_at": "2026-01-01T00:00:00Z",
            "content_hash": "",
            "test_id": "test_example",
        },
        "interactions": interactions or [],
    }
    return json.dumps(data)


def _write_cassette(tmp_path: Path, name: str, version: int = 1) -> Path:
    """Write a cassette file to tmp_path and return its path."""
    path = tmp_path / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_make_cassette_json(version), encoding="utf-8")
    return path


# --- register_migration ---


class TestRegisterMigration:
    def test_registers_function(self):
        @register_migration(from_version=99, to_version=100)
        def _migrate_99_to_100(data):
            return data

        assert 99 in _MIGRATIONS
        assert _MIGRATIONS[99][0] == 100

    def test_function_is_callable(self):
        @register_migration(from_version=98, to_version=99)
        def _migrate(data):
            data["migrated"] = True
            return data

        _, fn = _MIGRATIONS[98]
        result = fn({"meta": {}})
        assert result["migrated"] is True


# --- get_migration_path ---


class TestGetMigrationPath:
    def test_no_migration_needed(self):
        path = get_migration_path(CASSETTE_SCHEMA_VERSION)
        assert path == []

    def test_single_step(self):
        @register_migration(
            from_version=CASSETTE_SCHEMA_VERSION - 1,
            to_version=CASSETTE_SCHEMA_VERSION,
        )
        def _m(data):
            return data

        # Only works if current version > 1
        if CASSETTE_SCHEMA_VERSION > 1:
            path = get_migration_path(CASSETTE_SCHEMA_VERSION - 1)
            assert len(path) == 1
            assert path[0][0] == CASSETTE_SCHEMA_VERSION - 1
            assert path[0][1] == CASSETTE_SCHEMA_VERSION

    def test_multi_step_chain(self, monkeypatch):
        # Simulate a chain: 0 -> 1 -> 2 where current version is 2
        import checkagent.replay.migration as mod

        monkeypatch.setattr(mod, "CASSETTE_SCHEMA_VERSION", 2)

        @register_migration(from_version=0, to_version=1)
        def _m1(data):
            return data

        @register_migration(from_version=1, to_version=2)
        def _m2(data):
            return data

        path = get_migration_path(0)
        assert len(path) == 2
        assert path[0][:2] == (0, 1)
        assert path[1][:2] == (1, 2)

    def test_missing_migration_raises(self, monkeypatch):
        import checkagent.replay.migration as mod

        monkeypatch.setattr(mod, "CASSETTE_SCHEMA_VERSION", 100)
        with pytest.raises(ValueError, match="No migration registered from v50"):
            get_migration_path(50)


# --- migrate_cassette_data ---


class TestMigrateCassetteData:
    def test_already_current_version(self):
        data = {"meta": {"schema_version": CASSETTE_SCHEMA_VERSION}}
        result = migrate_cassette_data(data)
        assert result is data  # no copy, same object

    def test_newer_version_raises(self):
        data = {"meta": {"schema_version": CASSETTE_SCHEMA_VERSION + 1}}
        with pytest.raises(ValueError, match="newer than"):
            migrate_cassette_data(data)

    def test_applies_migration(self):
        target = CASSETTE_SCHEMA_VERSION

        @register_migration(from_version=0, to_version=target)
        def _m(data):
            data["upgraded"] = True
            return data

        data = {"meta": {"schema_version": 0}, "interactions": []}
        result = migrate_cassette_data(data)
        assert result["upgraded"] is True
        assert result["meta"]["schema_version"] == target

    def test_default_version_is_1(self):
        """Missing schema_version defaults to 1."""
        data = {"meta": {}}
        # version 1 == current, so no migration needed
        if CASSETTE_SCHEMA_VERSION == 1:
            result = migrate_cassette_data(data)
            assert result is data


# --- migrate_file ---


class TestMigrateFile:
    def test_already_current(self, tmp_path: Path):
        path = _write_cassette(tmp_path, "current.json", CASSETTE_SCHEMA_VERSION)
        result = migrate_file(path)
        assert result.success
        assert result.from_version == result.to_version

    def test_invalid_json(self, tmp_path: Path):
        path = tmp_path / "bad.json"
        path.write_text("not json!", encoding="utf-8")
        result = migrate_file(path)
        assert not result.success
        assert "Expecting value" in result.error

    def test_missing_file(self, tmp_path: Path):
        path = tmp_path / "missing.json"
        result = migrate_file(path)
        assert not result.success

    def test_creates_backup(self, tmp_path: Path):
        target = CASSETTE_SCHEMA_VERSION

        @register_migration(from_version=0, to_version=target)
        def _m(data):
            return data

        path = _write_cassette(tmp_path, "old.json", version=0)
        original_content = path.read_text(encoding="utf-8")

        result = migrate_file(path, backup=True)
        assert result.success
        assert result.from_version == 0
        assert result.to_version == target

        # Backup exists with original content
        bak = path.with_suffix(".json.bak")
        assert bak.exists()
        assert bak.read_text(encoding="utf-8") == original_content

        # Migrated file has new version
        migrated = json.loads(path.read_text(encoding="utf-8"))
        assert migrated["meta"]["schema_version"] == target

    def test_no_backup(self, tmp_path: Path):
        target = CASSETTE_SCHEMA_VERSION

        @register_migration(from_version=0, to_version=target)
        def _m(data):
            return data

        path = _write_cassette(tmp_path, "old.json", version=0)
        migrate_file(path, backup=False)

        bak = path.with_suffix(".json.bak")
        assert not bak.exists()

    def test_dry_run(self, tmp_path: Path):
        target = CASSETTE_SCHEMA_VERSION

        @register_migration(from_version=0, to_version=target)
        def _m(data):
            return data

        path = _write_cassette(tmp_path, "old.json", version=0)
        original_content = path.read_text(encoding="utf-8")

        result = migrate_file(path, dry_run=True)
        assert result.success
        assert result.from_version == 0
        assert result.to_version == target

        # File unchanged
        assert path.read_text(encoding="utf-8") == original_content

    def test_no_migration_path(self, tmp_path: Path):
        path = _write_cassette(tmp_path, "old.json", version=0)
        result = migrate_file(path)
        assert not result.success
        assert "No migration registered" in result.error


# --- migrate_directory ---


class TestMigrateDirectory:
    def test_empty_directory(self, tmp_path: Path):
        results = migrate_directory(tmp_path)
        assert results == []

    def test_skips_non_cassette_json(self, tmp_path: Path):
        # A JSON file without meta/interactions is not a cassette
        (tmp_path / "config.json").write_text('{"key": "value"}', encoding="utf-8")
        results = migrate_directory(tmp_path)
        assert results == []

    def test_finds_cassettes_recursively(self, tmp_path: Path):
        _write_cassette(tmp_path, "a.json")
        _write_cassette(tmp_path, "sub/b.json")
        _write_cassette(tmp_path, "sub/deep/c.json")

        results = migrate_directory(tmp_path)
        assert len(results) == 3

    def test_migrates_outdated_cassettes(self, tmp_path: Path):
        target = CASSETTE_SCHEMA_VERSION

        @register_migration(from_version=0, to_version=target)
        def _m(data):
            return data

        _write_cassette(tmp_path, "old.json", version=0)
        _write_cassette(tmp_path, "current.json", version=target)

        results = migrate_directory(tmp_path)
        assert len(results) == 2

        migrated = [r for r in results if r.from_version != r.to_version]
        skipped = [r for r in results if r.from_version == r.to_version]
        assert len(migrated) == 1
        assert len(skipped) == 1


# --- MigrationResult ---


class TestMigrationResult:
    def test_repr_success(self, tmp_path: Path):
        r = MigrationResult(
            tmp_path / "test.json", from_version=1, to_version=2
        )
        assert "v1->v2" in repr(r)
        assert "ok" in repr(r)

    def test_repr_failure(self, tmp_path: Path):
        r = MigrationResult(
            tmp_path / "test.json",
            from_version=1,
            to_version=1,
            success=False,
            error="boom",
        )
        assert "FAILED" in repr(r)
        assert "boom" in repr(r)


# --- CLI command ---


class TestMigrateCLI:
    def _invoke(self, args: list[str]):
        """Invoke migrate CLI with wide terminal to avoid Rich line wrapping."""
        import os

        from checkagent.cli.migrate import migrate_cmd

        old = os.environ.get("COLUMNS")
        os.environ["COLUMNS"] = "200"
        try:
            runner = CliRunner()
            return runner.invoke(migrate_cmd, args)
        finally:
            if old is None:
                os.environ.pop("COLUMNS", None)
            else:
                os.environ["COLUMNS"] = old

    def test_no_cassettes(self, tmp_path: Path):
        result = self._invoke([str(tmp_path)])
        assert result.exit_code == 0
        assert "No cassette files found" in result.output

    def test_dry_run_output(self, tmp_path: Path):
        target = CASSETTE_SCHEMA_VERSION

        @register_migration(from_version=0, to_version=target)
        def _m(data):
            return data

        _write_cassette(tmp_path, "old.json", version=0)

        result = self._invoke([str(tmp_path), "--dry-run"])
        assert result.exit_code == 0
        assert "Dry run" in result.output
        assert "would migrate" in result.output

    def test_migrate_with_summary(self, tmp_path: Path):
        target = CASSETTE_SCHEMA_VERSION

        @register_migration(from_version=0, to_version=target)
        def _m(data):
            return data

        _write_cassette(tmp_path, "old.json", version=0)
        _write_cassette(tmp_path, "current.json", version=target)

        result = self._invoke([str(tmp_path)])
        assert result.exit_code == 0
        assert "Migrated: 1" in result.output
        assert "Skipped: 1" in result.output
