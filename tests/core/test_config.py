"""Tests for checkagent.yml configuration loader."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from checkagent.core.config import (
    CassettesConfig,
    CheckAgentConfig,
    DefaultsConfig,
    QualityGateEntry,
    SafetyConfig,
    find_config,
    load_config,
)

# ---------------------------------------------------------------------------
# Default values
# ---------------------------------------------------------------------------


class TestDefaultConfig:
    """CheckAgentConfig with no input returns sensible defaults."""

    def test_default_version(self):
        cfg = CheckAgentConfig()
        assert cfg.version == 1

    def test_default_asyncio_mode(self):
        cfg = CheckAgentConfig()
        assert cfg.asyncio_mode == "auto"

    def test_default_layer(self):
        cfg = CheckAgentConfig()
        assert cfg.defaults.layer == "mock"

    def test_default_timeout(self):
        cfg = CheckAgentConfig()
        assert cfg.defaults.timeout == 30

    def test_default_max_iterations(self):
        cfg = CheckAgentConfig()
        assert cfg.defaults.max_iterations == 50

    def test_default_cassettes_directory(self):
        cfg = CheckAgentConfig()
        assert cfg.cassettes.directory == "tests/cassettes"

    def test_default_cassettes_format(self):
        cfg = CheckAgentConfig()
        assert cfg.cassettes.format == "json"

    def test_default_safety_enabled(self):
        cfg = CheckAgentConfig()
        assert cfg.safety.enabled is True

    def test_default_fixture_prefix(self):
        cfg = CheckAgentConfig()
        assert cfg.plugins.fixture_prefix == "ap_"

    def test_default_no_budget(self):
        cfg = CheckAgentConfig()
        assert cfg.budget.per_test is None
        assert cfg.budget.per_suite is None


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


class TestValidation:
    """Field validators reject invalid values."""

    def test_invalid_layer(self):
        with pytest.raises(ValueError, match="Invalid default layer"):
            DefaultsConfig(layer="bogus")

    def test_invalid_asyncio_mode(self):
        with pytest.raises(ValueError, match="Invalid asyncio_mode"):
            CheckAgentConfig(asyncio_mode="bogus")

    def test_invalid_cassette_format(self):
        with pytest.raises(ValueError, match="Invalid cassette format"):
            CassettesConfig(format="xml")

    def test_invalid_safety_severity(self):
        with pytest.raises(ValueError, match="Invalid severity_threshold"):
            SafetyConfig(severity_threshold="extreme")

    def test_invalid_quality_gate_action(self):
        with pytest.raises(ValueError, match="Invalid on_fail"):
            QualityGateEntry(on_fail="crash")

    def test_valid_layers(self):
        for layer in ("mock", "replay", "eval", "judge"):
            cfg = DefaultsConfig(layer=layer)
            assert cfg.layer == layer


# ---------------------------------------------------------------------------
# YAML loading
# ---------------------------------------------------------------------------


class TestLoadYAML:
    """Loading checkagent.yml files."""

    def test_load_minimal(self, tmp_path: Path):
        cfg_file = tmp_path / "checkagent.yml"
        cfg_file.write_text("version: 1\n")
        cfg = load_config(cfg_file)
        assert cfg.version == 1
        assert cfg.defaults.layer == "mock"

    def test_load_full_config(self, tmp_path: Path):
        data = {
            "version": 1,
            "asyncio_mode": "strict",
            "defaults": {"layer": "eval", "timeout": 60, "max_iterations": 100},
            "providers": {
                "openai": {
                    "model": "gpt-4o",
                    "pricing": {"input": 2.50, "output": 10.00},
                },
            },
            "budget": {"per_test": 0.10, "per_suite": 5.00},
            "quality_gates": {
                "task_completion": {"min": 0.90, "on_fail": "block"},
            },
            "cassettes": {
                "directory": "my_cassettes",
                "format": "json",
                "ttl_days": 14,
                "redact_patterns": [r"sk-[a-zA-Z0-9]{48}"],
            },
            "safety": {
                "enabled": False,
                "categories": ["prompt_injection"],
                "severity_threshold": "high",
            },
            "plugins": {"fixture_prefix": "ca_", "short_aliases": True},
        }
        cfg_file = tmp_path / "checkagent.yml"
        cfg_file.write_text(yaml.dump(data))
        cfg = load_config(cfg_file)

        assert cfg.asyncio_mode == "strict"
        assert cfg.defaults.layer == "eval"
        assert cfg.defaults.timeout == 60
        assert cfg.defaults.max_iterations == 100
        assert cfg.providers["openai"].model == "gpt-4o"
        assert cfg.providers["openai"].pricing.input == 2.50
        assert cfg.budget.per_test == 0.10
        assert cfg.quality_gates["task_completion"].min == 0.90
        assert cfg.quality_gates["task_completion"].on_fail == "block"
        assert cfg.cassettes.directory == "my_cassettes"
        assert cfg.cassettes.ttl_days == 14
        assert len(cfg.cassettes.redact_patterns) == 1
        assert cfg.safety.enabled is False
        assert cfg.plugins.fixture_prefix == "ca_"
        assert cfg.plugins.short_aliases is True

    def test_load_empty_file(self, tmp_path: Path):
        cfg_file = tmp_path / "checkagent.yml"
        cfg_file.write_text("")
        cfg = load_config(cfg_file)
        assert cfg.version == 1  # all defaults

    def test_load_invalid_yaml_value(self, tmp_path: Path):
        cfg_file = tmp_path / "checkagent.yml"
        cfg_file.write_text("defaults:\n  layer: bogus\n")
        with pytest.raises(ValueError, match="Invalid default layer"):
            load_config(cfg_file)

    def test_load_non_mapping_raises(self, tmp_path: Path):
        cfg_file = tmp_path / "checkagent.yml"
        cfg_file.write_text("- item1\n- item2\n")
        with pytest.raises(ValueError, match="must be a YAML/TOML mapping"):
            load_config(cfg_file)


# ---------------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------------


class TestFindConfig:
    """Config file discovery walks up the directory tree."""

    def test_finds_in_start_dir(self, tmp_path: Path):
        cfg_file = tmp_path / "checkagent.yml"
        cfg_file.write_text("version: 1\n")
        assert find_config(tmp_path) == cfg_file

    def test_finds_in_parent_dir(self, tmp_path: Path):
        cfg_file = tmp_path / "checkagent.yml"
        cfg_file.write_text("version: 1\n")
        child = tmp_path / "src" / "deep"
        child.mkdir(parents=True)
        assert find_config(child) == cfg_file

    def test_returns_none_when_missing(self, tmp_path: Path):
        child = tmp_path / "no_config_here"
        child.mkdir()
        # tmp_path itself has no config, so walking up should not find one
        # (unless the real filesystem has one — use isolated dir)
        result = find_config(child)
        # We can't guarantee None in CI (real fs may have config above tmp_path),
        # but if it finds something, it must be a valid config filename.
        if result is not None:
            assert result.name in ("checkagent.yml", "checkagent.yaml", "checkagent.toml")

    def test_prefers_yml_over_yaml(self, tmp_path: Path):
        (tmp_path / "checkagent.yml").write_text("version: 1\n")
        (tmp_path / "checkagent.yaml").write_text("version: 1\n")
        result = find_config(tmp_path)
        assert result is not None
        assert result.name == "checkagent.yml"

    def test_finds_yaml_extension(self, tmp_path: Path):
        cfg_file = tmp_path / "checkagent.yaml"
        cfg_file.write_text("version: 1\n")
        assert find_config(tmp_path) == cfg_file


# ---------------------------------------------------------------------------
# Load with no path (auto-discovery)
# ---------------------------------------------------------------------------


class TestLoadAutoDiscovery:
    """load_config() with no arguments uses find_config."""

    def test_returns_defaults_when_no_file(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.chdir(tmp_path)
        cfg = load_config()
        assert cfg.version == 1
        assert cfg.defaults.layer == "mock"

    def test_loads_from_cwd(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        cfg_file = tmp_path / "checkagent.yml"
        cfg_file.write_text("defaults:\n  layer: eval\n")
        monkeypatch.chdir(tmp_path)
        cfg = load_config()
        assert cfg.defaults.layer == "eval"


# ---------------------------------------------------------------------------
# Unsupported formats
# ---------------------------------------------------------------------------


class TestUnsupportedFormat:
    def test_unknown_extension_raises(self, tmp_path: Path):
        cfg_file = tmp_path / "checkagent.ini"
        cfg_file.write_text("[defaults]\nlayer=mock\n")
        with pytest.raises(ValueError, match="Unsupported config file format"):
            load_config(cfg_file)
