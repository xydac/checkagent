"""Configuration loader for CheckAgent.

Loads and validates checkagent.yml (or checkagent.toml) from the project root.
All configuration is represented as Pydantic models with sensible defaults.
"""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field, field_validator

# ---------------------------------------------------------------------------
# Sub-models
# ---------------------------------------------------------------------------


class DefaultsConfig(BaseModel):
    """Default settings applied to all tests unless overridden."""

    layer: str = "mock"
    timeout: int = 30
    max_iterations: int = 50

    @field_validator("layer")
    @classmethod
    def _valid_layer(cls, v: str) -> str:
        valid = {"mock", "replay", "eval", "judge"}
        if v not in valid:
            msg = f"Invalid default layer '{v}'. Must be one of: {', '.join(sorted(valid))}"
            raise ValueError(msg)
        return v


class ProviderPricing(BaseModel):
    """Token pricing for a provider (per 1M tokens, USD)."""

    input: float = 0.0
    output: float = 0.0


class ProviderConfig(BaseModel):
    """Configuration for a single LLM provider."""

    model: str = ""
    pricing: ProviderPricing = Field(default_factory=ProviderPricing)


class BudgetConfig(BaseModel):
    """Cost budget limits (USD)."""

    per_test: float | None = None
    per_suite: float | None = None
    per_ci_run: float | None = None


class QualityGateEntry(BaseModel):
    """A single quality gate threshold."""

    min: float | None = None
    max: float | None = None
    on_fail: str = "warn"

    @field_validator("on_fail")
    @classmethod
    def _valid_action(cls, v: str) -> str:
        valid = {"block", "warn", "ignore"}
        if v not in valid:
            msg = f"Invalid on_fail '{v}'. Must be one of: {', '.join(sorted(valid))}"
            raise ValueError(msg)
        return v


class CassettesConfig(BaseModel):
    """Cassette (record/replay) settings."""

    directory: str = "tests/cassettes"
    format: str = "json"
    ttl_days: int | None = None
    redact_patterns: list[str] = Field(default_factory=list)

    @field_validator("format")
    @classmethod
    def _valid_format(cls, v: str) -> str:
        valid = {"json", "yaml"}
        if v not in valid:
            msg = f"Invalid cassette format '{v}'. Must be one of: {', '.join(sorted(valid))}"
            raise ValueError(msg)
        return v


class SafetyConfig(BaseModel):
    """Safety testing configuration."""

    enabled: bool = True
    categories: list[str] = Field(
        default_factory=lambda: [
            "prompt_injection",
            "pii_leakage",
            "harmful_content",
            "tool_misuse",
        ]
    )
    severity_threshold: str = "medium"

    @field_validator("severity_threshold")
    @classmethod
    def _valid_severity(cls, v: str) -> str:
        valid = {"low", "medium", "high", "critical"}
        if v not in valid:
            msg = f"Invalid severity_threshold '{v}'. Must be one of: {', '.join(sorted(valid))}"
            raise ValueError(msg)
        return v


class PIIPatternConfig(BaseModel):
    """A named PII detection pattern."""

    name: str
    regex: str


class PIIConfig(BaseModel):
    """PII scrubbing configuration."""

    mode: str = "regex"
    patterns: list[PIIPatternConfig] = Field(default_factory=list)

    @field_validator("mode")
    @classmethod
    def _valid_mode(cls, v: str) -> str:
        valid = {"regex", "ner"}
        if v not in valid:
            msg = f"Invalid PII mode '{v}'. Must be one of: {', '.join(sorted(valid))}"
            raise ValueError(msg)
        return v


class PluginsConfig(BaseModel):
    """Plugin override settings."""

    fixture_prefix: str = "ap_"
    short_aliases: bool = False


# ---------------------------------------------------------------------------
# Root config model
# ---------------------------------------------------------------------------


class CheckAgentConfig(BaseModel):
    """Root configuration model for checkagent.yml."""

    version: int = 1
    asyncio_mode: str = "auto"
    defaults: DefaultsConfig = Field(default_factory=DefaultsConfig)
    providers: dict[str, ProviderConfig] = Field(default_factory=dict)
    budget: BudgetConfig = Field(default_factory=BudgetConfig)
    quality_gates: dict[str, QualityGateEntry] = Field(default_factory=dict)
    cassettes: CassettesConfig = Field(default_factory=CassettesConfig)
    safety: SafetyConfig = Field(default_factory=SafetyConfig)
    pii: PIIConfig = Field(default_factory=PIIConfig)
    plugins: PluginsConfig = Field(default_factory=PluginsConfig)

    @field_validator("asyncio_mode")
    @classmethod
    def _valid_asyncio_mode(cls, v: str) -> str:
        valid = {"auto", "strict"}
        if v not in valid:
            msg = f"Invalid asyncio_mode '{v}'. Must be one of: {', '.join(sorted(valid))}"
            raise ValueError(msg)
        return v


# ---------------------------------------------------------------------------
# File discovery and loading
# ---------------------------------------------------------------------------

CONFIG_FILENAMES = ("checkagent.yml", "checkagent.yaml", "checkagent.toml")


def find_config(start: Path | None = None) -> Path | None:
    """Walk up from *start* (default: cwd) looking for a config file.

    Returns the path to the first config file found, or None.
    """
    current = (start or Path.cwd()).resolve()
    for directory in (current, *current.parents):
        for name in CONFIG_FILENAMES:
            candidate = directory / name
            if candidate.is_file():
                return candidate
    return None


def load_config(path: Path | None = None) -> CheckAgentConfig:
    """Load and validate a CheckAgent config file.

    If *path* is None, searches for a config file starting from cwd.
    If no config file is found, returns the default configuration.
    """
    if path is None:
        path = find_config()

    if path is None:
        return CheckAgentConfig()

    return _load_file(path)


def _load_file(path: Path) -> CheckAgentConfig:
    """Load a config file from disk and validate it."""
    suffix = path.suffix.lower()
    text = path.read_text(encoding="utf-8")

    if suffix in (".yml", ".yaml"):
        data = yaml.safe_load(text)
    elif suffix == ".toml":
        try:
            import tomllib  # Python 3.11+
        except ImportError:
            import tomli as tomllib  # type: ignore[no-redef]
        data = tomllib.loads(text)
    else:
        msg = f"Unsupported config file format: {path.name}"
        raise ValueError(msg)

    if data is None:
        # Empty file — use defaults
        return CheckAgentConfig()

    if not isinstance(data, dict):
        msg = f"Config file must be a YAML/TOML mapping, got {type(data).__name__}"
        raise ValueError(msg)

    return CheckAgentConfig(**data)
