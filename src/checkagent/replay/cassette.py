"""Cassette data model for record-and-replay testing.

A cassette stores recorded LLM and tool interactions from an agent run,
enabling deterministic replay without live API calls.

Implements: F2.1 (Recording), F2.6 (Cassette Versioning), F2.8 (Git-Friendly).
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

# Current cassette schema version — bump when format changes.
CASSETTE_SCHEMA_VERSION = 1


class CassetteMeta(BaseModel):
    """Metadata block for a cassette file (_meta in JSON)."""

    schema_version: int = CASSETTE_SCHEMA_VERSION
    checkagent_version: str = ""
    recorded_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    content_hash: str = ""
    test_id: str = ""


class RecordedRequest(BaseModel):
    """A serialized outbound request (LLM or tool call)."""

    kind: str  # "llm" or "tool"
    method: str = ""  # e.g. "chat.completions.create"
    model: str | None = None
    body: dict[str, Any] = Field(default_factory=dict)


class RecordedResponse(BaseModel):
    """A serialized response (LLM or tool result)."""

    status: str = "ok"  # "ok" or "error"
    body: Any = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    duration_ms: float | None = None


class Interaction(BaseModel):
    """A single request-response pair within a cassette."""

    id: str = ""
    sequence: int = 0
    request: RecordedRequest
    response: RecordedResponse
    metadata: dict[str, Any] = Field(default_factory=dict)

    def compute_id(self) -> str:
        """Deterministic ID from method + normalized request body."""
        raw = json.dumps(
            {"method": self.request.method, "body": self.request.body},
            sort_keys=True,
            default=str,
        )
        return hashlib.sha256(raw.encode()).hexdigest()[:16]


class Cassette(BaseModel):
    """A complete cassette file: metadata + ordered interactions.

    Cassettes are JSON files stored at:
        cassettes/{test_id}/{short_hash}.json

    The short_hash is derived from the content for git-friendliness (F2.8).
    """

    meta: CassetteMeta = Field(default_factory=CassetteMeta)
    interactions: list[Interaction] = Field(default_factory=list)

    # --- Content hashing (F2.6) ---

    def compute_content_hash(self) -> str:
        """SHA-256 hash of the interactions payload."""
        raw = json.dumps(
            [i.model_dump() for i in self.interactions],
            sort_keys=True,
            default=str,
        )
        return hashlib.sha256(raw.encode()).hexdigest()

    def short_hash(self) -> str:
        """First 12 chars of content hash, used in filenames."""
        return self.compute_content_hash()[:12]

    def finalize(self) -> None:
        """Assign interaction IDs, sequence numbers, and content hash."""
        for idx, interaction in enumerate(self.interactions):
            interaction.sequence = idx
            if not interaction.id:
                interaction.id = interaction.compute_id()
        self.meta.content_hash = self.compute_content_hash()

    # --- Serialization ---

    def to_json(self, indent: int = 2) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.model_dump(), indent=indent, default=str)

    @classmethod
    def from_json(cls, data: str) -> Cassette:
        """Deserialize from JSON string."""
        return cls.model_validate(json.loads(data))

    # --- File I/O ---

    def save(self, path: Path) -> Path:
        """Write cassette to a JSON file, creating parent dirs."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_json())
        return path

    @classmethod
    def load(cls, path: Path) -> Cassette:
        """Load a cassette from a JSON file.

        Raises a warning if schema version is outdated.
        """
        cassette = cls.from_json(path.read_text())
        if cassette.meta.schema_version < CASSETTE_SCHEMA_VERSION:
            import warnings

            warnings.warn(
                f"Cassette {path.name} uses schema v"
                f"{cassette.meta.schema_version}, "
                f"current is v{CASSETTE_SCHEMA_VERSION}. "
                f"Run 'checkagent migrate-cassettes' to upgrade.",
                stacklevel=2,
            )
        return cassette

    def verify_integrity(self) -> bool:
        """Check that content hash matches stored hash."""
        if not self.meta.content_hash:
            return True  # no hash stored yet
        return self.compute_content_hash() == self.meta.content_hash

    # --- Cassette path helpers (F2.8) ---

    @staticmethod
    def cassette_path(
        base_dir: Path, test_id: str, content_hash: str
    ) -> Path:
        """Build a content-addressed cassette file path."""
        short = content_hash[:12]
        safe_id = test_id.replace("::", "/").replace(" ", "_")
        return base_dir / safe_id / f"{short}.json"


# --- Redaction (F2.1) ---

_DEFAULT_REDACT_KEYS = frozenset({
    "api_key", "api-key", "authorization", "token",
    "secret", "password", "credential", "x-api-key",
})


def redact_dict(
    d: dict[str, Any],
    keys: frozenset[str] = _DEFAULT_REDACT_KEYS,
) -> dict[str, Any]:
    """Recursively redact sensitive keys from a dict.

    Returns a new dict with matching keys replaced by '[REDACTED]'.
    """
    out: dict[str, Any] = {}
    for k, v in d.items():
        if k.lower() in keys:
            out[k] = "[REDACTED]"
        elif isinstance(v, dict):
            out[k] = redact_dict(v, keys)
        elif isinstance(v, list):
            out[k] = [
                redact_dict(item, keys) if isinstance(item, dict) else item
                for item in v
            ]
        else:
            out[k] = v
    return out
