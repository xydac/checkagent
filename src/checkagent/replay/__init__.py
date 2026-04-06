"""Record-and-replay layer for deterministic agent testing."""

from checkagent.replay.cassette import (
    CASSETTE_SCHEMA_VERSION,
    Cassette,
    CassetteMeta,
    Interaction,
    RecordedRequest,
    RecordedResponse,
    redact_dict,
)
from checkagent.replay.engine import (
    CassetteMismatchError,
    MatchStrategy,
    ReplayEngine,
)
from checkagent.replay.migration import (
    MigrationResult,
    migrate_cassette_data,
    migrate_directory,
    migrate_file,
    register_migration,
)
from checkagent.replay.recorder import CassetteRecorder, TimedCall

__all__ = [
    "CASSETTE_SCHEMA_VERSION",
    "Cassette",
    "CassetteMeta",
    "CassetteMismatchError",
    "CassetteRecorder",
    "Interaction",
    "MatchStrategy",
    "MigrationResult",
    "RecordedRequest",
    "RecordedResponse",
    "ReplayEngine",
    "TimedCall",
    "migrate_cassette_data",
    "migrate_directory",
    "migrate_file",
    "redact_dict",
    "register_migration",
]
