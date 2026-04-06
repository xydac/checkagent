"""Base protocol for trace importers.

Requirements: F6.2
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from checkagent.core.types import AgentRun


@runtime_checkable
class TraceImporter(Protocol):
    """Protocol for importing production traces into AgentRun format.

    Implementations normalize provider-specific trace schemas
    (Langfuse Generation, Phoenix Span, OTel SpanData, raw JSON)
    into CheckAgent's AgentRun type.
    """

    def import_traces(
        self,
        source: str,
        *,
        filters: dict[str, Any] | None = None,
        limit: int | None = None,
    ) -> list[AgentRun]:
        """Import traces from the given source.

        Args:
            source: Path to file, directory, or API endpoint.
            filters: Optional key-value filters (e.g. status=error).
            limit: Max number of traces to import.

        Returns:
            List of normalized AgentRun objects.
        """
        ...
