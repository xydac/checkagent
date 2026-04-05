"""StreamCollector — captures streaming events with assertion helpers.

Implements F13.1 and F13.3 from the PRD.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

from checkagent.core.types import StreamEvent, StreamEventType


class StreamCollector:
    """Collects StreamEvent objects from an async iterator.

    Provides assertion helpers for inspecting collected events::

        collector = StreamCollector()
        async for event in llm.stream("hello"):
            collector.add(event)

        assert collector.aggregated_text == "Hello world!"
        assert collector.total_chunks == 2

    Or use the ``collect_from`` shortcut::

        await collector.collect_from(llm.stream("hello"))
    """

    def __init__(self) -> None:
        self._events: list[StreamEvent] = []

    def add(self, event: StreamEvent) -> None:
        """Add a single event to the collection."""
        self._events.append(event)

    async def collect_from(self, stream: AsyncIterator[StreamEvent]) -> StreamCollector:
        """Consume all events from an async iterator. Returns self."""
        async for event in stream:
            self._events.append(event)
        return self

    @property
    def events(self) -> list[StreamEvent]:
        """All collected events."""
        return list(self._events)

    @property
    def total_events(self) -> int:
        """Total number of collected events."""
        return len(self._events)

    @property
    def total_chunks(self) -> int:
        """Number of TEXT_DELTA events (content chunks)."""
        return len(self.of_type(StreamEventType.TEXT_DELTA))

    def of_type(self, event_type: StreamEventType) -> list[StreamEvent]:
        """Filter events by type."""
        return [e for e in self._events if e.event_type == event_type]

    def first_of_type(self, event_type: StreamEventType) -> StreamEvent | None:
        """Get the first event of the given type, or None."""
        for e in self._events:
            if e.event_type == event_type:
                return e
        return None

    @property
    def aggregated_text(self) -> str:
        """Concatenated text from all TEXT_DELTA events."""
        return "".join(
            e.data for e in self._events
            if e.event_type == StreamEventType.TEXT_DELTA and e.data is not None
        )

    @property
    def time_to_first_token(self) -> float | None:
        """Seconds between RUN_START and first TEXT_DELTA, or None."""
        run_start = self.first_of_type(StreamEventType.RUN_START)
        first_delta = self.first_of_type(StreamEventType.TEXT_DELTA)
        if run_start is not None and first_delta is not None:
            return first_delta.timestamp - run_start.timestamp
        return None

    def tool_call_started(self, name: str) -> bool:
        """Check if a TOOL_CALL_START event with the given name was collected."""
        return any(
            e.event_type == StreamEventType.TOOL_CALL_START
            and isinstance(e.data, dict)
            and e.data.get("name") == name
            for e in self._events
        )

    @property
    def has_error(self) -> bool:
        """Whether any ERROR event was collected."""
        return any(e.event_type == StreamEventType.ERROR for e in self._events)

    @property
    def error_events(self) -> list[StreamEvent]:
        """All ERROR events."""
        return self.of_type(StreamEventType.ERROR)

    def reset(self) -> None:
        """Clear all collected events."""
        self._events.clear()
