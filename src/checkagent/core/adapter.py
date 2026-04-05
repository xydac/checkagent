"""Agent adapter protocol — the core abstraction for framework-agnostic testing."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from checkagent.core.types import AgentInput, AgentRun


@runtime_checkable
class AgentAdapter(Protocol):
    """Minimal interface any agent must implement for testing.

    All agent-facing APIs are async-first. Sync agents are supported
    via the GenericAdapter which wraps sync callables in a thread executor.

    Streaming is optional — adapters may also implement:
        async def run_stream(self, input: AgentInput) -> AsyncIterator[StreamEvent]

    If run_stream is not present, CheckAgent falls back to run() and
    synthesizes streaming events from the completed trace.
    """

    async def run(self, input: AgentInput) -> AgentRun:
        """Execute the agent and return a structured trace."""
        ...
