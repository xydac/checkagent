"""Agent adapter protocol — the core abstraction for framework-agnostic testing."""

from __future__ import annotations

from typing import AsyncIterator, Protocol, runtime_checkable

from checkagent.core.types import AgentInput, AgentRun, StreamEvent


@runtime_checkable
class AgentAdapter(Protocol):
    """Minimal interface any agent must implement for testing.

    All agent-facing APIs are async-first. Sync agents are supported
    via the GenericAdapter which wraps sync callables in a thread executor.
    """

    async def run(self, input: AgentInput) -> AgentRun:
        """Execute the agent and return a structured trace."""
        ...

    async def run_stream(self, input: AgentInput) -> AsyncIterator[StreamEvent]:
        """Execute the agent and yield streaming events.

        Optional. If not implemented, CheckAgent falls back to run()
        and synthesizes events from the completed trace.
        """
        ...
