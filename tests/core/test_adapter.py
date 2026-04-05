"""Tests for checkagent.core.adapter — the AgentAdapter protocol."""

from typing import AsyncIterator

from checkagent.core.adapter import AgentAdapter
from checkagent.core.types import AgentInput, AgentRun, StreamEvent, StreamEventType


class ConcreteAdapter:
    """A minimal concrete implementation for protocol testing."""

    async def run(self, input: AgentInput) -> AgentRun:
        return AgentRun(input=input, final_output="ok")

    async def run_stream(self, input: AgentInput) -> AsyncIterator[StreamEvent]:
        yield StreamEvent(event_type=StreamEventType.RUN_END)


class RunOnlyAdapter:
    """Adapter that only implements run(), not run_stream()."""

    async def run(self, input: AgentInput) -> AgentRun:
        return AgentRun(input=input, final_output="ok")


class TestAgentAdapterProtocol:
    def test_concrete_adapter_is_instance(self):
        adapter = ConcreteAdapter()
        assert isinstance(adapter, AgentAdapter)

    def test_run_only_adapter_is_instance(self):
        """run_stream is optional — adapters with only run() still conform."""
        adapter = RunOnlyAdapter()
        assert isinstance(adapter, AgentAdapter)

    def test_plain_object_is_not_adapter(self):
        assert not isinstance("not an adapter", AgentAdapter)
        assert not isinstance(42, AgentAdapter)

    async def test_concrete_adapter_run(self):
        adapter = ConcreteAdapter()
        inp = AgentInput(query="test")
        result = await adapter.run(inp)
        assert result.succeeded is True
        assert result.final_output == "ok"

    async def test_concrete_adapter_stream(self):
        adapter = ConcreteAdapter()
        inp = AgentInput(query="test")
        events = [e async for e in adapter.run_stream(inp)]
        assert len(events) == 1
        assert events[0].event_type == StreamEventType.RUN_END
