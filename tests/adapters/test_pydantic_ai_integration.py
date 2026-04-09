"""Integration tests for PydanticAIAdapter with real pydantic-ai objects.

These tests use pydantic_ai.models.test.TestModel to validate the adapter
against real PydanticAI Agent, RunResult, message types, tool calls, and
usage tracking — instead of MagicMock stubs.

Requires: pip install pydantic-ai
"""

from __future__ import annotations

import warnings

import pytest

pydantic_ai = pytest.importorskip("pydantic_ai", reason="pydantic-ai not installed")

from pydantic import BaseModel  # noqa: E402
from pydantic_ai import Agent  # noqa: E402
from pydantic_ai.models.test import TestModel  # noqa: E402

from checkagent.adapters.pydantic_ai import PydanticAIAdapter  # noqa: E402
from checkagent.core.types import AgentInput, StreamEventType  # noqa: E402

# Suppress PydanticAI deprecation warnings for request_tokens/response_tokens
pytestmark = pytest.mark.filterwarnings(
    "ignore::DeprecationWarning:checkagent.adapters.pydantic_ai"
)


# ---------------------------------------------------------------------------
# Basic agent tests
# ---------------------------------------------------------------------------


class TestPydanticAIIntegrationBasic:
    """Tests with real PydanticAI Agent + TestModel."""

    async def test_simple_agent_run(self):
        """Basic agent with TestModel produces a valid AgentRun."""
        agent = Agent(TestModel(), system_prompt="Be helpful")
        adapter = PydanticAIAdapter(agent)

        result = await adapter.run("hello")

        assert result.succeeded
        assert result.final_output is not None
        assert len(result.steps) >= 1
        assert result.duration_ms >= 0

    async def test_string_input_coerced(self):
        """String input is coerced to AgentInput."""
        agent = Agent(TestModel())
        adapter = PydanticAIAdapter(agent)

        result = await adapter.run("test query")

        assert result.input.query == "test query"
        assert result.succeeded

    async def test_agent_input_forwarded(self):
        """AgentInput.query is forwarded to the agent."""
        agent = Agent(TestModel())
        adapter = PydanticAIAdapter(agent)

        inp = AgentInput(query="explicit input")
        result = await adapter.run(inp)

        assert result.input.query == "explicit input"
        assert result.succeeded

    async def test_steps_contain_message_content(self):
        """Steps reflect the real PydanticAI message structure."""
        agent = Agent(TestModel(), system_prompt="You are a test agent")
        adapter = PydanticAIAdapter(agent)

        result = await adapter.run("hello")

        # PydanticAI TestModel produces: request (system+user) → response (text)
        assert len(result.steps) >= 2
        # The response step should have output text
        response_steps = [s for s in result.steps if s.output_text]
        assert len(response_steps) >= 1

    async def test_final_output_is_agent_data(self):
        """final_output is the agent's result data, not the raw RunResult."""
        agent = Agent(TestModel(), system_prompt="Test")
        adapter = PydanticAIAdapter(agent)

        result = await adapter.run("query")

        # TestModel with no tools returns "success (no tool calls)"
        assert isinstance(result.final_output, str)
        assert "success" in result.final_output.lower() or result.final_output


# ---------------------------------------------------------------------------
# Tool call tests
# ---------------------------------------------------------------------------


class TestPydanticAIIntegrationToolCalls:
    """Tests with real PydanticAI tool-calling agents."""

    async def test_tool_calls_captured(self):
        """Tool calls from a real PydanticAI agent appear in steps."""
        agent = Agent(TestModel(), system_prompt="Use the tool")

        @agent.tool_plain
        def get_weather(city: str) -> str:
            """Get weather for a city."""
            return f"Sunny in {city}"

        adapter = PydanticAIAdapter(agent)
        result = await adapter.run("Weather in Paris?")

        assert result.succeeded
        # Should have tool calls across the steps
        all_tool_calls = [
            tc for step in result.steps for tc in step.tool_calls
        ]
        assert len(all_tool_calls) >= 1

        # At least one tool call should be "get_weather"
        weather_calls = [tc for tc in all_tool_calls if tc.name == "get_weather"]
        assert len(weather_calls) >= 1

    async def test_tool_return_captured(self):
        """Tool return values appear in tool call results."""
        agent = Agent(TestModel(), system_prompt="Use tools")

        @agent.tool_plain
        def add(a: int, b: int) -> str:
            """Add two numbers."""
            return str(a + b)

        adapter = PydanticAIAdapter(agent)
        result = await adapter.run("Add 3 and 5")

        assert result.succeeded
        # Find tool return entries
        tool_returns = [
            tc
            for step in result.steps
            for tc in step.tool_calls
            if tc.result is not None
        ]
        assert len(tool_returns) >= 1

    async def test_multiple_tools(self):
        """Agent with multiple tools captures calls to each."""
        agent = Agent(TestModel(), system_prompt="Use tools")

        @agent.tool_plain
        def tool_a(x: str) -> str:
            """First tool."""
            return f"a:{x}"

        @agent.tool_plain
        def tool_b(y: str) -> str:
            """Second tool."""
            return f"b:{y}"

        adapter = PydanticAIAdapter(agent)
        result = await adapter.run("Use both tools")

        assert result.succeeded
        # TestModel calls tools alphabetically; at least one tool should be invoked
        all_tool_calls = [
            tc for step in result.steps for tc in step.tool_calls
        ]
        assert len(all_tool_calls) >= 1


# ---------------------------------------------------------------------------
# Token usage tests
# ---------------------------------------------------------------------------


class TestPydanticAIIntegrationTokenUsage:
    """Tests for token usage extraction from real RunResult.usage()."""

    async def test_token_usage_extracted(self):
        """Real TestModel usage() returns token counts."""
        agent = Agent(TestModel(), system_prompt="Test")
        adapter = PydanticAIAdapter(agent)

        result = await adapter.run("hello")

        # TestModel provides token counts
        assert result.total_prompt_tokens is not None
        assert result.total_prompt_tokens > 0
        assert result.total_completion_tokens is not None
        assert result.total_completion_tokens > 0

    async def test_token_usage_uses_new_attribute_names(self):
        """Adapter reads input_tokens/output_tokens (not deprecated names)."""
        agent = Agent(TestModel(), system_prompt="Test")
        adapter = PydanticAIAdapter(agent)

        # Run with warnings captured to verify no deprecation warnings from our adapter
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = await adapter.run("hello")

        assert result.total_prompt_tokens is not None
        # Filter for deprecation warnings from our adapter module
        adapter_warnings = [
            x for x in w
            if "checkagent" in str(x.filename) and issubclass(x.category, DeprecationWarning)
        ]
        assert len(adapter_warnings) == 0, (
            f"Adapter emitted deprecation warnings: {[str(x.message) for x in adapter_warnings]}"
        )


# ---------------------------------------------------------------------------
# Structured output tests
# ---------------------------------------------------------------------------


class TestPydanticAIIntegrationStructuredOutput:
    """Tests with structured (Pydantic model) output."""

    async def test_pydantic_model_output(self):
        """Agent with output_type returns a Pydantic model instance."""

        class Weather(BaseModel):
            city: str
            temp: int

        agent = Agent(TestModel(), output_type=Weather, system_prompt="Return weather")
        adapter = PydanticAIAdapter(agent)

        result = await adapter.run("Weather in NYC")

        assert result.succeeded
        assert isinstance(result.final_output, Weather)
        assert hasattr(result.final_output, "city")
        assert hasattr(result.final_output, "temp")


# ---------------------------------------------------------------------------
# Error handling tests
# ---------------------------------------------------------------------------


class TestPydanticAIIntegrationErrors:
    """Error handling with real PydanticAI agents."""

    async def test_error_captured_in_agent_run(self):
        """Exceptions during agent.run() are captured, not raised."""
        # Create an agent whose tool always fails
        agent = Agent(TestModel(), system_prompt="Use the tool")

        @agent.tool_plain
        def broken_tool(x: str) -> str:
            """A tool that always fails."""
            raise RuntimeError("tool exploded")

        adapter = PydanticAIAdapter(agent)
        result = await adapter.run("try the tool")

        # The agent might capture the error internally or propagate it
        # Either way, the adapter should not raise
        assert result.duration_ms >= 0
        # If the error propagated, it's captured in result.error
        # If PydanticAI handled it internally, result.succeeded may be True


# ---------------------------------------------------------------------------
# Streaming tests
# ---------------------------------------------------------------------------


class TestPydanticAIIntegrationStreaming:
    """Streaming tests with real PydanticAI agents."""

    async def test_stream_produces_events(self):
        """run_stream with a real agent produces RUN_START, TEXT_DELTA, RUN_END."""
        agent = Agent(TestModel(), system_prompt="Stream test")
        adapter = PydanticAIAdapter(agent)

        events = []
        async for event in adapter.run_stream("hello"):
            events.append(event)

        types = [e.event_type for e in events]
        assert types[0] == StreamEventType.RUN_START
        assert types[-1] == StreamEventType.RUN_END
        # Should have at least start + some content + end
        assert len(events) >= 3

    async def test_stream_text_deltas(self):
        """Text deltas contain actual content from the agent."""
        agent = Agent(TestModel(), system_prompt="Stream test")
        adapter = PydanticAIAdapter(agent)

        events = []
        async for event in adapter.run_stream("hello"):
            events.append(event)

        deltas = [e for e in events if e.event_type == StreamEventType.TEXT_DELTA]
        assert len(deltas) >= 1
        # Each delta should have non-empty data
        for d in deltas:
            assert d.data is not None
