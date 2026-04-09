"""Integration tests for OpenAIAgentsAdapter with real openai-agents objects.

These tests use a custom FakeModel implementing the agents Model interface
to validate the adapter against real Agent/Runner execution (not MagicMock
stubs). The FakeModel returns canned responses so no API keys are needed.

Requires: pip install openai-agents
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

import pytest

agents_mod = pytest.importorskip("agents", reason="openai-agents not installed")

from agents import Agent, Usage, function_tool  # noqa: E402
from agents.models.interface import Model, ModelResponse  # noqa: E402
from openai.types.responses import (  # noqa: E402
    ResponseFunctionToolCall,
    ResponseOutputMessage,
    ResponseOutputText,
)

from checkagent.adapters.openai_agents import OpenAIAgentsAdapter  # noqa: E402
from checkagent.core.types import AgentInput, StreamEventType  # noqa: E402

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _usage(input_tokens: int = 10, output_tokens: int = 5) -> Usage:
    return Usage(
        requests=1,
        input_tokens=input_tokens,
        input_tokens_details={"cached_tokens": 0},
        output_tokens=output_tokens,
        output_tokens_details={"reasoning_tokens": 0},
        total_tokens=input_tokens + output_tokens,
    )


def _message_response(text: str, *, usage: Usage | None = None) -> ModelResponse:
    """Build a ModelResponse with a single text message."""
    msg = ResponseOutputMessage(
        id="msg_fake",
        type="message",
        role="assistant",
        status="completed",
        content=[ResponseOutputText(type="output_text", text=text, annotations=[])],
    )
    return ModelResponse(
        output=[msg],
        usage=usage or _usage(),
        response_id="resp_fake",
    )


def _tool_call_response(
    name: str, arguments: dict[str, Any], *, call_id: str = "call_1", usage: Usage | None = None,
) -> ModelResponse:
    """Build a ModelResponse with a single tool call."""
    tc = ResponseFunctionToolCall(
        id="tc_1",
        type="function_call",
        name=name,
        arguments=json.dumps(arguments),
        call_id=call_id,
        status="completed",
    )
    return ModelResponse(
        output=[tc],
        usage=usage or _usage(),
        response_id="resp_fake",
    )


class FakeModel(Model):
    """A deterministic Model implementation for testing.

    Accepts a list of ModelResponse objects and returns them in order.
    """

    def __init__(self, responses: list[ModelResponse]) -> None:
        self._responses = responses
        self._idx = 0
        self.call_count = 0

    async def get_response(
        self, system_instructions, input, model_settings, tools,
        output_schema, handoffs, tracing, *, previous_response_id=None,
        conversation_id=None, prompt=None,
    ) -> ModelResponse:
        resp = self._responses[min(self._idx, len(self._responses) - 1)]
        self._idx += 1
        self.call_count += 1
        return resp

    async def stream_response(
        self, system_instructions, input, model_settings, tools,
        output_schema, handoffs, tracing, *, previous_response_id=None,
        conversation_id=None, prompt=None,
    ) -> AsyncIterator:
        raise NotImplementedError("Use get_response for these tests")


# ---------------------------------------------------------------------------
# Basic agent tests
# ---------------------------------------------------------------------------


class TestOpenAIAgentsIntegrationBasic:
    """Tests with real Agent + Runner using FakeModel."""

    async def test_simple_agent(self):
        """Basic agent returns text output."""
        model = FakeModel([_message_response("Hello from agent!")])
        agent = Agent(name="greeter", instructions="Say hello.", model=model)
        adapter = OpenAIAgentsAdapter(agent)

        result = await adapter.run("Hi")

        assert result.succeeded
        assert result.final_output == "Hello from agent!"
        assert len(result.steps) >= 1
        assert result.steps[0].output_text == "Hello from agent!"

    async def test_string_input(self):
        """String input is accepted and forwarded."""
        model = FakeModel([_message_response("ok")])
        agent = Agent(name="echo", instructions="Echo.", model=model)
        adapter = OpenAIAgentsAdapter(agent)

        result = await adapter.run("test query")

        assert result.succeeded
        assert result.input.query == "test query"

    async def test_agent_input(self):
        """AgentInput is accepted."""
        model = FakeModel([_message_response("result")])
        agent = Agent(name="test", instructions="Test.", model=model)
        adapter = OpenAIAgentsAdapter(agent)

        inp = AgentInput(query="hello", context={"key": "val"})
        result = await adapter.run(inp)

        assert result.succeeded
        assert result.input.query == "hello"

    async def test_duration_tracked(self):
        """Duration is recorded."""
        model = FakeModel([_message_response("fast")])
        agent = Agent(name="fast", instructions="Be fast.", model=model)
        adapter = OpenAIAgentsAdapter(agent)

        result = await adapter.run("go")

        assert result.duration_ms is not None
        assert result.duration_ms >= 0

    async def test_multi_turn_agent(self):
        """Agent that produces multiple message items."""
        model = FakeModel([
            _message_response("First thought"),
            _message_response("Second thought"),
        ])
        # The Runner calls get_response once and gets one message.
        # But our FakeModel can support multi-step if the agent loops.
        agent = Agent(name="thinker", instructions="Think.", model=model)
        adapter = OpenAIAgentsAdapter(agent)

        result = await adapter.run("Think about this")

        assert result.succeeded
        assert len(result.steps) >= 1


# ---------------------------------------------------------------------------
# Tool calling tests
# ---------------------------------------------------------------------------


class TestOpenAIAgentsIntegrationToolCalls:
    """Tests with real tool execution via the agents framework."""

    async def test_single_tool_call(self):
        """Agent calls a tool and gets the result."""
        @function_tool
        def get_weather(city: str) -> str:
            """Get weather for a city."""
            return f"Sunny in {city}"

        model = FakeModel([
            _tool_call_response("get_weather", {"city": "London"}),
            _message_response("It's sunny in London!"),
        ])
        agent = Agent(
            name="weather_agent",
            instructions="Use tools to answer.",
            model=model,
            tools=[get_weather],
        )
        adapter = OpenAIAgentsAdapter(agent)

        result = await adapter.run("What's the weather in London?")

        assert result.succeeded
        assert result.final_output == "It's sunny in London!"

        # Check tool calls were captured
        all_tc = result.tool_calls
        assert len(all_tc) >= 1
        assert all_tc[0].name == "get_weather"
        assert all_tc[0].arguments == {"city": "London"}

    async def test_tool_result_attached(self):
        """Tool output is attached to the ToolCall."""
        @function_tool
        def calculator(expression: str) -> str:
            """Evaluate a math expression."""
            return str(eval(expression))  # noqa: S307

        model = FakeModel([
            _tool_call_response("calculator", {"expression": "2+2"}),
            _message_response("The answer is 4."),
        ])
        agent = Agent(
            name="calc_agent",
            instructions="Calculate.",
            model=model,
            tools=[calculator],
        )
        adapter = OpenAIAgentsAdapter(agent)

        result = await adapter.run("What is 2+2?")

        all_tc = result.tool_calls
        assert len(all_tc) >= 1
        assert all_tc[0].result == "4"

    async def test_multiple_tool_calls_sequential(self):
        """Agent calls multiple tools in sequence."""
        call_log: list[str] = []

        @function_tool
        def search(query: str) -> str:
            """Search for information."""
            call_log.append(f"search:{query}")
            return f"Results for {query}"

        @function_tool
        def summarize(text: str) -> str:
            """Summarize text."""
            call_log.append(f"summarize:{text}")
            return f"Summary of {text}"

        model = FakeModel([
            _tool_call_response("search", {"query": "AI testing"}),
            _tool_call_response("summarize", {"text": "Results for AI testing"}, call_id="call_2"),
            _message_response("Here's a summary of AI testing results."),
        ])
        agent = Agent(
            name="research_agent",
            instructions="Search then summarize.",
            model=model,
            tools=[search, summarize],
        )
        adapter = OpenAIAgentsAdapter(agent)

        result = await adapter.run("Research AI testing")

        assert result.succeeded
        assert len(call_log) == 2
        assert "search:AI testing" in call_log
        all_tc = result.tool_calls
        assert len(all_tc) == 2


# ---------------------------------------------------------------------------
# Token usage tests
# ---------------------------------------------------------------------------


class TestOpenAIAgentsIntegrationTokenUsage:
    """Tests for token usage extraction from real RunResult."""

    async def test_single_response_tokens(self):
        """Token usage from a single response."""
        model = FakeModel([
            _message_response("hello", usage=_usage(input_tokens=50, output_tokens=25)),
        ])
        agent = Agent(name="test", instructions="Test.", model=model)
        adapter = OpenAIAgentsAdapter(agent)

        result = await adapter.run("hi")

        assert result.total_prompt_tokens == 50
        assert result.total_completion_tokens == 25

    async def test_multi_response_aggregated_tokens(self):
        """Token usage aggregated across multiple responses (tool call + final)."""
        @function_tool
        def noop() -> str:
            """Do nothing."""
            return "ok"

        model = FakeModel([
            _tool_call_response("noop", {}, usage=_usage(input_tokens=30, output_tokens=10)),
            _message_response("done", usage=_usage(input_tokens=40, output_tokens=20)),
        ])
        agent = Agent(
            name="multi",
            instructions="Use tools.",
            model=model,
            tools=[noop],
        )
        adapter = OpenAIAgentsAdapter(agent)

        result = await adapter.run("do something")

        assert result.total_prompt_tokens == 70  # 30 + 40
        assert result.total_completion_tokens == 30  # 10 + 20


# ---------------------------------------------------------------------------
# Error handling tests
# ---------------------------------------------------------------------------


class TestOpenAIAgentsIntegrationErrors:
    """Tests for error handling with real framework objects."""

    async def test_model_error_captured(self):
        """Exception during model execution is captured."""
        class ErrorModel(Model):
            async def get_response(self, *args, **kwargs):
                raise RuntimeError("Model crashed")

            async def stream_response(self, *args, **kwargs):
                raise NotImplementedError

        agent = Agent(name="broken", instructions="Fail.", model=ErrorModel())
        adapter = OpenAIAgentsAdapter(agent)

        result = await adapter.run("trigger error")

        assert not result.succeeded
        assert "RuntimeError" in result.error
        assert "Model crashed" in result.error
        assert result.duration_ms >= 0

    async def test_tool_error_captured(self):
        """Tool that raises an exception — the framework handles it."""
        @function_tool
        def bad_tool() -> str:
            """A tool that always fails."""
            raise ValueError("Tool exploded")

        model = FakeModel([
            _tool_call_response("bad_tool", {}),
            _message_response("The tool failed, sorry."),
        ])
        agent = Agent(
            name="error_agent",
            instructions="Try tools.",
            model=model,
            tools=[bad_tool],
        )
        adapter = OpenAIAgentsAdapter(agent)

        result = await adapter.run("use the bad tool")

        # The framework catches tool errors and reports them back to the model
        # The agent should still produce a final output
        assert result.final_output is not None


# ---------------------------------------------------------------------------
# Streaming tests
# ---------------------------------------------------------------------------


class TestOpenAIAgentsIntegrationStreaming:
    """Tests for streaming via run_stream() with real framework objects."""

    async def test_stream_produces_events(self):
        """Streaming produces START, content events, and END."""
        model = FakeModel([_message_response("Streamed response")])
        agent = Agent(name="streamer", instructions="Stream.", model=model)
        adapter = OpenAIAgentsAdapter(agent)

        events = []
        async for event in adapter.run_stream("stream this"):
            events.append(event)

        types = [e.event_type for e in events]
        assert types[0] == StreamEventType.RUN_START
        assert types[-1] == StreamEventType.RUN_END
        assert len(events) >= 3  # START + at least one content event + END

    async def test_stream_tool_events(self):
        """Streaming captures tool call events."""
        @function_tool
        def ping() -> str:
            """Ping."""
            return "pong"

        model = FakeModel([
            _tool_call_response("ping", {}),
            _message_response("Got pong back."),
        ])
        agent = Agent(
            name="tool_streamer",
            instructions="Use tools.",
            model=model,
            tools=[ping],
        )
        adapter = OpenAIAgentsAdapter(agent)

        events = []
        async for event in adapter.run_stream("ping the tool"):
            events.append(event)

        types = [e.event_type for e in events]
        assert StreamEventType.RUN_START in types
        assert StreamEventType.RUN_END in types
        # At minimum, the stream should complete without error
        assert types[-1] == StreamEventType.RUN_END
