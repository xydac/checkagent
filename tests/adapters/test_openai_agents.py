"""Tests for the OpenAI Agents SDK adapter.

openai-agents is an optional dependency — all tests mock the framework
objects to test the adapter's conversion logic.
"""

from __future__ import annotations

import sys
import types
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from checkagent.core.types import AgentInput, StreamEventType

# ---------------------------------------------------------------------------
# Fake agents module so the import guard passes
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _fake_agents_module():
    """Inject a fake agents module for the import check."""
    fake = types.ModuleType("agents")
    fake.Runner = MagicMock()
    sys.modules["agents"] = fake
    yield
    sys.modules.pop("agents", None)


# ---------------------------------------------------------------------------
# Helpers to build mock RunResult objects
# ---------------------------------------------------------------------------

def _make_message_item(content: str = "Hello") -> MagicMock:
    """Create a mock MessageOutputItem."""
    text_part = MagicMock()
    text_part.type = "output_text"
    text_part.text = content

    raw_item = MagicMock()
    raw_item.content = [text_part]

    item = MagicMock()
    item.type = "message_output_item"
    item.raw_item = raw_item
    return item


def _make_tool_call_item(name: str = "search", arguments: str = '{"q": "test"}') -> MagicMock:
    item = MagicMock()
    item.type = "tool_call_item"
    raw = MagicMock()
    raw.name = name
    raw.arguments = arguments
    item.raw_item = raw
    return item


def _make_tool_output_item(output: str = "tool result") -> MagicMock:
    item = MagicMock()
    item.type = "tool_call_output_item"
    item.output = output
    return item


def _make_run_result(
    final_output: str = "Done",
    items: list[Any] | None = None,
    raw_responses: list[Any] | None = None,
) -> MagicMock:
    result = MagicMock()
    result.final_output = final_output
    result.new_items = items or [_make_message_item(final_output)]
    result.raw_responses = raw_responses or []
    return result


def _make_usage(input_tokens: int = 10, output_tokens: int = 20) -> MagicMock:
    usage = MagicMock()
    usage.input_tokens = input_tokens
    usage.output_tokens = output_tokens
    return usage


# ---------------------------------------------------------------------------
# Tests: run()
# ---------------------------------------------------------------------------


class TestOpenAIAgentsAdapterRun:
    async def test_basic_run(self):
        from checkagent.adapters.openai_agents import OpenAIAgentsAdapter

        run_result = _make_run_result("Hello!")
        with patch("agents.Runner") as mock_runner:
            mock_runner.run = AsyncMock(return_value=run_result)

            adapter = OpenAIAgentsAdapter(MagicMock())
            result = await adapter.run("test query")

        assert result.succeeded
        assert result.final_output == "Hello!"
        assert len(result.steps) == 1
        assert result.steps[0].output_text == "Hello!"

    async def test_string_input_converted(self):
        from checkagent.adapters.openai_agents import OpenAIAgentsAdapter

        run_result = _make_run_result("ok")
        with patch("agents.Runner") as mock_runner:
            mock_runner.run = AsyncMock(return_value=run_result)

            adapter = OpenAIAgentsAdapter(MagicMock())
            result = await adapter.run("plain string")

        assert result.succeeded
        assert result.input.query == "plain string"

    async def test_agent_input(self):
        from checkagent.adapters.openai_agents import OpenAIAgentsAdapter

        run_result = _make_run_result("result")
        with patch("agents.Runner") as mock_runner:
            mock_runner.run = AsyncMock(return_value=run_result)

            adapter = OpenAIAgentsAdapter(MagicMock())
            inp = AgentInput(query="hello", context={"key": "val"})
            result = await adapter.run(inp)

        assert result.input.query == "hello"
        mock_runner.run.assert_awaited_once()

    async def test_tool_calls_extraction(self):
        from checkagent.adapters.openai_agents import OpenAIAgentsAdapter

        items = [
            _make_message_item("Let me search"),
            _make_tool_call_item("search", '{"query": "python"}'),
            _make_tool_output_item("search results here"),
        ]
        run_result = _make_run_result("Found it", items=items)

        with patch("agents.Runner") as mock_runner:
            mock_runner.run = AsyncMock(return_value=run_result)

            adapter = OpenAIAgentsAdapter(MagicMock())
            result = await adapter.run("search for python")

        # Message creates step 0, tool call attaches to it
        all_tool_calls = result.tool_calls
        assert len(all_tool_calls) == 1
        assert all_tool_calls[0].name == "search"
        assert all_tool_calls[0].arguments == {"query": "python"}

    async def test_tool_output_attached(self):
        from checkagent.adapters.openai_agents import OpenAIAgentsAdapter

        items = [
            _make_message_item("checking"),
            _make_tool_call_item("calc", '{"x": 1}'),
            _make_tool_output_item("42"),
        ]
        run_result = _make_run_result("answer is 42", items=items)

        with patch("agents.Runner") as mock_runner:
            mock_runner.run = AsyncMock(return_value=run_result)

            adapter = OpenAIAgentsAdapter(MagicMock())
            result = await adapter.run("what is 1+1")

        tc = result.tool_calls[0]
        assert tc.result == "42"

    async def test_token_usage(self):
        from checkagent.adapters.openai_agents import OpenAIAgentsAdapter

        resp = MagicMock()
        resp.usage = _make_usage(100, 200)
        run_result = _make_run_result("response", raw_responses=[resp])

        with patch("agents.Runner") as mock_runner:
            mock_runner.run = AsyncMock(return_value=run_result)

            adapter = OpenAIAgentsAdapter(MagicMock())
            result = await adapter.run("query")

        assert result.total_prompt_tokens == 100
        assert result.total_completion_tokens == 200

    async def test_multiple_responses_aggregate_tokens(self):
        from checkagent.adapters.openai_agents import OpenAIAgentsAdapter

        resp1 = MagicMock()
        resp1.usage = _make_usage(50, 80)
        resp2 = MagicMock()
        resp2.usage = _make_usage(30, 40)
        run_result = _make_run_result("done", raw_responses=[resp1, resp2])

        with patch("agents.Runner") as mock_runner:
            mock_runner.run = AsyncMock(return_value=run_result)

            adapter = OpenAIAgentsAdapter(MagicMock())
            result = await adapter.run("multi-step")

        assert result.total_prompt_tokens == 80
        assert result.total_completion_tokens == 120

    async def test_error_handling(self):
        from checkagent.adapters.openai_agents import OpenAIAgentsAdapter

        with patch("agents.Runner") as mock_runner:
            mock_runner.run = AsyncMock(side_effect=RuntimeError("agent crashed"))

            adapter = OpenAIAgentsAdapter(MagicMock())
            result = await adapter.run("oops")

        assert not result.succeeded
        assert "RuntimeError: agent crashed" in result.error
        assert result.duration_ms > 0

    async def test_duration_recorded(self):
        from checkagent.adapters.openai_agents import OpenAIAgentsAdapter

        run_result = _make_run_result("fast")
        with patch("agents.Runner") as mock_runner:
            mock_runner.run = AsyncMock(return_value=run_result)

            adapter = OpenAIAgentsAdapter(MagicMock())
            result = await adapter.run("query")

        assert result.duration_ms is not None
        assert result.duration_ms >= 0


# ---------------------------------------------------------------------------
# Tests: run_stream()
# ---------------------------------------------------------------------------


class TestOpenAIAgentsAdapterStream:
    async def test_basic_stream(self):
        from checkagent.adapters.openai_agents import OpenAIAgentsAdapter

        delta_event = MagicMock()
        delta_event.type = "raw_response_event"
        delta_event.data = MagicMock()
        delta_event.data.delta = "Hello"

        tool_event = MagicMock()
        tool_event.type = "run_item_stream_event"
        tool_item = MagicMock()
        tool_item.type = "tool_call_item"
        tool_item.raw_item = MagicMock()
        tool_item.raw_item.name = "search"
        tool_event.item = tool_item

        async def fake_stream_events():
            yield delta_event
            yield tool_event

        stream_result = MagicMock()
        stream_result.stream_events = fake_stream_events

        with patch("agents.Runner") as mock_runner:
            mock_runner.run_streamed = MagicMock(return_value=stream_result)

            adapter = OpenAIAgentsAdapter(MagicMock())
            events = []
            async for event in adapter.run_stream("query"):
                events.append(event)

        types = [e.event_type for e in events]
        assert types[0] == StreamEventType.RUN_START
        assert StreamEventType.TEXT_DELTA in types
        assert StreamEventType.TOOL_CALL_START in types
        assert types[-1] == StreamEventType.RUN_END

    async def test_stream_error_handling(self):
        from checkagent.adapters.openai_agents import OpenAIAgentsAdapter

        with patch("agents.Runner") as mock_runner:
            mock_runner.run_streamed = MagicMock(
                side_effect=RuntimeError("stream failed")
            )

            adapter = OpenAIAgentsAdapter(MagicMock())
            events = []
            async for event in adapter.run_stream("query"):
                events.append(event)

        types = [e.event_type for e in events]
        assert StreamEventType.ERROR in types
        assert types[-1] == StreamEventType.RUN_END


# ---------------------------------------------------------------------------
# Tests: import guard
# ---------------------------------------------------------------------------


class TestOpenAIAgentsImportGuard:
    async def test_import_error_without_agents(self):
        """_ensure_openai_agents raises ImportError if agents module missing."""
        from checkagent.adapters.openai_agents import _ensure_openai_agents

        # Block the import by inserting None (PEP 302 negative cache)
        saved = sys.modules.pop("agents", None)
        sys.modules["agents"] = None  # type: ignore[assignment]
        try:
            with pytest.raises(ImportError):
                _ensure_openai_agents()
        finally:
            if saved is not None:
                sys.modules["agents"] = saved
            else:
                sys.modules.pop("agents", None)
