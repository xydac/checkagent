"""Tests for the PydanticAI adapter.

pydantic-ai is an optional dependency — all tests mock the framework
objects to test the adapter's conversion logic.
"""

from __future__ import annotations

import sys
import types
from unittest.mock import AsyncMock, MagicMock

import pytest

from checkagent.core.types import AgentInput, StreamEventType

# ---------------------------------------------------------------------------
# Fake pydantic_ai module so the import guard passes
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _fake_pydantic_ai_module():
    """Inject a fake pydantic_ai module for the import check."""
    fake = types.ModuleType("pydantic_ai")
    sys.modules["pydantic_ai"] = fake
    yield
    sys.modules.pop("pydantic_ai", None)


# ---------------------------------------------------------------------------
# Helpers to build mock RunResult objects
# ---------------------------------------------------------------------------


def _make_text_part(content: str = "hello") -> MagicMock:
    part = MagicMock()
    part.part_kind = "text"
    part.content = content
    return part


def _make_tool_call_part(name: str = "search", args: dict | None = None) -> MagicMock:
    part = MagicMock()
    part.part_kind = "tool-call"
    part.tool_name = name
    part.args = args or {}
    return part


def _make_tool_return_part(name: str = "search", content: str = "result") -> MagicMock:
    part = MagicMock()
    part.part_kind = "tool-return"
    part.tool_name = name
    part.content = content
    return part


def _make_message(kind: str = "model-response", parts: list | None = None) -> MagicMock:
    msg = MagicMock()
    msg.kind = kind
    msg.parts = parts or []
    return msg


def _make_usage(request_tokens: int = 100, response_tokens: int = 50) -> MagicMock:
    usage = MagicMock()
    usage.request_tokens = request_tokens
    usage.response_tokens = response_tokens
    return usage


def _make_run_result(
    data: str = "final answer",
    messages: list | None = None,
    usage: MagicMock | None = None,
) -> MagicMock:
    result = MagicMock()
    result.data = data
    result.output = data

    def all_messages():
        return messages or []

    result.all_messages = all_messages
    result.usage = lambda: usage if usage else _make_usage(0, 0)
    return result


# ---------------------------------------------------------------------------
# Tests: basic run
# ---------------------------------------------------------------------------


class TestPydanticAIAdapterRun:
    async def test_basic_run(self):
        from checkagent.adapters.pydantic_ai import PydanticAIAdapter

        result = _make_run_result(data="hello world")
        agent = MagicMock()
        agent.run = AsyncMock(return_value=result)

        adapter = PydanticAIAdapter(agent)
        run = await adapter.run("hi")

        assert run.succeeded
        assert run.final_output == "hello world"

    async def test_string_to_agent_input(self):
        from checkagent.adapters.pydantic_ai import PydanticAIAdapter

        result = _make_run_result()
        agent = MagicMock()
        agent.run = AsyncMock(return_value=result)

        adapter = PydanticAIAdapter(agent)
        run = await adapter.run("test query")

        assert run.input.query == "test query"

    async def test_agent_input_forwarded(self):
        from checkagent.adapters.pydantic_ai import PydanticAIAdapter

        result = _make_run_result()
        agent = MagicMock()
        agent.run = AsyncMock(return_value=result)

        adapter = PydanticAIAdapter(agent)
        inp = AgentInput(query="hello")
        await adapter.run(inp)

        agent.run.assert_awaited_once_with("hello")

    async def test_steps_from_messages(self):
        from checkagent.adapters.pydantic_ai import PydanticAIAdapter

        messages = [
            _make_message("model-request", [_make_text_part("user input")]),
            _make_message("model-response", [_make_text_part("response text")]),
        ]
        result = _make_run_result(messages=messages)
        agent = MagicMock()
        agent.run = AsyncMock(return_value=result)

        adapter = PydanticAIAdapter(agent)
        run = await adapter.run("query")

        assert len(run.steps) == 2
        assert run.steps[1].output_text == "response text"

    async def test_tool_calls_extracted(self):
        from checkagent.adapters.pydantic_ai import PydanticAIAdapter

        messages = [
            _make_message("model-response", [
                _make_tool_call_part("search", {"q": "test"}),
            ]),
        ]
        result = _make_run_result(messages=messages)
        agent = MagicMock()
        agent.run = AsyncMock(return_value=result)

        adapter = PydanticAIAdapter(agent)
        run = await adapter.run("query")

        assert len(run.steps[0].tool_calls) == 1
        assert run.steps[0].tool_calls[0].name == "search"
        assert run.steps[0].tool_calls[0].arguments == {"q": "test"}

    async def test_tool_return_extracted(self):
        from checkagent.adapters.pydantic_ai import PydanticAIAdapter

        messages = [
            _make_message("tool-return", [
                _make_tool_return_part("search", "found it"),
            ]),
        ]
        result = _make_run_result(messages=messages)
        agent = MagicMock()
        agent.run = AsyncMock(return_value=result)

        adapter = PydanticAIAdapter(agent)
        run = await adapter.run("query")

        assert run.steps[0].tool_calls[0].result == "found it"

    async def test_token_usage(self):
        from checkagent.adapters.pydantic_ai import PydanticAIAdapter

        usage = _make_usage(200, 100)
        result = _make_run_result(usage=usage)
        agent = MagicMock()
        agent.run = AsyncMock(return_value=result)

        adapter = PydanticAIAdapter(agent)
        run = await adapter.run("query")

        assert run.total_prompt_tokens == 200
        assert run.total_completion_tokens == 100

    async def test_error_handling(self):
        from checkagent.adapters.pydantic_ai import PydanticAIAdapter

        agent = MagicMock()
        agent.run = AsyncMock(side_effect=ValueError("bad input"))

        adapter = PydanticAIAdapter(agent)
        run = await adapter.run("oops")

        assert not run.succeeded
        assert "ValueError: bad input" in run.error
        assert run.duration_ms >= 0

    async def test_duration_recorded(self):
        from checkagent.adapters.pydantic_ai import PydanticAIAdapter

        result = _make_run_result()
        agent = MagicMock()
        agent.run = AsyncMock(return_value=result)

        adapter = PydanticAIAdapter(agent)
        run = await adapter.run("query")

        assert run.duration_ms >= 0


# ---------------------------------------------------------------------------
# Tests: streaming
# ---------------------------------------------------------------------------


class TestPydanticAIAdapterStream:
    async def test_stream_with_run_stream(self):
        from checkagent.adapters.pydantic_ai import PydanticAIAdapter

        async def _fake_stream_text():
            yield "chunk1"
            yield "chunk2"

        stream_ctx = MagicMock()
        stream_ctx.stream_text = _fake_stream_text
        agent = MagicMock()

        # Make run_stream return an async context manager
        async_cm = MagicMock()
        async_cm.__aenter__ = AsyncMock(return_value=stream_ctx)
        async_cm.__aexit__ = AsyncMock(return_value=False)
        agent.run_stream = MagicMock(return_value=async_cm)

        adapter = PydanticAIAdapter(agent)
        events = []
        async for event in adapter.run_stream("query"):
            events.append(event)

        types = [e.event_type for e in events]
        assert types[0] == StreamEventType.RUN_START
        assert StreamEventType.TEXT_DELTA in types
        deltas = [e for e in events if e.event_type == StreamEventType.TEXT_DELTA]
        assert deltas[0].data == "chunk1"
        assert deltas[1].data == "chunk2"
        assert types[-1] == StreamEventType.RUN_END

    async def test_stream_fallback_no_run_stream(self):
        from checkagent.adapters.pydantic_ai import PydanticAIAdapter

        result = _make_run_result(data="fallback result")
        agent = MagicMock(spec=["run"])  # no run_stream
        agent.run = AsyncMock(return_value=result)

        adapter = PydanticAIAdapter(agent)
        events = []
        async for event in adapter.run_stream("query"):
            events.append(event)

        types = [e.event_type for e in events]
        assert types[0] == StreamEventType.RUN_START
        assert StreamEventType.TEXT_DELTA in types
        assert types[-1] == StreamEventType.RUN_END

    async def test_stream_error(self):
        from checkagent.adapters.pydantic_ai import PydanticAIAdapter

        async_cm = MagicMock()
        async_cm.__aenter__ = AsyncMock(side_effect=RuntimeError("stream boom"))
        async_cm.__aexit__ = AsyncMock(return_value=False)
        agent = MagicMock()
        agent.run_stream = MagicMock(return_value=async_cm)

        adapter = PydanticAIAdapter(agent)
        events = []
        async for event in adapter.run_stream("query"):
            events.append(event)

        types = [e.event_type for e in events]
        assert StreamEventType.ERROR in types
        assert types[-1] == StreamEventType.RUN_END


# ---------------------------------------------------------------------------
# Tests: import guard
# ---------------------------------------------------------------------------


class TestPydanticAIImportGuard:
    async def test_import_error_without_pydantic_ai(self):
        from checkagent.adapters.pydantic_ai import _ensure_pydantic_ai

        saved = sys.modules.get("pydantic_ai")
        sys.modules["pydantic_ai"] = None  # type: ignore[assignment]
        try:
            with pytest.raises(ImportError, match="pydantic-ai"):
                _ensure_pydantic_ai()
        finally:
            if saved is not None:
                sys.modules["pydantic_ai"] = saved
            else:
                sys.modules.pop("pydantic_ai", None)
