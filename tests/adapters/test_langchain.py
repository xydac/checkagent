"""Tests for the LangChain adapter.

LangChain is an optional dependency — all tests mock the framework objects
to test the adapter's conversion logic without requiring langchain-core.
"""

from __future__ import annotations

import sys
import types
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from checkagent.core.types import AgentInput, StreamEventType

# ---------------------------------------------------------------------------
# Fake langchain_core module so the import guard passes
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _fake_langchain_core():
    """Inject a fake langchain_core module for the import check."""
    fake = types.ModuleType("langchain_core")
    sys.modules["langchain_core"] = fake
    yield
    sys.modules.pop("langchain_core", None)


# ---------------------------------------------------------------------------
# Helpers to build mock Runnable objects
# ---------------------------------------------------------------------------

def _make_runnable(
    ainvoke_return: Any = "Hello from LangChain",
    *,
    has_ainvoke: bool = True,
    has_astream_events: bool = False,
    sync_invoke_return: Any = None,
) -> MagicMock:
    runnable = MagicMock()
    if has_ainvoke:
        runnable.ainvoke = AsyncMock(return_value=ainvoke_return)
    else:
        del runnable.ainvoke
        runnable.invoke = MagicMock(return_value=sync_invoke_return or ainvoke_return)

    if not has_astream_events:
        del runnable.astream_events

    return runnable


def _make_ai_message(
    content: str = "Hi there",
    tool_calls: list[dict[str, Any]] | None = None,
    usage_metadata: dict[str, int] | None = None,
) -> MagicMock:
    msg = MagicMock()
    msg.content = content
    msg.tool_calls = tool_calls or []
    msg.usage_metadata = usage_metadata
    msg.response_metadata = None
    return msg


# ---------------------------------------------------------------------------
# Tests: run()
# ---------------------------------------------------------------------------


class TestLangChainAdapterRun:
    async def test_basic_string_output(self):
        from checkagent.adapters.langchain import LangChainAdapter

        runnable = _make_runnable("Hello world")
        adapter = LangChainAdapter(runnable)

        result = await adapter.run("test query")

        assert result.succeeded
        assert result.final_output == "Hello world"
        assert len(result.steps) == 1
        assert result.steps[0].output_text == "Hello world"
        runnable.ainvoke.assert_awaited_once_with({"input": "test query"})

    async def test_agent_input_with_context(self):
        from checkagent.adapters.langchain import LangChainAdapter

        runnable = _make_runnable("response")
        adapter = LangChainAdapter(runnable)

        inp = AgentInput(query="hello", context={"user_id": "123"})
        result = await adapter.run(inp)

        assert result.succeeded
        runnable.ainvoke.assert_awaited_once_with({
            "input": "hello",
            "user_id": "123",
        })

    async def test_extra_inputs_constructor(self):
        """F-084: extra_inputs provides default variables for multi-variable chains."""
        from checkagent.adapters.langchain import LangChainAdapter

        runnable = _make_runnable("answer")
        adapter = LangChainAdapter(
            runnable, extra_inputs={"context": "some docs", "language": "en"}
        )

        result = await adapter.run("what is X?")

        assert result.succeeded
        runnable.ainvoke.assert_awaited_once_with({
            "input": "what is X?",
            "context": "some docs",
            "language": "en",
        })

    async def test_extra_inputs_with_context_override(self):
        """F-084: per-run context overrides constructor extra_inputs."""
        from checkagent.adapters.langchain import LangChainAdapter

        runnable = _make_runnable("answer")
        adapter = LangChainAdapter(
            runnable, extra_inputs={"context": "default docs", "language": "en"}
        )

        inp = AgentInput(query="q", context={"context": "custom docs"})
        result = await adapter.run(inp)

        assert result.succeeded
        runnable.ainvoke.assert_awaited_once_with({
            "input": "q",
            "context": "custom docs",  # overridden by context
            "language": "en",           # kept from extra_inputs
        })

    async def test_extra_inputs_with_custom_input_key(self):
        """F-084: extra_inputs works with custom input_key."""
        from checkagent.adapters.langchain import LangChainAdapter

        runnable = _make_runnable("ok")
        adapter = LangChainAdapter(
            runnable, input_key="question", extra_inputs={"context": "docs"}
        )

        await adapter.run("what?")
        runnable.ainvoke.assert_awaited_once_with({
            "question": "what?",
            "context": "docs",
        })

    async def test_extra_inputs_ignored_for_raw(self):
        """extra_inputs are ignored when input_key='__raw__'."""
        from checkagent.adapters.langchain import LangChainAdapter

        runnable = _make_runnable("ok")
        adapter = LangChainAdapter(
            runnable, input_key="__raw__", extra_inputs={"context": "docs"}
        )

        await adapter.run("plain")
        runnable.ainvoke.assert_awaited_once_with("plain")

    async def test_custom_input_key(self):
        from checkagent.adapters.langchain import LangChainAdapter

        runnable = _make_runnable("ok")
        adapter = LangChainAdapter(runnable, input_key="question")

        await adapter.run("what is 2+2?")
        runnable.ainvoke.assert_awaited_once_with({"question": "what is 2+2?"})

    async def test_raw_input_key(self):
        from checkagent.adapters.langchain import LangChainAdapter

        runnable = _make_runnable("ok")
        adapter = LangChainAdapter(runnable, input_key="__raw__")

        await adapter.run("plain string")
        runnable.ainvoke.assert_awaited_once_with("plain string")

    async def test_ai_message_output(self):
        from checkagent.adapters.langchain import LangChainAdapter

        msg = _make_ai_message("I can help with that")
        runnable = _make_runnable(msg)
        adapter = LangChainAdapter(runnable)

        result = await adapter.run("help me")

        assert result.steps[0].output_text == "I can help with that"

    async def test_tool_calls_extraction(self):
        from checkagent.adapters.langchain import LangChainAdapter

        msg = _make_ai_message(
            "Let me search",
            tool_calls=[
                {"name": "search", "args": {"query": "python"}},
                {"name": "calculator", "args": {"expr": "2+2"}},
            ],
        )
        runnable = _make_runnable(msg)
        adapter = LangChainAdapter(runnable)

        result = await adapter.run("search for python")

        assert len(result.steps[0].tool_calls) == 2
        assert result.steps[0].tool_calls[0].name == "search"
        assert result.steps[0].tool_calls[0].arguments == {"query": "python"}
        assert result.steps[0].tool_calls[1].name == "calculator"

    async def test_token_usage_from_usage_metadata(self):
        from checkagent.adapters.langchain import LangChainAdapter

        msg = _make_ai_message(
            "response",
            usage_metadata={"input_tokens": 10, "output_tokens": 20},
        )
        runnable = _make_runnable(msg)
        adapter = LangChainAdapter(runnable)

        result = await adapter.run("query")

        assert result.total_prompt_tokens == 10
        assert result.total_completion_tokens == 20

    async def test_token_usage_from_response_metadata(self):
        from checkagent.adapters.langchain import LangChainAdapter

        msg = MagicMock()
        msg.content = "response"
        msg.tool_calls = []
        msg.usage_metadata = None
        msg.response_metadata = {
            "token_usage": {"prompt_tokens": 15, "completion_tokens": 25}
        }

        runnable = _make_runnable(msg)
        adapter = LangChainAdapter(runnable)

        result = await adapter.run("query")

        assert result.total_prompt_tokens == 15
        assert result.total_completion_tokens == 25

    async def test_dict_output_with_output_key(self):
        from checkagent.adapters.langchain import LangChainAdapter

        runnable = _make_runnable({"output": "graph result", "state": "done"})
        adapter = LangChainAdapter(runnable)

        result = await adapter.run("run graph")

        assert result.steps[0].output_text == "graph result"

    async def test_dict_output_messages_key(self):
        from checkagent.adapters.langchain import LangChainAdapter

        msg = _make_ai_message("last message")
        runnable = _make_runnable({"messages": [msg]})
        adapter = LangChainAdapter(runnable)

        result = await adapter.run("query")

        assert result.steps[0].output_text == "last message"

    async def test_error_handling(self):
        from checkagent.adapters.langchain import LangChainAdapter

        runnable = MagicMock()
        runnable.ainvoke = AsyncMock(side_effect=ValueError("bad input"))
        adapter = LangChainAdapter(runnable)

        result = await adapter.run("oops")

        assert not result.succeeded
        assert "ValueError: bad input" in result.error
        assert result.duration_ms >= 0

    async def test_sync_runnable_fallback(self):
        from checkagent.adapters.langchain import LangChainAdapter

        runnable = _make_runnable("sync result", has_ainvoke=False)
        adapter = LangChainAdapter(runnable)

        result = await adapter.run("query")

        assert result.succeeded
        assert result.final_output == "sync result"
        runnable.invoke.assert_called_once()

    async def test_duration_recorded(self):
        from checkagent.adapters.langchain import LangChainAdapter

        runnable = _make_runnable("fast")
        adapter = LangChainAdapter(runnable)

        result = await adapter.run("query")

        assert result.duration_ms is not None
        assert result.duration_ms >= 0


# ---------------------------------------------------------------------------
# Tests: run_stream()
# ---------------------------------------------------------------------------


class TestLangChainAdapterStream:
    async def test_fallback_stream_no_astream_events(self):
        from checkagent.adapters.langchain import LangChainAdapter

        runnable = _make_runnable("streamed output", has_astream_events=False)
        adapter = LangChainAdapter(runnable)

        events = []
        async for event in adapter.run_stream("query"):
            events.append(event)

        types = [e.event_type for e in events]
        assert types[0] == StreamEventType.RUN_START
        assert StreamEventType.TEXT_DELTA in types
        assert types[-1] == StreamEventType.RUN_END

    async def test_stream_with_astream_events(self):
        from checkagent.adapters.langchain import LangChainAdapter

        chunk = MagicMock()
        chunk.content = "Hello"

        async def fake_astream_events(input, version="v2"):
            yield {"event": "on_chat_model_stream", "data": {"chunk": chunk}}
            yield {"event": "on_tool_start", "name": "search", "data": {}}
            yield {"event": "on_tool_end", "data": {"output": "result"}}

        runnable = MagicMock()
        runnable.ainvoke = AsyncMock(return_value="ok")
        runnable.astream_events = fake_astream_events

        adapter = LangChainAdapter(runnable)

        events = []
        async for event in adapter.run_stream("query"):
            events.append(event)

        types = [e.event_type for e in events]
        assert types[0] == StreamEventType.RUN_START
        assert StreamEventType.TEXT_DELTA in types
        assert StreamEventType.TOOL_CALL_START in types
        assert StreamEventType.TOOL_RESULT in types
        assert types[-1] == StreamEventType.RUN_END

        # Check text delta data
        text_events = [e for e in events if e.event_type == StreamEventType.TEXT_DELTA]
        assert text_events[0].data == "Hello"

    async def test_stream_with_extra_inputs(self):
        """F-084: extra_inputs are passed through in streaming mode."""
        from checkagent.adapters.langchain import LangChainAdapter

        runnable = _make_runnable("output", has_astream_events=False)
        adapter = LangChainAdapter(
            runnable, extra_inputs={"context": "docs"}
        )

        events = []
        async for event in adapter.run_stream("query"):
            events.append(event)

        types = [e.event_type for e in events]
        assert types[0] == StreamEventType.RUN_START
        assert types[-1] == StreamEventType.RUN_END

    async def test_stream_error_handling(self):
        from checkagent.adapters.langchain import LangChainAdapter

        async def failing_stream(input, version="v2"):
            raise RuntimeError("stream failed")
            yield  # noqa: F841 - unreachable yield makes this an async generator

        runnable = MagicMock()
        runnable.ainvoke = AsyncMock()
        runnable.astream_events = failing_stream

        adapter = LangChainAdapter(runnable)

        events = []
        async for event in adapter.run_stream("query"):
            events.append(event)

        types = [e.event_type for e in events]
        assert StreamEventType.ERROR in types
        assert types[-1] == StreamEventType.RUN_END


# ---------------------------------------------------------------------------
# Tests: import guard
# ---------------------------------------------------------------------------


class TestLangChainImportGuard:
    async def test_import_error_without_langchain(self):
        """_ensure_langchain raises ImportError if langchain_core missing."""
        from checkagent.adapters.langchain import _ensure_langchain

        saved = sys.modules.get("langchain_core")
        # Setting to None blocks the import (Python treats None entries as failed imports)
        sys.modules["langchain_core"] = None  # type: ignore[assignment]
        try:
            with pytest.raises(ImportError, match="langchain-core"):
                _ensure_langchain()
        finally:
            if saved is not None:
                sys.modules["langchain_core"] = saved
            else:
                sys.modules.pop("langchain_core", None)
