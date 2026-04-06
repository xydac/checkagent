"""Tests for the Anthropic Claude SDK adapter.

anthropic is an optional dependency — all tests mock the framework
objects to test the adapter's conversion logic.
"""

from __future__ import annotations

import sys
import types
from unittest.mock import AsyncMock, MagicMock

import pytest

from checkagent.core.types import AgentInput, StreamEventType

# ---------------------------------------------------------------------------
# Fake anthropic module so the import guard passes
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _fake_anthropic_module():
    """Inject a fake anthropic module for the import check."""
    fake = types.ModuleType("anthropic")
    sys.modules["anthropic"] = fake
    yield
    sys.modules.pop("anthropic", None)


# ---------------------------------------------------------------------------
# Helpers to build mock Anthropic Message objects
# ---------------------------------------------------------------------------


def _make_text_block(text: str = "Hello") -> MagicMock:
    block = MagicMock()
    block.type = "text"
    block.text = text
    return block


def _make_tool_use_block(
    name: str = "search", input_data: dict | None = None
) -> MagicMock:
    block = MagicMock()
    block.type = "tool_use"
    block.name = name
    block.input = input_data or {}
    return block


def _make_usage(input_tokens: int = 100, output_tokens: int = 50) -> MagicMock:
    usage = MagicMock()
    usage.input_tokens = input_tokens
    usage.output_tokens = output_tokens
    return usage


def _make_message(
    content: list | None = None,
    usage: MagicMock | None = None,
) -> MagicMock:
    msg = MagicMock()
    msg.content = content or [_make_text_block()]
    msg.usage = usage or _make_usage()
    return msg


def _make_client(message: MagicMock | None = None, is_async: bool = True) -> MagicMock:
    """Create a mock Anthropic client."""
    client = MagicMock()
    msg = message or _make_message()

    if is_async:
        client.messages.create = AsyncMock(return_value=msg)
    else:
        client.messages.create = MagicMock(return_value=msg)

    # Mark create as async or sync for inspect.iscoroutinefunction
    import inspect
    if is_async:
        assert inspect.iscoroutinefunction(client.messages.create)

    return client


# ---------------------------------------------------------------------------
# Tests: basic run
# ---------------------------------------------------------------------------


class TestAnthropicAdapterRun:
    async def test_basic_run(self):
        from checkagent.adapters.anthropic import AnthropicAdapter

        msg = _make_message([_make_text_block("Hello, world!")])
        client = _make_client(msg)

        adapter = AnthropicAdapter(client, model="claude-sonnet-4-20250514")
        result = await adapter.run("hi")

        assert result.succeeded
        assert result.steps[0].output_text == "Hello, world!"

    async def test_string_to_agent_input(self):
        from checkagent.adapters.anthropic import AnthropicAdapter

        client = _make_client()
        adapter = AnthropicAdapter(client)
        result = await adapter.run("test query")

        assert result.input.query == "test query"

    async def test_agent_input_forwarded(self):
        from checkagent.adapters.anthropic import AnthropicAdapter

        client = _make_client()
        adapter = AnthropicAdapter(client)
        inp = AgentInput(query="hello")
        await adapter.run(inp)

        call_kwargs = client.messages.create.call_args.kwargs
        assert call_kwargs["messages"] == [{"role": "user", "content": "hello"}]

    async def test_system_prompt(self):
        from checkagent.adapters.anthropic import AnthropicAdapter

        client = _make_client()
        adapter = AnthropicAdapter(client, system="You are helpful")
        await adapter.run("hello")

        call_kwargs = client.messages.create.call_args.kwargs
        assert call_kwargs["system"] == "You are helpful"

    async def test_no_system_prompt(self):
        from checkagent.adapters.anthropic import AnthropicAdapter

        client = _make_client()
        adapter = AnthropicAdapter(client)
        await adapter.run("hello")

        call_kwargs = client.messages.create.call_args.kwargs
        assert "system" not in call_kwargs

    async def test_tool_use_extracted(self):
        from checkagent.adapters.anthropic import AnthropicAdapter

        msg = _make_message([
            _make_text_block("Let me search"),
            _make_tool_use_block("web_search", {"query": "test"}),
        ])
        client = _make_client(msg)

        adapter = AnthropicAdapter(client)
        result = await adapter.run("search for test")

        assert len(result.steps[0].tool_calls) == 1
        assert result.steps[0].tool_calls[0].name == "web_search"
        assert result.steps[0].tool_calls[0].arguments == {"query": "test"}

    async def test_token_usage(self):
        from checkagent.adapters.anthropic import AnthropicAdapter

        usage = _make_usage(200, 100)
        msg = _make_message(usage=usage)
        client = _make_client(msg)

        adapter = AnthropicAdapter(client)
        result = await adapter.run("query")

        assert result.total_prompt_tokens == 200
        assert result.total_completion_tokens == 100

    async def test_multiple_text_blocks(self):
        from checkagent.adapters.anthropic import AnthropicAdapter

        msg = _make_message([
            _make_text_block("first"),
            _make_text_block("second"),
        ])
        client = _make_client(msg)

        adapter = AnthropicAdapter(client)
        result = await adapter.run("query")

        assert result.steps[0].output_text == "first\nsecond"

    async def test_error_handling(self):
        from checkagent.adapters.anthropic import AnthropicAdapter

        client = MagicMock()
        client.messages.create = AsyncMock(side_effect=RuntimeError("API error"))

        adapter = AnthropicAdapter(client)
        result = await adapter.run("oops")

        assert not result.succeeded
        assert "RuntimeError: API error" in result.error
        assert result.duration_ms >= 0

    async def test_duration_recorded(self):
        from checkagent.adapters.anthropic import AnthropicAdapter

        client = _make_client()
        adapter = AnthropicAdapter(client)
        result = await adapter.run("query")

        assert result.duration_ms >= 0

    async def test_max_tokens_forwarded(self):
        from checkagent.adapters.anthropic import AnthropicAdapter

        client = _make_client()
        adapter = AnthropicAdapter(client, max_tokens=2048)
        await adapter.run("hello")

        call_kwargs = client.messages.create.call_args.kwargs
        assert call_kwargs["max_tokens"] == 2048


# ---------------------------------------------------------------------------
# Tests: streaming
# ---------------------------------------------------------------------------


class TestAnthropicAdapterStream:
    async def test_stream_fallback_no_stream(self):
        from checkagent.adapters.anthropic import AnthropicAdapter

        msg = _make_message([_make_text_block("streamed text")])
        client = MagicMock(spec=["create"])  # no messages.stream
        client.create = AsyncMock(return_value=msg)

        adapter = AnthropicAdapter(client)
        events = []
        async for event in adapter.run_stream("query"):
            events.append(event)

        types = [e.event_type for e in events]
        assert types[0] == StreamEventType.RUN_START
        assert StreamEventType.TEXT_DELTA in types
        assert types[-1] == StreamEventType.RUN_END

    async def test_stream_with_stream_api(self):
        from checkagent.adapters.anthropic import AnthropicAdapter

        # Create mock stream events
        delta = MagicMock()
        delta.type = "text_delta"
        delta.text = "chunk"

        event1 = MagicMock()
        event1.type = "content_block_delta"
        event1.delta = delta

        event2 = MagicMock()
        event2.type = "message_stop"

        async def _fake_stream():
            yield event1
            yield event2

        stream_ctx = MagicMock()
        stream_ctx.__aenter__ = AsyncMock(return_value=_fake_stream())
        stream_ctx.__aexit__ = AsyncMock(return_value=False)

        client = MagicMock()
        client.messages.stream = MagicMock(return_value=stream_ctx)

        adapter = AnthropicAdapter(client)
        events = []
        async for event in adapter.run_stream("query"):
            events.append(event)

        types = [e.event_type for e in events]
        assert types[0] == StreamEventType.RUN_START
        assert StreamEventType.TEXT_DELTA in types
        deltas = [e for e in events if e.event_type == StreamEventType.TEXT_DELTA]
        assert deltas[0].data == "chunk"
        assert types[-1] == StreamEventType.RUN_END

    async def test_stream_error(self):
        from checkagent.adapters.anthropic import AnthropicAdapter

        stream_ctx = MagicMock()
        stream_ctx.__aenter__ = AsyncMock(side_effect=RuntimeError("stream error"))
        stream_ctx.__aexit__ = AsyncMock(return_value=False)

        client = MagicMock()
        client.messages.stream = MagicMock(return_value=stream_ctx)

        adapter = AnthropicAdapter(client)
        events = []
        async for event in adapter.run_stream("query"):
            events.append(event)

        types = [e.event_type for e in events]
        assert StreamEventType.ERROR in types
        assert types[-1] == StreamEventType.RUN_END


# ---------------------------------------------------------------------------
# Tests: import guard
# ---------------------------------------------------------------------------


class TestAnthropicImportGuard:
    async def test_import_error_without_anthropic(self):
        from checkagent.adapters.anthropic import _ensure_anthropic

        saved = sys.modules.get("anthropic")
        sys.modules["anthropic"] = None  # type: ignore[assignment]
        try:
            with pytest.raises(ImportError, match="anthropic"):
                _ensure_anthropic()
        finally:
            if saved is not None:
                sys.modules["anthropic"] = saved
            else:
                sys.modules.pop("anthropic", None)
