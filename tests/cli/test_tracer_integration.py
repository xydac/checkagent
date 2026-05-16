"""Integration tests for the execution tracer with real SDK classes.

These tests verify that install_patches() actually monkey-patches OpenAI and
Anthropic SDK classes and that calls through real client instances produce
trace events.  This closes F-120.
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import MagicMock

import pytest

from checkagent.core.tracer import (
    _ACTIVE_TRACE,
    begin_probe_trace,
    end_probe_trace,
    install_patches,
    is_installed,
    uninstall_patches,
)


@pytest.fixture(autouse=True)
def _clean_tracer():
    """Ensure tracer state is clean between tests."""
    _ACTIVE_TRACE.set(None)
    yield
    _ACTIVE_TRACE.set(None)
    if is_installed():
        uninstall_patches()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _openai_chat_response(
    content: str = "Hello!",
    model: str = "gpt-4o-mini",
    prompt_tokens: int = 10,
    completion_tokens: int = 5,
    tool_calls: list[dict[str, str]] | None = None,
) -> MagicMock:
    msg = MagicMock()
    msg.content = content
    if tool_calls:
        tc_list = []
        for tc in tool_calls:
            tc_mock = MagicMock()
            tc_mock.function.name = tc["name"]
            tc_mock.function.arguments = tc.get("arguments", "{}")
            tc_list.append(tc_mock)
        msg.tool_calls = tc_list
    else:
        msg.tool_calls = None
    choice = MagicMock()
    choice.message = msg
    usage = MagicMock()
    usage.prompt_tokens = prompt_tokens
    usage.completion_tokens = completion_tokens
    resp = MagicMock()
    resp.choices = [choice]
    resp.usage = usage
    resp.model = model
    return resp


def _anthropic_message_response(
    text: str = "Hello!",
    model: str = "claude-haiku-4-5-20251001",
    input_tokens: int = 15,
    output_tokens: int = 8,
    tool_use: list[dict[str, Any]] | None = None,
) -> MagicMock:
    blocks = []
    if text:
        text_block = MagicMock(spec=["text", "type"])
        text_block.text = text
        text_block.type = "text"
        blocks.append(text_block)
    if tool_use:
        for tu in tool_use:
            tu_block = MagicMock(spec=["type", "name", "input"])
            tu_block.type = "tool_use"
            tu_block.name = tu["name"]
            tu_block.input = tu.get("input", {})
            blocks.append(tu_block)
    resp = MagicMock()
    resp.content = blocks
    usage = MagicMock()
    usage.input_tokens = input_tokens
    usage.output_tokens = output_tokens
    resp.usage = usage
    resp.model = model
    return resp


# ---------------------------------------------------------------------------
# OpenAI integration — async
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_openai_async_real_client_traced() -> None:
    """install_patches() patches the real AsyncCompletions class and traces calls."""
    import openai.resources.chat.completions as oai_mod

    mock_resp = _openai_chat_response("I can help with that.", "gpt-4o")

    original_create = oai_mod.AsyncCompletions.create

    async def mock_create(self, *args, **kwargs):
        return mock_resp

    oai_mod.AsyncCompletions.create = mock_create
    install_patches()

    begin_probe_trace()
    instance = MagicMock(spec=oai_mod.AsyncCompletions)
    result = await oai_mod.AsyncCompletions.create(
        instance,
        model="gpt-4o",
        messages=[{"role": "user", "content": "What is 2+2?"}],
    )
    events = end_probe_trace()

    assert result is mock_resp
    llm_events = [e for e in events if e["type"] == "llm_call"]
    assert len(llm_events) == 1
    ev = llm_events[0]
    assert ev["provider"] == "openai"
    assert ev["model"] == "gpt-4o"
    assert "2+2" in ev["prompt_preview"]
    assert ev["response_preview"] == "I can help with that."
    assert ev["input_tokens"] == 10
    assert ev["output_tokens"] == 5
    assert ev["latency_ms"] >= 0

    uninstall_patches()
    oai_mod.AsyncCompletions.create = original_create


@pytest.mark.asyncio
async def test_openai_async_tool_calls_traced() -> None:
    """Tool calls in OpenAI responses are captured as separate trace events."""
    import openai.resources.chat.completions as oai_mod

    mock_resp = _openai_chat_response(
        content="",
        tool_calls=[
            {"name": "get_weather", "arguments": '{"city": "NYC"}'},
            {"name": "get_time", "arguments": '{"tz": "EST"}'},
        ],
    )
    original_create = oai_mod.AsyncCompletions.create

    async def mock_create(self, *args, **kwargs):
        return mock_resp

    oai_mod.AsyncCompletions.create = mock_create
    install_patches()

    begin_probe_trace()
    instance = MagicMock()
    await oai_mod.AsyncCompletions.create(
        instance, model="gpt-4o", messages=[{"role": "user", "content": "weather"}],
    )
    events = end_probe_trace()

    tool_events = [e for e in events if e["type"] == "tool_call"]
    assert len(tool_events) == 2
    assert tool_events[0]["name"] == "get_weather"
    assert "NYC" in tool_events[0]["arguments_preview"]
    assert tool_events[1]["name"] == "get_time"

    uninstall_patches()
    oai_mod.AsyncCompletions.create = original_create


# ---------------------------------------------------------------------------
# OpenAI integration — sync
# ---------------------------------------------------------------------------


def test_openai_sync_client_traced() -> None:
    """install_patches() also patches the sync Completions.create."""
    import openai.resources.chat.completions as oai_mod

    mock_resp = _openai_chat_response("Sync response", "gpt-4o-mini")
    original_create = oai_mod.Completions.create

    def mock_create(self, *args, **kwargs):
        return mock_resp

    oai_mod.Completions.create = mock_create
    install_patches()

    begin_probe_trace()
    instance = MagicMock()
    oai_mod.Completions.create(
        instance, model="gpt-4o-mini", messages=[{"role": "user", "content": "hi"}],
    )
    events = end_probe_trace()

    llm_events = [e for e in events if e["type"] == "llm_call"]
    assert len(llm_events) == 1
    assert llm_events[0]["provider"] == "openai"
    assert llm_events[0]["response_preview"] == "Sync response"

    uninstall_patches()
    oai_mod.Completions.create = original_create


# ---------------------------------------------------------------------------
# Anthropic integration — async
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_anthropic_async_real_client_traced() -> None:
    """install_patches() patches the real AsyncMessages class."""
    import anthropic.resources.messages as anth_mod

    mock_resp = _anthropic_message_response("Claude says hello", "claude-haiku-4-5-20251001")
    original_create = anth_mod.AsyncMessages.create

    async def mock_create(self, *args, **kwargs):
        return mock_resp

    anth_mod.AsyncMessages.create = mock_create
    install_patches()

    begin_probe_trace()
    instance = MagicMock()
    result = await anth_mod.AsyncMessages.create(
        instance,
        model="claude-haiku-4-5-20251001",
        messages=[{"role": "user", "content": "Hello Claude"}],
        max_tokens=100,
    )
    events = end_probe_trace()

    assert result is mock_resp
    llm_events = [e for e in events if e["type"] == "llm_call"]
    assert len(llm_events) == 1
    ev = llm_events[0]
    assert ev["provider"] == "anthropic"
    assert ev["model"] == "claude-haiku-4-5-20251001"
    assert "Claude" in ev["prompt_preview"] or "Hello" in ev["prompt_preview"]
    assert ev["response_preview"] == "Claude says hello"
    assert ev["input_tokens"] == 15
    assert ev["output_tokens"] == 8

    uninstall_patches()
    anth_mod.AsyncMessages.create = original_create


@pytest.mark.asyncio
async def test_anthropic_async_tool_use_traced() -> None:
    """Tool use blocks in Anthropic responses are captured."""
    import anthropic.resources.messages as anth_mod

    mock_resp = _anthropic_message_response(
        text="",
        tool_use=[{"name": "search_docs", "input": {"query": "test"}}],
    )
    original_create = anth_mod.AsyncMessages.create

    async def mock_create(self, *args, **kwargs):
        return mock_resp

    anth_mod.AsyncMessages.create = mock_create
    install_patches()

    begin_probe_trace()
    instance = MagicMock()
    await anth_mod.AsyncMessages.create(
        instance,
        model="claude-haiku-4-5-20251001",
        messages=[{"role": "user", "content": "search"}],
        max_tokens=100,
    )
    events = end_probe_trace()

    tool_events = [e for e in events if e["type"] == "tool_call"]
    assert len(tool_events) == 1
    assert tool_events[0]["name"] == "search_docs"

    uninstall_patches()
    anth_mod.AsyncMessages.create = original_create


# ---------------------------------------------------------------------------
# Anthropic integration — sync
# ---------------------------------------------------------------------------


def test_anthropic_sync_client_traced() -> None:
    """install_patches() patches the sync Messages.create."""
    import anthropic.resources.messages as anth_mod

    mock_resp = _anthropic_message_response("Sync Claude", "claude-haiku-4-5-20251001")
    original_create = anth_mod.Messages.create

    def mock_create(self, *args, **kwargs):
        return mock_resp

    anth_mod.Messages.create = mock_create
    install_patches()

    begin_probe_trace()
    instance = MagicMock()
    anth_mod.Messages.create(
        instance,
        model="claude-haiku-4-5-20251001",
        messages=[{"role": "user", "content": "hi"}],
        max_tokens=100,
    )
    events = end_probe_trace()

    llm_events = [e for e in events if e["type"] == "llm_call"]
    assert len(llm_events) == 1
    assert llm_events[0]["provider"] == "anthropic"
    assert llm_events[0]["response_preview"] == "Sync Claude"

    uninstall_patches()
    anth_mod.Messages.create = original_create


# ---------------------------------------------------------------------------
# Cross-provider: both patched simultaneously
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_both_providers_traced_in_single_probe() -> None:
    """A single probe that calls both OpenAI and Anthropic captures all events."""
    import anthropic.resources.messages as anth_mod
    import openai.resources.chat.completions as oai_mod

    oai_resp = _openai_chat_response("OpenAI answer")
    anth_resp = _anthropic_message_response("Anthropic answer")

    orig_oai = oai_mod.AsyncCompletions.create
    orig_anth = anth_mod.AsyncMessages.create

    async def mock_oai(self, *a, **kw):
        return oai_resp

    async def mock_anth(self, *a, **kw):
        return anth_resp

    oai_mod.AsyncCompletions.create = mock_oai
    anth_mod.AsyncMessages.create = mock_anth

    install_patches()

    begin_probe_trace()
    await oai_mod.AsyncCompletions.create(
        MagicMock(), model="gpt-4o", messages=[{"role": "user", "content": "q1"}],
    )
    await anth_mod.AsyncMessages.create(
        MagicMock(),
        model="claude-haiku-4-5-20251001",
        messages=[{"role": "user", "content": "q2"}],
        max_tokens=100,
    )
    events = end_probe_trace()

    providers = {e["provider"] for e in events if e["type"] == "llm_call"}
    assert providers == {"openai", "anthropic"}

    uninstall_patches()
    oai_mod.AsyncCompletions.create = orig_oai
    anth_mod.AsyncMessages.create = orig_anth


# ---------------------------------------------------------------------------
# Isolation: concurrent probes get separate traces
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_concurrent_probes_isolated_with_real_patches() -> None:
    """Two concurrent async probes each see only their own SDK calls."""
    import openai.resources.chat.completions as oai_mod

    original_create = oai_mod.AsyncCompletions.create

    call_count = 0

    async def mock_create(self, *a, **kw):
        nonlocal call_count
        call_count += 1
        return _openai_chat_response(f"response-{call_count}", kw.get("model", "?"))

    oai_mod.AsyncCompletions.create = mock_create
    install_patches()

    results: dict[str, list[dict[str, Any]]] = {}

    async def probe_a():
        begin_probe_trace()
        await oai_mod.AsyncCompletions.create(
            MagicMock(), model="model-a", messages=[{"role": "user", "content": "a"}],
        )
        results["a"] = end_probe_trace()

    async def probe_b():
        begin_probe_trace()
        await oai_mod.AsyncCompletions.create(
            MagicMock(), model="model-b", messages=[{"role": "user", "content": "b"}],
        )
        results["b"] = end_probe_trace()

    await asyncio.gather(probe_a(), probe_b())

    assert len(results["a"]) == 1
    assert results["a"][0]["model"] == "model-a"
    assert len(results["b"]) == 1
    assert results["b"][0]["model"] == "model-b"

    uninstall_patches()
    oai_mod.AsyncCompletions.create = original_create


# ---------------------------------------------------------------------------
# No active trace — events are silently dropped
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_no_active_trace_drops_events() -> None:
    """SDK calls outside begin/end don't accumulate events anywhere."""
    import openai.resources.chat.completions as oai_mod

    original_create = oai_mod.AsyncCompletions.create

    async def mock_create(self, *a, **kw):
        return _openai_chat_response()

    oai_mod.AsyncCompletions.create = mock_create
    install_patches()

    # Call without begin_probe_trace — events should be silently dropped
    _ACTIVE_TRACE.set(None)
    await oai_mod.AsyncCompletions.create(
        MagicMock(), model="gpt-4o", messages=[],
    )

    # Now start a trace — should be empty
    begin_probe_trace()
    events = end_probe_trace()
    assert events == []

    uninstall_patches()
    oai_mod.AsyncCompletions.create = original_create


# ---------------------------------------------------------------------------
# ca_tracer fixture
# ---------------------------------------------------------------------------


def test_ca_tracer_fixture_available(ca_tracer) -> None:
    """The ca_tracer fixture is registered by the plugin."""
    assert ca_tracer is not None
    assert hasattr(ca_tracer, "begin")
    assert hasattr(ca_tracer, "end")
    assert hasattr(ca_tracer, "llm_calls")
    assert hasattr(ca_tracer, "tool_calls")
    assert hasattr(ca_tracer, "events")


def test_ca_tracer_fixture_captures_events(ca_tracer) -> None:
    """ca_tracer fixture captures trace events via begin/end."""
    from checkagent.core.tracer import _record

    ca_tracer.begin()
    _record({"type": "llm_call", "provider": "openai", "model": "gpt-4o"})
    _record({"type": "tool_call", "name": "search"})
    ca_tracer.end()

    assert len(ca_tracer.events) == 2
    assert len(ca_tracer.llm_calls) == 1
    assert len(ca_tracer.tool_calls) == 1
    assert ca_tracer.llm_calls[0]["model"] == "gpt-4o"
    assert ca_tracer.tool_calls[0]["name"] == "search"


# ---------------------------------------------------------------------------
# TracerContext API
# ---------------------------------------------------------------------------


def test_tracer_context_begin_end() -> None:
    from checkagent.core.plugin import TracerContext
    from checkagent.core.tracer import _record

    ctx = TracerContext()
    assert ctx.events == []

    ctx.begin()
    _record({"type": "llm_call", "provider": "openai", "model": "gpt-4o"})
    _record({"type": "tool_call", "name": "search"})
    result = ctx.end()

    assert len(result) == 2
    assert ctx.events == result
    assert len(ctx.llm_calls) == 1
    assert len(ctx.tool_calls) == 1


def test_tracer_context_multiple_cycles() -> None:
    from checkagent.core.plugin import TracerContext
    from checkagent.core.tracer import _record

    ctx = TracerContext()

    ctx.begin()
    _record({"type": "llm_call", "provider": "openai", "model": "gpt-4o"})
    ctx.end()
    assert len(ctx.events) == 1

    ctx.begin()
    _record({"type": "llm_call", "provider": "anthropic", "model": "claude"})
    _record({"type": "llm_call", "provider": "anthropic", "model": "claude"})
    ctx.end()
    assert len(ctx.events) == 2
    assert ctx.llm_calls[0]["provider"] == "anthropic"


# ---------------------------------------------------------------------------
# Export verification
# ---------------------------------------------------------------------------


def test_tracer_exports_from_top_level() -> None:
    import checkagent

    assert hasattr(checkagent, "TracerContext")
    assert hasattr(checkagent, "install_patches")
    assert hasattr(checkagent, "uninstall_patches")
    assert hasattr(checkagent, "begin_probe_trace")
    assert hasattr(checkagent, "end_probe_trace")

    assert "TracerContext" in checkagent.__all__
    assert "install_patches" in checkagent.__all__
