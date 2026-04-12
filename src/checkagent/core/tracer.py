"""Execution tracer for checkagent scan.

Intercepts LLM calls (OpenAI, Anthropic) and tool calls during probe execution
using monkey-patching.  No user code changes required.

Usage::

    from checkagent.core.tracer import install_patches, uninstall_patches
    from checkagent.core.tracer import begin_probe_trace, end_probe_trace

    install_patches()          # once, before probes start
    ...
    begin_probe_trace()        # inside each probe coroutine
    await agent_fn(probe_input)
    events = end_probe_trace() # collect trace events for that probe
    ...
    uninstall_patches()        # restore originals when scan finishes

The tracer uses :mod:`contextvars` so concurrent probe coroutines each collect
their own events independently.
"""

from __future__ import annotations

import contextvars
import time
from typing import Any

# ---------------------------------------------------------------------------
# Context variable — per-coroutine event list
# ---------------------------------------------------------------------------

_ACTIVE_TRACE: contextvars.ContextVar[list[dict[str, Any]] | None] = (
    contextvars.ContextVar("_checkagent_active_trace", default=None)
)


def _record(event: dict[str, Any]) -> None:
    """Append *event* to the active probe trace, if any."""
    trace = _ACTIVE_TRACE.get()
    if trace is not None:
        trace.append(event)


def begin_probe_trace() -> None:
    """Start collecting trace events for the current async task/thread."""
    _ACTIVE_TRACE.set([])


def end_probe_trace() -> list[dict[str, Any]]:
    """Stop collecting events and return everything recorded since the last
    :func:`begin_probe_trace` call in this context."""
    events = _ACTIVE_TRACE.get() or []
    _ACTIVE_TRACE.set(None)
    return events


# ---------------------------------------------------------------------------
# Patch registry
# ---------------------------------------------------------------------------

_original_patches: dict[str, tuple[Any, str, Any]] = {}
_tracer_installed: bool = False


def install_patches() -> None:
    """Monkey-patch OpenAI and Anthropic client methods to record calls.

    Safe to call multiple times — subsequent calls are no-ops.
    """
    global _tracer_installed
    if _tracer_installed:
        return
    _try_patch_openai()
    _try_patch_anthropic()
    _tracer_installed = True


def uninstall_patches() -> None:
    """Restore all patched methods to their originals."""
    global _tracer_installed
    for _key, (obj, attr, orig) in _original_patches.items():
        setattr(obj, attr, orig)
    _original_patches.clear()
    _tracer_installed = False


def is_installed() -> bool:
    """Return True if patches are currently installed."""
    return _tracer_installed


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _register(key: str, obj: Any, attr: str, wrapper: Any) -> None:
    original = getattr(obj, attr)
    _original_patches[key] = (obj, attr, original)
    setattr(obj, attr, wrapper)


def _truncate_messages(messages: list[dict[str, Any]], max_chars: int = 200) -> str:
    """Summarise a messages list to a short string for trace display."""
    if not messages:
        return ""
    last = messages[-1]
    role = last.get("role", "?")
    content = last.get("content") or ""
    if isinstance(content, list):
        # OpenAI content blocks
        content = " ".join(
            block.get("text", "") if isinstance(block, dict) else str(block)
            for block in content
        )
    text = str(content)[:max_chars]
    return f"[{role}] {text}"


# ---------------------------------------------------------------------------
# OpenAI patches
# ---------------------------------------------------------------------------

def _try_patch_openai() -> None:
    try:
        import openai.resources.chat.completions as _oai  # noqa: PLC0415
    except ImportError:
        return

    # AsyncCompletions.create
    _orig_async = _oai.AsyncCompletions.create

    async def _openai_async_create(self: Any, *args: Any, **kwargs: Any) -> Any:
        t0 = time.monotonic()
        response = await _orig_async(self, *args, **kwargs)
        latency_ms = (time.monotonic() - t0) * 1000

        model = kwargs.get("model", "unknown")
        messages = kwargs.get("messages", [])
        response_text = ""
        input_tokens = 0
        output_tokens = 0
        tool_calls: list[dict[str, Any]] = []

        try:
            choice = response.choices[0]
            msg = choice.message
            response_text = msg.content or ""
            if msg.tool_calls:
                for tc in msg.tool_calls:
                    tool_calls.append({
                        "name": tc.function.name,
                        "arguments": tc.function.arguments[:200],
                    })
        except (AttributeError, IndexError):
            pass

        try:
            if response.usage:
                input_tokens = response.usage.prompt_tokens or 0
                output_tokens = response.usage.completion_tokens or 0
        except AttributeError:
            pass

        _record({
            "type": "llm_call",
            "provider": "openai",
            "model": model,
            "prompt_preview": _truncate_messages(messages),
            "response_preview": response_text[:300],
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "latency_ms": round(latency_ms, 1),
        })

        for tc in tool_calls:
            _record({
                "type": "tool_call",
                "source": "llm_request",
                "name": tc["name"],
                "arguments_preview": tc["arguments"],
            })

        return response

    _register("openai_async_create", _oai.AsyncCompletions, "create", _openai_async_create)

    # Completions.create (sync)
    _orig_sync = _oai.Completions.create

    def _openai_sync_create(self: Any, *args: Any, **kwargs: Any) -> Any:
        t0 = time.monotonic()
        response = _orig_sync(self, *args, **kwargs)
        latency_ms = (time.monotonic() - t0) * 1000

        model = kwargs.get("model", "unknown")
        messages = kwargs.get("messages", [])
        response_text = ""
        input_tokens = 0
        output_tokens = 0
        tool_calls: list[dict[str, Any]] = []

        try:
            choice = response.choices[0]
            msg = choice.message
            response_text = msg.content or ""
            if msg.tool_calls:
                for tc in msg.tool_calls:
                    tool_calls.append({
                        "name": tc.function.name,
                        "arguments": tc.function.arguments[:200],
                    })
        except (AttributeError, IndexError):
            pass

        try:
            if response.usage:
                input_tokens = response.usage.prompt_tokens or 0
                output_tokens = response.usage.completion_tokens or 0
        except AttributeError:
            pass

        _record({
            "type": "llm_call",
            "provider": "openai",
            "model": model,
            "prompt_preview": _truncate_messages(messages),
            "response_preview": response_text[:300],
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "latency_ms": round(latency_ms, 1),
        })

        for tc in tool_calls:
            _record({
                "type": "tool_call",
                "source": "llm_request",
                "name": tc["name"],
                "arguments_preview": tc["arguments"],
            })

        return response

    _register("openai_sync_create", _oai.Completions, "create", _openai_sync_create)


# ---------------------------------------------------------------------------
# Anthropic patches
# ---------------------------------------------------------------------------

def _try_patch_anthropic() -> None:
    try:
        import anthropic.resources.messages as _anth  # noqa: PLC0415
    except ImportError:
        return

    # AsyncMessages.create
    _orig_async = _anth.AsyncMessages.create

    async def _anthropic_async_create(self: Any, *args: Any, **kwargs: Any) -> Any:
        t0 = time.monotonic()
        response = await _orig_async(self, *args, **kwargs)
        latency_ms = (time.monotonic() - t0) * 1000

        model = kwargs.get("model", "unknown")
        messages = kwargs.get("messages", [])
        response_text = ""
        input_tokens = 0
        output_tokens = 0
        tool_calls: list[dict[str, Any]] = []

        try:
            for block in response.content:
                if hasattr(block, "text"):
                    response_text = block.text[:300]
                    break
                if hasattr(block, "type") and block.type == "tool_use":
                    tool_calls.append({
                        "name": block.name,
                        "arguments": str(getattr(block, "input", ""))[:200],
                    })
        except (AttributeError, TypeError):
            pass

        try:
            if response.usage:
                input_tokens = getattr(response.usage, "input_tokens", 0) or 0
                output_tokens = getattr(response.usage, "output_tokens", 0) or 0
        except AttributeError:
            pass

        _record({
            "type": "llm_call",
            "provider": "anthropic",
            "model": model,
            "prompt_preview": _truncate_messages(messages),
            "response_preview": response_text,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "latency_ms": round(latency_ms, 1),
        })

        for tc in tool_calls:
            _record({
                "type": "tool_call",
                "source": "llm_request",
                "name": tc["name"],
                "arguments_preview": tc["arguments"],
            })

        return response

    _register("anthropic_async_create", _anth.AsyncMessages, "create", _anthropic_async_create)

    # Messages.create (sync)
    _orig_sync = _anth.Messages.create

    def _anthropic_sync_create(self: Any, *args: Any, **kwargs: Any) -> Any:
        t0 = time.monotonic()
        response = _orig_sync(self, *args, **kwargs)
        latency_ms = (time.monotonic() - t0) * 1000

        model = kwargs.get("model", "unknown")
        messages = kwargs.get("messages", [])
        response_text = ""
        input_tokens = 0
        output_tokens = 0
        tool_calls: list[dict[str, Any]] = []

        try:
            for block in response.content:
                if hasattr(block, "text"):
                    response_text = block.text[:300]
                    break
                if hasattr(block, "type") and block.type == "tool_use":
                    tool_calls.append({
                        "name": block.name,
                        "arguments": str(getattr(block, "input", ""))[:200],
                    })
        except (AttributeError, TypeError):
            pass

        try:
            if response.usage:
                input_tokens = getattr(response.usage, "input_tokens", 0) or 0
                output_tokens = getattr(response.usage, "output_tokens", 0) or 0
        except AttributeError:
            pass

        _record({
            "type": "llm_call",
            "provider": "anthropic",
            "model": model,
            "prompt_preview": _truncate_messages(messages),
            "response_preview": response_text,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "latency_ms": round(latency_ms, 1),
        })

        for tc in tool_calls:
            _record({
                "type": "tool_call",
                "source": "llm_request",
                "name": tc["name"],
                "arguments_preview": tc["arguments"],
            })

        return response

    _register("anthropic_sync_create", _anth.Messages, "create", _anthropic_sync_create)
