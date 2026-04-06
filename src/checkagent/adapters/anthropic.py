"""Anthropic Claude SDK adapter — wraps AsyncAnthropic as AgentAdapter.

Requires ``anthropic`` to be installed.
"""

from __future__ import annotations

import asyncio
import functools
import inspect
import time
from collections.abc import AsyncIterator
from typing import Any

from checkagent.core.types import (
    AgentInput,
    AgentRun,
    Step,
    StreamEvent,
    StreamEventType,
    ToolCall,
)


def _ensure_anthropic() -> None:
    """Raise a clear error if anthropic is not installed."""
    try:
        import anthropic  # noqa: F401
    except ImportError:
        raise ImportError(
            "AnthropicAdapter requires anthropic. "
            "Install it with: pip install anthropic"
        ) from None


def _extract_text(message: Any) -> str:
    """Extract text content from an Anthropic Message."""
    content = getattr(message, "content", [])
    texts: list[str] = []
    for block in content:
        if isinstance(block, dict):
            if block.get("type") == "text":
                texts.append(block["text"])
        elif getattr(block, "type", None) == "text" and hasattr(block, "text"):
            texts.append(block.text)
    return "\n".join(texts) if texts else ""


def _extract_tool_calls(message: Any) -> list[ToolCall]:
    """Extract tool use blocks from an Anthropic Message."""
    calls: list[ToolCall] = []
    content = getattr(message, "content", [])
    for block in content:
        if isinstance(block, dict):
            if block.get("type") == "tool_use":
                calls.append(ToolCall(
                    name=block.get("name", "unknown"),
                    arguments=block.get("input", {}),
                ))
        elif getattr(block, "type", None) == "tool_use":
            calls.append(ToolCall(
                name=getattr(block, "name", "unknown"),
                arguments=getattr(block, "input", {}),
            ))
    return calls


def _extract_token_usage(message: Any) -> dict[str, int | None]:
    """Extract token usage from an Anthropic Message."""
    usage_info: dict[str, int | None] = {
        "prompt_tokens": None,
        "completion_tokens": None,
    }
    usage = getattr(message, "usage", None)
    if usage:
        usage_info["prompt_tokens"] = getattr(usage, "input_tokens", None)
        usage_info["completion_tokens"] = getattr(usage, "output_tokens", None)
    return usage_info


class AnthropicAdapter:
    """Wraps an Anthropic client as an AgentAdapter."""

    def __init__(
        self,
        client: Any,
        *,
        model: str = "claude-sonnet-4-20250514",
        system: str | None = None,
        max_tokens: int = 1024,
    ) -> None:
        _ensure_anthropic()
        self._client = client
        self._model = model
        self._system = system
        self._max_tokens = max_tokens

    async def run(self, input: AgentInput | str) -> AgentRun:
        """Send a message and return an AgentRun trace."""
        if isinstance(input, str):
            input = AgentInput(query=input)

        kwargs: dict[str, Any] = {
            "model": self._model,
            "max_tokens": self._max_tokens,
            "messages": [{"role": "user", "content": input.query}],
        }
        if self._system:
            kwargs["system"] = self._system

        start = time.perf_counter()
        try:
            if hasattr(self._client, "messages") and hasattr(
                self._client.messages, "create"
            ):
                create_fn = self._client.messages.create
            else:
                create_fn = self._client.create

            if inspect.iscoroutinefunction(create_fn):
                message = await create_fn(**kwargs)
            else:
                loop = asyncio.get_running_loop()
                message = await loop.run_in_executor(
                    None, functools.partial(create_fn, **kwargs)
                )
        except Exception as exc:
            elapsed = (time.perf_counter() - start) * 1000
            return AgentRun(
                input=input,
                error=f"{type(exc).__name__}: {exc}",
                duration_ms=elapsed,
            )

        elapsed = (time.perf_counter() - start) * 1000
        text = _extract_text(message)
        tool_calls = _extract_tool_calls(message)
        tokens = _extract_token_usage(message)

        step = Step(
            step_index=0,
            input_text=input.query,
            output_text=text,
            tool_calls=tool_calls,
            prompt_tokens=tokens["prompt_tokens"],
            completion_tokens=tokens["completion_tokens"],
        )

        return AgentRun(
            input=input,
            steps=[step],
            final_output=message,
            duration_ms=elapsed,
            total_prompt_tokens=tokens["prompt_tokens"],
            total_completion_tokens=tokens["completion_tokens"],
        )

    async def run_stream(self, input: AgentInput | str) -> AsyncIterator[StreamEvent]:
        """Stream execution events via Anthropic's streaming API."""
        if isinstance(input, str):
            input = AgentInput(query=input)

        kwargs: dict[str, Any] = {
            "model": self._model,
            "max_tokens": self._max_tokens,
            "messages": [{"role": "user", "content": input.query}],
        }
        if self._system:
            kwargs["system"] = self._system

        yield StreamEvent(event_type=StreamEventType.RUN_START)

        try:
            if hasattr(self._client, "messages") and hasattr(
                self._client.messages, "stream"
            ):
                stream_fn = self._client.messages.stream
            else:
                # Fallback: run and synthesize
                result = await self.run(input)
                if result.error:
                    yield StreamEvent(
                        event_type=StreamEventType.ERROR, data=result.error
                    )
                else:
                    yield StreamEvent(
                        event_type=StreamEventType.TEXT_DELTA,
                        data=_extract_text(result.final_output),
                        step_index=0,
                    )
                yield StreamEvent(event_type=StreamEventType.RUN_END, data=result)
                return

            async with stream_fn(**kwargs) as stream:
                async for event in stream:
                    event_type = getattr(event, "type", "")
                    if event_type == "content_block_delta":
                        delta = getattr(event, "delta", None)
                        if delta and getattr(delta, "type", "") == "text_delta":
                            yield StreamEvent(
                                event_type=StreamEventType.TEXT_DELTA,
                                data=getattr(delta, "text", ""),
                            )
        except Exception as exc:
            yield StreamEvent(
                event_type=StreamEventType.ERROR,
                data=f"{type(exc).__name__}: {exc}",
            )

        yield StreamEvent(event_type=StreamEventType.RUN_END)
