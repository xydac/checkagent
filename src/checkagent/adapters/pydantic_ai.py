"""PydanticAI adapter — wraps agent.run() as AgentAdapter.

Requires ``pydantic-ai`` to be installed. Converts PydanticAI's RunResult
into CheckAgent's AgentRun trace format.

Streaming is supported via ``agent.run_stream()``.
"""

from __future__ import annotations

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


def _ensure_pydantic_ai() -> None:
    """Raise a clear error if pydantic-ai is not installed."""
    try:
        import pydantic_ai  # noqa: F401
    except ImportError:
        raise ImportError(
            "PydanticAIAdapter requires pydantic-ai. "
            "Install it with: pip install pydantic-ai"
        ) from None


def _extract_steps(result: Any) -> list[Step]:
    """Extract steps from RunResult messages."""
    steps: list[Step] = []

    # RunResult has .all_messages() returning list of ModelMessage
    messages = []
    if hasattr(result, "all_messages"):
        messages = result.all_messages()
    elif hasattr(result, "messages"):
        messages = result.messages or []

    for i, msg in enumerate(messages):
        kind = getattr(msg, "kind", "") or type(msg).__name__
        parts = getattr(msg, "parts", []) or []

        tool_calls: list[ToolCall] = []
        output_parts: list[str] = []

        for part in parts:
            part_kind = getattr(part, "part_kind", "") or type(part).__name__
            if "tool-call" in part_kind or "ToolCall" in str(type(part)):
                tool_calls.append(ToolCall(
                    name=getattr(part, "tool_name", "unknown"),
                    arguments=getattr(part, "args", {}),
                ))
            elif "tool-return" in part_kind or "ToolReturn" in str(type(part)):
                tool_calls.append(ToolCall(
                    name=getattr(part, "tool_name", "unknown"),
                    arguments={},
                    result=getattr(part, "content", None),
                ))
            elif "text" in part_kind or hasattr(part, "content"):
                content = getattr(part, "content", str(part))
                output_parts.append(str(content))

        step = Step(
            step_index=i,
            input_text="",
            output_text="\n".join(output_parts) if output_parts else "",
            tool_calls=tool_calls,
            metadata={"kind": kind},
        )
        steps.append(step)

    return steps


def _extract_token_usage(result: Any) -> dict[str, int | None]:
    """Extract token usage from RunResult.usage()."""
    usage: dict[str, int | None] = {
        "prompt_tokens": None,
        "completion_tokens": None,
    }
    usage_obj = None
    if callable(getattr(result, "usage", None)):
        usage_obj = result.usage()
    elif hasattr(result, "usage"):
        usage_obj = result.usage

    if usage_obj:
        usage["prompt_tokens"] = getattr(usage_obj, "request_tokens", None)
        usage["completion_tokens"] = getattr(usage_obj, "response_tokens", None)
    return usage


class PydanticAIAdapter:
    """Wraps a PydanticAI Agent as an AgentAdapter.

    Parameters
    ----------
    agent : Any
        A PydanticAI ``Agent`` instance.
    """

    def __init__(self, agent: Any) -> None:
        _ensure_pydantic_ai()
        self._agent = agent

    async def run(self, input: AgentInput | str) -> AgentRun:
        """Execute the agent and return an AgentRun trace."""
        if isinstance(input, str):
            input = AgentInput(query=input)

        start = time.perf_counter()
        try:
            result = await self._agent.run(input.query)
        except Exception as exc:
            elapsed = (time.perf_counter() - start) * 1000
            return AgentRun(
                input=input,
                error=f"{type(exc).__name__}: {exc}",
                duration_ms=elapsed,
            )

        elapsed = (time.perf_counter() - start) * 1000
        steps = _extract_steps(result)
        tokens = _extract_token_usage(result)

        # Final output from result.data or result.output
        final = getattr(result, "data", None) or getattr(result, "output", None)

        return AgentRun(
            input=input,
            steps=steps,
            final_output=final,
            duration_ms=elapsed,
            total_prompt_tokens=tokens["prompt_tokens"],
            total_completion_tokens=tokens["completion_tokens"],
        )

    async def run_stream(self, input: AgentInput | str) -> AsyncIterator[StreamEvent]:
        """Stream execution events via PydanticAI's run_stream()."""
        if isinstance(input, str):
            input = AgentInput(query=input)

        yield StreamEvent(event_type=StreamEventType.RUN_START)

        if not hasattr(self._agent, "run_stream"):
            # Fallback: run and synthesize events
            result = await self.run(input)
            if result.error:
                yield StreamEvent(event_type=StreamEventType.ERROR, data=result.error)
            else:
                yield StreamEvent(
                    event_type=StreamEventType.TEXT_DELTA,
                    data=str(result.final_output),
                    step_index=0,
                )
            yield StreamEvent(event_type=StreamEventType.RUN_END, data=result)
            return

        try:
            async with self._agent.run_stream(input.query) as stream:
                async for text in stream.stream_text():
                    yield StreamEvent(
                        event_type=StreamEventType.TEXT_DELTA,
                        data=text,
                    )
        except Exception as exc:
            yield StreamEvent(
                event_type=StreamEventType.ERROR,
                data=f"{type(exc).__name__}: {exc}",
            )

        yield StreamEvent(event_type=StreamEventType.RUN_END)
