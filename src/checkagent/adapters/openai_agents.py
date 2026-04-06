"""OpenAI Agents SDK adapter — wraps Runner.run() as AgentAdapter.

Requires ``openai-agents`` to be installed. Converts the OpenAI Agents SDK
RunResult into CheckAgent's AgentRun trace format.

Streaming is supported via ``Runner.run_streamed()``.
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


def _ensure_openai_agents() -> None:
    """Raise a clear error if openai-agents is not installed."""
    try:
        import agents  # noqa: F401
    except ImportError:
        raise ImportError(
            "OpenAIAgentsAdapter requires openai-agents. "
            "Install it with: pip install openai-agents"
        ) from None


def _extract_steps(result: Any) -> list[Step]:
    """Convert RunResult items into CheckAgent Steps."""
    steps: list[Step] = []

    # RunResult has .new_items — list of RunItem objects
    items = getattr(result, "new_items", []) or []
    step_idx = 0

    for item in items:
        item_type = getattr(item, "type", None)

        if item_type == "message_output_item":
            # MessageOutputItem — contains the agent's text response
            raw_item = getattr(item, "raw_item", None)
            content = ""
            if raw_item and hasattr(raw_item, "content"):
                parts = raw_item.content or []
                content = "".join(
                    getattr(p, "text", "") for p in parts
                    if getattr(p, "type", "") == "output_text"
                )
            steps.append(Step(
                step_index=step_idx,
                output_text=content,
            ))
            step_idx += 1

        elif item_type == "tool_call_item":
            raw_item = getattr(item, "raw_item", None)
            tc = ToolCall(
                name=getattr(raw_item, "name", "unknown") if raw_item else "unknown",
                arguments=_parse_arguments(
                    getattr(raw_item, "arguments", "{}") if raw_item else "{}"
                ),
            )
            # Attach to previous step or create new one
            if steps:
                steps[-1].tool_calls.append(tc)
            else:
                steps.append(Step(step_index=step_idx, tool_calls=[tc]))
                step_idx += 1

        elif item_type == "tool_call_output_item":
            # Tool result — attach to the most recent tool call
            output = getattr(item, "output", None)
            if steps:
                for tc in reversed(steps[-1].tool_calls):
                    if tc.result is None:
                        tc.result = output
                        break

    return steps


def _parse_arguments(raw: str | dict[str, Any]) -> dict[str, Any]:
    """Parse tool call arguments from string or dict."""
    if isinstance(raw, dict):
        return raw
    try:
        import json
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return {"raw": raw}


def _get_final_output(result: Any) -> Any:
    """Extract the final output from a RunResult."""
    # RunResult.final_output is the primary output
    return getattr(result, "final_output", None)


def _get_token_usage(result: Any) -> dict[str, int | None]:
    """Extract token usage from RunResult."""
    usage: dict[str, int | None] = {
        "prompt_tokens": None,
        "completion_tokens": None,
    }
    # RunResult has .raw_responses with usage info
    raw_responses = getattr(result, "raw_responses", []) or []
    total_prompt = 0
    total_completion = 0
    has_usage = False

    for resp in raw_responses:
        resp_usage = getattr(resp, "usage", None)
        if resp_usage:
            has_usage = True
            total_prompt += getattr(resp_usage, "input_tokens", 0) or 0
            total_completion += getattr(resp_usage, "output_tokens", 0) or 0

    if has_usage:
        usage["prompt_tokens"] = total_prompt
        usage["completion_tokens"] = total_completion

    return usage


class OpenAIAgentsAdapter:
    """Wraps an OpenAI Agents SDK Agent as an AgentAdapter.

    Parameters
    ----------
    agent : Any
        An ``agents.Agent`` instance.
    """

    def __init__(self, agent: Any) -> None:
        _ensure_openai_agents()
        self._agent = agent

    async def run(self, input: AgentInput | str) -> AgentRun:
        """Execute the agent via Runner.run() and return an AgentRun trace."""
        if isinstance(input, str):
            input = AgentInput(query=input)

        from agents import Runner

        start = time.monotonic()
        try:
            result = await Runner.run(self._agent, input=input.query)
        except Exception as exc:
            elapsed = (time.monotonic() - start) * 1000
            return AgentRun(
                input=input,
                error=f"{type(exc).__name__}: {exc}",
                duration_ms=elapsed,
            )

        elapsed = (time.monotonic() - start) * 1000
        steps = _extract_steps(result)
        tokens = _get_token_usage(result)
        final = _get_final_output(result)

        return AgentRun(
            input=input,
            steps=steps,
            final_output=final,
            duration_ms=elapsed,
            total_prompt_tokens=tokens["prompt_tokens"],
            total_completion_tokens=tokens["completion_tokens"],
        )

    async def run_stream(self, input: AgentInput | str) -> AsyncIterator[StreamEvent]:
        """Stream execution via Runner.run_streamed()."""
        if isinstance(input, str):
            input = AgentInput(query=input)

        from agents import Runner

        yield StreamEvent(event_type=StreamEventType.RUN_START)

        try:
            result = Runner.run_streamed(self._agent, input=input.query)

            async for event in result.stream_events():
                event_type = getattr(event, "type", "")

                if event_type == "raw_response_event":
                    data = getattr(event, "data", None)
                    if data and hasattr(data, "delta"):
                        yield StreamEvent(
                            event_type=StreamEventType.TEXT_DELTA,
                            data=data.delta,
                        )
                elif event_type == "run_item_stream_event":
                    item = getattr(event, "item", None)
                    item_type = getattr(item, "type", "") if item else ""
                    if item_type == "tool_call_item":
                        raw = getattr(item, "raw_item", None)
                        yield StreamEvent(
                            event_type=StreamEventType.TOOL_CALL_START,
                            data={"name": getattr(raw, "name", "unknown") if raw else "unknown"},
                        )
                    elif item_type == "tool_call_output_item":
                        yield StreamEvent(
                            event_type=StreamEventType.TOOL_RESULT,
                            data=getattr(item, "output", None),
                        )

        except Exception as exc:
            yield StreamEvent(
                event_type=StreamEventType.ERROR,
                data=f"{type(exc).__name__}: {exc}",
            )

        yield StreamEvent(event_type=StreamEventType.RUN_END)
