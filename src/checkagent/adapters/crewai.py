"""CrewAI adapter — wraps Crew.kickoff() as AgentAdapter.

Requires ``crewai`` to be installed. Converts CrewAI's CrewOutput
into CheckAgent's AgentRun trace format.

Streaming is synthesized from the final result since CrewAI does not
expose a streaming API for individual steps.
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


def _ensure_crewai() -> None:
    """Raise a clear error if crewai is not installed."""
    try:
        import crewai  # noqa: F401
    except ImportError:
        raise ImportError(
            "CrewAIAdapter requires crewai. "
            "Install it with: pip install crewai"
        ) from None


def _extract_steps(result: Any) -> list[Step]:
    """Extract steps from CrewOutput.tasks_output or agent interactions."""
    steps: list[Step] = []

    # CrewOutput.tasks_output is a list of TaskOutput objects
    tasks_output = getattr(result, "tasks_output", None) or []
    for i, task_out in enumerate(tasks_output):
        raw_text = getattr(task_out, "raw", "") or ""
        description = getattr(task_out, "description", "") or ""
        agent_name = getattr(task_out, "agent", "") or ""

        step = Step(
            step_index=i,
            input_text=description,
            output_text=str(raw_text),
            metadata={"agent": str(agent_name)} if agent_name else {},
        )
        steps.append(step)

    # If no tasks_output, create a single step from the raw result
    if not steps:
        raw = getattr(result, "raw", str(result))
        steps.append(Step(step_index=0, input_text="", output_text=str(raw)))

    return steps


def _extract_tool_calls(result: Any) -> list[ToolCall]:
    """Extract tool calls from CrewOutput if available."""
    calls: list[ToolCall] = []
    tasks_output = getattr(result, "tasks_output", None) or []
    for task_out in tasks_output:
        # CrewAI TaskOutput may have tool_calls or tools_used
        tool_calls = getattr(task_out, "tool_calls", None) or []
        for tc in tool_calls:
            if isinstance(tc, dict):
                calls.append(ToolCall(
                    name=tc.get("name", "unknown"),
                    arguments=tc.get("arguments", tc.get("args", {})),
                ))
            else:
                calls.append(ToolCall(
                    name=getattr(tc, "name", "unknown"),
                    arguments=getattr(tc, "arguments", {}),
                ))
    return calls


def _extract_token_usage(result: Any) -> dict[str, int | None]:
    """Extract token usage from CrewOutput.token_usage."""
    usage: dict[str, int | None] = {
        "prompt_tokens": None,
        "completion_tokens": None,
    }
    token_usage = getattr(result, "token_usage", None)
    if token_usage and isinstance(token_usage, dict):
        usage["prompt_tokens"] = token_usage.get("prompt_tokens")
        usage["completion_tokens"] = token_usage.get("completion_tokens")
    elif token_usage:
        usage["prompt_tokens"] = getattr(token_usage, "prompt_tokens", None)
        usage["completion_tokens"] = getattr(token_usage, "completion_tokens", None)
    return usage


class CrewAIAdapter:
    """Wraps a CrewAI Crew as an AgentAdapter.

    Parameters
    ----------
    crew : Any
        A CrewAI ``Crew`` instance.
    """

    def __init__(self, crew: Any) -> None:
        _ensure_crewai()
        self._crew = crew

    async def run(self, input: AgentInput | str) -> AgentRun:
        """Execute the crew and return an AgentRun trace."""
        if isinstance(input, str):
            input = AgentInput(query=input)

        inputs = {"query": input.query, **input.context}

        start = time.perf_counter()
        try:
            if hasattr(self._crew, "kickoff_async"):
                result = await self._crew.kickoff_async(inputs=inputs)
            else:
                # Sync-only kickoff — run in executor
                import asyncio
                import functools

                loop = asyncio.get_running_loop()
                result = await loop.run_in_executor(
                    None, functools.partial(self._crew.kickoff, inputs=inputs)
                )
        except Exception as exc:
            elapsed = (time.perf_counter() - start) * 1000
            return AgentRun(
                input=input,
                error=f"{type(exc).__name__}: {exc}",
                duration_ms=elapsed,
            )

        elapsed = (time.perf_counter() - start) * 1000
        steps = _extract_steps(result)
        tool_calls = _extract_tool_calls(result)
        tokens = _extract_token_usage(result)

        # Attach tool calls to the last step if any
        if tool_calls and steps:
            steps[-1] = Step(
                step_index=steps[-1].step_index,
                input_text=steps[-1].input_text,
                output_text=steps[-1].output_text,
                tool_calls=tool_calls,
                metadata=steps[-1].metadata,
            )

        final_output = getattr(result, "raw", str(result))

        return AgentRun(
            input=input,
            steps=steps,
            final_output=final_output,
            duration_ms=elapsed,
            total_prompt_tokens=tokens["prompt_tokens"],
            total_completion_tokens=tokens["completion_tokens"],
        )

    async def run_stream(self, input: AgentInput | str) -> AsyncIterator[StreamEvent]:
        """Stream execution events (synthesized from final result)."""
        yield StreamEvent(event_type=StreamEventType.RUN_START)

        try:
            result = await self.run(input)
            if result.error:
                yield StreamEvent(event_type=StreamEventType.ERROR, data=result.error)
            else:
                for step in result.steps:
                    yield StreamEvent(
                        event_type=StreamEventType.TEXT_DELTA,
                        data=step.output_text,
                        step_index=step.step_index,
                    )
            yield StreamEvent(event_type=StreamEventType.RUN_END, data=result)
        except Exception as exc:
            yield StreamEvent(
                event_type=StreamEventType.ERROR,
                data=f"{type(exc).__name__}: {exc}",
            )
            yield StreamEvent(event_type=StreamEventType.RUN_END)
