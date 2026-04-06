"""LangChain/LangGraph adapter — wraps Runnable and StateGraph as AgentAdapter.

Requires ``langchain-core`` to be installed. This adapter auto-detects
RunnableSequence, StateGraph, and plain Runnable objects and converts
their output into CheckAgent's AgentRun trace format.

Streaming is supported via ``astream_events`` (LangChain v2 events API).
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


def _ensure_langchain() -> None:
    """Raise a clear error if langchain-core is not installed."""
    try:
        import langchain_core  # noqa: F401
    except ImportError:
        raise ImportError(
            "LangChainAdapter requires langchain-core. "
            "Install it with: pip install langchain-core"
        ) from None


def _extract_tool_calls(raw_output: Any) -> list[ToolCall]:
    """Extract tool calls from LangChain AIMessage or dict output."""
    calls: list[ToolCall] = []
    # AIMessage with tool_calls attribute
    if hasattr(raw_output, "tool_calls"):
        for tc in raw_output.tool_calls:
            if isinstance(tc, dict):
                calls.append(ToolCall(
                    name=tc.get("name", "unknown"),
                    arguments=tc.get("args", {}),
                ))
            else:
                calls.append(ToolCall(
                    name=getattr(tc, "name", "unknown"),
                    arguments=getattr(tc, "args", {}),
                ))
    return calls


def _extract_text(raw_output: Any) -> str:
    """Extract text content from various LangChain output types."""
    if isinstance(raw_output, str):
        return raw_output
    if hasattr(raw_output, "content"):
        return str(raw_output.content)
    if isinstance(raw_output, dict):
        # LangGraph state dicts — try common keys
        for key in ("output", "result", "response", "messages"):
            if key in raw_output:
                val = raw_output[key]
                if isinstance(val, list) and val:
                    return _extract_text(val[-1])
                return str(val)
        return str(raw_output)
    return str(raw_output)


def _extract_token_usage(raw_output: Any) -> dict[str, int | None]:
    """Extract token usage from LangChain output if available."""
    usage: dict[str, int | None] = {
        "prompt_tokens": None,
        "completion_tokens": None,
    }
    # AIMessage with usage_metadata
    if hasattr(raw_output, "usage_metadata") and raw_output.usage_metadata:
        meta = raw_output.usage_metadata
        if isinstance(meta, dict):
            usage["prompt_tokens"] = meta.get("input_tokens")
            usage["completion_tokens"] = meta.get("output_tokens")
        else:
            usage["prompt_tokens"] = getattr(meta, "input_tokens", None)
            usage["completion_tokens"] = getattr(meta, "output_tokens", None)
    # response_metadata fallback
    elif hasattr(raw_output, "response_metadata"):
        rm = raw_output.response_metadata or {}
        if "token_usage" in rm:
            tu = rm["token_usage"]
            usage["prompt_tokens"] = tu.get("prompt_tokens")
            usage["completion_tokens"] = tu.get("completion_tokens")
    return usage


class LangChainAdapter:
    """Wraps a LangChain Runnable (chain, agent, or graph) as an AgentAdapter.

    Parameters
    ----------
    runnable : Any
        A LangChain ``Runnable`` — could be a chain, agent executor,
        ``RunnableSequence``, or LangGraph ``StateGraph`` / compiled graph.
    input_key : str
        Key used to pass the query into the runnable's invoke dict.
        Defaults to ``"input"``.
    extra_inputs : dict[str, Any] | None
        Additional variables to include in every invocation dict. Useful for
        multi-variable prompt templates (e.g. RAG chains needing ``context``).
        Per-run values from ``AgentInput.context`` override these defaults.
    """

    def __init__(
        self,
        runnable: Any,
        *,
        input_key: str = "input",
        extra_inputs: dict[str, Any] | None = None,
    ) -> None:
        _ensure_langchain()
        self._runnable = runnable
        self._input_key = input_key
        self._extra_inputs: dict[str, Any] = extra_inputs or {}

    def _build_invoke_input(self, input: AgentInput) -> dict[str, Any] | str:
        """Build the invocation dict, merging extra_inputs and context."""
        if self._input_key == "__raw__":
            return input.query
        # Merge order: extra_inputs (defaults) < context (per-run) < input_key
        merged = {**self._extra_inputs, **input.context}
        merged[self._input_key] = input.query
        return merged

    async def run(self, input: AgentInput | str) -> AgentRun:
        """Execute the runnable and return an AgentRun trace."""
        if isinstance(input, str):
            input = AgentInput(query=input)

        invoke_input = self._build_invoke_input(input)

        start = time.perf_counter()
        try:
            if hasattr(self._runnable, "ainvoke"):
                raw = await self._runnable.ainvoke(invoke_input)
            else:
                # Sync-only runnable
                import asyncio
                import functools

                loop = asyncio.get_running_loop()
                raw = await loop.run_in_executor(
                    None, functools.partial(self._runnable.invoke, invoke_input)
                )
        except Exception as exc:
            elapsed = (time.perf_counter() - start) * 1000
            return AgentRun(
                input=input,
                error=f"{type(exc).__name__}: {exc}",
                duration_ms=elapsed,
            )

        elapsed = (time.perf_counter() - start) * 1000
        text = _extract_text(raw)
        tool_calls = _extract_tool_calls(raw)
        tokens = _extract_token_usage(raw)

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
            final_output=raw,
            duration_ms=elapsed,
            total_prompt_tokens=tokens["prompt_tokens"],
            total_completion_tokens=tokens["completion_tokens"],
        )

    async def run_stream(self, input: AgentInput | str) -> AsyncIterator[StreamEvent]:
        """Stream execution events via LangChain's astream_events API."""
        if isinstance(input, str):
            input = AgentInput(query=input)

        invoke_input = self._build_invoke_input(input)

        yield StreamEvent(event_type=StreamEventType.RUN_START)

        if not hasattr(self._runnable, "astream_events"):
            # Fallback: run and synthesize events
            result = await self.run(input)
            if result.error:
                yield StreamEvent(event_type=StreamEventType.ERROR, data=result.error)
            else:
                yield StreamEvent(
                    event_type=StreamEventType.TEXT_DELTA,
                    data=_extract_text(result.final_output),
                    step_index=0,
                )
            yield StreamEvent(event_type=StreamEventType.RUN_END, data=result)
            return

        try:
            async for event in self._runnable.astream_events(
                invoke_input, version="v2"
            ):
                kind = event.get("event", "")
                data = event.get("data", {})

                if kind == "on_chat_model_stream":
                    chunk = data.get("chunk")
                    if chunk and hasattr(chunk, "content"):
                        yield StreamEvent(
                            event_type=StreamEventType.TEXT_DELTA,
                            data=chunk.content,
                        )
                elif kind == "on_tool_start":
                    yield StreamEvent(
                        event_type=StreamEventType.TOOL_CALL_START,
                        data={"name": event.get("name", "unknown")},
                    )
                elif kind == "on_tool_end":
                    yield StreamEvent(
                        event_type=StreamEventType.TOOL_RESULT,
                        data=data.get("output"),
                    )
        except Exception as exc:
            yield StreamEvent(
                event_type=StreamEventType.ERROR,
                data=f"{type(exc).__name__}: {exc}",
            )

        yield StreamEvent(event_type=StreamEventType.RUN_END)
