"""Generic adapter — wraps any Python callable as an AgentAdapter.

Supports both sync and async callables. Sync functions are executed
in a thread pool executor to avoid blocking the event loop.
"""

from __future__ import annotations

import asyncio
import functools
import inspect
import time
from typing import Any, AsyncIterator, Callable, overload

from checkagent.core.types import AgentInput, AgentRun, Step, StreamEvent, StreamEventType


class GenericAdapter:
    """Wraps any Python callable to conform to the AgentAdapter protocol.

    The callable should accept a string (the query) and return a string
    or any value that becomes the final_output. Additional kwargs from
    AgentInput.context are forwarded if the callable accepts **kwargs.
    """

    def __init__(self, fn: Callable[..., Any]) -> None:
        self._fn = fn
        self._is_async = inspect.iscoroutinefunction(fn)
        self._accepts_kwargs = any(
            p.kind == inspect.Parameter.VAR_KEYWORD
            for p in inspect.signature(fn).parameters.values()
        )

    async def run(self, input: AgentInput) -> AgentRun:
        """Execute the wrapped callable and return an AgentRun trace."""
        start = time.monotonic()
        kwargs = input.context if self._accepts_kwargs else {}

        try:
            if self._is_async:
                result = await self._fn(input.query, **kwargs)
            else:
                loop = asyncio.get_running_loop()
                result = await loop.run_in_executor(
                    None, functools.partial(self._fn, input.query, **kwargs)
                )
        except Exception as exc:
            elapsed = (time.monotonic() - start) * 1000
            return AgentRun(
                input=input,
                error=f"{type(exc).__name__}: {exc}",
                duration_ms=elapsed,
            )

        elapsed = (time.monotonic() - start) * 1000
        return AgentRun(
            input=input,
            steps=[Step(step_index=0, input_text=input.query, output_text=str(result))],
            final_output=result,
            duration_ms=elapsed,
        )

    async def run_stream(self, input: AgentInput) -> AsyncIterator[StreamEvent]:
        """Fallback stream — runs the callable and synthesizes events."""
        yield StreamEvent(event_type=StreamEventType.RUN_START)

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


@overload
def wrap(fn: Callable[..., Any]) -> GenericAdapter: ...


@overload
def wrap() -> Callable[[Callable[..., Any]], GenericAdapter]: ...


def wrap(fn: Callable[..., Any] | None = None) -> GenericAdapter | Callable[[Callable[..., Any]], GenericAdapter]:
    """Wrap a callable as a GenericAdapter. Usable as decorator or function.

    @wrap
    async def my_agent(query: str) -> str:
        ...

    # or
    adapter = wrap(my_sync_function)
    """
    if fn is not None:
        return GenericAdapter(fn)
    return GenericAdapter  # type: ignore[return-value]
