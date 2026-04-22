"""Generic adapter — wraps any Python callable as an AgentAdapter.

Supports both sync and async callables. Sync functions are executed
in a thread pool executor to avoid blocking the event loop.

Non-callable framework agent objects (PydanticAI, LangChain, CrewAI,
OpenAI Agents) are auto-detected and routed to the correct adapter.
"""

from __future__ import annotations

import asyncio
import functools
import inspect
import time
from collections.abc import AsyncIterator, Callable
from typing import Any, overload

from checkagent.core.types import AgentInput, AgentRun, Step, StreamEvent, StreamEventType

# Maps module prefix → (adapter class name, import path, constructor arg name)
_FRAMEWORK_ADAPTERS: list[tuple[str, str, str]] = [
    ("pydantic_ai", "PydanticAIAdapter", "checkagent.adapters.pydantic_ai"),
    ("langchain", "LangChainAdapter", "checkagent.adapters.langchain"),
    ("crewai", "CrewAIAdapter", "checkagent.adapters.crewai"),
    ("agents", "OpenAIAgentsAdapter", "checkagent.adapters.openai_agents"),
]


def _try_framework_adapter(obj: Any) -> Any | None:
    """Return a framework-specific adapter if obj's type is recognised.

    Checks the object's module against known framework prefixes and
    lazily imports the matching adapter. Returns None if unrecognised.
    """
    module = getattr(type(obj), "__module__", None) or ""
    for prefix, adapter_name, import_path in _FRAMEWORK_ADAPTERS:
        # Match: exact, dot-separated subpackage (langchain.chains), or
        # underscore-separated variant (langchain_core, langchain_community)
        if module == prefix or module.startswith(prefix + ".") or module.startswith(prefix + "_"):
            import importlib
            mod = importlib.import_module(import_path)
            adapter_cls = getattr(mod, adapter_name)
            return adapter_cls(obj)
    return None


def _non_callable_error(obj: Any) -> TypeError:
    """Build a descriptive TypeError for non-callable objects."""
    type_name = f"{type(obj).__module__}.{type(obj).__qualname__}"
    lines = [
        f"wrap() requires a callable, but got {type_name!r}.",
        "",
        "Framework agent objects are not directly callable.",
        "Use a framework-specific adapter instead:",
        "  from checkagent.adapters.pydantic_ai import PydanticAIAdapter",
        "  from checkagent.adapters.langchain import LangChainAdapter",
        "  from checkagent.adapters.crewai import CrewAIAdapter",
        "  from checkagent.adapters.openai_agents import OpenAIAgentsAdapter",
        "",
        "Or pass a plain function/lambda:",
        "  wrap(lambda query: agent.run_sync(query).output)",
    ]
    return TypeError("\n".join(lines))


class GenericAdapter:
    """Wraps any Python callable to conform to the AgentAdapter protocol.

    The callable should accept a string (the query) and return a string
    or any value that becomes the final_output. Additional kwargs from
    AgentInput.context are forwarded if the callable accepts **kwargs.
    """

    def __init__(self, fn: Callable[..., Any]) -> None:
        if not callable(fn):
            adapter = _try_framework_adapter(fn)
            if adapter is not None:
                raise TypeError(
                    f"wrap() auto-detected {type(fn).__qualname__!r} — "
                    "use the returned adapter directly, not GenericAdapter"
                )
            raise _non_callable_error(fn)
        self._fn = fn
        self._is_async = inspect.iscoroutinefunction(fn)
        self._accepts_kwargs = any(
            p.kind == inspect.Parameter.VAR_KEYWORD
            for p in inspect.signature(fn).parameters.values()
        )

    async def run(self, input: AgentInput | str) -> AgentRun:
        """Execute the wrapped callable and return an AgentRun trace.

        Accepts either an AgentInput or a plain string (converted automatically).
        """
        if isinstance(input, str):
            input = AgentInput(query=input)
        start = time.perf_counter()
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
            elapsed = (time.perf_counter() - start) * 1000
            return AgentRun(
                input=input,
                error=f"{type(exc).__name__}: {exc}",
                duration_ms=elapsed,
            )

        elapsed = (time.perf_counter() - start) * 1000
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


def wrap(
    fn: Any | None = None,
) -> Any:
    """Wrap a callable as a GenericAdapter. Usable as decorator or function.

    For plain callables, returns a GenericAdapter:

        @wrap
        async def my_agent(query: str) -> str: ...

        adapter = wrap(my_sync_function)

    For framework agent objects (PydanticAI, LangChain, CrewAI, OpenAI Agents),
    auto-detects the framework and returns the appropriate adapter:

        from pydantic_ai import Agent
        agent = Agent(model="openai:gpt-4o-mini")
        adapter = wrap(agent)  # returns PydanticAIAdapter(agent)
    """
    if fn is not None:
        if not callable(fn):
            adapter = _try_framework_adapter(fn)
            if adapter is not None:
                return adapter
            raise _non_callable_error(fn)
        return GenericAdapter(fn)
    return GenericAdapter  # type: ignore[return-value]
