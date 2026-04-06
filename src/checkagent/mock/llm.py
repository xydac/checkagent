"""MockLLM — drop-in LLM replacement with pattern-based responses.

Supports exact, substring, and regex pattern matching, sequential
responses, default fallbacks, and full call recording for assertions.
Streaming mode returns async iterators of StreamEvent chunks (F1.6, F13.2).

Implements F1.1, F1.6, F13.2 from the PRD.
"""

from __future__ import annotations

import asyncio
import re
from enum import Enum
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from checkagent.mock.fault import FaultInjector

from checkagent.core.types import StreamEvent, StreamEventType


class MatchMode(str, Enum):
    """How a pattern rule matches against input text."""

    EXACT = "exact"
    SUBSTRING = "substring"
    REGEX = "regex"


class ResponseRule(BaseModel):
    """A single pattern → response mapping."""

    pattern: str
    response: str | list[str]
    match_mode: MatchMode = MatchMode.SUBSTRING
    model: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    # Internal counter for sequence responses — not a Pydantic field
    _call_count: int = 0

    def matches(self, text: str) -> bool:
        """Check if the input text matches this rule's pattern."""
        if self.match_mode == MatchMode.EXACT:
            return text == self.pattern
        elif self.match_mode == MatchMode.SUBSTRING:
            return self.pattern in text
        elif self.match_mode == MatchMode.REGEX:
            return bool(re.search(self.pattern, text))
        return False

    def get_response(self) -> str:
        """Get the next response, cycling through sequences."""
        if isinstance(self.response, list):
            idx = self._call_count % len(self.response)
            self._call_count += 1
            return self.response[idx]
        self._call_count += 1
        return self.response


class StreamConfig(BaseModel):
    """Configuration for streaming a response as chunks."""

    chunks: list[str]
    delay_ms: float = 0
    model: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class StreamRule(BaseModel):
    """A pattern → stream configuration mapping."""

    pattern: str
    stream_config: StreamConfig
    match_mode: MatchMode = MatchMode.SUBSTRING

    def matches(self, text: str) -> bool:
        if self.match_mode == MatchMode.EXACT:
            return text == self.pattern
        elif self.match_mode == MatchMode.SUBSTRING:
            return self.pattern in text
        elif self.match_mode == MatchMode.REGEX:
            return bool(re.search(self.pattern, text))
        return False


class LLMCall(BaseModel):
    """A recorded LLM call for assertion/inspection."""

    input_text: str
    response_text: str
    model: str | None = None
    rule_pattern: str | None = None
    was_default: bool = False
    streamed: bool = False
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def total_tokens(self) -> int | None:
        """Total tokens (prompt + completion), or None if not tracked."""
        if self.prompt_tokens is None and self.completion_tokens is None:
            return None
        return (self.prompt_tokens or 0) + (self.completion_tokens or 0)


class _InputMatcher:
    """Fluent builder returned by ``MockLLM.on_input()``.

    Captures the matching criteria and provides ``.respond()`` to register
    the rule back on the owning :class:`MockLLM`.

    Usage::

        llm.on_input(contains="book a meeting").respond("Meeting booked!")
        llm.on_input(pattern=r"flight.*\\d+").respond("Flight confirmed")
        llm.on_input(exact="hello").respond(["Hi!", "Hey!"])
    """

    def __init__(self, llm: MockLLM, match_mode: MatchMode, pattern: str) -> None:
        self._llm = llm
        self._match_mode = match_mode
        self._pattern = pattern

    def respond(
        self,
        response: str | list[str],
        *,
        model: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> MockLLM:
        """Register the response rule and return the MockLLM for further chaining."""
        return self._llm.add_rule(
            self._pattern,
            response,
            match_mode=self._match_mode,
            model=model,
            metadata=metadata,
        )

    def stream(
        self,
        chunks: list[str],
        *,
        delay_ms: float = 0,
        model: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> MockLLM:
        """Register a streaming response rule and return the MockLLM."""
        return self._llm.stream_response(
            self._pattern,
            chunks,
            delay_ms=delay_ms,
            match_mode=self._match_mode,
            model=model,
            metadata=metadata,
        )


class MockLLM:
    """A mock LLM provider that returns preconfigured responses.

    Responses are matched by pattern rules (regex, substring, exact match)
    or returned from a default response. All calls are recorded for
    assertions in tests.

    Fluent API (recommended)::

        llm = MockLLM(default_response="I don't know")
        llm.on_input(contains="weather").respond("It's sunny today")
        llm.on_input(pattern=r"book.*flight").respond("Flight booked!")

        response = await llm.complete("What's the weather?")
        assert response == "It's sunny today"

    Classic API::

        llm.add_rule("weather", "It's sunny today")
        llm.add_rule(r"book.*flight", "Flight booked!", match_mode=MatchMode.REGEX)

    Sequential responses::

        llm.on_input(contains="hello").respond(["Hi!", "Hey there!", "Greetings!"])
        # First call returns "Hi!", second returns "Hey there!", etc.
    """

    def __init__(
        self,
        *,
        default_response: str = "Mock response",
        default_model: str = "mock-model",
    ) -> None:
        self.default_response = default_response
        self.default_model = default_model
        self._rules: list[ResponseRule] = []
        self._stream_rules: list[StreamRule] = []
        self._calls: list[LLMCall] = []
        self._fault_injector: FaultInjector | None = None
        self._usage: tuple[int, int] | None = None
        self._auto_estimate_tokens: bool = False

    def attach_faults(self, injector: FaultInjector) -> MockLLM:
        """Wire a FaultInjector so LLM faults fire automatically on calls.

        When attached, ``complete()``, ``complete_sync()``, and ``stream()``
        check the injector before responding — no manual ``check_llm()``
        guards needed.

        ::

            llm.attach_faults(fault_injector)
            fault_injector.on_llm().server_error()
            await llm.complete("hello")  # raises LLMServerError automatically
        """
        self._fault_injector = injector
        return self

    def with_usage(
        self,
        *,
        prompt_tokens: int | None = None,
        completion_tokens: int | None = None,
        auto_estimate: bool = False,
    ) -> MockLLM:
        """Configure simulated token usage on recorded calls.

        When set, every ``LLMCall`` recorded by ``complete()``,
        ``complete_sync()``, or ``stream()`` will include token counts.
        This enables testing cost-tracking code paths with MockLLM.

        Args:
            prompt_tokens: Fixed prompt token count per call.
            completion_tokens: Fixed completion token count per call.
            auto_estimate: If True, estimate tokens from text length
                (``len(text) // 4 + 1``) instead of using fixed values.
                Cannot be combined with explicit token counts.

        ::

            llm = MockLLM().with_usage(prompt_tokens=100, completion_tokens=50)
            await llm.complete("hello")
            assert llm.last_call.prompt_tokens == 100

            # Or auto-estimate from text length
            llm = MockLLM().with_usage(auto_estimate=True)
        """
        if auto_estimate and (prompt_tokens is not None or completion_tokens is not None):
            raise ValueError(
                "Cannot set both auto_estimate=True and explicit token counts. "
                "Use auto_estimate=True alone, or set prompt_tokens/completion_tokens."
            )
        if auto_estimate:
            self._auto_estimate_tokens = True
            self._usage = None
        else:
            self._auto_estimate_tokens = False
            self._usage = (prompt_tokens or 0, completion_tokens or 0)
        return self

    def _make_call(self, **kwargs: Any) -> LLMCall:
        """Create an LLMCall with token usage if configured."""
        input_text = kwargs.get("input_text", "")
        response_text = kwargs.get("response_text", "")
        if self._auto_estimate_tokens:
            kwargs.setdefault("prompt_tokens", len(input_text) // 4 + 1)
            kwargs.setdefault("completion_tokens", len(response_text) // 4 + 1)
        elif self._usage is not None:
            kwargs.setdefault("prompt_tokens", self._usage[0])
            kwargs.setdefault("completion_tokens", self._usage[1])
        return LLMCall(**kwargs)

    def on_input(
        self,
        *,
        contains: str | None = None,
        pattern: str | None = None,
        exact: str | None = None,
    ) -> _InputMatcher:
        """Start a fluent rule definition by specifying how to match input.

        Exactly one of ``contains``, ``pattern``, or ``exact`` must be given.

        Returns an :class:`_InputMatcher` whose ``.respond()`` or ``.stream()``
        method completes the rule.

        ::

            llm.on_input(contains="weather").respond("Sunny!")
            llm.on_input(pattern=r"book.*\\d+").respond("Booked")
            llm.on_input(exact="hello").respond("Hi")
        """
        specified = sum(x is not None for x in (contains, pattern, exact))
        if specified != 1:
            raise ValueError(
                "Exactly one of 'contains', 'pattern', or 'exact' must be specified"
            )
        if contains is not None:
            return _InputMatcher(self, MatchMode.SUBSTRING, contains)
        if pattern is not None:
            return _InputMatcher(self, MatchMode.REGEX, pattern)
        # exact
        return _InputMatcher(self, MatchMode.EXACT, exact)  # type: ignore[arg-type]

    def add_rule(
        self,
        pattern: str,
        response: str | list[str],
        *,
        match_mode: MatchMode = MatchMode.SUBSTRING,
        model: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> MockLLM:
        """Add a pattern → response rule. Returns self for chaining."""
        self._rules.append(
            ResponseRule(
                pattern=pattern,
                response=response,
                match_mode=match_mode,
                model=model,
                metadata=metadata or {},
            )
        )
        return self

    def stream_response(
        self,
        pattern: str,
        chunks: list[str],
        *,
        delay_ms: float = 0,
        match_mode: MatchMode = MatchMode.SUBSTRING,
        model: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> MockLLM:
        """Configure a streaming response for a pattern. Returns self for chaining.

        When ``stream()`` is called with text matching *pattern*, an async
        iterator yields ``StreamEvent(TEXT_DELTA)`` for each chunk, wrapped
        by ``RUN_START`` / ``RUN_END`` events.

        ::

            llm.stream_response("weather", ["It's ", "sunny ", "today!"], delay_ms=10)
            async for event in llm.stream("What's the weather?"):
                print(event)
        """
        self._stream_rules.append(
            StreamRule(
                pattern=pattern,
                stream_config=StreamConfig(
                    chunks=chunks,
                    delay_ms=delay_ms,
                    model=model,
                    metadata=metadata or {},
                ),
                match_mode=match_mode,
            )
        )
        return self

    def _find_stream_rule(self, text: str) -> StreamRule | None:
        for rule in self._stream_rules:
            if rule.matches(text):
                return rule
        return None

    def stream(self, text: str) -> _StreamIterator:
        """Stream a mock response as an async iterator of StreamEvent chunks.

        Looks up stream rules first. If no stream rule matches, falls back
        to the regular response rules (or default) and yields the full
        response as a single TEXT_DELTA chunk.

        If a FaultInjector is attached, checks for LLM faults first.

        This is a synchronous method that returns an async iterator — no
        ``await`` needed::

            async for event in llm.stream("hello"):
                ...
        """
        if self._fault_injector is not None:
            self._fault_injector.check_llm()
        stream_rule = self._find_stream_rule(text)

        if stream_rule is not None:
            config = stream_rule.stream_config
            model = config.model or self.default_model
            full_text = "".join(config.chunks)

            self._calls.append(
                self._make_call(
                    input_text=text,
                    response_text=full_text,
                    model=model,
                    rule_pattern=stream_rule.pattern,
                    was_default=False,
                    streamed=True,
                    metadata=config.metadata,
                )
            )

            return _StreamIterator(config.chunks, config.delay_ms)

        # No stream rule — fall back to regular rules / default
        rule = self._find_rule(text)
        if rule is not None:
            response_text = rule.get_response()
            model = rule.model or self.default_model
            self._calls.append(
                self._make_call(
                    input_text=text,
                    response_text=response_text,
                    model=model,
                    rule_pattern=rule.pattern,
                    was_default=False,
                    streamed=True,
                    metadata=rule.metadata,
                )
            )
        else:
            response_text = self.default_response
            model = self.default_model
            self._calls.append(
                self._make_call(
                    input_text=text,
                    response_text=response_text,
                    model=model,
                    was_default=True,
                    streamed=True,
                )
            )

        return _StreamIterator([response_text], 0)

    def _find_rule(self, text: str) -> ResponseRule | None:
        """Find the first matching rule for the given text."""
        for rule in self._rules:
            if rule.matches(text):
                return rule
        return None

    async def complete(self, text: str) -> str:
        """Generate a mock completion for the given input text.

        If a FaultInjector is attached, checks for LLM faults first.
        """
        if self._fault_injector is not None:
            self._fault_injector.check_llm()
        rule = self._find_rule(text)

        if rule is not None:
            response_text = rule.get_response()
            model = rule.model or self.default_model
            self._calls.append(
                self._make_call(
                    input_text=text,
                    response_text=response_text,
                    model=model,
                    rule_pattern=rule.pattern,
                    was_default=False,
                    metadata=rule.metadata,
                )
            )
            return response_text

        # No rule matched — use default
        self._calls.append(
            self._make_call(
                input_text=text,
                response_text=self.default_response,
                model=self.default_model,
                was_default=True,
            )
        )
        return self.default_response

    def complete_sync(self, text: str) -> str:
        """Synchronous version of complete for non-async agents."""
        if self._fault_injector is not None:
            self._fault_injector.check_llm()
        rule = self._find_rule(text)

        if rule is not None:
            response_text = rule.get_response()
            model = rule.model or self.default_model
            self._calls.append(
                self._make_call(
                    input_text=text,
                    response_text=response_text,
                    model=model,
                    rule_pattern=rule.pattern,
                    was_default=False,
                    metadata=rule.metadata,
                )
            )
            return response_text

        self._calls.append(
            self._make_call(
                input_text=text,
                response_text=self.default_response,
                model=self.default_model,
                was_default=True,
            )
        )
        return self.default_response

    # --- Inspection / assertion helpers ---

    @property
    def calls(self) -> list[LLMCall]:
        """All recorded calls."""
        return list(self._calls)

    @property
    def call_count(self) -> int:
        """Total number of calls made."""
        return len(self._calls)

    @property
    def last_call(self) -> LLMCall | None:
        """The most recent call, or None if no calls have been made."""
        return self._calls[-1] if self._calls else None

    def get_calls_matching(self, pattern: str) -> list[LLMCall]:
        """Get all calls whose input_text contains the given substring."""
        return [c for c in self._calls if pattern in c.input_text]

    def was_called_with(self, text: str) -> bool:
        """Check if any call had the exact input_text."""
        return any(c.input_text == text for c in self._calls)

    def reset(self) -> None:
        """Clear all recorded calls and reset rule sequence counters."""
        self._calls.clear()
        for rule in self._rules:
            rule._call_count = 0

    def reset_calls(self) -> None:
        """Clear recorded calls but keep rule sequence counters."""
        self._calls.clear()


class _StreamIterator:
    """Async iterator that yields StreamEvent chunks with optional delay."""

    def __init__(self, chunks: list[str], delay_ms: float) -> None:
        self._chunks = chunks
        self._delay_ms = delay_ms
        self._index = 0
        self._started = False
        self._finished = False

    def __aiter__(self) -> _StreamIterator:
        return self

    async def __anext__(self) -> StreamEvent:
        if not self._started:
            self._started = True
            return StreamEvent(event_type=StreamEventType.RUN_START)

        if self._index < len(self._chunks):
            if self._delay_ms > 0 and self._index > 0:
                await asyncio.sleep(self._delay_ms / 1000)
            chunk = self._chunks[self._index]
            self._index += 1
            return StreamEvent(
                event_type=StreamEventType.TEXT_DELTA,
                data=chunk,
            )

        if not self._finished:
            self._finished = True
            return StreamEvent(event_type=StreamEventType.RUN_END)

        raise StopAsyncIteration
