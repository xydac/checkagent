"""MockLLM — drop-in LLM replacement with pattern-based responses.

Supports exact, substring, and regex pattern matching, sequential
responses, default fallbacks, and full call recording for assertions.

Implements F1.1 from the PRD.
"""

from __future__ import annotations

import re
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


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


class LLMCall(BaseModel):
    """A recorded LLM call for assertion/inspection."""

    input_text: str
    response_text: str
    model: str | None = None
    rule_pattern: str | None = None
    was_default: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)


class MockLLM:
    """A mock LLM provider that returns preconfigured responses.

    Responses are matched by pattern rules (regex, substring, exact match)
    or returned from a default response. All calls are recorded for
    assertions in tests.

    Usage::

        llm = MockLLM(default_response="I don't know")
        llm.add_rule("weather", "It's sunny today")
        llm.add_rule(r"book.*flight", "Flight booked!", match_mode=MatchMode.REGEX)

        response = await llm.complete("What's the weather?")
        assert response == "It's sunny today"
        assert llm.call_count == 1

    Sequential responses::

        llm.add_rule("hello", ["Hi!", "Hey there!", "Greetings!"])
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
        self._calls: list[LLMCall] = []

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

    def _find_rule(self, text: str) -> ResponseRule | None:
        """Find the first matching rule for the given text."""
        for rule in self._rules:
            if rule.matches(text):
                return rule
        return None

    async def complete(self, text: str) -> str:
        """Generate a mock completion for the given input text."""
        rule = self._find_rule(text)

        if rule is not None:
            response_text = rule.get_response()
            model = rule.model or self.default_model
            self._calls.append(
                LLMCall(
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
            LLMCall(
                input_text=text,
                response_text=self.default_response,
                model=self.default_model,
                was_default=True,
            )
        )
        return self.default_response

    def complete_sync(self, text: str) -> str:
        """Synchronous version of complete for non-async agents."""
        rule = self._find_rule(text)

        if rule is not None:
            response_text = rule.get_response()
            model = rule.model or self.default_model
            self._calls.append(
                LLMCall(
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
            LLMCall(
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
