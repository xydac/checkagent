"""FaultInjector — simulate failures in tool calls and LLM requests.

Provides a fluent API for injecting faults at tool and LLM boundaries:
timeouts, rate limits, malformed responses, intermittent failures, and more.

Implements F1.7, F14.1, F14.2 from the PRD.
"""

from __future__ import annotations

import asyncio
import random
import time
from enum import Enum
from typing import Any

from pydantic import BaseModel


class FaultType(str, Enum):
    """Types of faults that can be injected."""

    TIMEOUT = "timeout"
    RATE_LIMIT = "rate_limit"
    MALFORMED = "malformed"
    INTERMITTENT = "intermittent"
    SLOW = "slow"
    EMPTY = "empty"
    CONTEXT_OVERFLOW = "context_overflow"
    PARTIAL_RESPONSE = "partial_response"
    SERVER_ERROR = "server_error"
    CONTENT_FILTER = "content_filter"


class FaultRecord(BaseModel):
    """A recorded fault injection event."""

    target: str  # tool name or "llm"
    fault_type: FaultType
    triggered: bool = False
    call_index: int = 0  # which call number triggered this


class FaultConfig(BaseModel):
    """Configuration for a single fault rule."""

    fault_type: FaultType
    timeout_seconds: float = 5.0
    after_n: int = 0  # for rate_limit: succeed N times first
    fail_rate: float = 0.5  # for intermittent
    latency_ms: float = 100.0  # for slow
    malformed_data: Any = None  # for malformed
    error_message: str = ""
    seed: int | None = None  # for reproducible intermittent faults


class ToolFaultBuilder:
    """Fluent builder for configuring tool faults.

    Usage::

        ap_fault.on_tool("search").timeout(5)
        ap_fault.on_tool("search").rate_limit(after_n=3)
        ap_fault.on_tool("search").intermittent(fail_rate=0.3, seed=42)
    """

    def __init__(self, injector: FaultInjector, tool_name: str) -> None:
        self._injector = injector
        self._tool_name = tool_name

    def timeout(self, seconds: float = 5.0) -> FaultInjector:
        """Tool call raises TimeoutError after N seconds."""
        self._injector._add_fault(
            self._tool_name,
            FaultConfig(fault_type=FaultType.TIMEOUT, timeout_seconds=seconds),
        )
        return self._injector

    def rate_limit(self, after_n: int = 0) -> FaultInjector:
        """Returns rate limit error after N successful calls."""
        self._injector._add_fault(
            self._tool_name,
            FaultConfig(fault_type=FaultType.RATE_LIMIT, after_n=after_n),
        )
        return self._injector

    def returns_malformed(self, data: Any = None) -> FaultInjector:
        """Returns corrupt/unexpected response data."""
        self._injector._add_fault(
            self._tool_name,
            FaultConfig(fault_type=FaultType.MALFORMED, malformed_data=data),
        )
        return self._injector

    def intermittent(
        self, fail_rate: float = 0.5, *, seed: int | None = None
    ) -> FaultInjector:
        """Random failures at configured probability."""
        self._injector._add_fault(
            self._tool_name,
            FaultConfig(
                fault_type=FaultType.INTERMITTENT, fail_rate=fail_rate, seed=seed
            ),
        )
        return self._injector

    def slow(self, latency_ms: float = 100.0) -> FaultInjector:
        """Adds artificial latency to tool responses."""
        self._injector._add_fault(
            self._tool_name,
            FaultConfig(fault_type=FaultType.SLOW, latency_ms=latency_ms),
        )
        return self._injector

    def returns_empty(self) -> FaultInjector:
        """Tool returns empty/null response."""
        self._injector._add_fault(
            self._tool_name,
            FaultConfig(fault_type=FaultType.EMPTY),
        )
        return self._injector


class LLMFaultBuilder:
    """Fluent builder for configuring LLM faults.

    Usage::

        ap_fault.on_llm().context_overflow()
        ap_fault.on_llm().rate_limit(after_n=5)
        ap_fault.on_llm().server_error()
    """

    def __init__(self, injector: FaultInjector) -> None:
        self._injector = injector

    def context_overflow(self) -> FaultInjector:
        """Simulates context window exceeded error."""
        self._injector._add_fault(
            "llm",
            FaultConfig(
                fault_type=FaultType.CONTEXT_OVERFLOW,
                error_message="Context window exceeded: max 128000 tokens",
            ),
        )
        return self._injector

    def partial_response(self) -> FaultInjector:
        """Streaming response terminates mid-token."""
        self._injector._add_fault(
            "llm",
            FaultConfig(fault_type=FaultType.PARTIAL_RESPONSE),
        )
        return self._injector

    def rate_limit(self, after_n: int = 0) -> FaultInjector:
        """Returns 429 rate limit after N successful calls."""
        self._injector._add_fault(
            "llm",
            FaultConfig(fault_type=FaultType.RATE_LIMIT, after_n=after_n),
        )
        return self._injector

    def server_error(self, message: str = "Internal server error") -> FaultInjector:
        """Simulates HTTP 500/503 responses."""
        self._injector._add_fault(
            "llm",
            FaultConfig(
                fault_type=FaultType.SERVER_ERROR,
                error_message=message,
            ),
        )
        return self._injector

    def content_filter(self) -> FaultInjector:
        """Simulates content policy rejection."""
        self._injector._add_fault(
            "llm",
            FaultConfig(
                fault_type=FaultType.CONTENT_FILTER,
                error_message="Content filtered: response violated content policy",
            ),
        )
        return self._injector

    def intermittent(
        self, fail_rate: float = 0.5, *, seed: int | None = None
    ) -> FaultInjector:
        """Random LLM failures at configured probability.

        Args:
            fail_rate: Probability of failure per call (0.0-1.0).
            seed: Optional seed for reproducible fault sequences.
        """
        self._injector._add_fault(
            "llm",
            FaultConfig(
                fault_type=FaultType.INTERMITTENT, fail_rate=fail_rate, seed=seed
            ),
        )
        return self._injector

    def slow(self, latency_ms: float = 100.0) -> FaultInjector:
        """Adds artificial latency to LLM responses.

        In async mode (check_llm_async), performs a real asyncio.sleep delay.
        In sync mode (check_llm), raises LLMSlowError immediately.

        Args:
            latency_ms: Simulated latency in milliseconds.
        """
        self._injector._add_fault(
            "llm",
            FaultConfig(fault_type=FaultType.SLOW, latency_ms=latency_ms),
        )
        return self._injector


class FaultInjector:
    """Injects configurable faults into tool calls and LLM requests.

    Use the fluent API to configure faults, then call ``check_tool()``
    or ``check_llm()`` before each operation to see if a fault should fire.

    Usage::

        fault = FaultInjector()
        fault.on_tool("search").timeout(5)
        fault.on_llm().context_overflow()

        # In agent/test code:
        fault.check_tool("search")  # raises TimeoutError
        fault.check_llm()           # raises ContextOverflowError
    """

    def __init__(self) -> None:
        self._tool_faults: dict[str, list[_FaultState]] = {}
        self._llm_faults: list[_FaultState] = []
        self._records: list[FaultRecord] = []

    def on_tool(self, tool_name: str) -> ToolFaultBuilder:
        """Configure a fault for a specific tool."""
        return ToolFaultBuilder(self, tool_name)

    def on_llm(self) -> LLMFaultBuilder:
        """Configure a fault for LLM requests."""
        return LLMFaultBuilder(self)

    def _add_fault(self, target: str, config: FaultConfig) -> None:
        """Internal: register a fault configuration."""
        state = _FaultState(config=config)
        if target == "llm":
            self._llm_faults.append(state)
        else:
            if target not in self._tool_faults:
                self._tool_faults[target] = []
            self._tool_faults[target].append(state)

    def check_tool(self, tool_name: str) -> None:
        """Check if a fault should fire for this tool call. Raises on fault.

        Call this before executing a tool. If a fault is configured and
        should trigger, raises the appropriate exception.
        """
        faults = self._tool_faults.get(tool_name, [])
        for state in faults:
            state.call_count += 1
            if self._should_trigger(state):
                self._records.append(
                    FaultRecord(
                        target=tool_name,
                        fault_type=state.config.fault_type,
                        triggered=True,
                        call_index=state.call_count,
                    )
                )
                self._raise_tool_fault(tool_name, state.config)
            else:
                self._records.append(
                    FaultRecord(
                        target=tool_name,
                        fault_type=state.config.fault_type,
                        triggered=False,
                        call_index=state.call_count,
                    )
                )

    async def check_tool_async(self, tool_name: str) -> None:
        """Async version of check_tool — supports slow fault with real delay."""
        faults = self._tool_faults.get(tool_name, [])
        for state in faults:
            state.call_count += 1
            if self._should_trigger(state):
                self._records.append(
                    FaultRecord(
                        target=tool_name,
                        fault_type=state.config.fault_type,
                        triggered=True,
                        call_index=state.call_count,
                    )
                )
                await self._raise_tool_fault_async(tool_name, state.config)
            else:
                self._records.append(
                    FaultRecord(
                        target=tool_name,
                        fault_type=state.config.fault_type,
                        triggered=False,
                        call_index=state.call_count,
                    )
                )

    def check_llm(self) -> None:
        """Check if a fault should fire for this LLM call. Raises on fault."""
        for state in self._llm_faults:
            state.call_count += 1
            if self._should_trigger(state):
                self._records.append(
                    FaultRecord(
                        target="llm",
                        fault_type=state.config.fault_type,
                        triggered=True,
                        call_index=state.call_count,
                    )
                )
                self._raise_llm_fault(state.config)
            else:
                self._records.append(
                    FaultRecord(
                        target="llm",
                        fault_type=state.config.fault_type,
                        triggered=False,
                        call_index=state.call_count,
                    )
                )

    async def check_llm_async(self) -> None:
        """Async version of check_llm — supports slow fault with real delay."""
        for state in self._llm_faults:
            state.call_count += 1
            if self._should_trigger(state):
                self._records.append(
                    FaultRecord(
                        target="llm",
                        fault_type=state.config.fault_type,
                        triggered=True,
                        call_index=state.call_count,
                    )
                )
                await self._raise_llm_fault_async(state.config)
            else:
                self._records.append(
                    FaultRecord(
                        target="llm",
                        fault_type=state.config.fault_type,
                        triggered=False,
                        call_index=state.call_count,
                    )
                )

    def _should_trigger(self, state: _FaultState) -> bool:
        """Determine if a fault should trigger on this call."""
        config = state.config

        if config.fault_type == FaultType.RATE_LIMIT:
            # Succeed for the first `after_n` calls, then fail
            return state.call_count > config.after_n

        if config.fault_type == FaultType.INTERMITTENT:
            rng = state.rng or random.Random()
            return rng.random() < config.fail_rate

        # All other fault types trigger every time
        return True

    def _raise_tool_fault(self, tool_name: str, config: FaultConfig) -> None:
        """Raise the appropriate exception for a tool fault."""
        if config.fault_type == FaultType.TIMEOUT:
            raise ToolTimeoutError(tool_name, config.timeout_seconds)
        elif config.fault_type == FaultType.RATE_LIMIT:
            raise ToolRateLimitError(tool_name)
        elif config.fault_type == FaultType.MALFORMED:
            raise ToolMalformedResponseError(tool_name, config.malformed_data)
        elif config.fault_type == FaultType.INTERMITTENT:
            raise ToolIntermittentError(tool_name)
        elif config.fault_type == FaultType.SLOW:
            # Sync latency simulation — block the thread like a real slow call
            time.sleep(config.latency_ms / 1000.0)
            return
        elif config.fault_type == FaultType.EMPTY:
            raise ToolEmptyResponseError(tool_name)

    async def _raise_tool_fault_async(
        self, tool_name: str, config: FaultConfig
    ) -> None:
        """Async version — supports actual delay for slow faults."""
        if config.fault_type == FaultType.SLOW:
            await asyncio.sleep(config.latency_ms / 1000.0)
            return  # slow doesn't raise — it just delays
        # All other faults raise the same as sync
        self._raise_tool_fault(tool_name, config)

    def _raise_llm_fault(self, config: FaultConfig) -> None:
        """Raise the appropriate exception for an LLM fault."""
        if config.fault_type == FaultType.CONTEXT_OVERFLOW:
            raise LLMContextOverflowError(config.error_message)
        elif config.fault_type == FaultType.PARTIAL_RESPONSE:
            raise LLMPartialResponseError()
        elif config.fault_type == FaultType.RATE_LIMIT:
            raise LLMRateLimitError()
        elif config.fault_type == FaultType.SERVER_ERROR:
            raise LLMServerError(config.error_message)
        elif config.fault_type == FaultType.CONTENT_FILTER:
            raise LLMContentFilterError(config.error_message)
        elif config.fault_type == FaultType.INTERMITTENT:
            raise LLMIntermittentError()
        elif config.fault_type == FaultType.SLOW:
            # Sync latency simulation — block the thread like a real slow call
            time.sleep(config.latency_ms / 1000.0)
            return

    async def _raise_llm_fault_async(self, config: FaultConfig) -> None:
        """Async version — supports actual delay for slow faults."""
        if config.fault_type == FaultType.SLOW:
            await asyncio.sleep(config.latency_ms / 1000.0)
            return  # slow doesn't raise — it just delays
        # All other faults raise the same as sync
        self._raise_llm_fault(config)

    # --- Inspection ---

    @property
    def records(self) -> list[FaultRecord]:
        """All fault injection records (triggered and non-triggered)."""
        return list(self._records)

    @property
    def triggered_records(self) -> list[FaultRecord]:
        """Only records where the fault actually fired."""
        return [r for r in self._records if r.triggered]

    @property
    def trigger_count(self) -> int:
        """Number of times a fault was triggered."""
        return len(self.triggered_records)

    @property
    def triggered(self) -> bool:
        """Whether any fault was triggered. Safe to use in ``if fi.triggered:``."""
        return any(r.triggered for r in self._records)

    def was_triggered(self, target: str | None = None) -> bool:
        """Check if any fault was triggered, optionally for a specific target."""
        if target is None:
            return any(r.triggered for r in self._records)
        return any(r.triggered and r.target == target for r in self._records)

    def has_faults_for(self, tool_name: str) -> bool:
        """Check if any faults are configured for a tool."""
        return tool_name in self._tool_faults

    def has_llm_faults(self) -> bool:
        """Check if any LLM faults are configured."""
        return len(self._llm_faults) > 0

    def reset(self) -> None:
        """Clear all faults and records."""
        self._tool_faults.clear()
        self._llm_faults.clear()
        self._records.clear()

    def reset_records(self) -> None:
        """Clear records but keep fault configurations."""
        self._records.clear()
        # Also reset call counters
        for faults in self._tool_faults.values():
            for state in faults:
                state.call_count = 0
        for state in self._llm_faults:
            state.call_count = 0


class _FaultState:
    """Internal: tracks call count and RNG state for a fault rule."""

    def __init__(self, config: FaultConfig) -> None:
        self.config = config
        self.call_count = 0
        self.rng: random.Random | None = None
        if config.seed is not None:
            self.rng = random.Random(config.seed)


# --- Exceptions ---


class FaultInjectionError(Exception):
    """Base exception for all fault injection errors."""


class ToolFaultError(FaultInjectionError):
    """Base exception for tool fault injection errors."""

    def __init__(self, tool_name: str, message: str) -> None:
        self.tool_name = tool_name
        super().__init__(f"FaultInjection({tool_name}): {message}")


class ToolTimeoutError(ToolFaultError):
    """Simulated tool timeout."""

    def __init__(self, tool_name: str, seconds: float) -> None:
        self.seconds = seconds
        super().__init__(tool_name, f"timeout after {seconds}s")


class ToolRateLimitError(ToolFaultError):
    """Simulated tool rate limit (HTTP 429)."""

    def __init__(self, tool_name: str) -> None:
        super().__init__(tool_name, "rate limit exceeded (429)")


class ToolMalformedResponseError(ToolFaultError):
    """Simulated malformed tool response."""

    def __init__(self, tool_name: str, data: Any = None) -> None:
        self.malformed_data = data
        super().__init__(tool_name, f"malformed response: {data!r}")


class ToolIntermittentError(ToolFaultError):
    """Simulated intermittent tool failure."""

    def __init__(self, tool_name: str) -> None:
        super().__init__(tool_name, "intermittent failure")


class ToolSlowError(ToolFaultError):
    """Raised in sync mode when a slow fault is configured (use async for real delay)."""

    def __init__(self, tool_name: str, latency_ms: float) -> None:
        self.latency_ms = latency_ms
        super().__init__(tool_name, f"slow response ({latency_ms}ms) — use async for real delay")


class ToolEmptyResponseError(ToolFaultError):
    """Simulated empty/null tool response."""

    def __init__(self, tool_name: str) -> None:
        super().__init__(tool_name, "empty response")


class LLMFaultError(FaultInjectionError):
    """Base exception for LLM fault injection errors."""


class LLMContextOverflowError(LLMFaultError):
    """Simulated context window exceeded."""

    def __init__(self, message: str = "Context window exceeded") -> None:
        super().__init__(message)


class LLMPartialResponseError(LLMFaultError):
    """Simulated truncated streaming response."""

    def __init__(self) -> None:
        super().__init__("Partial response: streaming terminated mid-token")


class LLMRateLimitError(LLMFaultError):
    """Simulated LLM rate limit (HTTP 429)."""

    def __init__(self) -> None:
        super().__init__("Rate limit exceeded (429)")


class LLMServerError(LLMFaultError):
    """Simulated LLM server error (HTTP 500/503)."""

    def __init__(self, message: str = "Internal server error") -> None:
        super().__init__(message)


class LLMContentFilterError(LLMFaultError):
    """Simulated content policy rejection."""

    def __init__(self, message: str = "Content filtered") -> None:
        super().__init__(message)


class LLMIntermittentError(LLMFaultError):
    """Simulated intermittent LLM failure."""

    def __init__(self) -> None:
        super().__init__("Intermittent LLM failure")


class LLMSlowError(LLMFaultError):
    """Raised in sync mode when a slow LLM fault is configured (use async for real delay)."""

    def __init__(self, latency_ms: float) -> None:
        self.latency_ms = latency_ms
        super().__init__(f"Slow LLM response ({latency_ms}ms) — use async for real delay")
