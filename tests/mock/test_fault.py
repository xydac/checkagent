"""Tests for FaultInjector — fault injection for tool calls and LLM requests."""

from __future__ import annotations

import asyncio

import pytest

from checkagent.mock.fault import (
    FaultInjectionError,
    FaultInjector,
    FaultType,
    LLMContentFilterError,
    LLMContextOverflowError,
    LLMFaultError,
    LLMIntermittentError,
    LLMPartialResponseError,
    LLMRateLimitError,
    LLMServerError,
    LLMSlowError,
    ToolEmptyResponseError,
    ToolFaultError,
    ToolIntermittentError,
    ToolMalformedResponseError,
    ToolRateLimitError,
    ToolSlowError,
    ToolTimeoutError,
)

# --- Fluent API ---


class TestFluentAPI:
    """Test the fluent builder API."""

    def test_on_tool_returns_builder(self):
        fault = FaultInjector()
        builder = fault.on_tool("search")
        assert builder is not None

    def test_on_llm_returns_builder(self):
        fault = FaultInjector()
        builder = fault.on_llm()
        assert builder is not None

    def test_chaining_tool_faults(self):
        fault = FaultInjector()
        result = fault.on_tool("search").timeout(5)
        assert result is fault  # returns the injector for chaining

    def test_chaining_llm_faults(self):
        fault = FaultInjector()
        result = fault.on_llm().context_overflow()
        assert result is fault

    def test_multiple_tool_faults(self):
        fault = FaultInjector()
        fault.on_tool("search").timeout(5)
        fault.on_tool("fetch").rate_limit(after_n=3)
        assert fault.has_faults_for("search")
        assert fault.has_faults_for("fetch")
        assert not fault.has_faults_for("unknown")

    def test_has_llm_faults(self):
        fault = FaultInjector()
        assert not fault.has_llm_faults()
        fault.on_llm().server_error()
        assert fault.has_llm_faults()


# --- Tool Faults ---


class TestToolTimeout:
    """Test tool timeout fault."""

    def test_timeout_raises(self):
        fault = FaultInjector()
        fault.on_tool("search").timeout(10)
        with pytest.raises(ToolTimeoutError) as exc_info:
            fault.check_tool("search")
        assert exc_info.value.tool_name == "search"
        assert exc_info.value.seconds == 10

    def test_timeout_message(self):
        fault = FaultInjector()
        fault.on_tool("api").timeout(3.5)
        with pytest.raises(ToolTimeoutError, match="timeout after 3.5s"):
            fault.check_tool("api")

    def test_timeout_triggers_every_call(self):
        fault = FaultInjector()
        fault.on_tool("search").timeout(5)
        for _ in range(3):
            with pytest.raises(ToolTimeoutError):
                fault.check_tool("search")
        assert fault.trigger_count == 3


class TestToolRateLimit:
    """Test tool rate limit fault."""

    def test_rate_limit_after_n(self):
        fault = FaultInjector()
        fault.on_tool("api").rate_limit(after_n=2)

        # First 2 calls succeed (no exception)
        fault.check_tool("api")
        fault.check_tool("api")

        # 3rd call fails
        with pytest.raises(ToolRateLimitError):
            fault.check_tool("api")

    def test_rate_limit_immediate(self):
        fault = FaultInjector()
        fault.on_tool("api").rate_limit(after_n=0)
        with pytest.raises(ToolRateLimitError):
            fault.check_tool("api")

    def test_rate_limit_continues_failing(self):
        fault = FaultInjector()
        fault.on_tool("api").rate_limit(after_n=1)
        fault.check_tool("api")  # succeeds
        for _ in range(5):
            with pytest.raises(ToolRateLimitError):
                fault.check_tool("api")


class TestToolMalformed:
    """Test malformed response fault."""

    def test_malformed_raises(self):
        fault = FaultInjector()
        fault.on_tool("api").returns_malformed({"broken": True})
        with pytest.raises(ToolMalformedResponseError) as exc_info:
            fault.check_tool("api")
        assert exc_info.value.malformed_data == {"broken": True}

    def test_malformed_none_data(self):
        fault = FaultInjector()
        fault.on_tool("api").returns_malformed()
        with pytest.raises(ToolMalformedResponseError) as exc_info:
            fault.check_tool("api")
        assert exc_info.value.malformed_data is None


class TestToolIntermittent:
    """Test intermittent failure fault."""

    def test_intermittent_with_seed_reproducible(self):
        """With a seed, intermittent faults are reproducible."""
        results1 = []
        fault1 = FaultInjector()
        fault1.on_tool("api").intermittent(fail_rate=0.5, seed=42)
        for _ in range(20):
            try:
                fault1.check_tool("api")
                results1.append(False)
            except ToolIntermittentError:
                results1.append(True)

        results2 = []
        fault2 = FaultInjector()
        fault2.on_tool("api").intermittent(fail_rate=0.5, seed=42)
        for _ in range(20):
            try:
                fault2.check_tool("api")
                results2.append(False)
            except ToolIntermittentError:
                results2.append(True)

        assert results1 == results2

    def test_intermittent_high_rate_mostly_fails(self):
        fault = FaultInjector()
        fault.on_tool("api").intermittent(fail_rate=1.0, seed=1)
        # 100% fail rate should always raise
        with pytest.raises(ToolIntermittentError):
            fault.check_tool("api")

    def test_intermittent_zero_rate_never_fails(self):
        fault = FaultInjector()
        fault.on_tool("api").intermittent(fail_rate=0.0, seed=1)
        # 0% fail rate should never raise
        for _ in range(10):
            fault.check_tool("api")  # no exception


class TestToolSlow:
    """Test slow tool fault."""

    def test_slow_sync_delays(self):
        """Sync check_tool delays with time.sleep (no exception)."""
        import time

        fault = FaultInjector()
        fault.on_tool("api").slow(latency_ms=50)
        start = time.monotonic()
        fault.check_tool("api")  # should not raise
        elapsed = time.monotonic() - start
        assert elapsed >= 0.04  # at least ~40ms (with tolerance)

    @pytest.mark.asyncio
    async def test_slow_async_delays(self):
        """Async check_tool_async actually delays (no exception)."""
        fault = FaultInjector()
        fault.on_tool("api").slow(latency_ms=50)
        start = asyncio.get_event_loop().time()
        await fault.check_tool_async("api")
        elapsed = asyncio.get_event_loop().time() - start
        assert elapsed >= 0.04  # at least ~40ms (with tolerance)


class TestToolEmpty:
    """Test empty response fault."""

    def test_empty_raises(self):
        fault = FaultInjector()
        fault.on_tool("api").returns_empty()
        with pytest.raises(ToolEmptyResponseError, match="empty response"):
            fault.check_tool("api")


class TestNoFaultConfigured:
    """Test behavior when no fault is configured."""

    def test_no_fault_no_error(self):
        fault = FaultInjector()
        fault.check_tool("unknown_tool")  # should not raise

    def test_no_llm_fault_no_error(self):
        fault = FaultInjector()
        fault.check_llm()  # should not raise


# --- LLM Faults ---


class TestLLMContextOverflow:
    """Test LLM context overflow fault."""

    def test_context_overflow_raises(self):
        fault = FaultInjector()
        fault.on_llm().context_overflow()
        with pytest.raises(LLMContextOverflowError, match="Context window exceeded"):
            fault.check_llm()


class TestLLMPartialResponse:
    """Test LLM partial response fault."""

    def test_partial_response_raises(self):
        fault = FaultInjector()
        fault.on_llm().partial_response()
        with pytest.raises(LLMPartialResponseError, match="mid-token"):
            fault.check_llm()


class TestLLMRateLimit:
    """Test LLM rate limit fault."""

    def test_rate_limit_after_n(self):
        fault = FaultInjector()
        fault.on_llm().rate_limit(after_n=3)
        for _ in range(3):
            fault.check_llm()  # succeeds
        with pytest.raises(LLMRateLimitError):
            fault.check_llm()

    def test_rate_limit_immediate(self):
        fault = FaultInjector()
        fault.on_llm().rate_limit(after_n=0)
        with pytest.raises(LLMRateLimitError):
            fault.check_llm()


class TestLLMServerError:
    """Test LLM server error fault."""

    def test_server_error_raises(self):
        fault = FaultInjector()
        fault.on_llm().server_error()
        with pytest.raises(LLMServerError, match="Internal server error"):
            fault.check_llm()

    def test_server_error_custom_message(self):
        fault = FaultInjector()
        fault.on_llm().server_error("Service unavailable (503)")
        with pytest.raises(LLMServerError, match="503"):
            fault.check_llm()


class TestLLMContentFilter:
    """Test LLM content filter fault."""

    def test_content_filter_raises(self):
        fault = FaultInjector()
        fault.on_llm().content_filter()
        with pytest.raises(LLMContentFilterError, match="content policy"):
            fault.check_llm()


# --- Records & Inspection ---


class TestRecords:
    """Test fault record tracking."""

    def test_records_empty_initially(self):
        fault = FaultInjector()
        assert fault.records == []
        assert fault.trigger_count == 0

    def test_triggered_record(self):
        fault = FaultInjector()
        fault.on_tool("search").timeout(5)
        with pytest.raises(ToolTimeoutError):
            fault.check_tool("search")
        assert len(fault.records) == 1
        assert fault.records[0].target == "search"
        assert fault.records[0].fault_type == FaultType.TIMEOUT
        assert fault.records[0].triggered is True

    def test_non_triggered_record(self):
        """Rate limit before threshold records as non-triggered."""
        fault = FaultInjector()
        fault.on_tool("api").rate_limit(after_n=2)
        fault.check_tool("api")  # succeeds
        assert len(fault.records) == 1
        assert fault.records[0].triggered is False

    def test_llm_record(self):
        fault = FaultInjector()
        fault.on_llm().server_error()
        with pytest.raises(LLMServerError):
            fault.check_llm()
        assert fault.records[0].target == "llm"
        assert fault.records[0].fault_type == FaultType.SERVER_ERROR

    def test_was_triggered(self):
        fault = FaultInjector()
        fault.on_tool("search").timeout(5)
        assert not fault.was_triggered()
        assert not fault.was_triggered("search")
        with pytest.raises(ToolTimeoutError):
            fault.check_tool("search")
        assert fault.was_triggered()
        assert fault.was_triggered("search")
        assert not fault.was_triggered("other")

    def test_triggered_property(self):
        """The `triggered` property is safe for bare `if fi.triggered:` usage."""
        fault = FaultInjector()
        fault.on_tool("search").timeout(5)
        assert not fault.triggered
        assert fault.triggered is False  # actually bool, not truthy method
        with pytest.raises(ToolTimeoutError):
            fault.check_tool("search")
        assert fault.triggered
        assert fault.triggered is True

    def test_triggered_records_filters(self):
        fault = FaultInjector()
        fault.on_tool("api").rate_limit(after_n=2)
        fault.check_tool("api")  # not triggered
        fault.check_tool("api")  # not triggered
        with pytest.raises(ToolRateLimitError):
            fault.check_tool("api")  # triggered
        assert len(fault.records) == 3
        assert len(fault.triggered_records) == 1


# --- Reset ---


class TestReset:
    """Test reset and reset_records."""

    def test_reset_clears_everything(self):
        fault = FaultInjector()
        fault.on_tool("search").timeout(5)
        fault.on_llm().server_error()
        with pytest.raises(ToolTimeoutError):
            fault.check_tool("search")
        fault.reset()
        assert not fault.has_faults_for("search")
        assert not fault.has_llm_faults()
        assert fault.records == []
        # No fault configured — should not raise
        fault.check_tool("search")

    def test_reset_records_keeps_faults(self):
        fault = FaultInjector()
        fault.on_tool("search").timeout(5)
        with pytest.raises(ToolTimeoutError):
            fault.check_tool("search")
        assert fault.trigger_count == 1
        fault.reset_records()
        assert fault.trigger_count == 0
        assert fault.has_faults_for("search")
        # Fault still fires
        with pytest.raises(ToolTimeoutError):
            fault.check_tool("search")

    def test_reset_records_resets_call_counters(self):
        """After reset_records, rate limit counters start fresh."""
        fault = FaultInjector()
        fault.on_tool("api").rate_limit(after_n=2)
        fault.check_tool("api")
        fault.check_tool("api")
        with pytest.raises(ToolRateLimitError):
            fault.check_tool("api")
        fault.reset_records()
        # Counter reset — first 2 calls succeed again
        fault.check_tool("api")
        fault.check_tool("api")
        with pytest.raises(ToolRateLimitError):
            fault.check_tool("api")


# --- Exception Hierarchy ---


class TestExceptionHierarchy:
    """Test that exception classes form a proper hierarchy."""

    def test_tool_faults_inherit_from_tool_fault_error(self):
        assert issubclass(ToolTimeoutError, ToolFaultError)
        assert issubclass(ToolRateLimitError, ToolFaultError)
        assert issubclass(ToolMalformedResponseError, ToolFaultError)
        assert issubclass(ToolIntermittentError, ToolFaultError)
        assert issubclass(ToolSlowError, ToolFaultError)
        assert issubclass(ToolEmptyResponseError, ToolFaultError)

    def test_llm_faults_inherit_from_llm_fault_error(self):
        assert issubclass(LLMContextOverflowError, LLMFaultError)
        assert issubclass(LLMPartialResponseError, LLMFaultError)
        assert issubclass(LLMRateLimitError, LLMFaultError)
        assert issubclass(LLMServerError, LLMFaultError)
        assert issubclass(LLMContentFilterError, LLMFaultError)

    def test_all_inherit_from_fault_injection_error(self):
        assert issubclass(ToolFaultError, FaultInjectionError)
        assert issubclass(LLMFaultError, FaultInjectionError)

    def test_catch_all_tool_faults(self):
        fault = FaultInjector()
        fault.on_tool("api").timeout(5)
        with pytest.raises(ToolFaultError):
            fault.check_tool("api")

    def test_catch_all_faults(self):
        fault = FaultInjector()
        fault.on_llm().server_error()
        with pytest.raises(FaultInjectionError):
            fault.check_llm()


# --- Async ---


class TestAsync:
    """Test async fault checking."""

    @pytest.mark.asyncio
    async def test_async_tool_timeout(self):
        fault = FaultInjector()
        fault.on_tool("search").timeout(5)
        with pytest.raises(ToolTimeoutError):
            await fault.check_tool_async("search")

    @pytest.mark.asyncio
    async def test_async_no_fault(self):
        fault = FaultInjector()
        await fault.check_tool_async("anything")  # no exception

    @pytest.mark.asyncio
    async def test_async_rate_limit(self):
        fault = FaultInjector()
        fault.on_tool("api").rate_limit(after_n=1)
        await fault.check_tool_async("api")  # succeeds
        with pytest.raises(ToolRateLimitError):
            await fault.check_tool_async("api")


# --- Multiple Faults on Same Target ---


class TestMultipleFaults:
    """Test multiple faults configured on the same target."""

    def test_multiple_tool_faults_both_trigger(self):
        """When two faults are on the same tool, both are checked."""
        fault = FaultInjector()
        fault.on_tool("api").timeout(5)
        # The first fault (timeout) triggers and raises before the second is checked
        with pytest.raises(ToolTimeoutError):
            fault.check_tool("api")

    def test_multiple_llm_faults(self):
        fault = FaultInjector()
        fault.on_llm().context_overflow()
        # First fault triggers
        with pytest.raises(LLMContextOverflowError):
            fault.check_llm()


# --- LLM Intermittent Faults ---


class TestLLMIntermittent:
    """Test LLM intermittent fault — probabilistic failures."""

    def test_intermittent_with_seed_deterministic(self):
        """Seeded intermittent fault produces reproducible results."""
        fault = FaultInjector()
        fault.on_llm().intermittent(fail_rate=0.5, seed=42)
        results = []
        for _ in range(10):
            try:
                fault.check_llm()
                results.append("ok")
            except LLMIntermittentError:
                results.append("fail")
        # Same seed → same sequence
        fault2 = FaultInjector()
        fault2.on_llm().intermittent(fail_rate=0.5, seed=42)
        results2 = []
        for _ in range(10):
            try:
                fault2.check_llm()
                results2.append("ok")
            except LLMIntermittentError:
                results2.append("fail")
        assert results == results2

    def test_intermittent_always_fails(self):
        """fail_rate=1.0 always triggers."""
        fault = FaultInjector()
        fault.on_llm().intermittent(fail_rate=1.0, seed=0)
        with pytest.raises(LLMIntermittentError, match="Intermittent LLM failure"):
            fault.check_llm()

    def test_intermittent_never_fails(self):
        """fail_rate=0.0 never triggers."""
        fault = FaultInjector()
        fault.on_llm().intermittent(fail_rate=0.0, seed=0)
        for _ in range(20):
            fault.check_llm()  # should never raise

    def test_intermittent_records_triggered(self):
        """Records reflect triggered vs non-triggered intermittent faults."""
        fault = FaultInjector()
        fault.on_llm().intermittent(fail_rate=1.0, seed=0)
        with pytest.raises(LLMIntermittentError):
            fault.check_llm()
        assert fault.trigger_count == 1
        assert fault.records[0].fault_type == FaultType.INTERMITTENT
        assert fault.records[0].target == "llm"

    def test_intermittent_inherits_from_llm_fault_error(self):
        assert issubclass(LLMIntermittentError, LLMFaultError)
        assert issubclass(LLMIntermittentError, FaultInjectionError)

    def test_intermittent_chaining(self):
        """intermittent() returns FaultInjector for chaining."""
        fault = FaultInjector()
        result = fault.on_llm().intermittent(fail_rate=0.5, seed=1)
        assert result is fault


# --- LLM Slow Faults ---


class TestLLMSlow:
    """Test LLM slow fault — latency simulation."""

    def test_slow_sync_delays(self):
        """Sync check_llm delays with time.sleep (no exception)."""
        import time

        fault = FaultInjector()
        fault.on_llm().slow(latency_ms=50)
        start = time.monotonic()
        fault.check_llm()  # should not raise
        elapsed = time.monotonic() - start
        assert elapsed >= 0.04  # at least ~40ms (with tolerance)

    def test_slow_sync_records(self):
        """Sync slow fault records as triggered."""
        fault = FaultInjector()
        fault.on_llm().slow(latency_ms=10)
        fault.check_llm()
        assert fault.trigger_count == 1
        assert fault.records[0].fault_type == FaultType.SLOW

    async def test_slow_async_delays(self):
        """Async check_llm_async performs real delay without raising."""
        fault = FaultInjector()
        fault.on_llm().slow(latency_ms=50)
        import time

        start = time.monotonic()
        await fault.check_llm_async()  # should delay, not raise
        elapsed_ms = (time.monotonic() - start) * 1000
        assert elapsed_ms >= 40  # allow some timing slack

    async def test_slow_async_records(self):
        """Async slow fault records as triggered."""
        fault = FaultInjector()
        fault.on_llm().slow(latency_ms=10)
        await fault.check_llm_async()
        assert fault.trigger_count == 1
        assert fault.records[0].fault_type == FaultType.SLOW

    def test_slow_inherits_from_llm_fault_error(self):
        assert issubclass(LLMSlowError, LLMFaultError)
        assert issubclass(LLMSlowError, FaultInjectionError)

    def test_slow_chaining(self):
        """slow() returns FaultInjector for chaining."""
        fault = FaultInjector()
        result = fault.on_llm().slow(latency_ms=100)
        assert result is fault

    def test_slow_default_latency(self):
        """Default latency is 100ms — verify via timing."""
        import time

        fault = FaultInjector()
        fault.on_llm().slow()
        start = time.monotonic()
        fault.check_llm()
        elapsed = time.monotonic() - start
        assert elapsed >= 0.08  # default 100ms with tolerance


# --- LLM check_llm_async ---


class TestCheckLLMAsync:
    """Test async LLM fault checking."""

    async def test_async_context_overflow(self):
        """Async check raises same exceptions as sync."""
        fault = FaultInjector()
        fault.on_llm().context_overflow()
        with pytest.raises(LLMContextOverflowError):
            await fault.check_llm_async()

    async def test_async_intermittent(self):
        """Async intermittent fault works."""
        fault = FaultInjector()
        fault.on_llm().intermittent(fail_rate=1.0, seed=0)
        with pytest.raises(LLMIntermittentError):
            await fault.check_llm_async()

    async def test_async_no_fault_no_error(self):
        """No faults configured → no error in async mode."""
        fault = FaultInjector()
        await fault.check_llm_async()  # should not raise
