"""Tests for FaultInjector integration with MockTool and MockLLM.

Validates that attach_faults() wires fault injection so faults fire
automatically when mocks are called — no manual check_tool()/check_llm()
guards needed.
"""

import pytest

from checkagent.mock.fault import (
    FaultInjector,
    LLMContentFilterError,
    LLMContextOverflowError,
    LLMIntermittentError,
    LLMRateLimitError,
    LLMServerError,
    LLMSlowError,
    ToolEmptyResponseError,
    ToolIntermittentError,
    ToolMalformedResponseError,
    ToolRateLimitError,
    ToolTimeoutError,
)
from checkagent.mock.llm import MockLLM
from checkagent.mock.tool import MockTool

# --- MockTool + FaultInjector integration ---


class TestMockToolFaultIntegration:
    """MockTool.attach_faults() auto-fires tool faults on call()."""

    async def test_timeout_fires_on_async_call(self):
        tool = MockTool()
        fault = FaultInjector()
        tool.attach_faults(fault)
        fault.on_tool("search").timeout(5)
        tool.on_call("search").respond({"results": []})

        with pytest.raises(ToolTimeoutError):
            await tool.call("search", {"query": "test"})

    def test_timeout_fires_on_sync_call(self):
        tool = MockTool()
        fault = FaultInjector()
        tool.attach_faults(fault)
        fault.on_tool("search").timeout(5)
        tool.on_call("search").respond({"results": []})

        with pytest.raises(ToolTimeoutError):
            tool.call_sync("search", {"query": "test"})

    async def test_rate_limit_fires_after_n(self):
        tool = MockTool()
        fault = FaultInjector()
        tool.attach_faults(fault)
        fault.on_tool("api").rate_limit(after_n=2)
        tool.on_call("api").respond({"ok": True})

        # First 2 calls succeed
        assert await tool.call("api", {}) == {"ok": True}
        assert await tool.call("api", {}) == {"ok": True}

        # Third call hits rate limit
        with pytest.raises(ToolRateLimitError):
            await tool.call("api", {})

    async def test_malformed_response(self):
        tool = MockTool()
        fault = FaultInjector()
        tool.attach_faults(fault)
        fault.on_tool("search").returns_malformed({"corrupt": True})
        tool.on_call("search").respond({"results": []})

        with pytest.raises(ToolMalformedResponseError) as exc_info:
            await tool.call("search", {})
        assert exc_info.value.malformed_data == {"corrupt": True}

    async def test_empty_response(self):
        tool = MockTool()
        fault = FaultInjector()
        tool.attach_faults(fault)
        fault.on_tool("search").returns_empty()
        tool.on_call("search").respond({"results": []})

        with pytest.raises(ToolEmptyResponseError):
            await tool.call("search", {})

    async def test_intermittent_with_seed(self):
        tool = MockTool()
        fault = FaultInjector()
        tool.attach_faults(fault)
        fault.on_tool("search").intermittent(fail_rate=0.5, seed=42)
        tool.on_call("search").respond({"results": []})

        # With seed=42, results are deterministic
        results = []
        for _ in range(10):
            try:
                await tool.call("search", {})
                results.append("ok")
            except ToolIntermittentError:
                results.append("fail")

        # At least some should succeed and some should fail
        assert "ok" in results
        assert "fail" in results

    async def test_no_fault_without_attach(self):
        """Without attach_faults(), no fault checking happens."""
        tool = MockTool()
        fault = FaultInjector()
        # NOT attached
        fault.on_tool("search").timeout(5)
        tool.on_call("search").respond({"results": []})

        # Should succeed — fault not wired in
        result = await tool.call("search", {})
        assert result == {"results": []}

    async def test_fault_only_affects_configured_tools(self):
        """Faults for 'search' don't affect 'calendar'."""
        tool = MockTool()
        fault = FaultInjector()
        tool.attach_faults(fault)
        fault.on_tool("search").timeout(5)
        tool.on_call("search").respond({"results": []})
        tool.on_call("calendar").respond({"events": []})

        # calendar should work fine
        result = await tool.call("calendar", {})
        assert result == {"events": []}

        # search should fail
        with pytest.raises(ToolTimeoutError):
            await tool.call("search", {})

    async def test_attach_faults_returns_self(self):
        """attach_faults() supports chaining."""
        tool = MockTool()
        fault = FaultInjector()
        result = tool.attach_faults(fault)
        assert result is tool

    async def test_fault_records_tracked(self):
        """Attached faults still record to FaultInjector.records."""
        tool = MockTool()
        fault = FaultInjector()
        tool.attach_faults(fault)
        fault.on_tool("search").timeout(5)
        tool.on_call("search").respond({"results": []})

        with pytest.raises(ToolTimeoutError):
            await tool.call("search", {})

        assert fault.trigger_count == 1
        assert fault.was_triggered("search")

    async def test_fault_fires_before_tool_error(self):
        """Fault fires even if tool is configured with an error response."""
        tool = MockTool()
        fault = FaultInjector()
        tool.attach_faults(fault)
        fault.on_tool("bad").timeout(5)
        tool.on_call("bad").error("Service unavailable")

        # Should raise ToolTimeoutError, not ToolExecutionError
        with pytest.raises(ToolTimeoutError):
            await tool.call("bad", {})

    async def test_fault_fires_before_unknown_tool_error(self):
        """Fault fires even for unregistered tools."""
        tool = MockTool()
        fault = FaultInjector()
        tool.attach_faults(fault)
        fault.on_tool("missing").timeout(5)

        with pytest.raises(ToolTimeoutError):
            await tool.call("missing", {})


# --- MockLLM + FaultInjector integration ---


class TestMockLLMFaultIntegration:
    """MockLLM.attach_faults() auto-fires LLM faults on complete()/stream()."""

    async def test_server_error_on_complete(self):
        llm = MockLLM()
        fault = FaultInjector()
        llm.attach_faults(fault)
        fault.on_llm().server_error()

        with pytest.raises(LLMServerError):
            await llm.complete("hello")

    def test_server_error_on_complete_sync(self):
        llm = MockLLM()
        fault = FaultInjector()
        llm.attach_faults(fault)
        fault.on_llm().server_error()

        with pytest.raises(LLMServerError):
            llm.complete_sync("hello")

    async def test_context_overflow(self):
        llm = MockLLM()
        fault = FaultInjector()
        llm.attach_faults(fault)
        fault.on_llm().context_overflow()

        with pytest.raises(LLMContextOverflowError):
            await llm.complete("hello")

    async def test_rate_limit_after_n(self):
        llm = MockLLM()
        fault = FaultInjector()
        llm.attach_faults(fault)
        fault.on_llm().rate_limit(after_n=2)
        llm.on_input(contains="hello").respond("Hi!")

        # First 2 calls succeed
        assert await llm.complete("hello") == "Hi!"
        assert await llm.complete("hello") == "Hi!"

        # Third call hits rate limit
        with pytest.raises(LLMRateLimitError):
            await llm.complete("hello")

    async def test_content_filter(self):
        llm = MockLLM()
        fault = FaultInjector()
        llm.attach_faults(fault)
        fault.on_llm().content_filter()

        with pytest.raises(LLMContentFilterError):
            await llm.complete("hello")

    def test_stream_checks_faults(self):
        llm = MockLLM()
        fault = FaultInjector()
        llm.attach_faults(fault)
        fault.on_llm().server_error()

        with pytest.raises(LLMServerError):
            llm.stream("hello")

    async def test_no_fault_without_attach(self):
        """Without attach_faults(), no fault checking happens."""
        llm = MockLLM()
        fault = FaultInjector()
        fault.on_llm().server_error()
        llm.on_input(contains="hello").respond("Hi!")

        result = await llm.complete("hello")
        assert result == "Hi!"

    async def test_attach_faults_returns_self(self):
        llm = MockLLM()
        fault = FaultInjector()
        result = llm.attach_faults(fault)
        assert result is llm

    async def test_fault_records_tracked(self):
        llm = MockLLM()
        fault = FaultInjector()
        llm.attach_faults(fault)
        fault.on_llm().server_error()

        with pytest.raises(LLMServerError):
            await llm.complete("hello")

        assert fault.trigger_count == 1
        assert fault.was_triggered("llm")


# --- Combined integration ---


class TestCombinedFaultIntegration:
    """Both MockTool and MockLLM wired to the same FaultInjector."""

    async def test_shared_injector(self):
        tool = MockTool()
        llm = MockLLM()
        fault = FaultInjector()
        tool.attach_faults(fault)
        llm.attach_faults(fault)

        fault.on_tool("search").timeout(5)
        fault.on_llm().server_error()

        tool.on_call("search").respond({"results": []})
        llm.on_input(contains="hello").respond("Hi!")

        with pytest.raises(ToolTimeoutError):
            await tool.call("search", {})

        with pytest.raises(LLMServerError):
            await llm.complete("hello")

        assert fault.trigger_count == 2

    async def test_tool_fault_does_not_affect_llm(self):
        tool = MockTool()
        llm = MockLLM()
        fault = FaultInjector()
        tool.attach_faults(fault)
        llm.attach_faults(fault)

        fault.on_tool("search").timeout(5)
        # No LLM faults configured

        llm.on_input(contains="hello").respond("Hi!")

        # LLM should work fine
        result = await llm.complete("hello")
        assert result == "Hi!"

    async def test_fluent_chaining_with_attach(self):
        """Fluent builder pattern works end-to-end."""
        fault = FaultInjector()
        tool = MockTool().attach_faults(fault)
        tool.on_call("search").respond({"results": []})
        fault.on_tool("search").rate_limit(after_n=1)

        assert await tool.call("search", {}) == {"results": []}
        with pytest.raises(ToolRateLimitError):
            await tool.call("search", {})


# --- LLM intermittent + slow integration ---


class TestMockLLMIntermittentSlowIntegration:
    """MockLLM + FaultInjector for intermittent and slow LLM faults."""

    async def test_intermittent_on_complete(self):
        llm = MockLLM()
        fault = FaultInjector()
        llm.attach_faults(fault)
        fault.on_llm().intermittent(fail_rate=1.0, seed=0)

        with pytest.raises(LLMIntermittentError):
            await llm.complete("hello")

    async def test_intermittent_partial_failures(self):
        """With fail_rate < 1.0, some calls succeed."""
        llm = MockLLM()
        fault = FaultInjector()
        llm.attach_faults(fault)
        llm.on_input(contains="hello").respond("Hi!")
        fault.on_llm().intermittent(fail_rate=0.0, seed=0)

        # fail_rate=0 → all succeed
        result = await llm.complete("hello")
        assert result == "Hi!"

    async def test_slow_on_complete_async(self):
        """Slow fault adds real delay in async MockLLM.complete()."""
        import time

        llm = MockLLM()
        fault = FaultInjector()
        llm.attach_faults(fault)
        llm.on_input(contains="hello").respond("Hi!")
        fault.on_llm().slow(latency_ms=50)

        start = time.monotonic()
        result = await llm.complete("hello")
        elapsed_ms = (time.monotonic() - start) * 1000
        assert result == "Hi!"
        assert elapsed_ms >= 40  # timing slack

    def test_slow_on_complete_sync_raises(self):
        """Sync complete_sync raises LLMSlowError (no real delay)."""
        llm = MockLLM()
        fault = FaultInjector()
        llm.attach_faults(fault)
        fault.on_llm().slow(latency_ms=200)

        with pytest.raises(LLMSlowError, match="200"):
            llm.complete_sync("hello")

    def test_slow_on_stream_sync_raises(self):
        """stream() is sync, so slow fault raises LLMSlowError."""
        llm = MockLLM()
        fault = FaultInjector()
        llm.attach_faults(fault)
        fault.on_llm().slow(latency_ms=100)

        with pytest.raises(LLMSlowError):
            llm.stream("hello")
