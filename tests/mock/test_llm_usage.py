"""Tests for MockLLM token usage simulation (RQ2 cost tracking)."""

from __future__ import annotations

import pytest

from checkagent.mock.llm import MockLLM


class TestWithUsageFixedTokens:
    """Tests for with_usage() with explicit token counts."""

    @pytest.mark.asyncio
    async def test_complete_records_fixed_tokens(self):
        llm = MockLLM().with_usage(prompt_tokens=100, completion_tokens=50)
        await llm.complete("hello")
        assert llm.last_call.prompt_tokens == 100
        assert llm.last_call.completion_tokens == 50
        assert llm.last_call.total_tokens == 150

    def test_complete_sync_records_fixed_tokens(self):
        llm = MockLLM().with_usage(prompt_tokens=200, completion_tokens=80)
        llm.complete_sync("hello")
        assert llm.last_call.prompt_tokens == 200
        assert llm.last_call.completion_tokens == 80
        assert llm.last_call.total_tokens == 280

    @pytest.mark.asyncio
    async def test_stream_records_fixed_tokens(self):
        llm = MockLLM().with_usage(prompt_tokens=150, completion_tokens=60)
        llm.on_input(contains="weather").stream(["It's ", "sunny"])
        chunks = []
        async for event in llm.stream("weather"):
            if event.data:
                chunks.append(event.data)
        assert chunks == ["It's ", "sunny"]
        assert llm.last_call.prompt_tokens == 150
        assert llm.last_call.completion_tokens == 60

    @pytest.mark.asyncio
    async def test_stream_fallback_records_fixed_tokens(self):
        llm = MockLLM(default_response="ok").with_usage(
            prompt_tokens=50, completion_tokens=10
        )
        async for _ in llm.stream("anything"):
            pass
        assert llm.last_call.prompt_tokens == 50
        assert llm.last_call.completion_tokens == 10

    @pytest.mark.asyncio
    async def test_multiple_calls_all_have_tokens(self):
        llm = MockLLM().with_usage(prompt_tokens=100, completion_tokens=50)
        await llm.complete("first")
        await llm.complete("second")
        assert all(c.prompt_tokens == 100 for c in llm.calls)
        assert all(c.completion_tokens == 50 for c in llm.calls)


class TestWithUsageAutoEstimate:
    """Tests for with_usage(auto_estimate=True)."""

    @pytest.mark.asyncio
    async def test_auto_estimate_scales_with_text_length(self):
        llm = MockLLM(default_response="short").with_usage(auto_estimate=True)
        await llm.complete("a longer input string here")
        call = llm.last_call
        # len("a longer input string here") = 26, // 4 + 1 = 7
        assert call.prompt_tokens == 26 // 4 + 1
        # len("short") = 5, // 4 + 1 = 2
        assert call.completion_tokens == 5 // 4 + 1

    def test_auto_estimate_sync(self):
        llm = MockLLM(default_response="ok").with_usage(auto_estimate=True)
        llm.complete_sync("test input")
        call = llm.last_call
        assert call.prompt_tokens == len("test input") // 4 + 1
        assert call.completion_tokens == len("ok") // 4 + 1

    @pytest.mark.asyncio
    async def test_auto_estimate_with_rule(self):
        llm = MockLLM().with_usage(auto_estimate=True)
        llm.on_input(contains="book").respond("Booking confirmed for tomorrow")
        await llm.complete("book a meeting")
        call = llm.last_call
        assert call.prompt_tokens == len("book a meeting") // 4 + 1
        assert call.completion_tokens == len("Booking confirmed for tomorrow") // 4 + 1


class TestWithoutUsage:
    """Tests that token fields are None when usage is not configured."""

    @pytest.mark.asyncio
    async def test_no_usage_config_returns_none(self):
        llm = MockLLM()
        await llm.complete("hello")
        assert llm.last_call.prompt_tokens is None
        assert llm.last_call.completion_tokens is None
        assert llm.last_call.total_tokens is None

    def test_no_usage_config_sync_returns_none(self):
        llm = MockLLM()
        llm.complete_sync("hello")
        assert llm.last_call.prompt_tokens is None
        assert llm.last_call.total_tokens is None


class TestUsageChaining:
    """Tests that with_usage() chains with other MockLLM methods."""

    @pytest.mark.asyncio
    async def test_chain_with_on_input(self):
        llm = (
            MockLLM()
            .with_usage(prompt_tokens=50, completion_tokens=25)
            .on_input(contains="hi")
            .respond("hello back")
        )
        # on_input().respond() returns MockLLM, so this chains
        result = await llm.complete("hi there")
        assert result == "hello back"
        assert llm.last_call.prompt_tokens == 50

    @pytest.mark.asyncio
    async def test_chain_with_attach_faults(self):
        from checkagent.mock.fault import FaultInjector

        fault = FaultInjector()
        llm = MockLLM().with_usage(prompt_tokens=100, completion_tokens=50)
        llm.attach_faults(fault)
        await llm.complete("hello")
        assert llm.last_call.prompt_tokens == 100


class TestLLMCallTotalTokens:
    """Tests for LLMCall.total_tokens property."""

    def test_both_none(self):
        from checkagent.mock.llm import LLMCall

        call = LLMCall(input_text="a", response_text="b")
        assert call.total_tokens is None

    def test_one_set(self):
        from checkagent.mock.llm import LLMCall

        call = LLMCall(
            input_text="a", response_text="b", prompt_tokens=10
        )
        assert call.total_tokens == 10

    def test_both_set(self):
        from checkagent.mock.llm import LLMCall

        call = LLMCall(
            input_text="a",
            response_text="b",
            prompt_tokens=100,
            completion_tokens=50,
        )
        assert call.total_tokens == 150


class TestWithUsageValidation:
    """Tests for with_usage() argument validation (F-081)."""

    def test_auto_estimate_with_prompt_tokens_raises(self):
        with pytest.raises(ValueError, match="Cannot set both"):
            MockLLM().with_usage(auto_estimate=True, prompt_tokens=100)

    def test_auto_estimate_with_completion_tokens_raises(self):
        with pytest.raises(ValueError, match="Cannot set both"):
            MockLLM().with_usage(auto_estimate=True, completion_tokens=50)

    def test_auto_estimate_with_both_tokens_raises(self):
        with pytest.raises(ValueError, match="Cannot set both"):
            MockLLM().with_usage(
                auto_estimate=True, prompt_tokens=100, completion_tokens=50
            )

    def test_auto_estimate_alone_works(self):
        llm = MockLLM().with_usage(auto_estimate=True)
        assert llm._auto_estimate_tokens is True

    def test_fixed_tokens_alone_works(self):
        llm = MockLLM().with_usage(prompt_tokens=100, completion_tokens=50)
        assert llm._usage == (100, 50)
