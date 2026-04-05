"""Tests for MockLLM — pattern-based mock LLM provider."""

from __future__ import annotations

import pytest

from checkagent.mock.llm import MatchMode, MockLLM, ResponseRule

# --- ResponseRule tests ---


class TestResponseRule:
    def test_substring_match(self):
        rule = ResponseRule(pattern="weather", response="Sunny")
        assert rule.matches("What's the weather today?")
        assert not rule.matches("How are you?")

    def test_exact_match(self):
        rule = ResponseRule(
            pattern="hello", response="Hi", match_mode=MatchMode.EXACT
        )
        assert rule.matches("hello")
        assert not rule.matches("hello world")
        assert not rule.matches("say hello")

    def test_regex_match(self):
        rule = ResponseRule(
            pattern=r"book.*flight", response="Booked", match_mode=MatchMode.REGEX
        )
        assert rule.matches("book a flight to NYC")
        assert rule.matches("book my flight")
        assert not rule.matches("booking hotel")

    def test_single_response(self):
        rule = ResponseRule(pattern="hi", response="Hello")
        assert rule.get_response() == "Hello"
        assert rule.get_response() == "Hello"

    def test_sequence_response_cycles(self):
        rule = ResponseRule(pattern="hi", response=["A", "B", "C"])
        assert rule.get_response() == "A"
        assert rule.get_response() == "B"
        assert rule.get_response() == "C"
        # Cycles back
        assert rule.get_response() == "A"

    def test_case_sensitive_substring(self):
        rule = ResponseRule(pattern="Hello", response="Hi")
        assert rule.matches("Hello world")
        assert not rule.matches("hello world")

    def test_regex_case_insensitive_flag(self):
        rule = ResponseRule(
            pattern=r"(?i)hello", response="Hi", match_mode=MatchMode.REGEX
        )
        assert rule.matches("Hello")
        assert rule.matches("HELLO")
        assert rule.matches("hello")


# --- MockLLM basic tests ---


class TestMockLLMBasic:
    @pytest.mark.asyncio
    async def test_default_response(self):
        llm = MockLLM()
        result = await llm.complete("anything")
        assert result == "Mock response"

    @pytest.mark.asyncio
    async def test_custom_default_response(self):
        llm = MockLLM(default_response="I don't know")
        result = await llm.complete("unknown query")
        assert result == "I don't know"

    @pytest.mark.asyncio
    async def test_substring_rule(self):
        llm = MockLLM()
        llm.add_rule("weather", "It's sunny")
        result = await llm.complete("What's the weather?")
        assert result == "It's sunny"

    @pytest.mark.asyncio
    async def test_exact_rule(self):
        llm = MockLLM()
        llm.add_rule("hello", "Hi there!", match_mode=MatchMode.EXACT)
        assert await llm.complete("hello") == "Hi there!"
        # Should not match partial
        assert await llm.complete("hello world") == "Mock response"

    @pytest.mark.asyncio
    async def test_regex_rule(self):
        llm = MockLLM()
        llm.add_rule(r"book\s+\w+\s+flight", "Flight booked!", match_mode=MatchMode.REGEX)
        assert await llm.complete("book a flight") == "Flight booked!"
        assert await llm.complete("book my flight") == "Flight booked!"
        assert await llm.complete("cancel flight") == "Mock response"

    @pytest.mark.asyncio
    async def test_first_matching_rule_wins(self):
        llm = MockLLM()
        llm.add_rule("hello", "First rule")
        llm.add_rule("hello", "Second rule")
        result = await llm.complete("hello")
        assert result == "First rule"

    @pytest.mark.asyncio
    async def test_chaining(self):
        llm = MockLLM()
        llm.add_rule("a", "A").add_rule("b", "B").add_rule("c", "C")
        assert await llm.complete("a") == "A"
        assert await llm.complete("b") == "B"
        assert await llm.complete("c") == "C"


# --- Sequence responses ---


class TestMockLLMSequence:
    @pytest.mark.asyncio
    async def test_sequence_responses(self):
        llm = MockLLM()
        llm.add_rule("greet", ["Hi!", "Hey!", "Hello!"])
        assert await llm.complete("greet me") == "Hi!"
        assert await llm.complete("greet again") == "Hey!"
        assert await llm.complete("greet once more") == "Hello!"

    @pytest.mark.asyncio
    async def test_sequence_cycles(self):
        llm = MockLLM()
        llm.add_rule("x", ["A", "B"])
        assert await llm.complete("x") == "A"
        assert await llm.complete("x") == "B"
        assert await llm.complete("x") == "A"
        assert await llm.complete("x") == "B"


# --- Call recording ---


class TestMockLLMRecording:
    @pytest.mark.asyncio
    async def test_call_count(self):
        llm = MockLLM()
        assert llm.call_count == 0
        await llm.complete("a")
        await llm.complete("b")
        assert llm.call_count == 2

    @pytest.mark.asyncio
    async def test_calls_list(self):
        llm = MockLLM()
        llm.add_rule("weather", "Sunny")
        await llm.complete("weather today")
        calls = llm.calls
        assert len(calls) == 1
        assert calls[0].input_text == "weather today"
        assert calls[0].response_text == "Sunny"
        assert calls[0].rule_pattern == "weather"
        assert calls[0].was_default is False

    @pytest.mark.asyncio
    async def test_default_call_recording(self):
        llm = MockLLM(default_response="fallback")
        await llm.complete("unknown")
        assert llm.last_call is not None
        assert llm.last_call.was_default is True
        assert llm.last_call.response_text == "fallback"
        assert llm.last_call.rule_pattern is None

    @pytest.mark.asyncio
    async def test_last_call(self):
        llm = MockLLM()
        assert llm.last_call is None
        await llm.complete("first")
        await llm.complete("second")
        assert llm.last_call is not None
        assert llm.last_call.input_text == "second"

    @pytest.mark.asyncio
    async def test_was_called_with(self):
        llm = MockLLM()
        await llm.complete("hello world")
        assert llm.was_called_with("hello world")
        assert not llm.was_called_with("hello")

    @pytest.mark.asyncio
    async def test_get_calls_matching(self):
        llm = MockLLM()
        await llm.complete("book flight to NYC")
        await llm.complete("book hotel in NYC")
        await llm.complete("cancel flight")
        matches = llm.get_calls_matching("book")
        assert len(matches) == 2

    @pytest.mark.asyncio
    async def test_model_recorded(self):
        llm = MockLLM(default_model="gpt-4o-mock")
        llm.add_rule("test", "result", model="claude-mock")
        await llm.complete("test input")
        assert llm.last_call.model == "claude-mock"

    @pytest.mark.asyncio
    async def test_default_model_recorded(self):
        llm = MockLLM(default_model="gpt-4o-mock")
        await llm.complete("anything")
        assert llm.last_call.model == "gpt-4o-mock"


# --- Reset ---


class TestMockLLMReset:
    @pytest.mark.asyncio
    async def test_reset_clears_calls_and_counters(self):
        llm = MockLLM()
        llm.add_rule("x", ["A", "B"])
        await llm.complete("x")
        assert llm.call_count == 1
        llm.reset()
        assert llm.call_count == 0
        # Sequence counter also reset
        assert await llm.complete("x") == "A"

    @pytest.mark.asyncio
    async def test_reset_calls_keeps_sequence_counter(self):
        llm = MockLLM()
        llm.add_rule("x", ["A", "B"])
        await llm.complete("x")  # A
        llm.reset_calls()
        assert llm.call_count == 0
        # Sequence counter preserved — next is B
        assert await llm.complete("x") == "B"


# --- Sync interface ---


class TestMockLLMSync:
    def test_complete_sync(self):
        llm = MockLLM()
        llm.add_rule("hello", "Hi!")
        assert llm.complete_sync("hello") == "Hi!"
        assert llm.call_count == 1
        assert llm.last_call.input_text == "hello"

    def test_complete_sync_default(self):
        llm = MockLLM(default_response="default")
        assert llm.complete_sync("unknown") == "default"
        assert llm.last_call.was_default is True

    def test_complete_sync_sequence(self):
        llm = MockLLM()
        llm.add_rule("x", ["A", "B"])
        assert llm.complete_sync("x") == "A"
        assert llm.complete_sync("x") == "B"
        assert llm.complete_sync("x") == "A"


# --- Metadata ---


class TestMockLLMMetadata:
    @pytest.mark.asyncio
    async def test_rule_metadata_recorded(self):
        llm = MockLLM()
        llm.add_rule("test", "result", metadata={"latency_ms": 100})
        await llm.complete("test")
        assert llm.last_call.metadata == {"latency_ms": 100}

    @pytest.mark.asyncio
    async def test_default_call_has_empty_metadata(self):
        llm = MockLLM()
        await llm.complete("anything")
        assert llm.last_call.metadata == {}
