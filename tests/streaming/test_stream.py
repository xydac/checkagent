"""Tests for MockLLM streaming and StreamCollector.

Covers F1.6 (streaming mock responses), F13.1 (stream collection),
F13.2 (streaming mock LLM), and F13.3 (stream event assertions).
"""

from __future__ import annotations

import asyncio
import time

import pytest

from checkagent.core.types import StreamEvent, StreamEventType
from checkagent.mock.llm import MockLLM, MatchMode
from checkagent.streaming.collector import StreamCollector


# ── MockLLM.stream() basics ──────────────────────────────────────────


class TestMockLLMStreamBasics:
    """Basic streaming from MockLLM."""

    async def test_stream_with_configured_chunks(self):
        llm = MockLLM()
        llm.stream_response("weather", ["It's ", "sunny ", "today!"])

        events = []
        async for event in llm.stream("What's the weather?"):
            events.append(event)

        types = [e.event_type for e in events]
        assert types[0] == StreamEventType.RUN_START
        assert types[-1] == StreamEventType.RUN_END
        assert all(t == StreamEventType.TEXT_DELTA for t in types[1:-1])
        assert len(events) == 5  # START + 3 chunks + END

    async def test_stream_chunk_data(self):
        llm = MockLLM()
        llm.stream_response("hello", ["Hi ", "there!"])

        texts = []
        async for event in llm.stream("hello"):
            if event.event_type == StreamEventType.TEXT_DELTA:
                texts.append(event.data)

        assert texts == ["Hi ", "there!"]

    async def test_stream_falls_back_to_regular_rule(self):
        llm = MockLLM()
        llm.add_rule("hello", "Hi there!")

        events = []
        async for event in llm.stream("hello"):
            events.append(event)

        deltas = [e for e in events if e.event_type == StreamEventType.TEXT_DELTA]
        assert len(deltas) == 1
        assert deltas[0].data == "Hi there!"

    async def test_stream_falls_back_to_default(self):
        llm = MockLLM(default_response="I don't know")

        events = []
        async for event in llm.stream("anything"):
            events.append(event)

        deltas = [e for e in events if e.event_type == StreamEventType.TEXT_DELTA]
        assert len(deltas) == 1
        assert deltas[0].data == "I don't know"

    async def test_stream_records_call(self):
        llm = MockLLM()
        llm.stream_response("hello", ["Hi ", "world"])

        async for _ in llm.stream("hello"):
            pass

        assert llm.call_count == 1
        assert llm.last_call is not None
        assert llm.last_call.streamed is True
        assert llm.last_call.response_text == "Hi world"
        assert llm.last_call.rule_pattern == "hello"

    async def test_stream_default_records_call(self):
        llm = MockLLM()

        async for _ in llm.stream("anything"):
            pass

        assert llm.last_call is not None
        assert llm.last_call.was_default is True
        assert llm.last_call.streamed is True


class TestMockLLMStreamConfig:
    """stream_response() configuration options."""

    async def test_stream_response_chaining(self):
        llm = MockLLM()
        result = llm.stream_response("a", ["x"]).stream_response("b", ["y"])
        assert result is llm

    async def test_stream_response_exact_match(self):
        llm = MockLLM()
        llm.stream_response("hello", ["Hi!"], match_mode=MatchMode.EXACT)

        # Exact match works
        events = []
        async for e in llm.stream("hello"):
            events.append(e)
        deltas = [e for e in events if e.event_type == StreamEventType.TEXT_DELTA]
        assert deltas[0].data == "Hi!"

        # Substring does not match — falls back to default
        events2 = []
        async for e in llm.stream("say hello"):
            events2.append(e)
        assert any(e.event_type == StreamEventType.TEXT_DELTA and e.data == "Mock response"
                   for e in events2)

    async def test_stream_response_regex_match(self):
        llm = MockLLM()
        llm.stream_response(r"book.*flight", ["Booked!"], match_mode=MatchMode.REGEX)

        events = []
        async for e in llm.stream("I want to book a flight"):
            events.append(e)
        deltas = [e for e in events if e.event_type == StreamEventType.TEXT_DELTA]
        assert deltas[0].data == "Booked!"

    async def test_stream_response_with_model(self):
        llm = MockLLM()
        llm.stream_response("hello", ["Hi!"], model="gpt-4")

        async for _ in llm.stream("hello"):
            pass

        assert llm.last_call.model == "gpt-4"

    async def test_stream_response_with_metadata(self):
        llm = MockLLM()
        llm.stream_response("hello", ["Hi!"], metadata={"provider": "openai"})

        async for _ in llm.stream("hello"):
            pass

        assert llm.last_call.metadata == {"provider": "openai"}

    async def test_stream_first_matching_rule_wins(self):
        llm = MockLLM()
        llm.stream_response("hello", ["First!"])
        llm.stream_response("hello", ["Second!"])

        events = []
        async for e in llm.stream("hello"):
            events.append(e)
        deltas = [e for e in events if e.event_type == StreamEventType.TEXT_DELTA]
        assert deltas[0].data == "First!"

    async def test_stream_rule_takes_priority_over_regular_rule(self):
        llm = MockLLM()
        llm.add_rule("hello", "Regular response")
        llm.stream_response("hello", ["Streamed!"])

        async for _ in llm.stream("hello"):
            pass

        assert llm.last_call.response_text == "Streamed!"

    async def test_single_chunk_stream(self):
        llm = MockLLM()
        llm.stream_response("hello", ["One chunk"])

        events = []
        async for e in llm.stream("hello"):
            events.append(e)

        assert len(events) == 3  # START + 1 chunk + END


class TestMockLLMStreamDelay:
    """Delay between chunks."""

    async def test_stream_with_delay(self):
        llm = MockLLM()
        llm.stream_response("hello", ["a", "b", "c"], delay_ms=20)

        start = time.monotonic()
        async for _ in llm.stream("hello"):
            pass
        elapsed_ms = (time.monotonic() - start) * 1000

        # 2 delays (between chunks 1-2 and 2-3), ~40ms total
        assert elapsed_ms >= 30  # allow some tolerance

    async def test_stream_no_delay_by_default(self):
        llm = MockLLM()
        llm.stream_response("hello", ["a", "b", "c"])

        start = time.monotonic()
        async for _ in llm.stream("hello"):
            pass
        elapsed_ms = (time.monotonic() - start) * 1000

        assert elapsed_ms < 50  # should be near-instant


# ── StreamCollector ───────────────────────────────────────────────────


class TestStreamCollectorBasics:
    """StreamCollector event collection."""

    async def test_collect_from_stream(self):
        llm = MockLLM()
        llm.stream_response("hello", ["Hi ", "there!"])

        collector = StreamCollector()
        await collector.collect_from(llm.stream("hello"))

        assert collector.total_events == 4  # START + 2 deltas + END
        assert collector.total_chunks == 2

    async def test_add_individual_events(self):
        collector = StreamCollector()
        collector.add(StreamEvent(event_type=StreamEventType.RUN_START))
        collector.add(StreamEvent(event_type=StreamEventType.TEXT_DELTA, data="Hi"))

        assert collector.total_events == 2

    async def test_collect_from_returns_self(self):
        llm = MockLLM()
        llm.stream_response("hello", ["Hi!"])

        collector = StreamCollector()
        result = await collector.collect_from(llm.stream("hello"))
        assert result is collector

    async def test_events_are_copies(self):
        collector = StreamCollector()
        collector.add(StreamEvent(event_type=StreamEventType.TEXT_DELTA, data="Hi"))
        events = collector.events
        events.clear()
        assert collector.total_events == 1  # original unaffected


class TestStreamCollectorAssertions:
    """StreamCollector assertion helpers (F13.3)."""

    async def test_aggregated_text(self):
        llm = MockLLM()
        llm.stream_response("hello", ["Hello ", "world", "!"])

        collector = StreamCollector()
        await collector.collect_from(llm.stream("hello"))

        assert collector.aggregated_text == "Hello world!"

    async def test_aggregated_text_empty(self):
        collector = StreamCollector()
        assert collector.aggregated_text == ""

    async def test_of_type(self):
        llm = MockLLM()
        llm.stream_response("hello", ["a", "b"])

        collector = StreamCollector()
        await collector.collect_from(llm.stream("hello"))

        deltas = collector.of_type(StreamEventType.TEXT_DELTA)
        assert len(deltas) == 2
        starts = collector.of_type(StreamEventType.RUN_START)
        assert len(starts) == 1

    async def test_first_of_type(self):
        llm = MockLLM()
        llm.stream_response("hello", ["a", "b"])

        collector = StreamCollector()
        await collector.collect_from(llm.stream("hello"))

        first_delta = collector.first_of_type(StreamEventType.TEXT_DELTA)
        assert first_delta is not None
        assert first_delta.data == "a"

    async def test_first_of_type_missing(self):
        collector = StreamCollector()
        assert collector.first_of_type(StreamEventType.ERROR) is None

    async def test_time_to_first_token(self):
        llm = MockLLM()
        llm.stream_response("hello", ["Hi!"], delay_ms=0)

        collector = StreamCollector()
        await collector.collect_from(llm.stream("hello"))

        ttft = collector.time_to_first_token
        assert ttft is not None
        assert ttft >= 0

    async def test_time_to_first_token_none_without_events(self):
        collector = StreamCollector()
        assert collector.time_to_first_token is None

    async def test_tool_call_started(self):
        collector = StreamCollector()
        collector.add(StreamEvent(
            event_type=StreamEventType.TOOL_CALL_START,
            data={"name": "search", "arguments": {"q": "test"}},
        ))

        assert collector.tool_call_started("search") is True
        assert collector.tool_call_started("unknown") is False

    async def test_has_error(self):
        collector = StreamCollector()
        assert collector.has_error is False

        collector.add(StreamEvent(event_type=StreamEventType.ERROR, data="boom"))
        assert collector.has_error is True

    async def test_error_events(self):
        collector = StreamCollector()
        collector.add(StreamEvent(event_type=StreamEventType.ERROR, data="err1"))
        collector.add(StreamEvent(event_type=StreamEventType.TEXT_DELTA, data="ok"))
        collector.add(StreamEvent(event_type=StreamEventType.ERROR, data="err2"))

        assert len(collector.error_events) == 2

    async def test_reset(self):
        llm = MockLLM()
        llm.stream_response("hello", ["Hi!"])

        collector = StreamCollector()
        await collector.collect_from(llm.stream("hello"))
        assert collector.total_events > 0

        collector.reset()
        assert collector.total_events == 0
        assert collector.aggregated_text == ""


# ── Fixture integration ──────────────────────────────────────────────


class TestStreamFixtures:
    """Test the ap_stream_collector fixture."""

    async def test_fixture_provides_collector(self, ap_stream_collector):
        assert isinstance(ap_stream_collector, StreamCollector)
        assert ap_stream_collector.total_events == 0

    async def test_fixture_with_mock_llm(self, ap_mock_llm, ap_stream_collector):
        ap_mock_llm.stream_response("hello", ["Hi ", "there!"])
        await ap_stream_collector.collect_from(ap_mock_llm.stream("hello"))

        assert ap_stream_collector.aggregated_text == "Hi there!"
        assert ap_stream_collector.total_chunks == 2
        assert ap_mock_llm.call_count == 1
