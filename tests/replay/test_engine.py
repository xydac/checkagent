"""Tests for the replay engine (F2.2)."""

from __future__ import annotations

import pytest

from checkagent.replay.cassette import (
    Cassette,
    Interaction,
    RecordedRequest,
    RecordedResponse,
)
from checkagent.replay.engine import (
    CassetteMismatchError,
    MatchStrategy,
    ReplayEngine,
)


def _make_cassette(interactions: list[Interaction]) -> Cassette:
    """Build a finalized cassette from interactions."""
    cassette = Cassette(interactions=interactions)
    cassette.finalize()
    return cassette


def _llm_interaction(
    method: str = "chat", body: dict | None = None, response: str = "ok"
) -> Interaction:
    return Interaction(
        request=RecordedRequest(kind="llm", method=method, body=body or {}),
        response=RecordedResponse(body=response),
    )


def _tool_interaction(
    name: str = "search", args: dict | None = None, result: str = "found"
) -> Interaction:
    return Interaction(
        request=RecordedRequest(kind="tool", method=name, body=args or {}),
        response=RecordedResponse(body=result),
    )


# --- Exact matching ---


class TestExactMatch:
    def test_match_single_llm_call(self):
        cassette = _make_cassette([
            _llm_interaction("chat", {"prompt": "hi"}, "hello")
        ])
        engine = ReplayEngine(cassette, MatchStrategy.EXACT)
        result = engine.match(
            RecordedRequest(kind="llm", method="chat", body={"prompt": "hi"})
        )
        assert result.response.body == "hello"
        assert engine.all_used

    def test_match_tool_call(self):
        cassette = _make_cassette([
            _tool_interaction("search", {"q": "weather"}, "sunny")
        ])
        engine = ReplayEngine(cassette, MatchStrategy.EXACT)
        result = engine.match(
            RecordedRequest(kind="tool", method="search", body={"q": "weather"})
        )
        assert result.response.body == "sunny"

    def test_no_match_different_body(self):
        cassette = _make_cassette([
            _llm_interaction("chat", {"prompt": "hi"})
        ])
        engine = ReplayEngine(cassette, MatchStrategy.EXACT)
        with pytest.raises(CassetteMismatchError, match="chat"):
            engine.match(
                RecordedRequest(
                    kind="llm", method="chat", body={"prompt": "bye"}
                )
            )

    def test_no_match_different_method(self):
        cassette = _make_cassette([
            _llm_interaction("chat", {"prompt": "hi"})
        ])
        engine = ReplayEngine(cassette, MatchStrategy.EXACT)
        with pytest.raises(CassetteMismatchError):
            engine.match(
                RecordedRequest(
                    kind="llm", method="embeddings", body={"prompt": "hi"}
                )
            )

    def test_no_match_different_kind(self):
        cassette = _make_cassette([
            _llm_interaction("chat", {"prompt": "hi"})
        ])
        engine = ReplayEngine(cassette, MatchStrategy.EXACT)
        with pytest.raises(CassetteMismatchError):
            engine.match(
                RecordedRequest(
                    kind="tool", method="chat", body={"prompt": "hi"}
                )
            )

    def test_multiple_calls_matched_in_order(self):
        cassette = _make_cassette([
            _llm_interaction("chat", {"prompt": "a"}, "resp_a"),
            _llm_interaction("chat", {"prompt": "b"}, "resp_b"),
        ])
        engine = ReplayEngine(cassette, MatchStrategy.EXACT)
        r1 = engine.match(
            RecordedRequest(kind="llm", method="chat", body={"prompt": "a"})
        )
        r2 = engine.match(
            RecordedRequest(kind="llm", method="chat", body={"prompt": "b"})
        )
        assert r1.response.body == "resp_a"
        assert r2.response.body == "resp_b"
        assert engine.all_used

    def test_already_used_interaction_not_reused(self):
        cassette = _make_cassette([
            _llm_interaction("chat", {"prompt": "hi"}, "first"),
        ])
        engine = ReplayEngine(cassette, MatchStrategy.EXACT)
        engine.match(
            RecordedRequest(kind="llm", method="chat", body={"prompt": "hi"})
        )
        with pytest.raises(CassetteMismatchError):
            engine.match(
                RecordedRequest(
                    kind="llm", method="chat", body={"prompt": "hi"}
                )
            )


# --- Sequence matching ---


class TestSequenceMatch:
    def test_match_by_position(self):
        cassette = _make_cassette([
            _llm_interaction("chat", {"prompt": "a"}, "resp_a"),
            _tool_interaction("search", {"q": "x"}, "found"),
        ])
        engine = ReplayEngine(cassette, MatchStrategy.SEQUENCE)
        r1 = engine.match(
            RecordedRequest(kind="llm", method="anything", body={})
        )
        r2 = engine.match(
            RecordedRequest(kind="tool", method="whatever", body={})
        )
        assert r1.response.body == "resp_a"
        assert r2.response.body == "found"

    def test_exhausted_sequence(self):
        cassette = _make_cassette([
            _llm_interaction("chat", {}, "only_one"),
        ])
        engine = ReplayEngine(cassette, MatchStrategy.SEQUENCE)
        engine.match(RecordedRequest(kind="llm", method="chat", body={}))
        with pytest.raises(CassetteMismatchError, match="sequence"):
            engine.match(RecordedRequest(kind="llm", method="chat", body={}))


# --- Subset matching ---


class TestSubsetMatch:
    def test_match_with_extra_keys_in_request(self):
        cassette = _make_cassette([
            _llm_interaction("chat", {"model": "gpt-4"}, "ok")
        ])
        engine = ReplayEngine(cassette, MatchStrategy.SUBSET)
        result = engine.match(
            RecordedRequest(
                kind="llm",
                method="chat",
                body={"model": "gpt-4", "temperature": 0.7},
            )
        )
        assert result.response.body == "ok"

    def test_no_match_missing_key(self):
        cassette = _make_cassette([
            _llm_interaction("chat", {"model": "gpt-4", "stream": True})
        ])
        engine = ReplayEngine(cassette, MatchStrategy.SUBSET)
        with pytest.raises(CassetteMismatchError):
            engine.match(
                RecordedRequest(
                    kind="llm", method="chat", body={"model": "gpt-4"}
                )
            )

    def test_no_match_wrong_value(self):
        cassette = _make_cassette([
            _llm_interaction("chat", {"model": "gpt-4"})
        ])
        engine = ReplayEngine(cassette, MatchStrategy.SUBSET)
        with pytest.raises(CassetteMismatchError):
            engine.match(
                RecordedRequest(
                    kind="llm", method="chat", body={"model": "gpt-3.5"}
                )
            )

    def test_nested_subset(self):
        cassette = _make_cassette([
            _llm_interaction(
                "chat",
                {"params": {"temperature": 0.5}},
                "warm",
            )
        ])
        engine = ReplayEngine(cassette, MatchStrategy.SUBSET)
        result = engine.match(
            RecordedRequest(
                kind="llm",
                method="chat",
                body={"params": {"temperature": 0.5, "top_p": 0.9}},
            )
        )
        assert result.response.body == "warm"

    def test_empty_recorded_body_matches_anything(self):
        cassette = _make_cassette([
            _llm_interaction("chat", {}, "any")
        ])
        engine = ReplayEngine(cassette, MatchStrategy.SUBSET)
        result = engine.match(
            RecordedRequest(
                kind="llm", method="chat", body={"whatever": "value"}
            )
        )
        assert result.response.body == "any"


# --- Engine state ---


class TestEngineState:
    def test_remaining_count(self):
        cassette = _make_cassette([
            _llm_interaction("a", {}, "1"),
            _llm_interaction("b", {}, "2"),
            _llm_interaction("c", {}, "3"),
        ])
        engine = ReplayEngine(cassette, MatchStrategy.SEQUENCE)
        assert engine.remaining == 3
        engine.match(RecordedRequest(kind="llm", method="a", body={}))
        assert engine.remaining == 2

    def test_reset(self):
        cassette = _make_cassette([
            _llm_interaction("chat", {"p": "hi"}, "hello"),
        ])
        engine = ReplayEngine(cassette, MatchStrategy.EXACT)
        engine.match(
            RecordedRequest(kind="llm", method="chat", body={"p": "hi"})
        )
        assert engine.all_used
        engine.reset()
        assert engine.remaining == 1
        assert not engine.all_used

    def test_error_message_includes_method(self):
        cassette = _make_cassette([])
        engine = ReplayEngine(cassette, MatchStrategy.EXACT)
        with pytest.raises(
            CassetteMismatchError, match="my_method"
        ):
            engine.match(
                RecordedRequest(kind="llm", method="my_method", body={})
            )

    def test_error_message_includes_strategy(self):
        cassette = _make_cassette([])
        engine = ReplayEngine(cassette, MatchStrategy.SUBSET)
        with pytest.raises(CassetteMismatchError, match="subset"):
            engine.match(
                RecordedRequest(kind="llm", method="x", body={})
            )
