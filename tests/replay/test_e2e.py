"""End-to-end record-and-replay tests using MockLLM — no real API keys needed.

These tests close the Layer 2 gap identified in meta-review cycle 190:
'cassette recording and playback has scaffolding but no working end-to-end
test that actually records a real interaction.'

The workflow under test:
  1. Run an agent backed by MockLLM
  2. Record each LLM call via CassetteRecorder
  3. Save cassette to disk
  4. Load cassette, replay via ReplayEngine
  5. Assert replayed responses match recorded ones
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from checkagent.mock.llm import MockLLM
from checkagent.replay.cassette import Cassette
from checkagent.replay.engine import CassetteMismatchError, MatchStrategy, ReplayEngine
from checkagent.replay.recorder import CassetteRecorder

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _record_mock_session(
    llm: MockLLM, inputs: list[str], recorder: CassetteRecorder
) -> list[str]:
    """Run inputs through MockLLM and record each call synchronously."""
    responses = []
    for text in inputs:
        resp = llm.complete_sync(text)
        recorder.record_llm_call(
            method="chat.completions.create",
            request_body={"messages": [{"role": "user", "content": text}]},
            response_body=resp,
            prompt_tokens=len(text.split()),
            completion_tokens=len(resp.split()),
        )
        responses.append(resp)
    return responses


async def _record_mock_session_async(
    llm: MockLLM, inputs: list[str], recorder: CassetteRecorder
) -> list[str]:
    """Run inputs through MockLLM and record each call asynchronously."""
    responses = []
    for text in inputs:
        resp = await llm.complete(text)
        recorder.record_llm_call(
            method="chat.completions.create",
            request_body={"messages": [{"role": "user", "content": text}]},
            response_body=resp,
            prompt_tokens=len(text.split()),
            completion_tokens=len(resp.split()),
        )
        responses.append(resp)
    return responses


# ---------------------------------------------------------------------------
# Core e2e: record → save → load → replay
# ---------------------------------------------------------------------------

class TestRecordReplayEndToEnd:
    def test_single_turn_record_and_replay(self, tmp_path: Path):
        """Record one LLM call, save to disk, reload, replay — responses must match."""
        llm = MockLLM()
        llm.on_input(contains="hello").respond("Hi there!")

        recorder = CassetteRecorder(test_id="tests/replay::test_single_turn")
        live_responses = _record_mock_session(llm, ["hello"], recorder)

        cassette = recorder.finalize()
        cassette_path = tmp_path / "single_turn.json"
        cassette.save(cassette_path)

        loaded = Cassette.load(cassette_path)
        assert loaded.verify_integrity()
        assert len(loaded.interactions) == 1

        engine = ReplayEngine(loaded, strategy=MatchStrategy.SEQUENCE)
        from checkagent.replay.cassette import RecordedRequest
        replayed = engine.match(RecordedRequest(
            kind="llm",
            method="chat.completions.create",
            body={"messages": [{"role": "user", "content": "hello"}]},
        ))

        assert replayed.response.body == live_responses[0]

    def test_multi_turn_record_and_replay(self, tmp_path: Path):
        """Record multiple turns, replay in order — all responses must match."""
        llm = MockLLM()
        llm.on_input(contains="weather").respond("Sunny today!")
        llm.on_input(contains="news").respond("Nothing new.")
        llm.on_input(contains="joke").respond("Why did the AI cross the road?")

        inputs = ["weather", "news", "joke"]
        recorder = CassetteRecorder(test_id="tests/replay::test_multi_turn")
        live_responses = _record_mock_session(llm, inputs, recorder)

        cassette = recorder.finalize()
        path = tmp_path / "multi_turn.json"
        cassette.save(path)

        loaded = Cassette.load(path)
        assert len(loaded.interactions) == 3

        engine = ReplayEngine(loaded, strategy=MatchStrategy.SEQUENCE)
        from checkagent.replay.cassette import RecordedRequest
        replayed_responses = []
        for inp in inputs:
            result = engine.match(RecordedRequest(
                kind="llm",
                method="chat.completions.create",
                body={"messages": [{"role": "user", "content": inp}]},
            ))
            replayed_responses.append(result.response.body)

        assert replayed_responses == live_responses

    def test_cassette_json_is_human_readable(self, tmp_path: Path):
        """Saved cassette is valid JSON with expected structure."""
        llm = MockLLM(default_response="default answer")
        recorder = CassetteRecorder(test_id="tests/replay::test_json")
        _record_mock_session(llm, ["anything"], recorder)

        cassette = recorder.finalize()
        path = tmp_path / "readable.json"
        cassette.save(path)

        data = json.loads(path.read_text())
        assert "meta" in data
        assert data["meta"]["schema_version"] == 1
        assert "interactions" in data
        assert len(data["interactions"]) == 1
        assert data["interactions"][0]["request"]["kind"] == "llm"

    def test_default_response_recorded_and_replayed(self, tmp_path: Path):
        """Unmatched inputs use MockLLM default; default is captured and replayed."""
        llm = MockLLM(default_response="I don't know.")

        recorder = CassetteRecorder(test_id="tests/replay::test_default")
        live = _record_mock_session(llm, ["something unknown"], recorder)

        cassette = recorder.finalize()
        path = tmp_path / "default.json"
        cassette.save(path)

        loaded = Cassette.load(path)
        engine = ReplayEngine(loaded, strategy=MatchStrategy.SEQUENCE)
        from checkagent.replay.cassette import RecordedRequest
        result = engine.match(RecordedRequest(
            kind="llm",
            method="chat.completions.create",
            body={"messages": [{"role": "user", "content": "something unknown"}]},
        ))

        assert result.response.body == live[0]

    def test_replay_exhaustion_raises(self, tmp_path: Path):
        """Attempting to replay more calls than recorded raises CassetteMismatchError."""
        llm = MockLLM(default_response="ok")
        recorder = CassetteRecorder(test_id="tests/replay::test_exhaustion")
        _record_mock_session(llm, ["one call only"], recorder)

        cassette = recorder.finalize()
        path = tmp_path / "exhaustion.json"
        cassette.save(path)

        loaded = Cassette.load(path)
        engine = ReplayEngine(loaded, strategy=MatchStrategy.SEQUENCE)
        from checkagent.replay.cassette import RecordedRequest
        req = RecordedRequest(kind="llm", method="chat.completions.create", body={})

        engine.match(req)  # first call succeeds
        with pytest.raises(CassetteMismatchError):
            engine.match(req)  # second call raises — cassette exhausted

    async def test_async_record_and_replay(self, tmp_path: Path):
        """Async recording path produces a replayable cassette."""
        llm = MockLLM()
        llm.on_input(contains="async question").respond("async answer!")

        recorder = CassetteRecorder(test_id="tests/replay::test_async")
        live = await _record_mock_session_async(llm, ["async question"], recorder)

        cassette = recorder.finalize()
        path = tmp_path / "async.json"
        cassette.save(path)

        loaded = Cassette.load(path)
        assert loaded.verify_integrity()

        engine = ReplayEngine(loaded, strategy=MatchStrategy.SEQUENCE)
        from checkagent.replay.cassette import RecordedRequest
        result = engine.match(RecordedRequest(
            kind="llm",
            method="chat.completions.create",
            body={"messages": [{"role": "user", "content": "async question"}]},
        ))
        assert result.response.body == live[0]


class TestSubsetMatchReplay:
    def test_subset_match_ignores_extra_request_fields(self, tmp_path: Path):
        """Subset strategy matches even when the live request has extra fields."""
        llm = MockLLM(default_response="42")
        recorder = CassetteRecorder(test_id="tests/replay::test_subset")
        _record_mock_session(llm, ["query"], recorder)

        cassette = recorder.finalize()
        path = tmp_path / "subset.json"
        cassette.save(path)

        loaded = Cassette.load(path)
        engine = ReplayEngine(loaded, strategy=MatchStrategy.SUBSET)
        from checkagent.replay.cassette import RecordedRequest
        result = engine.match(RecordedRequest(
            kind="llm",
            method="chat.completions.create",
            body={
                "messages": [{"role": "user", "content": "query"}],
                "temperature": 0.7,  # extra field not in cassette
                "max_tokens": 100,   # extra field not in cassette
            },
        ))
        assert result.response.body == "42"
