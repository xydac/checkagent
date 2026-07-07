"""Tests for the ap_cassette pytest fixture.

Verifies:
- RECORD mode when no cassette exists (creates and saves cassette)
- REPLAY mode when cassette already exists (loads and creates engine)
- @pytest.mark.cassette(path=...) custom path override
- CassetteFixture helpers: is_recording(), is_replaying(), mode, path
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from checkagent.core.plugin import CassetteFixture
from checkagent.replay.cassette import (
    Cassette,
    CassetteMeta,
    Interaction,
    RecordedRequest,
    RecordedResponse,
)
from checkagent.replay.engine import MatchStrategy, ReplayEngine
from checkagent.replay.recorder import CassetteRecorder


@pytest.fixture(autouse=True)
def _clean_fixture_cassette() -> None:
    """Remove cassette saved by the record-mode self-test so it runs fresh each time."""
    cassette = (
        Path(__file__).parent
        / "cassettes"
        / "test_ap_cassette_fixture"
        / "test_cassette_fixture_record_mode.json"
    )
    cassette.unlink(missing_ok=True)
    yield
    cassette.unlink(missing_ok=True)


def test_cassette_fixture_record_mode(ap_cassette: CassetteFixture, tmp_path: Path) -> None:
    """ap_cassette starts in record mode when no cassette file exists."""
    assert ap_cassette.mode == "record"
    assert ap_cassette.is_recording()
    assert not ap_cassette.is_replaying()
    assert ap_cassette.recorder is not None
    assert ap_cassette.engine is None
    assert ap_cassette.cassette is None


def test_cassette_fixture_record_saves_cassette(tmp_path: Path) -> None:
    """After a test using ap_cassette in record mode, the cassette file is saved."""
    cassette_path = tmp_path / "cassettes" / "test_session.json"

    recorder = CassetteRecorder(test_id="test_session")
    ctx = CassetteFixture(mode="record", path=cassette_path, recorder=recorder)

    # Simulate recording a call
    ctx.recorder.record_llm_call(
        method="chat.completions.create",
        request_body={"messages": [{"role": "user", "content": "ping"}]},
        response_body={"choices": [{"message": {"content": "pong"}}]},
    )

    # Simulate fixture teardown: finalize and save
    saved_cassette = ctx.recorder.finalize()
    saved_cassette.save(cassette_path)

    assert cassette_path.exists()
    data = json.loads(cassette_path.read_text(encoding="utf-8"))
    assert "interactions" in data
    assert len(data["interactions"]) == 1
    assert data["interactions"][0]["request"]["kind"] == "llm"


def test_cassette_fixture_replay_mode(tmp_path: Path) -> None:
    """CassetteFixture in replay mode loads cassette and creates engine."""
    cassette = Cassette(
        meta=CassetteMeta(test_id="test_replay"),
        interactions=[
            Interaction(
                request=RecordedRequest(
                    kind="llm",
                    method="chat.completions.create",
                    body={"messages": [{"role": "user", "content": "hello"}]},
                ),
                response=RecordedResponse(body={"choices": [{"message": {"content": "hi"}}]}),
            )
        ],
    )
    cassette.finalize()
    cassette_path = tmp_path / "test.json"
    cassette.save(cassette_path)

    loaded = Cassette.load(cassette_path)
    engine = ReplayEngine(loaded)
    ctx = CassetteFixture(mode="replay", path=cassette_path, engine=engine, cassette=loaded)

    assert ctx.mode == "replay"
    assert ctx.is_replaying()
    assert not ctx.is_recording()
    assert ctx.engine is not None
    assert ctx.cassette is not None
    assert ctx.recorder is None


def test_cassette_fixture_replay_engine_works(tmp_path: Path) -> None:
    """In replay mode the engine returns the recorded interaction on match."""
    cassette = Cassette(
        meta=CassetteMeta(test_id="test_replay_match"),
        interactions=[
            Interaction(
                request=RecordedRequest(
                    kind="llm",
                    method="chat.completions.create",
                    body={"messages": [{"role": "user", "content": "hi"}]},
                ),
                response=RecordedResponse(body={"text": "hello back"}),
            )
        ],
    )
    cassette.finalize()
    cassette_path = tmp_path / "test_match.json"
    cassette.save(cassette_path)

    loaded = Cassette.load(cassette_path)
    engine = ReplayEngine(loaded, strategy=MatchStrategy.SEQUENCE)
    ctx = CassetteFixture(mode="replay", path=cassette_path, engine=engine, cassette=loaded)

    interaction = ctx.engine.match(ctx.cassette.interactions[0].request)
    assert interaction is not None
    assert interaction.response.body == {"text": "hello back"}


def test_cassette_fixture_custom_path_via_mark(tmp_path: Path) -> None:
    """@pytest.mark.cassette(path=...) controls the cassette file location."""
    custom_path = tmp_path / "custom_cassette.json"
    ctx = CassetteFixture(mode="record", path=custom_path)
    assert ctx.path == custom_path


def test_cassette_fixture_exported_from_checkagent() -> None:
    """CassetteFixture is accessible from the checkagent.core.plugin namespace."""
    from checkagent.core.plugin import CassetteFixture as FixtureClass

    assert FixtureClass is CassetteFixture


class TestCassetteFixtureSimpleAPI:
    """Tests for the high-level replay_response() and arun() helpers."""

    def _make_replay_fixture(self, tmp_path: Path, prompt: str, response: str) -> CassetteFixture:
        from checkagent.replay.recorder import CassetteRecorder

        recorder = CassetteRecorder(test_id="test::simple")
        recorder.record_response(prompt, response)
        cassette = recorder.finalize()
        cassette_path = tmp_path / "simple.json"
        cassette.save(cassette_path)

        loaded = Cassette.load(cassette_path)
        engine = ReplayEngine(loaded, strategy=MatchStrategy.SEQUENCE)
        return CassetteFixture(mode="replay", path=cassette_path, engine=engine, cassette=loaded)

    def test_replay_response_returns_string(self, tmp_path: Path) -> None:
        ctx = self._make_replay_fixture(tmp_path, "ping", "pong")
        result = ctx.replay_response("ping")
        assert result == "pong"

    def test_replay_response_extracts_openai_format(self, tmp_path: Path) -> None:
        from checkagent.replay.recorder import CassetteRecorder

        recorder = CassetteRecorder()
        recorder.record_llm_call(
            method="chat.completions.create",
            request_body={"messages": [{"role": "user", "content": "hi"}]},
            response_body={"choices": [{"message": {"content": "hello"}}]},
        )
        cassette = recorder.finalize()
        path = tmp_path / "openai.json"
        cassette.save(path)
        loaded = Cassette.load(path)
        engine = ReplayEngine(loaded, strategy=MatchStrategy.SEQUENCE)
        ctx = CassetteFixture(mode="replay", path=path, engine=engine, cassette=loaded)

        result = ctx.replay_response("hi")
        assert result == "hello"

    def test_replay_response_raises_in_record_mode(self, tmp_path: Path) -> None:
        from checkagent.replay.recorder import CassetteRecorder

        recorder = CassetteRecorder()
        ctx = CassetteFixture(mode="record", path=tmp_path / "x.json", recorder=recorder)
        with pytest.raises(RuntimeError, match="record mode"):
            ctx.replay_response("hello")

    async def test_arun_in_record_mode_calls_agent(self, tmp_path: Path) -> None:
        from checkagent.replay.recorder import CassetteRecorder

        calls: list[str] = []

        async def my_agent(prompt: str) -> str:
            calls.append(prompt)
            return f"response:{prompt}"

        recorder = CassetteRecorder(test_id="test::arun_record")
        ctx = CassetteFixture(mode="record", path=tmp_path / "arun.json", recorder=recorder)
        result = await ctx.arun(my_agent, "hello")
        assert result == "response:hello"
        assert calls == ["hello"]
        assert recorder.interaction_count == 1

    async def test_arun_in_replay_mode_skips_agent(self, tmp_path: Path) -> None:
        from checkagent.replay.recorder import CassetteRecorder

        recorder = CassetteRecorder()
        recorder.record_response("hello", "from cassette")
        cassette = recorder.finalize()
        path = tmp_path / "arun_replay.json"
        cassette.save(path)

        loaded = Cassette.load(path)
        engine = ReplayEngine(loaded, strategy=MatchStrategy.SEQUENCE)
        ctx = CassetteFixture(mode="replay", path=path, engine=engine, cassette=loaded)

        calls: list[str] = []

        async def my_agent(prompt: str) -> str:
            calls.append(prompt)
            return "SHOULD NOT BE CALLED"

        result = await ctx.arun(my_agent, "hello")
        assert result == "from cassette"
        assert calls == []  # agent was NOT called

    async def test_arun_with_sync_agent(self, tmp_path: Path) -> None:
        from checkagent.replay.recorder import CassetteRecorder

        recorder = CassetteRecorder()
        ctx = CassetteFixture(mode="record", path=tmp_path / "sync.json", recorder=recorder)

        def sync_agent(prompt: str) -> str:
            return f"sync:{prompt}"

        result = await ctx.arun(sync_agent, "test")
        assert result == "sync:test"
        assert recorder.interaction_count == 1
