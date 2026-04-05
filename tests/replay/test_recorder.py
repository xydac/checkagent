"""Tests for the cassette recorder (F2.1)."""

from __future__ import annotations

import pytest

from checkagent.replay.recorder import CassetteRecorder, TimedCall


class TestCassetteRecorder:
    def test_empty_recorder(self):
        recorder = CassetteRecorder(test_id="test::empty")
        assert recorder.interaction_count == 0
        cassette = recorder.finalize()
        assert len(cassette.interactions) == 0
        assert cassette.meta.test_id == "test::empty"

    def test_record_llm_call(self):
        recorder = CassetteRecorder(test_id="test::llm")
        interaction = recorder.record_llm_call(
            method="chat.completions.create",
            request_body={"messages": [{"role": "user", "content": "hi"}]},
            response_body={"choices": [{"message": {"content": "hello"}}]},
            model="gpt-4",
            prompt_tokens=5,
            completion_tokens=3,
            duration_ms=120.5,
        )
        assert interaction.request.kind == "llm"
        assert interaction.request.method == "chat.completions.create"
        assert interaction.request.model == "gpt-4"
        assert interaction.response.prompt_tokens == 5
        assert interaction.response.completion_tokens == 3
        assert interaction.response.duration_ms == 120.5
        assert recorder.interaction_count == 1

    def test_record_tool_call(self):
        recorder = CassetteRecorder()
        interaction = recorder.record_tool_call(
            tool_name="search",
            arguments={"query": "weather"},
            result={"temperature": 72},
            duration_ms=50.0,
        )
        assert interaction.request.kind == "tool"
        assert interaction.request.method == "search"
        assert interaction.response.body == {"temperature": 72}
        assert recorder.interaction_count == 1

    def test_record_multiple_interactions(self):
        recorder = CassetteRecorder(test_id="test::multi")
        recorder.record_llm_call(
            method="chat.completions.create",
            request_body={"messages": [{"role": "user", "content": "hi"}]},
            response_body={"choices": [{"message": {"content": "use search"}}]},
        )
        recorder.record_tool_call(
            tool_name="search",
            arguments={"query": "weather"},
            result="sunny",
        )
        recorder.record_llm_call(
            method="chat.completions.create",
            request_body={"messages": [{"role": "user", "content": "thanks"}]},
            response_body={"choices": [{"message": {"content": "done"}}]},
        )
        assert recorder.interaction_count == 3
        cassette = recorder.finalize()
        assert len(cassette.interactions) == 3
        assert cassette.interactions[0].sequence == 0
        assert cassette.interactions[1].sequence == 1
        assert cassette.interactions[2].sequence == 2

    def test_finalize_assigns_ids_and_hash(self):
        recorder = CassetteRecorder(test_id="test::hash")
        recorder.record_llm_call(
            method="chat",
            request_body={"prompt": "hello"},
            response_body="world",
        )
        cassette = recorder.finalize()
        assert cassette.interactions[0].id != ""
        assert cassette.meta.content_hash != ""
        assert cassette.verify_integrity()

    def test_cannot_record_after_finalize(self):
        recorder = CassetteRecorder()
        recorder.finalize()
        with pytest.raises(RuntimeError, match="Cannot record after finalize"):
            recorder.record_llm_call(
                method="chat", request_body={}, response_body=""
            )
        with pytest.raises(RuntimeError, match="Cannot record after finalize"):
            recorder.record_tool_call(
                tool_name="x", arguments={}, result=""
            )

    def test_redaction_default_keys(self):
        recorder = CassetteRecorder()
        interaction = recorder.record_llm_call(
            method="chat",
            request_body={"api_key": "sk-secret", "prompt": "hello"},
            response_body="world",
        )
        assert interaction.request.body["api_key"] == "[REDACTED]"
        assert interaction.request.body["prompt"] == "hello"

    def test_redaction_custom_keys(self):
        recorder = CassetteRecorder(
            redact_keys=frozenset({"my_secret"})
        )
        interaction = recorder.record_llm_call(
            method="chat",
            request_body={"my_secret": "hidden", "api_key": "visible"},
            response_body="ok",
        )
        assert interaction.request.body["my_secret"] == "[REDACTED]"
        # api_key is NOT in custom keys, so it stays
        assert interaction.request.body["api_key"] == "visible"

    def test_record_error_status(self):
        recorder = CassetteRecorder()
        interaction = recorder.record_llm_call(
            method="chat",
            request_body={},
            response_body={"error": "rate limited"},
            status="error",
        )
        assert interaction.response.status == "error"

    def test_record_with_metadata(self):
        recorder = CassetteRecorder()
        interaction = recorder.record_tool_call(
            tool_name="calc",
            arguments={"expr": "2+2"},
            result=4,
            metadata={"retry_count": 1},
        )
        assert interaction.metadata["retry_count"] == 1

    def test_finalize_roundtrip(self, tmp_path):
        """Record, finalize, save, reload, and verify."""
        recorder = CassetteRecorder(test_id="test::roundtrip")
        recorder.record_llm_call(
            method="chat",
            request_body={"prompt": "hi"},
            response_body="hello",
            prompt_tokens=2,
            completion_tokens=1,
        )
        recorder.record_tool_call(
            tool_name="calc",
            arguments={"x": 1},
            result=42,
        )
        cassette = recorder.finalize()

        path = tmp_path / "test.json"
        cassette.save(path)

        from checkagent.replay.cassette import Cassette

        loaded = Cassette.load(path)
        assert loaded.verify_integrity()
        assert len(loaded.interactions) == 2
        assert loaded.meta.test_id == "test::roundtrip"


class TestTimedCall:
    def test_basic_timing(self):
        with TimedCall() as tc:
            total = sum(range(1000))
            _ = total
        assert tc.duration_ms >= 0

    def test_timing_is_positive_for_slow_ops(self):
        import time

        with TimedCall() as tc:
            time.sleep(0.01)
        assert tc.duration_ms >= 5  # at least 5ms
