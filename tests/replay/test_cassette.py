"""Tests for cassette data model (F2.1, F2.6, F2.8)."""

from __future__ import annotations

import json
import warnings
from pathlib import Path

from checkagent.replay.cassette import (
    CASSETTE_SCHEMA_VERSION,
    Cassette,
    CassetteMeta,
    Interaction,
    RecordedRequest,
    RecordedResponse,
    redact_dict,
)


def _make_interaction(
    kind: str = "llm",
    method: str = "chat.completions.create",
    body: dict | None = None,
    response_body: str = "Hello",
    seq: int = 0,
) -> Interaction:
    return Interaction(
        sequence=seq,
        request=RecordedRequest(
            kind=kind, method=method, body=body or {}
        ),
        response=RecordedResponse(body=response_body),
    )


class TestCassetteMeta:
    def test_defaults(self):
        meta = CassetteMeta()
        assert meta.schema_version == CASSETTE_SCHEMA_VERSION
        assert meta.content_hash == ""
        assert meta.recorded_at != ""

    def test_custom_values(self):
        meta = CassetteMeta(
            schema_version=1,
            checkagent_version="0.1.0",
            test_id="test_foo::test_bar",
        )
        assert meta.test_id == "test_foo::test_bar"
        assert meta.checkagent_version == "0.1.0"


class TestRecordedRequest:
    def test_llm_request(self):
        req = RecordedRequest(
            kind="llm",
            method="chat.completions.create",
            model="gpt-4o",
            body={"messages": [{"role": "user", "content": "hi"}]},
        )
        assert req.kind == "llm"
        assert req.model == "gpt-4o"
        assert "messages" in req.body

    def test_tool_request(self):
        req = RecordedRequest(
            kind="tool",
            method="search",
            body={"query": "weather"},
        )
        assert req.kind == "tool"
        assert req.model is None


class TestRecordedResponse:
    def test_ok_response(self):
        resp = RecordedResponse(
            body="Hello!",
            prompt_tokens=10,
            completion_tokens=5,
        )
        assert resp.status == "ok"
        assert resp.prompt_tokens == 10

    def test_error_response(self):
        resp = RecordedResponse(status="error", body="rate limited")
        assert resp.status == "error"


class TestInteraction:
    def test_compute_id_deterministic(self):
        i1 = _make_interaction(method="chat", body={"a": 1})
        i2 = _make_interaction(method="chat", body={"a": 1})
        assert i1.compute_id() == i2.compute_id()

    def test_compute_id_differs_on_method(self):
        i1 = _make_interaction(method="chat")
        i2 = _make_interaction(method="embed")
        assert i1.compute_id() != i2.compute_id()

    def test_compute_id_differs_on_body(self):
        i1 = _make_interaction(body={"x": 1})
        i2 = _make_interaction(body={"x": 2})
        assert i1.compute_id() != i2.compute_id()

    def test_compute_id_length(self):
        i = _make_interaction()
        assert len(i.compute_id()) == 16


class TestCassette:
    def test_empty_cassette(self):
        c = Cassette()
        assert c.interactions == []
        assert c.meta.schema_version == CASSETTE_SCHEMA_VERSION

    def test_finalize_assigns_ids_and_sequences(self):
        c = Cassette(interactions=[
            _make_interaction(method="a"),
            _make_interaction(method="b"),
        ])
        c.finalize()
        assert c.interactions[0].sequence == 0
        assert c.interactions[1].sequence == 1
        assert c.interactions[0].id != ""
        assert c.interactions[1].id != ""
        assert c.meta.content_hash != ""

    def test_content_hash_stable(self):
        c = Cassette(interactions=[_make_interaction()])
        h1 = c.compute_content_hash()
        h2 = c.compute_content_hash()
        assert h1 == h2

    def test_content_hash_changes_with_data(self):
        c1 = Cassette(interactions=[_make_interaction(body={"x": 1})])
        c2 = Cassette(interactions=[_make_interaction(body={"x": 2})])
        assert c1.compute_content_hash() != c2.compute_content_hash()

    def test_short_hash(self):
        c = Cassette(interactions=[_make_interaction()])
        assert len(c.short_hash()) == 12

    def test_verify_integrity_no_hash(self):
        c = Cassette()
        assert c.verify_integrity() is True

    def test_verify_integrity_valid(self):
        c = Cassette(interactions=[_make_interaction()])
        c.finalize()
        assert c.verify_integrity() is True

    def test_verify_integrity_tampered(self):
        c = Cassette(interactions=[_make_interaction()])
        c.finalize()
        c.interactions[0].response.body = "tampered"
        assert c.verify_integrity() is False


class TestCassetteSerialization:
    def test_to_json_roundtrip(self):
        c = Cassette(interactions=[
            _make_interaction(method="chat", body={"msg": "hi"}),
        ])
        c.finalize()
        json_str = c.to_json()
        loaded = Cassette.from_json(json_str)
        assert loaded.meta.content_hash == c.meta.content_hash
        assert len(loaded.interactions) == 1
        assert loaded.interactions[0].request.method == "chat"

    def test_to_json_is_valid_json(self):
        c = Cassette(interactions=[_make_interaction()])
        parsed = json.loads(c.to_json())
        assert "meta" in parsed
        assert "interactions" in parsed

    def test_from_json_empty(self):
        c = Cassette.from_json('{"meta": {}, "interactions": []}')
        assert c.interactions == []


class TestCassetteFileIO:
    def test_save_and_load(self, tmp_path: Path):
        c = Cassette(
            meta=CassetteMeta(test_id="test_example"),
            interactions=[_make_interaction()],
        )
        c.finalize()
        path = tmp_path / "test.json"
        c.save(path)
        assert path.exists()

        loaded = Cassette.load(path)
        assert loaded.meta.test_id == "test_example"
        assert loaded.verify_integrity()

    def test_save_creates_parent_dirs(self, tmp_path: Path):
        c = Cassette()
        path = tmp_path / "a" / "b" / "cassette.json"
        c.save(path)
        assert path.exists()

    def test_load_outdated_schema_warns(self, tmp_path: Path):
        c = Cassette()
        c.meta.schema_version = 0  # outdated
        path = tmp_path / "old.json"
        c.save(path)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            Cassette.load(path)
            assert len(w) == 1
            assert "migrate-cassettes" in str(w[0].message)

    def test_load_current_schema_no_warning(self, tmp_path: Path):
        c = Cassette()
        path = tmp_path / "current.json"
        c.save(path)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            Cassette.load(path)
            assert len(w) == 0


class TestCassettePath:
    def test_basic_path(self, tmp_path: Path):
        p = Cassette.cassette_path(tmp_path, "test_foo", "abcdef123456")
        assert p == tmp_path / "test_foo" / "abcdef123456.json"

    def test_namespaced_test_id(self, tmp_path: Path):
        p = Cassette.cassette_path(
            tmp_path, "tests/test_agent::test_run", "aabbcc112233"
        )
        assert p == tmp_path / "tests/test_agent/test_run" / "aabbcc112233.json"

    def test_short_hash_used(self, tmp_path: Path):
        p = Cassette.cassette_path(
            tmp_path,
            "t",
            "abcdef1234567890extra",
        )
        assert p.stem == "abcdef123456"


class TestRedaction:
    def test_redacts_api_key(self):
        d = {"api_key": "sk-secret", "model": "gpt-4o"}
        result = redact_dict(d)
        assert result["api_key"] == "[REDACTED]"
        assert result["model"] == "gpt-4o"

    def test_redacts_nested(self):
        d = {"headers": {"Authorization": "Bearer xyz"}}
        result = redact_dict(d)
        assert result["headers"]["Authorization"] == "[REDACTED]"

    def test_redacts_in_list(self):
        d = {"items": [{"token": "abc", "name": "foo"}]}
        result = redact_dict(d)
        assert result["items"][0]["token"] == "[REDACTED]"
        assert result["items"][0]["name"] == "foo"

    def test_case_insensitive(self):
        d = {"API_KEY": "secret", "Token": "abc"}
        result = redact_dict(d)
        assert result["API_KEY"] == "[REDACTED]"
        assert result["Token"] == "[REDACTED]"

    def test_custom_keys(self):
        d = {"my_secret": "val", "name": "ok"}
        result = redact_dict(d, keys=frozenset({"my_secret"}))
        assert result["my_secret"] == "[REDACTED]"
        assert result["name"] == "ok"

    def test_preserves_original(self):
        d = {"api_key": "sk-123"}
        redact_dict(d)
        assert d["api_key"] == "sk-123"  # original unchanged

    def test_empty_dict(self):
        assert redact_dict({}) == {}

    def test_no_matching_keys(self):
        d = {"name": "test", "value": 42}
        assert redact_dict(d) == d
