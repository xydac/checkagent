"""Tests for the Arize Phoenix API trace importer."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from checkagent.trace_import.phoenix_importer import PhoenixAPIImporter


def _make_span(
    span_id: str = "s1",
    trace_id: str = "tr1",
    name: str = "chain.invoke",
    span_kind: str = "CHAIN",
    inp: object = None,
    out: object = None,
    parent_id: str | None = None,
    status_code: str = "OK",
    start: str = "2026-01-01T00:00:00.000Z",
    end: str = "2026-01-01T00:00:02.000Z",
    attrs: dict | None = None,
) -> dict:
    return {
        "id": span_id,
        "context": {"trace_id": trace_id, "span_id": span_id},
        "name": name,
        "spanKind": span_kind,
        "parentId": parent_id,
        "startTime": start,
        "endTime": end,
        "statusCode": status_code,
        "input": inp or {"value": "What is 2+2?"},
        "output": out or {"value": "4"},
        "attributes": attrs or {},
    }


def _mock_response(spans: list[dict], next_cursor: str | None = None):
    body = json.dumps({"data": spans, "next": next_cursor}).encode()
    mock = MagicMock()
    mock.__enter__ = lambda s: s
    mock.__exit__ = MagicMock(return_value=False)
    mock.read.return_value = body
    return mock


class TestPhoenixImporterNormalization:
    def test_single_root_span(self):
        span = _make_span(inp={"value": "Hello"}, out={"value": "Hi"})
        importer = PhoenixAPIImporter()
        with patch("urllib.request.urlopen", return_value=_mock_response([span])):
            runs = importer.import_traces()
        assert len(runs) == 1
        assert runs[0].input.query == "Hello"
        assert runs[0].final_output == "Hi"

    def test_duration_ms_from_timestamps(self):
        span = _make_span(
            start="2026-01-01T00:00:00.000Z",
            end="2026-01-01T00:00:01.500Z",
        )
        importer = PhoenixAPIImporter()
        with patch("urllib.request.urlopen", return_value=_mock_response([span])):
            runs = importer.import_traces()
        assert runs[0].duration_ms == pytest.approx(1500.0, abs=1.0)

    def test_metadata_includes_source(self):
        span = _make_span(trace_id="phoenix-trace-1")
        importer = PhoenixAPIImporter()
        with patch("urllib.request.urlopen", return_value=_mock_response([span])):
            runs = importer.import_traces()
        assert runs[0].metadata["source"] == "phoenix"
        assert runs[0].metadata["trace_id"] == "phoenix-trace-1"

    def test_child_spans_become_steps(self):
        root = _make_span(span_id="root", trace_id="tr1", name="chain")
        child = _make_span(
            span_id="child",
            trace_id="tr1",
            name="llm.call",
            parent_id="root",
            attrs={"llm.model_name": "gpt-4", "llm.token_count.prompt": 50},
        )
        importer = PhoenixAPIImporter()
        with patch("urllib.request.urlopen", return_value=_mock_response([root, child])):
            runs = importer.import_traces()
        assert len(runs) == 1
        assert len(runs[0].steps) == 1
        assert runs[0].steps[0].model == "gpt-4"
        assert runs[0].steps[0].prompt_tokens == 50

    def test_tool_span_becomes_tool_call(self):
        root = _make_span(span_id="root", trace_id="tr1")
        tool_span = _make_span(
            span_id="tool1",
            trace_id="tr1",
            name="search_database",
            span_kind="TOOL",
            inp={"value": {"query": "Alice"}},
            out={"value": "Found: Alice Smith"},
            parent_id="root",
        )
        importer = PhoenixAPIImporter()
        with patch("urllib.request.urlopen", return_value=_mock_response([root, tool_span])):
            runs = importer.import_traces()
        assert len(runs[0].steps[0].tool_calls) == 1
        tc = runs[0].steps[0].tool_calls[0]
        assert tc.name == "search_database"
        assert tc.result == "Found: Alice Smith"

    def test_error_span(self):
        span = _make_span(status_code="ERROR")
        span["statusMessage"] = "Timeout after 30s"
        importer = PhoenixAPIImporter()
        with patch("urllib.request.urlopen", return_value=_mock_response([span])):
            runs = importer.import_traces()
        assert runs[0].error == "Timeout after 30s"

    def test_traces_grouped_by_trace_id(self):
        spans = [
            _make_span(span_id="s1", trace_id="tr1", name="chain-a"),
            _make_span(span_id="s2", trace_id="tr2", name="chain-b"),
        ]
        importer = PhoenixAPIImporter()
        with patch("urllib.request.urlopen", return_value=_mock_response(spans)):
            runs = importer.import_traces()
        assert len(runs) == 2


class TestPhoenixImporterFilters:
    def test_filter_error(self):
        spans = [
            _make_span(span_id="ok", trace_id="tr1", status_code="OK"),
            _make_span(span_id="err", trace_id="tr2", status_code="ERROR"),
        ]
        importer = PhoenixAPIImporter()
        with patch("urllib.request.urlopen", return_value=_mock_response(spans)):
            runs = importer.import_traces(filters={"status": "error"})
        assert len(runs) == 1
        assert runs[0].error is not None

    def test_filter_success(self):
        spans = [
            _make_span(span_id="ok", trace_id="tr1", status_code="OK"),
            _make_span(span_id="err", trace_id="tr2", status_code="ERROR"),
        ]
        spans[1]["statusMessage"] = "Error"
        importer = PhoenixAPIImporter()
        with patch("urllib.request.urlopen", return_value=_mock_response(spans)):
            runs = importer.import_traces(filters={"status": "success"})
        assert len(runs) == 1
        assert runs[0].error is None


class TestPhoenixImporterCLI:
    def test_import_trace_phoenix_source(self, tmp_path):
        from click.testing import CliRunner

        from checkagent.cli.import_trace import import_trace_cmd

        spans = [_make_span()]
        runner = CliRunner()
        with patch("urllib.request.urlopen", return_value=_mock_response(spans)):
            result = runner.invoke(
                import_trace_cmd,
                [
                    "--source",
                    "phoenix",
                    "--api-url",
                    "http://localhost:6006",
                    "-o",
                    str(tmp_path / "out.json"),
                ],
            )
        assert result.exit_code == 0, result.output
        assert "Found 1 traces" in result.output

    def test_api_url_used_for_host(self, tmp_path):
        from click.testing import CliRunner

        from checkagent.cli.import_trace import import_trace_cmd

        spans = [_make_span()]
        runner = CliRunner()
        with patch("urllib.request.urlopen", return_value=_mock_response(spans)) as mock_open:
            runner.invoke(
                import_trace_cmd,
                [
                    "--source",
                    "phoenix",
                    "--api-url",
                    "http://my-phoenix:9999",
                    "-o",
                    str(tmp_path / "out.json"),
                ],
            )
        called_url = mock_open.call_args[0][0].full_url
        assert "my-phoenix:9999" in called_url
