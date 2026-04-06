"""Tests for OpenTelemetry OTLP JSON importer."""

from __future__ import annotations

import json

import pytest

from checkagent.trace_import.otel_importer import OtelJsonImporter


@pytest.fixture
def importer():
    return OtelJsonImporter()


@pytest.fixture
def tmp_otel(tmp_path):
    """Helper to write an OTLP JSON file."""

    def _write(data):
        p = tmp_path / "otel.json"
        p.write_text(json.dumps(data), encoding="utf-8")
        return str(p)

    return _write


def _make_otlp(spans):
    """Build a minimal OTLP JSON structure."""
    return {
        "resourceSpans": [
            {
                "scopeSpans": [
                    {"spans": spans}
                ]
            }
        ]
    }


def _make_span(
    trace_id="abc123",
    span_id="span1",
    name="agent_run",
    parent_span_id="",
    attributes=None,
    status=None,
    start_ns=1000000000,
    end_ns=2000000000,
):
    span = {
        "traceId": trace_id,
        "spanId": span_id,
        "name": name,
        "parentSpanId": parent_span_id,
        "startTimeUnixNano": str(start_ns),
        "endTimeUnixNano": str(end_ns),
        "attributes": attributes or [],
        "status": status or {},
    }
    return span


class TestOtelJsonImporter:
    def test_single_trace_root_only(self, importer, tmp_otel):
        spans = [
            _make_span(
                attributes=[
                    {"key": "input", "value": {"stringValue": "Hello"}},
                    {"key": "output", "value": {"stringValue": "World"}},
                ]
            )
        ]
        runs = importer.import_traces(tmp_otel(_make_otlp(spans)))
        assert len(runs) == 1
        assert runs[0].input.query == "Hello"
        assert runs[0].final_output == "World"
        assert runs[0].duration_ms == 1000.0  # 1e9 ns diff

    def test_root_with_child_steps(self, importer, tmp_otel):
        spans = [
            _make_span(
                span_id="root",
                attributes=[
                    {"key": "input", "value": {"stringValue": "Query"}},
                ],
            ),
            _make_span(
                span_id="child1",
                parent_span_id="root",
                name="llm_call",
                attributes=[
                    {"key": "output", "value": {"stringValue": "Step output"}},
                    {"key": "llm.model", "value": {"stringValue": "gpt-4"}},
                ],
            ),
        ]
        runs = importer.import_traces(tmp_otel(_make_otlp(spans)))
        assert len(runs) == 1
        assert len(runs[0].steps) == 1
        assert runs[0].steps[0].model == "gpt-4"

    def test_tool_span_detected(self, importer, tmp_otel):
        spans = [
            _make_span(span_id="root"),
            _make_span(
                span_id="child",
                parent_span_id="root",
                name="tool_search",
                attributes=[
                    {
                        "key": "tool.arguments",
                        "value": {"stringValue": '{"q": "test"}'},
                    },
                    {
                        "key": "tool.result",
                        "value": {"stringValue": "found 3 results"},
                    },
                ],
            ),
        ]
        runs = importer.import_traces(tmp_otel(_make_otlp(spans)))
        assert len(runs[0].steps[0].tool_calls) == 1
        tc = runs[0].steps[0].tool_calls[0]
        assert tc.name == "tool_search"
        assert tc.arguments == {"q": "test"}
        assert tc.result == "found 3 results"

    def test_function_span_detected(self, importer, tmp_otel):
        spans = [
            _make_span(span_id="root"),
            _make_span(
                span_id="child",
                parent_span_id="root",
                name="function_calculator",
            ),
        ]
        runs = importer.import_traces(tmp_otel(_make_otlp(spans)))
        assert len(runs[0].steps[0].tool_calls) == 1
        assert runs[0].steps[0].tool_calls[0].name == "function_calculator"

    def test_error_status(self, importer, tmp_otel):
        spans = [
            _make_span(
                status={"code": 2, "message": "Agent crashed"},
            )
        ]
        runs = importer.import_traces(tmp_otel(_make_otlp(spans)))
        assert runs[0].error == "Agent crashed"

    def test_filter_error_only(self, importer, tmp_otel):
        spans = [
            _make_span(
                trace_id="t1",
                attributes=[
                    {"key": "input", "value": {"stringValue": "OK"}},
                ],
            ),
            _make_span(
                trace_id="t2",
                attributes=[
                    {"key": "input", "value": {"stringValue": "Bad"}},
                ],
                status={"code": 2, "message": "fail"},
            ),
        ]
        runs = importer.import_traces(
            tmp_otel(_make_otlp(spans)), filters={"status": "error"}
        )
        assert len(runs) == 1
        assert runs[0].error == "fail"

    def test_limit(self, importer, tmp_otel):
        spans = [
            _make_span(trace_id=f"t{i}") for i in range(5)
        ]
        runs = importer.import_traces(tmp_otel(_make_otlp(spans)), limit=2)
        assert len(runs) == 2

    def test_flat_spans_format(self, importer, tmp_otel):
        """Handles flat {spans: [...]} format."""
        data = {
            "spans": [
                _make_span(
                    attributes=[
                        {"key": "input", "value": {"stringValue": "flat"}},
                    ]
                )
            ]
        }
        runs = importer.import_traces(tmp_otel(data))
        assert len(runs) == 1

    def test_snake_case_keys(self, importer, tmp_otel):
        """Handles snake_case OTLP keys (resource_spans, scope_spans)."""
        data = {
            "resource_spans": [
                {
                    "scope_spans": [
                        {
                            "spans": [
                                _make_span(
                                    attributes=[
                                        {
                                            "key": "input",
                                            "value": {
                                                "stringValue": "snake"
                                            },
                                        },
                                    ]
                                )
                            ]
                        }
                    ]
                }
            ]
        }
        runs = importer.import_traces(tmp_otel(data))
        assert len(runs) == 1

    def test_dict_attributes(self, importer, tmp_otel):
        """Handles already-flattened dict attributes."""
        spans = [
            _make_span(attributes={"input": "dict attrs", "model": "gpt-4"})
        ]
        runs = importer.import_traces(tmp_otel(_make_otlp(spans)))
        assert runs[0].input.query == "dict attrs"

    def test_file_not_found(self, importer):
        with pytest.raises(FileNotFoundError):
            importer.import_traces("/nonexistent/otel.json")

    def test_metadata_includes_trace_id(self, importer, tmp_otel):
        spans = [_make_span(trace_id="my-trace-123")]
        runs = importer.import_traces(tmp_otel(_make_otlp(spans)))
        assert runs[0].metadata["trace_id"] == "my-trace-123"
        assert runs[0].metadata["source"] == "otel"

    def test_token_attributes(self, importer, tmp_otel):
        spans = [
            _make_span(span_id="root"),
            _make_span(
                span_id="child",
                parent_span_id="root",
                name="llm_call",
                attributes=[
                    {
                        "key": "llm.prompt_tokens",
                        "value": {"intValue": 100},
                    },
                    {
                        "key": "llm.completion_tokens",
                        "value": {"intValue": 50},
                    },
                ],
            ),
        ]
        runs = importer.import_traces(tmp_otel(_make_otlp(spans)))
        assert runs[0].steps[0].prompt_tokens == 100
        assert runs[0].steps[0].completion_tokens == 50
