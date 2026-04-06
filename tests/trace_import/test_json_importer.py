"""Tests for JSON/JSONL file importer."""

from __future__ import annotations

import json

import pytest

from checkagent.trace_import.json_importer import JsonFileImporter


@pytest.fixture
def importer():
    return JsonFileImporter()


@pytest.fixture
def tmp_json(tmp_path):
    """Helper to write a JSON file and return its path."""

    def _write(data, suffix=".json"):
        p = tmp_path / f"traces{suffix}"
        p.write_text(json.dumps(data), encoding="utf-8")
        return str(p)

    return _write


@pytest.fixture
def tmp_jsonl(tmp_path):
    """Helper to write a JSONL file and return its path."""

    def _write(lines):
        p = tmp_path / "traces.jsonl"
        text = "\n".join(json.dumps(line) for line in lines)
        p.write_text(text, encoding="utf-8")
        return str(p)

    return _write


class TestJsonFileImporter:
    def test_native_format(self, importer, tmp_json):
        data = [
            {
                "input": {"query": "Hello agent"},
                "steps": [
                    {
                        "step_index": 0,
                        "output_text": "Hi there",
                        "tool_calls": [
                            {"name": "greet", "arguments": {"name": "user"}}
                        ],
                    }
                ],
                "final_output": "Hi there!",
            }
        ]
        runs = importer.import_traces(tmp_json(data))
        assert len(runs) == 1
        assert runs[0].input.query == "Hello agent"
        assert runs[0].final_output == "Hi there!"
        assert len(runs[0].steps) == 1
        assert runs[0].steps[0].tool_calls[0].name == "greet"

    def test_single_object(self, importer, tmp_json):
        data = {
            "input": {"query": "Single trace"},
            "steps": [],
            "final_output": "Done",
        }
        runs = importer.import_traces(tmp_json(data))
        assert len(runs) == 1
        assert runs[0].input.query == "Single trace"

    def test_string_input(self, importer, tmp_json):
        data = [
            {
                "input": "Just a string query",
                "steps": [],
                "final_output": "Response",
            }
        ]
        runs = importer.import_traces(tmp_json(data))
        assert runs[0].input.query == "Just a string query"

    def test_jsonl_format(self, importer, tmp_jsonl):
        lines = [
            {"input": {"query": "Q1"}, "steps": [], "final_output": "A1"},
            {"input": {"query": "Q2"}, "steps": [], "final_output": "A2"},
        ]
        runs = importer.import_traces(tmp_jsonl(lines))
        assert len(runs) == 2
        assert runs[0].input.query == "Q1"
        assert runs[1].input.query == "Q2"

    def test_span_based_format(self, importer, tmp_json):
        data = [
            {
                "input": "Span-based query",
                "spans": [
                    {
                        "name": "llm_call",
                        "output": "Generated text",
                        "model": "gpt-4",
                    },
                    {
                        "name": "tool_call",
                        "tool_calls": [
                            {"name": "search", "arguments": {"q": "test"}}
                        ],
                    },
                ],
                "output": "Final answer",
            }
        ]
        runs = importer.import_traces(tmp_json(data))
        assert len(runs) == 1
        assert len(runs[0].steps) == 2
        assert runs[0].final_output == "Final answer"

    def test_flat_format(self, importer, tmp_json):
        data = [{"query": "Simple prompt", "response": "Simple response"}]
        runs = importer.import_traces(tmp_json(data))
        assert len(runs) == 1
        assert runs[0].input.query == "Simple prompt"
        assert runs[0].final_output == "Simple response"

    def test_filter_by_status_error(self, importer, tmp_json):
        data = [
            {"input": {"query": "OK"}, "steps": [], "final_output": "Good"},
            {
                "input": {"query": "Bad"},
                "steps": [],
                "error": "Something failed",
            },
        ]
        runs = importer.import_traces(
            tmp_json(data), filters={"status": "error"}
        )
        assert len(runs) == 1
        assert runs[0].input.query == "Bad"

    def test_filter_by_status_success(self, importer, tmp_json):
        data = [
            {"input": {"query": "OK"}, "steps": [], "final_output": "Good"},
            {
                "input": {"query": "Bad"},
                "steps": [],
                "error": "Something failed",
            },
        ]
        runs = importer.import_traces(
            tmp_json(data), filters={"status": "success"}
        )
        assert len(runs) == 1
        assert runs[0].input.query == "OK"

    def test_filter_by_tags(self, importer, tmp_json):
        data = [
            {
                "input": {"query": "Tagged"},
                "steps": [],
                "tags": ["production", "critical"],
            },
            {"input": {"query": "Untagged"}, "steps": []},
        ]
        runs = importer.import_traces(
            tmp_json(data), filters={"tags": ["critical"]}
        )
        assert len(runs) == 1
        assert runs[0].input.query == "Tagged"

    def test_limit(self, importer, tmp_json):
        data = [
            {"input": {"query": f"Q{i}"}, "steps": []} for i in range(10)
        ]
        runs = importer.import_traces(tmp_json(data), limit=3)
        assert len(runs) == 3

    def test_file_not_found(self, importer):
        with pytest.raises(FileNotFoundError):
            importer.import_traces("/nonexistent/file.json")

    def test_token_metadata(self, importer, tmp_json):
        data = [
            {
                "input": {"query": "Token test"},
                "steps": [
                    {
                        "step_index": 0,
                        "prompt_tokens": 100,
                        "completion_tokens": 50,
                        "model": "gpt-4",
                    }
                ],
                "total_prompt_tokens": 100,
                "total_completion_tokens": 50,
                "duration_ms": 1234.5,
            }
        ]
        runs = importer.import_traces(tmp_json(data))
        assert runs[0].total_prompt_tokens == 100
        assert runs[0].total_completion_tokens == 50
        assert runs[0].duration_ms == 1234.5
        assert runs[0].steps[0].model == "gpt-4"

    def test_tool_call_with_result_and_error(self, importer, tmp_json):
        data = [
            {
                "input": {"query": "Tool test"},
                "steps": [
                    {
                        "tool_calls": [
                            {
                                "name": "search",
                                "arguments": {"q": "test"},
                                "result": {"items": []},
                            },
                            {
                                "name": "broken",
                                "error": "timeout",
                                "duration_ms": 5000,
                            },
                        ]
                    }
                ],
            }
        ]
        runs = importer.import_traces(tmp_json(data))
        tcs = runs[0].steps[0].tool_calls
        assert tcs[0].result == {"items": []}
        assert tcs[0].succeeded is True
        assert tcs[1].error == "timeout"
        assert tcs[1].succeeded is False

    def test_empty_jsonl_lines_skipped(self, importer, tmp_path):
        p = tmp_path / "traces.jsonl"
        lines = [
            '{"input": {"query": "A"}, "steps": []}',
            "",
            '{"input": {"query": "B"}, "steps": []}',
            "   ",
        ]
        p.write_text("\n".join(lines), encoding="utf-8")
        runs = importer.import_traces(str(p))
        assert len(runs) == 2
