"""Tests for the Langfuse API trace importer."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from checkagent.trace_import.langfuse_importer import LangfuseAPIImporter


def _make_trace(
    trace_id: str = "t1",
    name: str = "my-chain",
    inp: object = "What is the capital?",
    out: object = "Paris",
    observations: list | None = None,
    latency: float = 1.5,
) -> dict:
    return {
        "id": trace_id,
        "name": name,
        "input": inp,
        "output": out,
        "latency": latency,
        "observations": observations or [],
    }


def _mock_response(traces: list[dict], page: int = 1, total_pages: int = 1):
    """Build a mock urlopen response for a page of traces."""
    body = json.dumps(
        {
            "data": traces,
            "meta": {
                "page": page,
                "limit": 50,
                "totalItems": len(traces),
                "totalPages": total_pages,
            },
        }
    ).encode()
    mock = MagicMock()
    mock.__enter__ = lambda s: s
    mock.__exit__ = MagicMock(return_value=False)
    mock.read.return_value = body
    return mock


class TestLangfuseImporterNormalization:
    def test_string_input(self):
        importer = LangfuseAPIImporter(public_key="pk", secret_key="sk")
        trace = _make_trace(inp="Hello world", out="Hi there")
        with patch("urllib.request.urlopen", return_value=_mock_response([trace])):
            runs = importer.import_traces()
        assert len(runs) == 1
        assert runs[0].input.query == "Hello world"
        assert runs[0].final_output == "Hi there"

    def test_dict_input_query_key(self):
        importer = LangfuseAPIImporter(public_key="pk", secret_key="sk")
        trace = _make_trace(inp={"query": "What is 2+2?"}, out="4")
        with patch("urllib.request.urlopen", return_value=_mock_response([trace])):
            runs = importer.import_traces()
        assert runs[0].input.query == "What is 2+2?"

    def test_dict_input_messages(self):
        importer = LangfuseAPIImporter(public_key="pk", secret_key="sk")
        trace = _make_trace(
            inp={"messages": [{"role": "user", "content": "Explain recursion"}]},
            out="Recursion is...",
        )
        with patch("urllib.request.urlopen", return_value=_mock_response([trace])):
            runs = importer.import_traces()
        assert runs[0].input.query == "Explain recursion"

    def test_duration_ms(self):
        importer = LangfuseAPIImporter(public_key="pk", secret_key="sk")
        trace = _make_trace(latency=2.0)
        with patch("urllib.request.urlopen", return_value=_mock_response([trace])):
            runs = importer.import_traces()
        assert runs[0].duration_ms == pytest.approx(2000.0)

    def test_metadata_includes_source(self):
        importer = LangfuseAPIImporter(public_key="pk", secret_key="sk")
        trace = _make_trace(trace_id="abc123")
        with patch("urllib.request.urlopen", return_value=_mock_response([trace])):
            runs = importer.import_traces()
        assert runs[0].metadata["source"] == "langfuse"
        assert runs[0].metadata["trace_id"] == "abc123"

    def test_observations_become_steps(self):
        obs = [
            {
                "id": "o1",
                "type": "GENERATION",
                "name": "openai-chat",
                "input": {"messages": [{"role": "user", "content": "hi"}]},
                "output": "hello",
                "model": "gpt-4",
                "usage": {"input": 10, "output": 5},
                "latency": 0.8,
            }
        ]
        importer = LangfuseAPIImporter(public_key="pk", secret_key="sk")
        trace = _make_trace(observations=obs)
        with patch("urllib.request.urlopen", return_value=_mock_response([trace])):
            runs = importer.import_traces()
        assert len(runs[0].steps) == 1
        step = runs[0].steps[0]
        assert step.model == "gpt-4"
        assert step.prompt_tokens == 10
        assert step.completion_tokens == 5

    def test_span_observation_becomes_tool_call(self):
        obs = [
            {
                "id": "o2",
                "type": "SPAN",
                "name": "database-lookup",
                "input": {"table": "users", "id": 42},
                "output": {"name": "Alice"},
                "latency": 0.2,
            }
        ]
        importer = LangfuseAPIImporter(public_key="pk", secret_key="sk")
        trace = _make_trace(observations=obs)
        with patch("urllib.request.urlopen", return_value=_mock_response([trace])):
            runs = importer.import_traces()
        assert len(runs[0].steps[0].tool_calls) == 1
        assert runs[0].steps[0].tool_calls[0].name == "database-lookup"


class TestLangfuseImporterFilters:
    def test_filter_success(self):
        traces = [
            _make_trace(trace_id="t1", out="ok"),
            _make_trace(trace_id="t2", out=None),
        ]
        importer = LangfuseAPIImporter(public_key="pk", secret_key="sk")
        with patch("urllib.request.urlopen", return_value=_mock_response(traces)):
            runs = importer.import_traces(filters={"status": "success"})
        assert len(runs) == 1
        assert runs[0].metadata["trace_id"] == "t1"

    def test_limit(self):
        traces = [_make_trace(trace_id=f"t{i}") for i in range(5)]
        importer = LangfuseAPIImporter(public_key="pk", secret_key="sk")
        with patch("urllib.request.urlopen", return_value=_mock_response(traces)):
            runs = importer.import_traces(limit=3)
        assert len(runs) == 3


class TestLangfuseImporterCredentials:
    def test_missing_credentials_raises(self):
        importer = LangfuseAPIImporter(public_key="", secret_key="")
        with pytest.raises(RuntimeError, match="credentials required"):
            importer.import_traces()

    def test_reads_env_vars(self, monkeypatch):
        monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-test")
        monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-test")
        importer = LangfuseAPIImporter()
        assert importer._public_key == "pk-test"
        assert importer._secret_key == "sk-test"


class TestLangfuseImporterCLI:
    def test_import_trace_langfuse_source(self, tmp_path):
        from click.testing import CliRunner

        from checkagent.cli.import_trace import import_trace_cmd

        traces = [_make_trace(inp="Hello", out="World")]
        runner = CliRunner()
        with patch("urllib.request.urlopen", return_value=_mock_response(traces)):
            result = runner.invoke(
                import_trace_cmd,
                ["--source", "langfuse", "--api-key", "pk:sk", "-o", str(tmp_path / "out.json")],
            )
        assert result.exit_code == 0, result.output
        assert "Found 1 traces" in result.output

    def test_import_trace_no_file_no_source_errors(self):
        from click.testing import CliRunner

        from checkagent.cli.import_trace import import_trace_cmd

        runner = CliRunner()
        result = runner.invoke(import_trace_cmd, [])
        assert result.exit_code != 0
