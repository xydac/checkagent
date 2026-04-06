"""Tests for the import-trace CLI command."""

from __future__ import annotations

import json
from pathlib import Path

from click.testing import CliRunner

from checkagent.cli import main


@staticmethod
def _write_traces(tmp_path: Path, traces: list[dict], name: str = "traces.json") -> str:
    p = tmp_path / name
    p.write_text(json.dumps(traces), encoding="utf-8")
    return str(p)


class TestImportTraceCli:
    def test_basic_import(self, tmp_path):
        traces = [
            {
                "input": {"query": "Hello agent"},
                "steps": [],
                "final_output": "Hi there",
            }
        ]
        path = _write_traces(tmp_path, traces)
        runner = CliRunner()
        result = runner.invoke(
            main,
            ["import-trace", path, "-o", str(tmp_path / "out.json")],
        )
        assert result.exit_code == 0
        assert "1 traces" in result.output

        out = json.loads((tmp_path / "out.json").read_text())
        assert len(out["cases"]) == 1

    def test_filter_status(self, tmp_path):
        traces = [
            {"input": {"query": "OK"}, "steps": [], "final_output": "Good"},
            {"input": {"query": "Bad"}, "steps": [], "error": "fail"},
        ]
        path = _write_traces(tmp_path, traces)
        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                "import-trace",
                path,
                "--filter-status",
                "error",
                "-o",
                str(tmp_path / "out.json"),
            ],
        )
        assert result.exit_code == 0
        out = json.loads((tmp_path / "out.json").read_text())
        assert len(out["cases"]) == 1

    def test_limit(self, tmp_path):
        traces = [
            {"input": {"query": f"Q{i}"}, "steps": []} for i in range(10)
        ]
        path = _write_traces(tmp_path, traces)
        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                "import-trace",
                path,
                "--limit",
                "3",
                "-o",
                str(tmp_path / "out.json"),
            ],
        )
        assert result.exit_code == 0
        out = json.loads((tmp_path / "out.json").read_text())
        assert len(out["cases"]) == 3

    def test_no_pii_scrub_flag(self, tmp_path):
        traces = [
            {
                "input": {"query": "Email john@example.com"},
                "steps": [],
            }
        ]
        path = _write_traces(tmp_path, traces)
        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                "import-trace",
                path,
                "--no-pii-scrub",
                "-o",
                str(tmp_path / "out.json"),
            ],
        )
        assert result.exit_code == 0
        out = json.loads((tmp_path / "out.json").read_text())
        assert "john@example.com" in out["cases"][0]["input"]

    def test_custom_dataset_name(self, tmp_path):
        traces = [{"input": {"query": "Test"}, "steps": []}]
        path = _write_traces(tmp_path, traces)
        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                "import-trace",
                path,
                "--dataset-name",
                "my-dataset",
                "-o",
                str(tmp_path / "out.json"),
            ],
        )
        assert result.exit_code == 0
        out = json.loads((tmp_path / "out.json").read_text())
        assert out["name"] == "my-dataset"

    def test_custom_tags(self, tmp_path):
        traces = [{"input": {"query": "Test"}, "steps": []}]
        path = _write_traces(tmp_path, traces)
        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                "import-trace",
                path,
                "--tag",
                "regression",
                "--tag",
                "prod",
                "-o",
                str(tmp_path / "out.json"),
            ],
        )
        assert result.exit_code == 0
        out = json.loads((tmp_path / "out.json").read_text())
        tags = out["cases"][0]["tags"]
        assert "regression" in tags
        assert "prod" in tags

    def test_otel_auto_detect(self, tmp_path):
        data = {
            "resourceSpans": [
                {
                    "scopeSpans": [
                        {
                            "spans": [
                                {
                                    "traceId": "abc",
                                    "spanId": "s1",
                                    "name": "agent",
                                    "parentSpanId": "",
                                    "startTimeUnixNano": "1000000",
                                    "endTimeUnixNano": "2000000",
                                    "attributes": [
                                        {
                                            "key": "input",
                                            "value": {
                                                "stringValue": "OTel query"
                                            },
                                        }
                                    ],
                                    "status": {},
                                }
                            ]
                        }
                    ]
                }
            ]
        }
        p = tmp_path / "otel.json"
        p.write_text(json.dumps(data), encoding="utf-8")
        runner = CliRunner()
        result = runner.invoke(
            main,
            ["import-trace", str(p), "-o", str(tmp_path / "out.json")],
        )
        assert result.exit_code == 0
        assert "otel format" in result.output

    def test_empty_traces(self, tmp_path):
        path = _write_traces(tmp_path, [])
        runner = CliRunner()
        result = runner.invoke(
            main,
            ["import-trace", path, "-o", str(tmp_path / "out.json")],
        )
        assert result.exit_code == 0
        assert "No traces found" in result.output

    def test_default_output_path(self, tmp_path, monkeypatch):
        traces = [{"input": {"query": "Test"}, "steps": []}]
        path = _write_traces(tmp_path, traces)
        # Run from tmp_path so datasets/imported/ is created there
        monkeypatch.chdir(tmp_path)
        runner = CliRunner()
        result = runner.invoke(main, ["import-trace", path])
        assert result.exit_code == 0
        assert (tmp_path / "datasets" / "imported" / "traces.json").exists()
