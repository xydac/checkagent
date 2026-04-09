"""Tests for the ``checkagent scan`` CLI command."""

from __future__ import annotations

import textwrap
from pathlib import Path

from click.testing import CliRunner

from checkagent.cli import main
from checkagent.cli.scan import _evaluate_output, _resolve_callable, scan_cmd

# ---------------------------------------------------------------------------
# Helper: write a temp module with agent callables
# ---------------------------------------------------------------------------


def _write_agent_module(tmp_path: Path) -> Path:
    """Write a temp module with safe, unsafe, and sync agent callables."""
    mod = tmp_path / "scan_test_agents.py"
    mod.write_text(textwrap.dedent("""\
        async def safe_agent(query):
            return "I can help you with that."

        async def unsafe_agent(query):
            if "system prompt" in query.lower():
                return "My system prompt is: You are a helpful assistant."
            return "I can help you with that."

        def sync_agent(query):
            return "Sync response: " + query[:20]

        async def error_agent(query):
            raise ValueError("Agent crashed")

        async def dict_agent(query):
            return {"output": "I can help.", "status": "ok"}

        not_callable = 42
    """))
    return mod


# ---------------------------------------------------------------------------
# Unit tests: _resolve_callable
# ---------------------------------------------------------------------------


class TestResolveCallable:
    def test_colon_syntax(self, tmp_path: Path, monkeypatch) -> None:
        _write_agent_module(tmp_path)
        monkeypatch.syspath_prepend(str(tmp_path))
        fn = _resolve_callable("scan_test_agents:safe_agent")
        assert callable(fn)

    def test_dot_syntax(self, tmp_path: Path, monkeypatch) -> None:
        _write_agent_module(tmp_path)
        monkeypatch.syspath_prepend(str(tmp_path))
        fn = _resolve_callable("scan_test_agents.safe_agent")
        assert callable(fn)

    def test_missing_module(self) -> None:
        import click
        import pytest

        with pytest.raises(click.exceptions.BadParameter, match="Cannot import"):
            _resolve_callable("nonexistent_module_xyz:fn")

    def test_missing_attr(self, tmp_path: Path, monkeypatch) -> None:
        _write_agent_module(tmp_path)
        monkeypatch.syspath_prepend(str(tmp_path))
        import click
        import pytest

        with pytest.raises(click.exceptions.BadParameter, match="no attribute"):
            _resolve_callable("scan_test_agents:nonexistent_fn")

    def test_not_callable(self, tmp_path: Path, monkeypatch) -> None:
        _write_agent_module(tmp_path)
        monkeypatch.syspath_prepend(str(tmp_path))
        import click
        import pytest

        with pytest.raises(click.exceptions.BadParameter, match="not callable"):
            _resolve_callable("scan_test_agents:not_callable")

    def test_no_separator(self) -> None:
        import click
        import pytest

        with pytest.raises(click.exceptions.BadParameter, match="Cannot parse"):
            _resolve_callable("just_a_name")


# ---------------------------------------------------------------------------
# Unit tests: _evaluate_output
# ---------------------------------------------------------------------------


class TestEvaluateOutput:
    def test_clean_output(self) -> None:
        findings = _evaluate_output("I can help you with that.")
        assert findings == []

    def test_injection_detected(self) -> None:
        findings = _evaluate_output("My system prompt is: You are a helpful assistant.")
        assert len(findings) > 0
        categories = {f.category.value for f in findings}
        assert "prompt_injection" in categories or "pii_leakage" in categories

    def test_pii_detected(self) -> None:
        findings = _evaluate_output("The user's email is alice@example.com")
        assert len(findings) > 0
        assert any("email" in f.description.lower() for f in findings)


# ---------------------------------------------------------------------------
# Integration tests: scan_cmd via CliRunner
# ---------------------------------------------------------------------------


class TestScanCommand:
    def test_safe_agent_passes(self, tmp_path: Path, monkeypatch) -> None:
        _write_agent_module(tmp_path)
        monkeypatch.syspath_prepend(str(tmp_path))
        runner = CliRunner()
        result = runner.invoke(scan_cmd, [
            "scan_test_agents:safe_agent",
            "--category", "injection",
            "--timeout", "2",
        ])
        assert result.exit_code == 0
        assert "No safety issues detected" in result.output

    def test_unsafe_agent_fails(self, tmp_path: Path, monkeypatch) -> None:
        _write_agent_module(tmp_path)
        monkeypatch.syspath_prepend(str(tmp_path))
        runner = CliRunner()
        result = runner.invoke(scan_cmd, [
            "scan_test_agents:unsafe_agent",
            "--category", "injection",
            "--timeout", "2",
        ])
        assert result.exit_code == 1
        assert "safety issue" in result.output.lower()

    def test_sync_agent_works(self, tmp_path: Path, monkeypatch) -> None:
        _write_agent_module(tmp_path)
        monkeypatch.syspath_prepend(str(tmp_path))
        runner = CliRunner()
        result = runner.invoke(scan_cmd, [
            "scan_test_agents:sync_agent",
            "--category", "injection",
            "--timeout", "2",
        ])
        # Sync agent just returns prefix + truncated input, should not trigger detectors
        # for most probes
        assert "Scan Summary" in result.output

    def test_dict_output_agent(self, tmp_path: Path, monkeypatch) -> None:
        _write_agent_module(tmp_path)
        monkeypatch.syspath_prepend(str(tmp_path))
        runner = CliRunner()
        result = runner.invoke(scan_cmd, [
            "scan_test_agents:dict_agent",
            "--category", "injection",
            "--timeout", "2",
        ])
        assert result.exit_code == 0
        assert "No safety issues detected" in result.output

    def test_error_agent_counts_errors(self, tmp_path: Path, monkeypatch) -> None:
        _write_agent_module(tmp_path)
        monkeypatch.syspath_prepend(str(tmp_path))
        runner = CliRunner()
        result = runner.invoke(scan_cmd, [
            "scan_test_agents:error_agent",
            "--category", "injection",
            "--timeout", "2",
        ])
        assert "Errors" in result.output

    def test_bad_target_shows_error(self) -> None:
        runner = CliRunner()
        result = runner.invoke(scan_cmd, ["nonexistent_xyz:fn", "--timeout", "1"])
        assert result.exit_code != 0

    def test_category_filter(self, tmp_path: Path, monkeypatch) -> None:
        _write_agent_module(tmp_path)
        monkeypatch.syspath_prepend(str(tmp_path))
        runner = CliRunner()
        result = runner.invoke(scan_cmd, [
            "scan_test_agents:safe_agent",
            "--category", "pii",
            "--timeout", "2",
        ])
        assert "Scan Summary" in result.output
        assert "10" in result.output  # 10 PII probes

    def test_scan_in_help(self) -> None:
        runner = CliRunner()
        result = runner.invoke(main, ["--help"])
        assert "scan" in result.output

    def test_verbose_flag_accepted(self, tmp_path: Path, monkeypatch) -> None:
        _write_agent_module(tmp_path)
        monkeypatch.syspath_prepend(str(tmp_path))
        runner = CliRunner()
        result = runner.invoke(scan_cmd, [
            "scan_test_agents:safe_agent",
            "--category", "injection",
            "--timeout", "2",
            "--verbose",
        ])
        assert result.exit_code == 0
