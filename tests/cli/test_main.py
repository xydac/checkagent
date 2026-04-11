"""Tests for the checkagent CLI main group."""

from __future__ import annotations

from click.testing import CliRunner

from checkagent.cli import main


class TestMainGroup:
    """Test the top-level CLI group."""

    def test_help_shows_commands(self) -> None:
        runner = CliRunner()
        result = runner.invoke(main, ["--help"])
        assert result.exit_code == 0
        assert "init" in result.output
        assert "run" in result.output

    def test_version_flag(self) -> None:
        runner = CliRunner()
        result = runner.invoke(main, ["--version"])
        assert result.exit_code == 0
        assert "0.1.2" in result.output

    def test_no_args_shows_help(self) -> None:
        runner = CliRunner()
        result = runner.invoke(main, [])
        assert result.exit_code == 0
        assert "Usage" in result.output
