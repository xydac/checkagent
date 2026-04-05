"""Tests for checkagent demo CLI command."""

from __future__ import annotations

from click.testing import CliRunner

from checkagent.cli.demo import demo_cmd


class TestDemoCommand:
    """Test the checkagent demo zero-config demo."""

    def test_demo_runs_successfully(self) -> None:
        runner = CliRunner()
        result = runner.invoke(demo_cmd, [], catch_exceptions=False)
        assert result.exit_code == 0

    def test_demo_output_shows_panel(self) -> None:
        runner = CliRunner()
        result = runner.invoke(demo_cmd, [], catch_exceptions=False)
        assert "CheckAgent Demo" in result.output

    def test_demo_shows_all_passed(self) -> None:
        runner = CliRunner()
        result = runner.invoke(demo_cmd, [], catch_exceptions=False)
        assert "passed" in result.output

    def test_demo_shows_next_steps(self) -> None:
        runner = CliRunner()
        result = runner.invoke(demo_cmd, [], catch_exceptions=False)
        assert "checkagent init" in result.output
        assert "checkagent run" in result.output

    def test_demo_verbose_flag(self) -> None:
        runner = CliRunner()
        result = runner.invoke(demo_cmd, ["-v"], catch_exceptions=False)
        assert result.exit_code == 0
