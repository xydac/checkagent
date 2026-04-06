"""Tests for checkagent init CLI command."""

from __future__ import annotations

from pathlib import Path

from click.testing import CliRunner

from checkagent.cli.init import init_cmd


class TestInitCommand:
    """Test the checkagent init scaffolding."""

    def test_creates_files_in_empty_directory(self, tmp_path: Path) -> None:
        runner = CliRunner()
        result = runner.invoke(init_cmd, [str(tmp_path)])

        assert result.exit_code == 0
        assert (tmp_path / "checkagent.yml").exists()
        assert (tmp_path / "pyproject.toml").exists()
        assert (tmp_path / "sample_agent.py").exists()
        assert (tmp_path / "tests" / "test_sample.py").exists()
        assert (tmp_path / "tests" / "conftest.py").exists()
        assert (tmp_path / "tests" / "cassettes" / ".gitkeep").exists()

    def test_config_is_valid_yaml(self, tmp_path: Path) -> None:
        import yaml

        runner = CliRunner()
        runner.invoke(init_cmd, [str(tmp_path)])

        config = yaml.safe_load((tmp_path / "checkagent.yml").read_text())
        assert config["version"] == 1
        assert config["defaults"]["layer"] == "mock"

    def test_skips_existing_files(self, tmp_path: Path) -> None:
        # Create a file first
        (tmp_path / "checkagent.yml").write_text("custom: true")

        runner = CliRunner()
        result = runner.invoke(init_cmd, [str(tmp_path)])

        assert result.exit_code == 0
        # Original file should be preserved
        assert (tmp_path / "checkagent.yml").read_text() == "custom: true"
        assert "skip" in result.output

    def test_force_overwrites_existing_files(self, tmp_path: Path) -> None:
        (tmp_path / "checkagent.yml").write_text("custom: true")

        runner = CliRunner()
        result = runner.invoke(init_cmd, [str(tmp_path), "--force"])

        assert result.exit_code == 0
        content = (tmp_path / "checkagent.yml").read_text()
        assert "version: 1" in content

    def test_defaults_to_current_directory(self, tmp_path: Path, monkeypatch) -> None:
        monkeypatch.chdir(tmp_path)
        runner = CliRunner()
        result = runner.invoke(init_cmd, [])

        assert result.exit_code == 0
        assert (tmp_path / "checkagent.yml").exists()

    def test_output_shows_next_steps(self, tmp_path: Path) -> None:
        runner = CliRunner()
        result = runner.invoke(init_cmd, [str(tmp_path)])

        assert "Next steps" in result.output
        assert "pytest tests/ -v" in result.output

    def test_sample_agent_is_valid_python(self, tmp_path: Path) -> None:
        runner = CliRunner()
        runner.invoke(init_cmd, [str(tmp_path)])

        agent_code = (tmp_path / "sample_agent.py").read_text()
        compile(agent_code, "sample_agent.py", "exec")

    def test_sample_test_is_valid_python(self, tmp_path: Path) -> None:
        runner = CliRunner()
        runner.invoke(init_cmd, [str(tmp_path)])

        test_code = (tmp_path / "tests" / "test_sample.py").read_text()
        compile(test_code, "test_sample.py", "exec")

    def test_created_files_listed_in_output(self, tmp_path: Path) -> None:
        runner = CliRunner()
        result = runner.invoke(init_cmd, [str(tmp_path)])

        assert "checkagent.yml" in result.output
        assert "sample_agent.py" in result.output
        assert "test_sample.py" in result.output

    def test_pyproject_has_pytest_config(self, tmp_path: Path) -> None:
        """F-005: Generated pyproject.toml must set asyncio_mode and pythonpath."""
        runner = CliRunner()
        runner.invoke(init_cmd, [str(tmp_path)])

        content = (tmp_path / "pyproject.toml").read_text()
        assert "asyncio_mode" in content
        assert '"auto"' in content
        assert "pythonpath" in content

    def test_generated_tests_pass(self, tmp_path: Path) -> None:
        """F-005: Generated tests must pass out of the box."""
        import subprocess

        runner = CliRunner()
        runner.invoke(init_cmd, [str(tmp_path)])

        proc = subprocess.run(
            ["pytest", "tests/", "-q", "--tb=short"],
            capture_output=True,
            text=True,
            cwd=str(tmp_path),
        )
        assert proc.returncode == 0, f"Generated tests failed:\n{proc.stdout}\n{proc.stderr}"
        assert "passed" in proc.stdout
