"""Tests for ``checkagent ci-init`` command."""

from __future__ import annotations

from pathlib import Path

import yaml
from click.testing import CliRunner

from checkagent.cli import main
from checkagent.cli.ci_init import ci_init_cmd


class TestCiInitCommand:
    """Tests for the ci-init subcommand."""

    def test_help_text(self) -> None:
        runner = CliRunner()
        result = runner.invoke(ci_init_cmd, ["--help"])
        assert result.exit_code == 0
        assert "CI/CD" in result.output or "ci" in result.output.lower()
        assert "--platform" in result.output
        assert "--scan-target" in result.output

    def test_registered_in_main_group(self) -> None:
        runner = CliRunner()
        result = runner.invoke(main, ["--help"])
        assert result.exit_code == 0
        assert "ci-init" in result.output

    def test_github_is_default_platform(self, tmp_path: Path) -> None:
        runner = CliRunner()
        with runner.isolated_filesystem(temp_dir=tmp_path):
            result = runner.invoke(ci_init_cmd, [])
        assert result.exit_code == 0
        assert ".github/workflows/checkagent.yml" in result.output

    def test_creates_github_workflow_file(self, tmp_path: Path) -> None:
        runner = CliRunner()
        result = runner.invoke(ci_init_cmd, ["--directory", str(tmp_path)])
        assert result.exit_code == 0
        wf = tmp_path / ".github" / "workflows" / "checkagent.yml"
        assert wf.exists()
        content = wf.read_text()
        assert "checkagent scan" in content
        assert "pytest" in content

    def test_github_workflow_is_valid_yaml(self, tmp_path: Path) -> None:
        runner = CliRunner()
        runner.invoke(ci_init_cmd, ["--directory", str(tmp_path)])
        wf = tmp_path / ".github" / "workflows" / "checkagent.yml"
        data = yaml.safe_load(wf.read_text())
        # YAML 'on' key parses as boolean True in some loaders; jobs must always be present
        assert "jobs" in data

    def test_creates_gitlab_ci_file(self, tmp_path: Path) -> None:
        runner = CliRunner()
        result = runner.invoke(ci_init_cmd, ["--platform", "gitlab", "--directory", str(tmp_path)])
        assert result.exit_code == 0
        gl = tmp_path / ".gitlab-ci.yml"
        assert gl.exists()
        content = gl.read_text()
        assert "checkagent scan" in content
        assert "pytest" in content

    def test_gitlab_ci_is_valid_yaml(self, tmp_path: Path) -> None:
        runner = CliRunner()
        runner.invoke(ci_init_cmd, ["--platform", "gitlab", "--directory", str(tmp_path)])
        gl = tmp_path / ".gitlab-ci.yml"
        data = yaml.safe_load(gl.read_text())
        assert "stages" in data

    def test_both_platform_creates_both_files(self, tmp_path: Path) -> None:
        runner = CliRunner()
        result = runner.invoke(ci_init_cmd, ["--platform", "both", "--directory", str(tmp_path)])
        assert result.exit_code == 0
        assert (tmp_path / ".github" / "workflows" / "checkagent.yml").exists()
        assert (tmp_path / ".gitlab-ci.yml").exists()

    def test_custom_scan_target_in_workflow(self, tmp_path: Path) -> None:
        runner = CliRunner()
        runner.invoke(ci_init_cmd, [
            "--directory", str(tmp_path),
            "--scan-target", "my_agent:run",
        ])
        wf = tmp_path / ".github" / "workflows" / "checkagent.yml"
        assert "my_agent:run" in wf.read_text()

    def test_custom_scan_target_in_gitlab(self, tmp_path: Path) -> None:
        runner = CliRunner()
        runner.invoke(ci_init_cmd, [
            "--platform", "gitlab",
            "--directory", str(tmp_path),
            "--scan-target", "my_agent:run",
        ])
        gl = tmp_path / ".gitlab-ci.yml"
        assert "my_agent:run" in gl.read_text()

    def test_skip_existing_without_force(self, tmp_path: Path) -> None:
        runner = CliRunner()
        # First invocation creates the file
        runner.invoke(ci_init_cmd, ["--directory", str(tmp_path)])
        wf = tmp_path / ".github" / "workflows" / "checkagent.yml"
        wf.write_text("# sentinel")

        # Second invocation without --force should skip
        result = runner.invoke(ci_init_cmd, ["--directory", str(tmp_path)])
        assert result.exit_code == 0
        assert "skip" in result.output
        assert wf.read_text() == "# sentinel"

    def test_force_overwrites_existing(self, tmp_path: Path) -> None:
        runner = CliRunner()
        wf = tmp_path / ".github" / "workflows" / "checkagent.yml"
        wf.parent.mkdir(parents=True)
        wf.write_text("# sentinel")

        result = runner.invoke(ci_init_cmd, ["--directory", str(tmp_path), "--force"])
        assert result.exit_code == 0
        assert wf.read_text() != "# sentinel"

    def test_output_shows_next_steps(self, tmp_path: Path) -> None:
        runner = CliRunner()
        result = runner.invoke(ci_init_cmd, ["--directory", str(tmp_path)])
        assert "Next steps" in result.output

    def test_openai_api_key_mentioned_in_workflow(self, tmp_path: Path) -> None:
        runner = CliRunner()
        runner.invoke(ci_init_cmd, ["--directory", str(tmp_path)])
        wf = tmp_path / ".github" / "workflows" / "checkagent.yml"
        assert "OPENAI_API_KEY" in wf.read_text()

    def test_repeat_flag_in_github_workflow(self, tmp_path: Path) -> None:
        """Generated GitHub workflow should include --repeat for LLM-backed agents."""
        runner = CliRunner()
        runner.invoke(ci_init_cmd, ["--directory", str(tmp_path)])
        wf = (tmp_path / ".github" / "workflows" / "checkagent.yml").read_text()
        assert "--repeat 3" in wf

    def test_repeat_flag_in_gitlab_ci(self, tmp_path: Path) -> None:
        """Generated GitLab CI should include --repeat for LLM-backed agents."""
        runner = CliRunner()
        runner.invoke(ci_init_cmd, ["--directory", str(tmp_path), "--platform", "gitlab"])
        ci = (tmp_path / ".gitlab-ci.yml").read_text()
        assert "--repeat 3" in ci
