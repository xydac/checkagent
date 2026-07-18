"""Tests for checkagent probe-list command."""

from __future__ import annotations

import json

from click.testing import CliRunner

from checkagent.cli.probe_list import probe_list_cmd


class TestProbeListCmd:
    def test_default_output_shows_categories(self) -> None:
        runner = CliRunner()
        result = runner.invoke(probe_list_cmd, [])
        assert result.exit_code == 0
        assert "prompt_injection" in result.output
        assert "jailbreak" in result.output
        assert "pii_leakage" in result.output
        assert "data_enumeration" in result.output

    def test_shows_total_count(self) -> None:
        runner = CliRunner()
        result = runner.invoke(probe_list_cmd, [])
        assert result.exit_code == 0
        assert "101" in result.output  # total probes

    def test_shows_owasp_mapping(self) -> None:
        runner = CliRunner()
        result = runner.invoke(probe_list_cmd, [])
        assert result.exit_code == 0
        assert "LLM01" in result.output
        assert "LLM06" in result.output

    def test_filter_by_category_key(self) -> None:
        runner = CliRunner()
        result = runner.invoke(probe_list_cmd, ["--category", "injection"])
        assert result.exit_code == 0
        assert "prompt_injection" in result.output
        assert "jailbreak" not in result.output

    def test_filter_by_display_name(self) -> None:
        runner = CliRunner()
        result = runner.invoke(probe_list_cmd, ["--category", "pii_leakage"])
        assert result.exit_code == 0
        assert "pii_leakage" in result.output

    def test_invalid_category_raises_usage_error(self) -> None:
        runner = CliRunner()
        result = runner.invoke(probe_list_cmd, ["--category", "nonexistent"])
        assert result.exit_code != 0
        assert "Unknown category" in result.output or "Unknown category" in str(result.exception)

    def test_examples_flag_shows_probe_inputs(self) -> None:
        runner = CliRunner()
        result = runner.invoke(probe_list_cmd, ["--examples"])
        assert result.exit_code == 0
        assert "Example Probe Inputs" in result.output
        # Should show at least one actual probe input text
        assert "instructions" in result.output.lower() or "ignore" in result.output.lower()

    def test_json_output_structure(self) -> None:
        runner = CliRunner()
        result = runner.invoke(probe_list_cmd, ["--json"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert "total_probes" in data
        assert "categories" in data
        assert data["total_probes"] > 0
        assert len(data["categories"]) >= 4

    def test_json_category_fields(self) -> None:
        runner = CliRunner()
        result = runner.invoke(probe_list_cmd, ["--json"])
        data = json.loads(result.output)
        cat = data["categories"][0]
        assert "name" in cat
        assert "count" in cat
        assert "description" in cat
        assert "owasp" in cat
        assert "examples" in cat

    def test_json_with_examples(self) -> None:
        runner = CliRunner()
        result = runner.invoke(probe_list_cmd, ["--json", "--examples"])
        data = json.loads(result.output)
        # At least one category should have examples
        has_examples = any(len(c["examples"]) > 0 for c in data["categories"])
        assert has_examples

    def test_json_category_filter(self) -> None:
        runner = CliRunner()
        result = runner.invoke(probe_list_cmd, ["--json", "--category", "jailbreak"])
        data = json.loads(result.output)
        assert len(data["categories"]) == 1
        assert data["categories"][0]["name"] == "jailbreak"
        assert data["categories"][0]["count"] > 0
