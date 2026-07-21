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


class TestProbeListVerbose:
    def test_verbose_shows_all_probes(self) -> None:
        runner = CliRunner()
        result = runner.invoke(probe_list_cmd, ["--verbose", "--category", "injection"])
        assert result.exit_code == 0
        # Should show "All Probe Inputs" heading
        assert "All Probe Inputs" in result.output

    def test_verbose_shows_line_numbers(self) -> None:
        runner = CliRunner()
        result = runner.invoke(probe_list_cmd, ["--verbose", "--category", "injection"])
        assert result.exit_code == 0
        # Line numbers like "  1." should appear
        assert "1." in result.output

    def test_verbose_shows_count_note(self) -> None:
        runner = CliRunner()
        result = runner.invoke(probe_list_cmd, ["--verbose", "--category", "injection"])
        assert result.exit_code == 0
        # Total count annotation "(N total)" should appear
        assert "total" in result.output

    def test_verbose_json_includes_all_probes(self) -> None:
        runner = CliRunner()
        result = runner.invoke(probe_list_cmd, ["--verbose", "--json", "--category", "injection"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        cat = data["categories"][0]
        # probes key with full input + description
        assert "probes" in cat
        assert len(cat["probes"]) == cat["count"]
        assert "input" in cat["probes"][0]
        assert "description" in cat["probes"][0]

    def test_verbose_more_probes_than_examples(self) -> None:
        runner = CliRunner()
        r_verbose = runner.invoke(probe_list_cmd, ["--verbose", "--category", "injection"])
        r_examples = runner.invoke(probe_list_cmd, ["--examples", "--category", "injection"])
        # --verbose should show more text (all probes, not just 3)
        assert len(r_verbose.output) > len(r_examples.output)

    def test_verbose_examples_json_no_duplication(self) -> None:
        """F-160: --verbose --examples --json should not duplicate all probes in examples field."""
        runner = CliRunner()
        result = runner.invoke(
            probe_list_cmd, ["--verbose", "--examples", "--json", "--category", "injection"]
        )
        assert result.exit_code == 0
        data = json.loads(result.output)
        cat = data["categories"][0]
        # probes has the full set
        assert len(cat["probes"]) == cat["count"]
        # examples is capped at 3 (not duplicated with probes)
        assert len(cat["examples"]) <= 3

    def test_verbose_json_examples_still_populated_with_examples_flag(self) -> None:
        """--verbose --examples --json: examples field has up to 3 entries (not empty)."""
        runner = CliRunner()
        result = runner.invoke(
            probe_list_cmd, ["--verbose", "--examples", "--json", "--category", "injection"]
        )
        assert result.exit_code == 0
        data = json.loads(result.output)
        cat = data["categories"][0]
        # examples should have some entries (not 0) since --examples was passed
        assert len(cat["examples"]) > 0
