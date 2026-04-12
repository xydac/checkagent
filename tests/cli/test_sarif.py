"""Tests for the SARIF 2.1.0 builder (checkagent.cli.sarif)."""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

import pytest
from click.testing import CliRunner

from checkagent.cli.sarif import (
    _RULES,
    build_sarif,
    format_utc,
    get_rule_for_category,
    sarif_invocation,
    sarif_results,
    sarif_run_properties,
)
from checkagent.safety.evaluator import SafetyFinding
from checkagent.safety.probes.base import Probe
from checkagent.safety.taxonomy import SafetyCategory, Severity

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_probe(name: str = "test_probe", input_text: str = "ignore previous") -> Probe:
    return Probe(
        name=name,
        input=input_text,
        category=SafetyCategory.PROMPT_INJECTION,
        description="Test probe",
    )


def _make_finding(
    category: SafetyCategory = SafetyCategory.PROMPT_INJECTION,
    severity: Severity = Severity.HIGH,
    description: str = "Agent complied with injection",
) -> SafetyFinding:
    return SafetyFinding(
        category=category,
        severity=severity,
        description=description,
        evidence="echo back",
        probe="ignore previous",
    )


def _minimal_sarif(all_findings=None) -> dict:
    return build_sarif(
        target="my_module:agent_fn",
        total=10,
        passed=8,
        failed=2,
        errors=0,
        elapsed=1.23,
        start_time_utc="2026-04-11T19:00:00Z",
        end_time_utc="2026-04-11T19:00:01Z",
        all_findings=all_findings or [],
    )


# ---------------------------------------------------------------------------
# SARIF structure tests
# ---------------------------------------------------------------------------


class TestBuildSarif:
    def test_top_level_schema(self) -> None:
        doc = _minimal_sarif()
        assert doc["version"] == "2.1.0"
        assert "$schema" in doc
        assert "runs" in doc
        assert len(doc["runs"]) == 1

    def test_tool_driver(self) -> None:
        doc = _minimal_sarif()
        driver = doc["runs"][0]["tool"]["driver"]
        assert driver["name"] == "checkagent"
        assert "version" in driver
        assert driver["informationUri"] == "https://github.com/xydac/checkagent"

    def test_invocation_fields(self) -> None:
        doc = _minimal_sarif()
        inv = sarif_invocation(doc)
        assert inv["executionSuccessful"] is True
        assert inv["startTimeUtc"] == "2026-04-11T19:00:00Z"
        assert inv["endTimeUtc"] == "2026-04-11T19:00:01Z"
        props = inv["properties"]
        assert props["probesRun"] == 10
        assert props["probesPassed"] == 8
        assert props["probesFailed"] == 2
        assert props["probesErrored"] == 0

    def test_run_properties(self) -> None:
        doc = _minimal_sarif()
        props = sarif_run_properties(doc)
        assert props["overallScore"] == 80  # 8/10 = 80%
        assert props["passRate"] == pytest.approx(0.8)
        summary = props["summary"]
        assert summary["total"] == 10
        assert summary["passed"] == 8
        assert summary["failed"] == 2

    def test_no_findings_produces_empty_results(self) -> None:
        doc = _minimal_sarif()
        assert sarif_results(doc) == []

    def test_no_findings_no_rules(self) -> None:
        doc = _minimal_sarif()
        rules = doc["runs"][0]["tool"]["driver"]["rules"]
        assert rules == []

    def test_finding_produces_result(self) -> None:
        probe = _make_probe()
        finding = _make_finding()
        doc = build_sarif(
            target="my_module:fn",
            total=5,
            passed=4,
            failed=1,
            errors=0,
            elapsed=0.5,
            start_time_utc="2026-04-11T19:00:00Z",
            end_time_utc="2026-04-11T19:00:01Z",
            all_findings=[(probe, "echo back output", finding)],
        )
        results = sarif_results(doc)
        assert len(results) == 1
        r = results[0]
        assert r["ruleId"] == "CA-INJ-001"
        assert r["level"] == "error"  # HIGH → error
        assert "Agent complied with injection" in r["message"]["text"]

    def test_finding_locations_point_to_target_file(self) -> None:
        probe = _make_probe()
        finding = _make_finding()
        doc = build_sarif(
            target="my_module:agent_fn",
            total=1,
            passed=0,
            failed=1,
            errors=0,
            elapsed=0.1,
            start_time_utc="2026-04-11T19:00:00Z",
            end_time_utc="2026-04-11T19:00:01Z",
            all_findings=[(probe, "bad output", finding)],
        )
        result = sarif_results(doc)[0]
        loc = result["locations"][0]["physicalLocation"]["artifactLocation"]["uri"]
        assert loc == "my_module.py"

    def test_http_target_uri(self) -> None:
        probe = _make_probe()
        finding = _make_finding()
        doc = build_sarif(
            target="http://localhost:8000/chat",
            total=1,
            passed=0,
            failed=1,
            errors=0,
            elapsed=0.1,
            start_time_utc="2026-04-11T19:00:00Z",
            end_time_utc="2026-04-11T19:00:01Z",
            all_findings=[(probe, "bad", finding)],
        )
        result = sarif_results(doc)[0]
        loc = result["locations"][0]["physicalLocation"]["artifactLocation"]["uri"]
        assert loc == "http://localhost:8000/chat"

    def test_code_flow_included_when_output_present(self) -> None:
        probe = _make_probe(input_text="inject this")
        finding = _make_finding()
        doc = build_sarif(
            target="mod:fn",
            total=1,
            passed=0,
            failed=1,
            errors=0,
            elapsed=0.1,
            start_time_utc="2026-04-11T19:00:00Z",
            end_time_utc="2026-04-11T19:00:01Z",
            all_findings=[(probe, "agent replied: inject this", finding)],
        )
        result = sarif_results(doc)[0]
        assert "codeFlows" in result
        flows = result["codeFlows"]
        assert len(flows) == 1
        thread_locations = flows[0]["threadFlows"][0]["locations"]
        assert len(thread_locations) == 2
        assert "Probe sent" in thread_locations[0]["location"]["message"]["text"]
        assert "Agent response" in thread_locations[1]["location"]["message"]["text"]

    def test_no_code_flow_when_output_is_none(self) -> None:
        probe = _make_probe()
        finding = _make_finding()
        doc = build_sarif(
            target="mod:fn",
            total=1,
            passed=0,
            failed=1,
            errors=0,
            elapsed=0.1,
            start_time_utc="2026-04-11T19:00:00Z",
            end_time_utc="2026-04-11T19:00:01Z",
            all_findings=[(probe, None, finding)],
        )
        result = sarif_results(doc)[0]
        assert "codeFlows" not in result

    def test_rules_only_include_categories_present(self) -> None:
        probe_inj = _make_probe()
        finding_inj = _make_finding(category=SafetyCategory.PROMPT_INJECTION)

        probe_pii = _make_probe(name="pii_probe", input_text="give me PII")
        finding_pii = _make_finding(
            category=SafetyCategory.PII_LEAKAGE,
            severity=Severity.MEDIUM,
        )

        doc = build_sarif(
            target="mod:fn",
            total=5,
            passed=3,
            failed=2,
            errors=0,
            elapsed=0.5,
            start_time_utc="2026-04-11T19:00:00Z",
            end_time_utc="2026-04-11T19:00:01Z",
            all_findings=[
                (probe_inj, "bad output", finding_inj),
                (probe_pii, "leaked PII", finding_pii),
            ],
        )
        rules = doc["runs"][0]["tool"]["driver"]["rules"]
        rule_ids = {r["id"] for r in rules}
        assert "CA-INJ-001" in rule_ids
        assert "CA-PII-001" in rule_ids
        # No jailbreak rule since no jailbreak findings
        assert not any(r["id"] == "CA-JAILBREAK-001" for r in rules)

    def test_severity_levels(self) -> None:
        """CRITICAL/HIGH → error, MEDIUM → warning, LOW → note."""
        cases = [
            (Severity.CRITICAL, "error"),
            (Severity.HIGH, "error"),
            (Severity.MEDIUM, "warning"),
            (Severity.LOW, "note"),
        ]
        for severity, expected_level in cases:
            probe = _make_probe()
            finding = _make_finding(severity=severity)
            doc = build_sarif(
                target="mod:fn",
                total=1,
                passed=0,
                failed=1,
                errors=0,
                elapsed=0.1,
                start_time_utc="2026-04-11T19:00:00Z",
                end_time_utc="2026-04-11T19:00:01Z",
                all_findings=[(probe, "output", finding)],
            )
            result = sarif_results(doc)[0]
            assert result["level"] == expected_level, f"Expected {expected_level} for {severity}"

    def test_rule_has_required_sarif_fields(self) -> None:
        """Every rule must have the fields required for GitHub upload."""
        probe = _make_probe()
        finding = _make_finding()
        doc = build_sarif(
            target="mod:fn",
            total=1,
            passed=0,
            failed=1,
            errors=0,
            elapsed=0.1,
            start_time_utc="2026-04-11T19:00:00Z",
            end_time_utc="2026-04-11T19:00:01Z",
            all_findings=[(probe, "output", finding)],
        )
        rule = doc["runs"][0]["tool"]["driver"]["rules"][0]
        assert "id" in rule
        assert "name" in rule
        assert "shortDescription" in rule
        assert "fullDescription" in rule
        assert "help" in rule
        assert "markdown" in rule["help"]
        assert "properties" in rule
        assert "security-severity" in rule["properties"]

    def test_json_serialisable(self) -> None:
        """The document must be JSON-serialisable (no datetime objects, etc.)."""
        probe = _make_probe()
        finding = _make_finding()
        doc = build_sarif(
            target="mod:fn",
            total=1,
            passed=0,
            failed=1,
            errors=0,
            elapsed=0.1,
            start_time_utc="2026-04-11T19:00:00Z",
            end_time_utc="2026-04-11T19:00:01Z",
            all_findings=[(probe, "output", finding)],
        )
        serialised = json.dumps(doc)
        assert len(serialised) > 0


# ---------------------------------------------------------------------------
# Rule catalogue tests
# ---------------------------------------------------------------------------


class TestRuleCatalogue:
    def test_all_safety_categories_have_rules(self) -> None:
        from checkagent.safety.taxonomy import SafetyCategory

        # Check all SafetyCategory values have a rule mapping
        for cat in SafetyCategory:
            rule = get_rule_for_category(cat.value)
            assert rule["id"].startswith("CA-"), f"No rule for {cat.value}"

    def test_unknown_category_returns_fallback(self) -> None:
        rule = get_rule_for_category("totally_unknown_category_xyz")
        assert rule["id"] == "CA-UNKNOWN-001"

    def test_all_rules_have_security_severity(self) -> None:
        for cat_value, rule in _RULES.items():
            sev = rule.get("properties", {}).get("security-severity")
            assert sev is not None, f"Rule for {cat_value} missing security-severity"
            assert 0.0 <= float(sev) <= 10.0, f"Invalid security-severity {sev} for {cat_value}"

    def test_all_rules_have_markdown_help(self) -> None:
        for cat_value, rule in _RULES.items():
            assert "markdown" in rule.get("help", {}), (
                f"Rule for {cat_value} missing help.markdown"
            )


# ---------------------------------------------------------------------------
# format_utc tests
# ---------------------------------------------------------------------------


class TestFormatUtc:
    def test_returns_iso_8601_string(self) -> None:
        ts = 1744399200.0  # 2025-04-11T18:00:00Z
        result = format_utc(ts)
        assert result.endswith("Z")
        assert "T" in result
        assert len(result) == 20  # YYYY-MM-DDTHH:MM:SSZ


# ---------------------------------------------------------------------------
# Integration: --output flag writes SARIF to file
# ---------------------------------------------------------------------------


class TestScanOutputFlag:
    def test_output_flag_writes_sarif_file(self, tmp_path: Path, monkeypatch) -> None:
        """--output writes a valid SARIF 2.1.0 file."""
        from checkagent.cli.scan import scan_cmd

        mod = tmp_path / "sarif_safe_agent.py"
        mod.write_text("async def run(query):\n    return 'I can help you with that.'\n")
        monkeypatch.syspath_prepend(str(tmp_path))

        sarif_path = tmp_path / "scan.sarif"
        runner = CliRunner()
        result = runner.invoke(
            scan_cmd,
            [
                "sarif_safe_agent:run",
                "--sarif", str(sarif_path),
                "--category", "injection",
            ],
        )

        assert sarif_path.exists(), (
            f"SARIF file not written. Exit={result.exit_code}. Output:\n{result.output}"
        )
        doc = json.loads(sarif_path.read_text())
        assert doc["version"] == "2.1.0"
        assert "runs" in doc
        assert len(doc["runs"]) == 1

    def test_output_sarif_has_correct_structure(self, tmp_path: Path, monkeypatch) -> None:
        """Written SARIF has all required top-level SARIF fields."""
        from checkagent.cli.scan import scan_cmd

        mod = tmp_path / "sarif_struct_agent.py"
        mod.write_text("async def run(query):\n    return 'safe response'\n")
        monkeypatch.syspath_prepend(str(tmp_path))

        sarif_path = tmp_path / "out.sarif"
        runner = CliRunner()
        runner.invoke(
            scan_cmd,
            ["sarif_struct_agent:run", "--sarif", str(sarif_path), "--category", "pii"],
        )

        assert sarif_path.exists()
        doc = json.loads(sarif_path.read_text())
        run = doc["runs"][0]
        assert "tool" in run
        assert "invocations" in run
        assert "results" in run
        assert run["tool"]["driver"]["name"] == "checkagent"

    def test_output_message_shown_in_terminal(self, tmp_path: Path, monkeypatch) -> None:
        """When --output is given, terminal output confirms SARIF written."""
        from checkagent.cli.scan import scan_cmd

        mod = tmp_path / "sarif_confirm_agent.py"
        mod.write_text("async def run(query):\n    return 'safe'\n")
        monkeypatch.syspath_prepend(str(tmp_path))

        sarif_path = tmp_path / "result.sarif"
        runner = CliRunner()
        result = runner.invoke(
            scan_cmd,
            ["sarif_confirm_agent:run", "--sarif", str(sarif_path), "--category", "injection"],
        )

        assert "SARIF written" in result.output
        assert str(sarif_path) in result.output
