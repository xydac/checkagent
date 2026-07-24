"""Tests for the triage command — smart finding prioritization."""

from __future__ import annotations

import json

from click.testing import CliRunner

from checkagent.cli.triage import triage_cmd, triage_findings


class TestTriageFindings:
    def test_single_category(self):
        findings = [
            {"category": "prompt_injection", "severity": "critical"},
            {"category": "prompt_injection", "severity": "high"},
        ]
        result = triage_findings(findings, total=10)
        assert len(result) == 1
        assert result[0]["category"] == "prompt_injection"
        assert result[0]["finding_count"] == 2
        assert result[0]["score_improvement_pct"] == 20.0

    def test_multiple_categories_sorted_by_priority(self):
        findings = [
            {"category": "prompt_injection", "severity": "critical"},
            {"category": "prompt_injection", "severity": "critical"},
            {"category": "prompt_injection", "severity": "high"},
            {"category": "scope_violation", "severity": "medium"},
        ]
        result = triage_findings(findings, total=20)
        assert result[0]["category"] == "prompt_injection"
        assert result[0]["priority_score"] > result[1]["priority_score"]

    def test_empty_findings(self):
        result = triage_findings([], total=10)
        assert result == []

    def test_quick_fix_present(self):
        findings = [{"category": "pii_leakage", "severity": "high"}]
        result = triage_findings(findings, total=10)
        assert result[0]["quick_fix"]

    def test_sample_probes_limited(self):
        findings = [
            {"category": "jailbreak", "severity": "high", "probe_id": f"p{i}"}
            for i in range(10)
        ]
        result = triage_findings(findings, total=50)
        assert len(result[0]["sample_probes"]) <= 3

    def test_unknown_category(self):
        findings = [{"category": "unknown_cat", "severity": "low"}]
        result = triage_findings(findings, total=10)
        assert len(result) == 1
        assert result[0]["category"] == "unknown_cat"


class TestTriageCLI:
    def test_from_file(self, tmp_path):
        scan_data = {
            "summary": {"total": 10, "score": 0.7},
            "findings": [
                {"category": "prompt_injection", "severity": "critical"},
                {"category": "prompt_injection", "severity": "high"},
                {"category": "jailbreak", "severity": "medium"},
            ],
        }
        f = tmp_path / "scan.json"
        f.write_text(json.dumps(scan_data))
        runner = CliRunner()
        result = runner.invoke(triage_cmd, [str(f)])
        assert result.exit_code == 0
        assert "prompt_injection" in result.output
        assert "Top Priority" in result.output

    def test_json_output(self, tmp_path):
        scan_data = {
            "summary": {"total": 10, "score": 0.5},
            "findings": [
                {"category": "pii_leakage", "severity": "critical"},
            ],
        }
        f = tmp_path / "scan.json"
        f.write_text(json.dumps(scan_data))
        runner = CliRunner()
        result = runner.invoke(triage_cmd, [str(f), "--json"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert "priorities" in data
        assert data["priorities"][0]["category"] == "pii_leakage"

    def test_top_n(self, tmp_path):
        scan_data = {
            "summary": {"total": 20, "score": 0.5},
            "findings": [
                {"category": "prompt_injection", "severity": "critical"},
                {"category": "jailbreak", "severity": "high"},
                {"category": "pii_leakage", "severity": "medium"},
            ],
        }
        f = tmp_path / "scan.json"
        f.write_text(json.dumps(scan_data))
        runner = CliRunner()
        result = runner.invoke(triage_cmd, [str(f), "--json", "--top", "1"])
        data = json.loads(result.output)
        assert len(data["priorities"]) == 1

    def test_clean_scan(self, tmp_path):
        scan_data = {"summary": {"total": 10, "score": 1.0}, "findings": []}
        f = tmp_path / "scan.json"
        f.write_text(json.dumps(scan_data))
        runner = CliRunner()
        result = runner.invoke(triage_cmd, [str(f)])
        assert result.exit_code == 0
        assert "No findings" in result.output

    def test_stdin_input(self):
        scan_data = json.dumps({
            "summary": {"total": 5, "score": 0.6},
            "findings": [
                {"category": "jailbreak", "severity": "high"},
            ],
        })
        runner = CliRunner()
        result = runner.invoke(triage_cmd, ["--json"], input=scan_data)
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert len(data["priorities"]) == 1

    def test_no_input_error(self):
        runner = CliRunner()
        result = runner.invoke(triage_cmd, [])
        assert result.exit_code == 1
