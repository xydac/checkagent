"""Tests for the audit command — full security audit pipeline."""

from __future__ import annotations

import json
from unittest.mock import patch

from click.testing import CliRunner

from checkagent.cli.audit import _render_audit_markdown, audit_cmd

# ── Fixtures ────────────────────────────────────────────────────────────────

_CLEAN_SCAN = {
    "summary": {"total": 10, "passed": 10, "failed": 0, "score": 1.0},
    "findings": [],
}

_FINDINGS_SCAN = {
    "summary": {"total": 101, "passed": 74, "failed": 27, "score": 0.733},
    "findings": [
        {"category": "prompt_injection", "severity": "CRITICAL",
         "probe_id": "inject-01", "finding": "No refusal"},
        {"category": "prompt_injection", "severity": "HIGH",
         "probe_id": "inject-02", "finding": "No refusal"},
        {"category": "jailbreak", "severity": "CRITICAL",
         "probe_id": "jail-01", "finding": "Jailbreak succeeded"},
        {"category": "pii_leakage", "severity": "HIGH",
         "probe_id": "pii-01", "finding": "PII exposed"},
    ],
}


def _make_audit(scan: dict) -> dict:
    from checkagent.cli.simulate import simulate_attacks
    from checkagent.cli.triage import triage_findings

    findings = scan.get("findings", [])
    total = scan.get("summary", {}).get("total", 0)
    triage = triage_findings(findings, total) if findings else []
    simulation = simulate_attacks(scan)
    return {"scan": scan, "triage": triage, "simulation": simulation}


# ── Unit tests for helper functions ─────────────────────────────────────────

class TestRunAuditMarkdown:
    def test_clean_agent_markdown(self):
        audit = _make_audit(_CLEAN_SCAN)
        md = _render_audit_markdown(audit, "my_module:agent")
        assert "# CheckAgent Security Audit" in md
        assert "`my_module:agent`" in md
        assert "100%" in md

    def test_findings_agent_markdown_has_triage(self):
        audit = _make_audit(_FINDINGS_SCAN)
        md = _render_audit_markdown(audit, "my_module:agent")
        assert "## Triage" in md
        assert "prompt_injection" in md

    def test_findings_agent_markdown_has_attack_chains(self):
        audit = _make_audit(_FINDINGS_SCAN)
        md = _render_audit_markdown(audit, "my_module:agent")
        # With injection + jailbreak + pii, chains should exist
        assert "Attack Chain" in md

    def test_findings_table_in_markdown(self):
        audit = _make_audit(_FINDINGS_SCAN)
        md = _render_audit_markdown(audit, "test:fn")
        assert "## Findings" in md
        assert "inject-01" in md

    def test_target_label_in_markdown(self):
        audit = _make_audit(_CLEAN_SCAN)
        md = _render_audit_markdown(audit, "http://localhost:8000/chat")
        assert "http://localhost:8000/chat" in md

    def test_checkagent_footer_in_markdown(self):
        audit = _make_audit(_CLEAN_SCAN)
        md = _render_audit_markdown(audit, "x")
        assert "CheckAgent" in md


# ── CLI tests ────────────────────────────────────────────────────────────────

class TestAuditCmdCli:
    def setup_method(self):
        self.runner = CliRunner()

    def _mock_audit(self, scan: dict):
        """Patch _run_scan_json to return a fixed scan result."""
        def _fake_run(target, url, category, timeout, repeat, **kwargs):
            return scan
        return patch("checkagent.cli.audit._run_scan_json", side_effect=_fake_run)

    def test_requires_target_or_url(self):
        result = self.runner.invoke(audit_cmd, [])
        assert result.exit_code != 0
        assert "TARGET" in result.output or "url" in result.output.lower() or \
               "Error" in result.output

    def test_json_output_clean_agent(self):
        with self._mock_audit(_CLEAN_SCAN):
            result = self.runner.invoke(
                audit_cmd, ["my_module:agent", "--json"]
            )
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert "scan" in data
        assert "triage" in data
        assert "simulation" in data

    def test_json_output_findings(self):
        with self._mock_audit(_FINDINGS_SCAN):
            result = self.runner.invoke(
                audit_cmd, ["my_module:agent", "--json"]
            )
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert len(data["triage"]) > 0
        assert data["simulation"]["chain_count"] >= 0

    def test_terminal_output_shows_grade(self):
        with self._mock_audit(_FINDINGS_SCAN):
            result = self.runner.invoke(audit_cmd, ["my_module:agent"])
        assert result.exit_code == 0
        assert "Grade" in result.output
        assert "%" in result.output

    def test_terminal_output_shows_triage(self):
        with self._mock_audit(_FINDINGS_SCAN):
            result = self.runner.invoke(audit_cmd, ["my_module:agent"])
        assert result.exit_code == 0
        assert "Triage" in result.output
        assert "prompt_injection" in result.output

    def test_terminal_output_clean_agent_no_triage(self):
        with self._mock_audit(_CLEAN_SCAN):
            result = self.runner.invoke(audit_cmd, ["my_module:agent"])
        assert result.exit_code == 0
        assert "passed all safety probes" in result.output

    def test_report_writes_markdown(self, tmp_path):
        rpath = tmp_path / "audit.md"
        with self._mock_audit(_FINDINGS_SCAN):
            result = self.runner.invoke(
                audit_cmd, ["my_module:agent", "--report", str(rpath)]
            )
        assert result.exit_code == 0
        assert rpath.exists()
        content = rpath.read_text()
        assert "# CheckAgent Security Audit" in content

    def test_report_writes_json(self, tmp_path):
        rpath = tmp_path / "audit.json"
        with self._mock_audit(_FINDINGS_SCAN):
            result = self.runner.invoke(
                audit_cmd, ["my_module:agent", "--report", str(rpath)]
            )
        assert result.exit_code == 0
        assert rpath.exists()
        data = json.loads(rpath.read_text())
        assert "scan" in data

    def test_url_flag_passes_through(self):
        captured = {}

        def _capture(target, url, category, timeout, repeat, **kwargs):
            captured["url"] = url
            captured["target"] = target
            return _CLEAN_SCAN

        with patch("checkagent.cli.audit._run_scan_json", side_effect=_capture):
            result = self.runner.invoke(
                audit_cmd, ["--url", "http://localhost:8000/chat"]
            )
        assert result.exit_code == 0
        assert captured["url"] == "http://localhost:8000/chat"
        assert captured["target"] is None

    def test_repeat_flag_passes_through(self):
        captured = {}

        def _capture(target, url, category, timeout, repeat, **kwargs):
            captured["repeat"] = repeat
            return _CLEAN_SCAN

        with patch("checkagent.cli.audit._run_scan_json", side_effect=_capture):
            result = self.runner.invoke(
                audit_cmd, ["my_module:agent", "--repeat", "3"]
            )
        assert result.exit_code == 0
        assert captured["repeat"] == 3

    def test_audit_shows_attack_chains_section(self):
        with self._mock_audit(_FINDINGS_SCAN):
            result = self.runner.invoke(audit_cmd, ["my_module:agent"])
        assert result.exit_code == 0
        assert "Attack Chain" in result.output

    def test_json_output_includes_share_card(self):
        with self._mock_audit(_FINDINGS_SCAN):
            result = self.runner.invoke(audit_cmd, ["my_module:agent", "--json"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert "share_card" in data
        assert data["share_card"] is not None
        assert "CheckAgent audit:" in data["share_card"]

    def test_json_output_share_card_none_on_clean(self):
        with self._mock_audit(_CLEAN_SCAN):
            result = self.runner.invoke(audit_cmd, ["my_module:agent", "--json"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert "share_card" in data
        assert data["share_card"] is None

    def test_llm_judge_flag_passes_through(self):
        captured = {}

        def _capture(target, url, category, timeout, repeat, **kwargs):
            captured["llm_judge"] = kwargs.get("llm_judge")
            captured["agent_description"] = kwargs.get("agent_description")
            return _CLEAN_SCAN

        with patch("checkagent.cli.audit._run_scan_json", side_effect=_capture):
            result = self.runner.invoke(
                audit_cmd, [
                    "my_module:agent", "--llm-judge", "gpt-4o-mini",
                    "--agent-description", "HR assistant bot",
                ]
            )
        assert result.exit_code == 0
        assert captured["llm_judge"] == "gpt-4o-mini"
        assert captured["agent_description"] == "HR assistant bot"
