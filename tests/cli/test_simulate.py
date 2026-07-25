"""Tests for the simulate command — attack chain simulation."""

from __future__ import annotations

import json

from click.testing import CliRunner

from checkagent.cli.simulate import (
    _coverage_gaps,  # noqa: E402
    _find_applicable_chains,  # noqa: E402
    _risk_level,  # noqa: E402
    simulate_attacks,
    simulate_cmd,
)

# -- Unit tests for chain logic --

class TestFindApplicableChains:
    def test_injection_plus_leak_matches(self):
        cats = {"system_prompt_leak", "prompt_injection"}
        chains = _find_applicable_chains(cats)
        names = [c["name"] for c in chains]
        assert "Prompt Exfiltration → Targeted Injection" in names

    def test_injection_plus_data_enum(self):
        cats = {"prompt_injection", "data_enumeration"}
        chains = _find_applicable_chains(cats)
        assert any("Data Exfiltration" in c["name"] for c in chains)

    def test_no_match_for_single_category(self):
        cats = {"prompt_injection"}
        chains = _find_applicable_chains(cats)
        assert len(chains) == 0

    def test_superset_categories_match_all_subsets(self):
        cats = {
            "prompt_injection", "system_prompt_leak", "data_enumeration",
            "jailbreak", "pii_leakage", "scope_violation", "tool_boundary",
        }
        chains = _find_applicable_chains(cats)
        assert len(chains) >= 5

    def test_empty_categories(self):
        chains = _find_applicable_chains(set())
        assert chains == []

    def test_sorted_by_impact_then_steps(self):
        cats = {
            "prompt_injection", "system_prompt_leak", "data_enumeration",
            "scope_violation",
        }
        chains = _find_applicable_chains(cats)
        if len(chains) >= 2:
            for i in range(len(chains) - 1):
                sev = {"critical": 4, "high": 3, "medium": 2, "low": 1}
                a_impact = sev[chains[i]["impact"]]
                b_impact = sev[chains[i + 1]["impact"]]
                assert a_impact >= b_impact


class TestCoverageGaps:
    def test_near_miss_detected(self):
        cats = {"system_prompt_leak"}
        gaps = _coverage_gaps(cats)
        assert len(gaps) > 0
        assert any("prompt_injection" in g["missing_category"] for g in gaps)

    def test_no_near_miss_when_fully_secure(self):
        gaps = _coverage_gaps(set())
        assert len(gaps) == 0

    def test_no_near_miss_when_fully_vulnerable(self):
        cats = {
            "prompt_injection", "system_prompt_leak", "data_enumeration",
            "jailbreak", "pii_leakage", "scope_violation", "tool_boundary",
        }
        gaps = _coverage_gaps(cats)
        assert len(gaps) == 0


class TestRiskLevel:
    def test_critical_with_critical_chains(self):
        chains = [{"impact": "critical"}]
        assert _risk_level(chains) == "critical"

    def test_high_with_high_chains(self):
        chains = [{"impact": "high"}]
        assert _risk_level(chains) == "high"

    def test_low_with_no_chains(self):
        assert _risk_level([]) == "low"


class TestSimulateAttacks:
    def test_full_simulation(self):
        scan_data = {
            "summary": {"total": 50, "score": 0.5},
            "findings": [
                {"category": "prompt_injection", "severity": "critical", "probe_id": "pi-001"},
                {"category": "prompt_injection", "severity": "high", "probe_id": "pi-002"},
                {"category": "system_prompt_leak", "severity": "high", "probe_id": "sp-001"},
                {"category": "data_enumeration", "severity": "medium", "probe_id": "de-001"},
            ],
        }
        result = simulate_attacks(scan_data)
        assert result["risk_level"] == "critical"
        assert result["chain_count"] >= 2
        assert result["finding_count"] == 4
        assert "prompt_injection" in result["categories_found"]

    def test_evidence_probes_attached(self):
        scan_data = {
            "findings": [
                {"category": "prompt_injection", "probe_id": "pi-001"},
                {"category": "system_prompt_leak", "probe_id": "sp-001"},
            ],
        }
        result = simulate_attacks(scan_data)
        for chain in result["chains"]:
            for step in chain["steps"]:
                if step["category"] in ("prompt_injection", "system_prompt_leak"):
                    assert step["evidence_count"] >= 1

    def test_no_findings_returns_low_risk(self):
        result = simulate_attacks({"findings": []})
        assert result["risk_level"] == "low"
        assert result["chain_count"] == 0

    def test_near_misses_included(self):
        scan_data = {
            "findings": [
                {"category": "system_prompt_leak", "severity": "high"},
            ],
        }
        result = simulate_attacks(scan_data)
        assert len(result["near_misses"]) > 0

    def test_single_category_no_chains(self):
        scan_data = {
            "findings": [
                {"category": "pii_leakage", "severity": "medium"},
            ],
        }
        result = simulate_attacks(scan_data)
        assert result["chain_count"] == 0


# -- CLI tests --

class TestSimulateCLI:
    def test_from_file_terminal(self, tmp_path):
        scan_data = {
            "summary": {"total": 20, "score": 0.6},
            "findings": [
                {"category": "prompt_injection", "severity": "critical", "probe_id": "pi-001"},
                {"category": "system_prompt_leak", "severity": "high", "probe_id": "sp-001"},
            ],
        }
        f = tmp_path / "scan.json"
        f.write_text(json.dumps(scan_data))
        runner = CliRunner()
        result = runner.invoke(simulate_cmd, [str(f)])
        assert result.exit_code == 0
        assert "Attack Simulation" in result.output
        assert "Chain 1" in result.output

    def test_json_output(self, tmp_path):
        scan_data = {
            "summary": {"total": 20, "score": 0.6},
            "findings": [
                {"category": "prompt_injection", "severity": "critical"},
                {"category": "data_enumeration", "severity": "medium"},
            ],
        }
        f = tmp_path / "scan.json"
        f.write_text(json.dumps(scan_data))
        runner = CliRunner()
        result = runner.invoke(simulate_cmd, [str(f), "--json"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert "chains" in data
        assert data["risk_level"] == "critical"
        assert data["chain_count"] >= 1

    def test_chain_filter(self, tmp_path):
        scan_data = {
            "summary": {"total": 20, "score": 0.6},
            "findings": [
                {"category": "prompt_injection", "severity": "critical"},
                {"category": "system_prompt_leak", "severity": "high"},
                {"category": "data_enumeration", "severity": "medium"},
            ],
        }
        f = tmp_path / "scan.json"
        f.write_text(json.dumps(scan_data))
        runner = CliRunner()
        result = runner.invoke(simulate_cmd, [str(f), "--json", "--chain", "chain-01"])
        data = json.loads(result.output)
        assert data["chain_count"] == 1
        assert data["chains"][0]["id"] == "chain-01"

    def test_clean_scan_no_chains(self, tmp_path):
        scan_data = {"summary": {"total": 10, "score": 1.0}, "findings": []}
        f = tmp_path / "scan.json"
        f.write_text(json.dumps(scan_data))
        runner = CliRunner()
        result = runner.invoke(simulate_cmd, [str(f)])
        assert result.exit_code == 0
        assert "No multi-step attack chains" in result.output

    def test_stdin_input(self):
        scan_data = json.dumps({
            "summary": {"total": 5, "score": 0.6},
            "findings": [
                {"category": "jailbreak", "severity": "high"},
                {"category": "pii_leakage", "severity": "high"},
            ],
        })
        runner = CliRunner()
        result = runner.invoke(simulate_cmd, ["--json"], input=scan_data)
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["chain_count"] >= 1

    def test_no_input_error(self):
        runner = CliRunner()
        result = runner.invoke(simulate_cmd, [])
        assert result.exit_code == 1

    def test_near_misses_shown(self, tmp_path):
        scan_data = {
            "summary": {"total": 5, "score": 0.8},
            "findings": [
                {"category": "system_prompt_leak", "severity": "medium"},
            ],
        }
        f = tmp_path / "scan.json"
        f.write_text(json.dumps(scan_data))
        runner = CliRunner()
        result = runner.invoke(simulate_cmd, [str(f)])
        assert result.exit_code == 0
        assert "Near Misses" in result.output
