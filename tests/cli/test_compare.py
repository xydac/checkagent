"""Tests for checkagent compare command."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from checkagent.cli.compare import build_comparison, compare_cmd
from checkagent.cli.history import save_scan_result


def _make_finding(category: str = "prompt_injection", desc: str = "test") -> dict:
    return {"category": category, "description": desc, "severity": "high", "probe": desc}


def _save(
    target: str,
    base: Path,
    passed: int = 90,
    failed: int = 10,
    findings: list | None = None,
) -> None:
    total = passed + failed
    save_scan_result(
        target,
        passed=passed,
        failed=failed,
        errors=0,
        total=total,
        elapsed=1.0,
        base_dir=base,
        findings=findings or [],
    )


class TestBuildComparison:
    def test_basic_comparison(self) -> None:
        a = {
            "target": "a:fn",
            "date": "2026-07-14",
            "summary": {"score": 0.9, "passed": 90, "failed": 10, "total": 100},
            "findings": [_make_finding("prompt_injection", "inj1")],
        }
        b = {
            "target": "b:fn",
            "date": "2026-07-14",
            "summary": {"score": 0.8, "passed": 80, "failed": 20, "total": 100},
            "findings": [
                _make_finding("prompt_injection", "inj1"),
                _make_finding("pii_leakage", "pii1"),
            ],
        }
        result = build_comparison(a, b)
        assert result["winner"] == "agent_a"
        assert result["score_delta"] == pytest.approx(-0.1)
        assert result["agent_a"]["score"] == 0.9
        assert result["agent_b"]["score"] == 0.8

    def test_tie(self) -> None:
        a = {"target": "a", "summary": {"score": 0.5, "passed": 50, "failed": 50, "total": 100}}
        b = {"target": "b", "summary": {"score": 0.5, "passed": 50, "failed": 50, "total": 100}}
        result = build_comparison(a, b)
        assert result["winner"] == "tie"

    def test_category_breakdown(self) -> None:
        a = {
            "target": "a",
            "summary": {"score": 0.9, "passed": 90, "failed": 10, "total": 100},
            "findings": [_make_finding("prompt_injection"), _make_finding("prompt_injection")],
        }
        b = {
            "target": "b",
            "summary": {"score": 0.95, "passed": 95, "failed": 5, "total": 100},
            "findings": [_make_finding("pii_leakage")],
        }
        result = build_comparison(a, b)
        cats = {c["category"]: c for c in result["categories"]}
        assert "prompt_injection" in cats
        assert "pii_leakage" in cats
        assert cats["prompt_injection"]["agent_a_findings"] == 2
        assert cats["prompt_injection"]["agent_b_findings"] == 0

    def test_unique_findings(self) -> None:
        a = {
            "target": "a",
            "summary": {"score": 0.9, "passed": 90, "failed": 10, "total": 100},
            "findings": [
                _make_finding("prompt_injection", "shared"),
                _make_finding("pii_leakage", "only_a"),
            ],
        }
        b = {
            "target": "b",
            "summary": {"score": 0.9, "passed": 90, "failed": 10, "total": 100},
            "findings": [
                _make_finding("prompt_injection", "shared"),
                _make_finding("jailbreak", "only_b"),
            ],
        }
        result = build_comparison(a, b)
        assert "only_a" in result["only_agent_a"]
        assert "only_b" in result["only_agent_b"]
        assert "shared" not in result["only_agent_a"]

    def test_empty_findings(self) -> None:
        a = {"target": "a", "summary": {"score": 1.0, "passed": 100, "failed": 0, "total": 100}}
        b = {"target": "b", "summary": {"score": 1.0, "passed": 100, "failed": 0, "total": 100}}
        result = build_comparison(a, b)
        assert result["categories"] == []
        assert result["only_agent_a"] == []
        assert result["only_agent_b"] == []

    def test_probe_id_findings_f153(self) -> None:
        """F-153: probe_id-keyed findings (real scan format) populate only_agent_* correctly."""
        def _pid(probe_id: str, category: str = "prompt_injection") -> dict:
            return {
                "probe_id": probe_id,
                "category": category,
                "severity": "high",
                "finding": "test",
                "probe_input": "test",
                "response": "test",
            }

        a = {
            "target": "a",
            "summary": {"score": 0.5, "passed": 50, "failed": 50, "total": 100},
            "findings": [_pid("probe-shared"), _pid("probe-only-a", "pii_leakage")],
        }
        b = {
            "target": "b",
            "summary": {"score": 0.5, "passed": 50, "failed": 50, "total": 100},
            "findings": [_pid("probe-shared"), _pid("probe-only-b", "jailbreak")],
        }
        result = build_comparison(a, b)
        assert "probe-only-a" in result["only_agent_a"]
        assert "probe-only-b" in result["only_agent_b"]
        assert "probe-shared" not in result["only_agent_a"]
        assert "probe-shared" not in result["only_agent_b"]
        assert "" not in result["only_agent_a"]
        assert "" not in result["only_agent_b"]

    def test_no_empty_string_when_no_probe_key(self) -> None:
        """F-153: findings lacking probe_id/probe/description don't add '' to only_agent_*."""
        a = {
            "target": "a",
            "summary": {"score": 0.0, "passed": 0, "failed": 35, "total": 35},
            "findings": [
                {"category": "prompt_injection", "severity": "high", "finding": "x"}
                for _ in range(35)
            ],
        }
        b = {
            "target": "b",
            "summary": {"score": 1.0, "passed": 35, "failed": 0, "total": 35},
            "findings": [],
        }
        result = build_comparison(a, b)
        assert "" not in result["only_agent_a"]
        assert result["only_agent_a"] == []


class TestCompareCmd:
    def test_compare_with_history(self, tmp_path: Path) -> None:
        _save("agent_a:fn", tmp_path, passed=90, failed=10, findings=[_make_finding()])
        _save(
            "agent_b:fn", tmp_path, passed=80, failed=20,
            findings=[_make_finding(), _make_finding("pii_leakage")],
        )

        runner = CliRunner()
        result = runner.invoke(compare_cmd, [
            "agent_a:fn", "agent_b:fn",
            "--base-dir", str(tmp_path),
        ])
        assert result.exit_code == 0
        assert "Safety Comparison" in result.output
        assert "Winner" in result.output

    def test_compare_json_output(self, tmp_path: Path) -> None:
        _save("a:fn", tmp_path, passed=95, failed=5)
        _save("b:fn", tmp_path, passed=80, failed=20)

        runner = CliRunner()
        result = runner.invoke(compare_cmd, [
            "a:fn", "b:fn",
            "--base-dir", str(tmp_path),
            "--json",
        ])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["winner"] == "agent_a"
        assert "score_delta" in data

    def test_missing_target_a(self, tmp_path: Path) -> None:
        _save("b:fn", tmp_path)
        runner = CliRunner()
        result = runner.invoke(compare_cmd, [
            "missing:fn", "b:fn",
            "--base-dir", str(tmp_path),
        ])
        assert result.exit_code != 0
        assert "No scan history" in result.output

    def test_missing_target_b(self, tmp_path: Path) -> None:
        _save("a:fn", tmp_path)
        runner = CliRunner()
        result = runner.invoke(compare_cmd, [
            "a:fn", "missing:fn",
            "--base-dir", str(tmp_path),
        ])
        assert result.exit_code != 0
        assert "No scan history" in result.output

    def test_tie_display(self, tmp_path: Path) -> None:
        _save("a:fn", tmp_path, passed=50, failed=50)
        _save("b:fn", tmp_path, passed=50, failed=50)
        runner = CliRunner()
        result = runner.invoke(compare_cmd, [
            "a:fn", "b:fn",
            "--base-dir", str(tmp_path),
        ])
        assert result.exit_code == 0
        assert "Tie" in result.output

    def test_compare_in_help(self) -> None:
        runner = CliRunner()
        result = runner.invoke(compare_cmd, ["--help"])
        assert "Compare safety scan results" in result.output

    def test_url_a_url_b_flags_f155(self, tmp_path: Path) -> None:
        """F-155: --url-a/--url-b work as alternatives to positional arguments."""
        _save("http://a/chat", tmp_path, passed=90, failed=10)
        _save("http://b/chat", tmp_path, passed=80, failed=20)

        runner = CliRunner()
        result = runner.invoke(compare_cmd, [
            "--url-a", "http://a/chat",
            "--url-b", "http://b/chat",
            "--base-dir", str(tmp_path),
        ])
        assert result.exit_code == 0
        assert "Comparison" in result.output or "Winner" in result.output

    def test_url_a_url_b_json_output(self, tmp_path: Path) -> None:
        """F-155: --url-a/--url-b with --json produces valid output."""
        _save("http://a/chat", tmp_path, passed=95, failed=5)
        _save("http://b/chat", tmp_path, passed=80, failed=20)

        runner = CliRunner()
        result = runner.invoke(compare_cmd, [
            "--url-a", "http://a/chat",
            "--url-b", "http://b/chat",
            "--base-dir", str(tmp_path),
            "--json",
        ])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["winner"] == "agent_a"
        assert "margin" in data

    def test_missing_both_targets_raises(self, tmp_path: Path) -> None:
        """F-155: no target_a at all → UsageError with helpful message."""
        runner = CliRunner()
        result = runner.invoke(compare_cmd, ["--base-dir", str(tmp_path)])
        assert result.exit_code != 0
        assert "TARGET_A" in result.output or "url-a" in result.output

    def test_margin_field_f156(self) -> None:
        """F-156: margin field is always non-negative, representing absolute score diff."""
        a = {
            "target": "a:fn",
            "summary": {"score": 1.0, "passed": 100, "failed": 0, "total": 100},
        }
        b = {
            "target": "b:fn",
            "summary": {"score": 0.03, "passed": 3, "failed": 97, "total": 100},
        }
        result = build_comparison(a, b)
        # score_delta = b - a = 0.03 - 1.0 (negative when a wins)
        assert result["score_delta"] < 0
        # margin is always positive
        assert result["margin"] > 0
        assert result["margin"] == pytest.approx(abs(result["score_delta"]))
        assert result["winner"] == "agent_a"
