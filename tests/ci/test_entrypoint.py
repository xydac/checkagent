"""Tests for CI entrypoint (JUnit parsing, report generation)."""

from __future__ import annotations

from pathlib import Path

import pytest

from checkagent.ci.entrypoint import main, parse_junit_xml

SAMPLE_JUNIT = """\
<?xml version="1.0" encoding="utf-8"?>
<testsuites>
  <testsuite name="pytest" errors="0" failures="1" skipped="2" tests="10" time="3.456">
    <testcase classname="tests.test_agent" name="test_basic" time="0.1"/>
    <testcase classname="tests.test_agent" name="test_fail" time="0.2">
      <failure message="assert False"/>
    </testcase>
  </testsuite>
</testsuites>
"""

SAMPLE_BARE_SUITE = """\
<?xml version="1.0" encoding="utf-8"?>
<testsuite name="pytest" errors="1" failures="0" skipped="0" tests="5" time="1.0">
</testsuite>
"""


class TestParseJunitXml:
    """Tests for JUnit XML parsing."""

    def test_parses_testsuites_wrapper(self, tmp_path: Path):
        xml_path = tmp_path / "results.xml"
        xml_path.write_text(SAMPLE_JUNIT)
        summary = parse_junit_xml(xml_path)
        assert summary.total == 10
        assert summary.failed == 1
        assert summary.skipped == 2
        assert summary.passed == 7
        assert summary.errors == 0
        assert summary.duration_s == pytest.approx(3.456)

    def test_parses_bare_testsuite(self, tmp_path: Path):
        xml_path = tmp_path / "results.xml"
        xml_path.write_text(SAMPLE_BARE_SUITE)
        summary = parse_junit_xml(xml_path)
        assert summary.total == 5
        assert summary.errors == 1
        assert summary.passed == 4

    def test_pass_rate(self, tmp_path: Path):
        xml_path = tmp_path / "results.xml"
        xml_path.write_text(SAMPLE_JUNIT)
        summary = parse_junit_xml(xml_path)
        assert summary.pass_rate == pytest.approx(0.7)


class TestMain:
    """Tests for the CLI entrypoint."""

    def test_generates_report_from_junit(self, tmp_path: Path):
        xml_path = tmp_path / "results.xml"
        xml_path.write_text(SAMPLE_JUNIT)
        output_dir = tmp_path / "report"

        exit_code = main(["--junit", str(xml_path), "--output-dir", str(output_dir)])
        assert exit_code == 0
        assert (output_dir / "summary.md").exists()

        content = (output_dir / "summary.md").read_text()
        assert "CheckAgent Test Report" in content
        assert "Test Results" in content

    def test_no_junit_produces_minimal_report(self, tmp_path: Path):
        output_dir = tmp_path / "report"
        exit_code = main(["--output-dir", str(output_dir)])
        assert exit_code == 0
        assert (output_dir / "summary.md").exists()

    def test_quality_gate_block_returns_nonzero(self, tmp_path: Path):
        # Create config with strict gate
        config_path = tmp_path / "checkagent.yml"
        config_path.write_text(
            "version: 1\n"
            "quality_gates:\n"
            "  pass_rate:\n"
            "    min: 0.95\n"
            "    on_fail: block\n"
        )
        xml_path = tmp_path / "results.xml"
        xml_path.write_text(SAMPLE_JUNIT)  # pass rate is 0.7
        output_dir = tmp_path / "report"

        exit_code = main([
            "--junit", str(xml_path),
            "--config", str(config_path),
            "--output-dir", str(output_dir),
        ])
        assert exit_code == 1  # gate blocked
