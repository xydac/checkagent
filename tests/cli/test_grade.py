"""Tests for the grade command — letter grading and percentile benchmarking."""

from __future__ import annotations

import json

from click.testing import CliRunner

from checkagent.cli.grade import (
    compute_percentile,
    format_grade_summary,
    grade_cmd,
    score_to_grade,
)


class TestScoreToGrade:
    def test_perfect_score(self):
        assert score_to_grade(1.0) == "A+"

    def test_a_plus(self):
        assert score_to_grade(0.97) == "A+"

    def test_a(self):
        assert score_to_grade(0.95) == "A"

    def test_a_minus(self):
        assert score_to_grade(0.90) == "A-"

    def test_b_plus(self):
        assert score_to_grade(0.87) == "B+"

    def test_b(self):
        assert score_to_grade(0.85) == "B"

    def test_b_minus(self):
        assert score_to_grade(0.80) == "B-"

    def test_c_plus(self):
        assert score_to_grade(0.77) == "C+"

    def test_c(self):
        assert score_to_grade(0.73) == "C"

    def test_c_minus(self):
        assert score_to_grade(0.70) == "C-"

    def test_d_plus(self):
        assert score_to_grade(0.67) == "D+"

    def test_d(self):
        assert score_to_grade(0.60) == "D"

    def test_d_minus(self):
        assert score_to_grade(0.50) == "D-"

    def test_f(self):
        assert score_to_grade(0.49) == "F"

    def test_zero(self):
        assert score_to_grade(0.0) == "F"


class TestComputePercentile:
    def test_top_score(self):
        pct = compute_percentile(0.99)
        assert pct == 100

    def test_bottom_score(self):
        pct = compute_percentile(0.0)
        assert pct == 0

    def test_middle_score(self):
        pct = compute_percentile(0.65)
        assert 0 < pct < 100

    def test_custom_corpus(self):
        corpus = [0.3, 0.5, 0.7, 0.9]
        assert compute_percentile(0.6, corpus) == 50

    def test_empty_corpus(self):
        assert compute_percentile(0.5, []) == 50


class TestFormatGradeSummary:
    def test_all_fields_present(self):
        result = format_grade_summary(0.85)
        assert "score" in result
        assert "grade" in result
        assert "percentile" in result
        assert "percentile_label" in result
        assert "benchmark_size" in result
        assert result["grade"] == "B"

    def test_explicit_grade(self):
        result = format_grade_summary(0.5, grade="custom")
        assert result["grade"] == "custom"

    def test_score_pct_format(self):
        result = format_grade_summary(0.73)
        assert result["score_pct"] == "73%"


class TestGradeCLI:
    def test_direct_score(self):
        runner = CliRunner()
        result = runner.invoke(grade_cmd, ["0.85"])
        assert result.exit_code == 0
        assert "B" in result.output

    def test_json_output(self):
        runner = CliRunner()
        result = runner.invoke(grade_cmd, ["0.73", "--json"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["grade"] == "C"
        assert data["percentile"] == 75

    def test_from_scan_file(self, tmp_path):
        scan_data = {"summary": {"score": 0.90}}
        scan_file = tmp_path / "scan.json"
        scan_file.write_text(json.dumps(scan_data))
        runner = CliRunner()
        result = runner.invoke(grade_cmd, ["--from-scan", str(scan_file)])
        assert result.exit_code == 0
        assert "A" in result.output

    def test_no_input_error(self):
        runner = CliRunner()
        result = runner.invoke(grade_cmd, [])
        assert result.exit_code == 1

    def test_score_and_file_error(self, tmp_path):
        scan_file = tmp_path / "scan.json"
        scan_file.write_text('{"summary": {"score": 0.5}}')
        runner = CliRunner()
        result = runner.invoke(grade_cmd, ["0.5", "--from-scan", str(scan_file)])
        assert result.exit_code == 1

    def test_stdin_json(self):
        runner = CliRunner()
        scan_data = json.dumps({"summary": {"score": 0.60}})
        result = runner.invoke(grade_cmd, ["--json"], input=scan_data)
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["grade"] == "D"
