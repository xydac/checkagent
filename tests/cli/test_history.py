"""Tests for the scan history persistence module."""

from __future__ import annotations

import json
import time

import pytest

from checkagent.cli.history import (
    _target_id,
    compute_delta,
    format_delta_line,
    list_history,
    load_previous_result,
    save_scan_result,
)


class TestTargetId:
    def test_stable_across_calls(self):
        tid = _target_id("my_agent:fn")
        assert tid == _target_id("my_agent:fn")

    def test_different_targets_differ(self):
        assert _target_id("agent_a:fn") != _target_id("agent_b:fn")

    def test_length_is_12(self):
        assert len(_target_id("anything")) == 12


class TestSaveLoadResult:
    def test_save_creates_files(self, tmp_path):
        path = save_scan_result(
            "my_agent:fn",
            passed=80,
            failed=20,
            errors=0,
            total=100,
            elapsed=1.23,
            base_dir=tmp_path,
        )
        assert path.exists()
        latest = tmp_path / ".checkagent" / "history" / _target_id("my_agent:fn") / "latest.json"
        assert latest.exists()

    def test_saved_record_content(self, tmp_path):
        save_scan_result(
            "my_agent:fn",
            passed=80,
            failed=20,
            errors=0,
            total=100,
            elapsed=1.23,
            base_dir=tmp_path,
        )
        latest = tmp_path / ".checkagent" / "history" / _target_id("my_agent:fn") / "latest.json"
        record = json.loads(latest.read_text())
        assert record["target"] == "my_agent:fn"
        assert record["summary"]["passed"] == 80
        assert record["summary"]["failed"] == 20
        assert record["summary"]["total"] == 100
        assert record["summary"]["score"] == pytest.approx(0.8, abs=0.001)

    def test_load_returns_none_when_no_history(self, tmp_path):
        result = load_previous_result("never_scanned:fn", base_dir=tmp_path)
        assert result is None

    def test_load_returns_latest(self, tmp_path):
        save_scan_result(
            "my_agent:fn",
            passed=60,
            failed=40,
            errors=0,
            total=100,
            elapsed=1.0,
            base_dir=tmp_path,
        )
        result = load_previous_result("my_agent:fn", base_dir=tmp_path)
        assert result is not None
        assert result["summary"]["passed"] == 60

    def test_load_before_timestamp_skips_newer(self, tmp_path):
        ts1 = time.time() - 100
        ts2 = time.time() - 50
        save_scan_result(
            "my_agent:fn",
            passed=60,
            failed=40,
            errors=0,
            total=100,
            elapsed=1.0,
            timestamp=ts1,
            base_dir=tmp_path,
        )
        save_scan_result(
            "my_agent:fn",
            passed=70,
            failed=30,
            errors=0,
            total=100,
            elapsed=1.0,
            timestamp=ts2,
            base_dir=tmp_path,
        )
        # Request the result BEFORE ts2 — should get ts1 record
        result = load_previous_result("my_agent:fn", before_timestamp=ts2, base_dir=tmp_path)
        assert result is not None
        assert result["summary"]["passed"] == 60

    def test_load_before_timestamp_when_only_one(self, tmp_path):
        ts = time.time()
        save_scan_result(
            "my_agent:fn",
            passed=60,
            failed=40,
            errors=0,
            total=100,
            elapsed=1.0,
            timestamp=ts,
            base_dir=tmp_path,
        )
        # Only result IS at ts — nothing before it
        result = load_previous_result("my_agent:fn", before_timestamp=ts, base_dir=tmp_path)
        assert result is None


class TestComputeDelta:
    def _make_previous(self, passed: int, total: int, failed: int = 0) -> dict:
        score = passed / total if total > 0 else 0.0
        return {
            "date": "2026-04-27",
            "summary": {
                "passed": passed,
                "failed": failed,
                "total": total,
                "score": score,
            },
        }

    def test_improvement(self):
        prev = self._make_previous(70, 100)
        delta = compute_delta(80, 100, prev)
        assert delta["score_delta"] == pytest.approx(0.1, abs=0.001)
        assert delta["previous_score"] == pytest.approx(0.7, abs=0.001)
        assert delta["current_score"] == pytest.approx(0.8, abs=0.001)

    def test_regression(self):
        prev = self._make_previous(80, 100)
        delta = compute_delta(70, 100, prev)
        assert delta["score_delta"] == pytest.approx(-0.1, abs=0.001)

    def test_no_change(self):
        prev = self._make_previous(80, 100)
        delta = compute_delta(80, 100, prev)
        assert delta["score_delta"] == pytest.approx(0.0, abs=0.001)

    def test_no_change_not_negative_zero(self):
        """score_delta must be +0.0, never -0.0 (F-118)."""
        import json
        prev = self._make_previous(80, 100)
        delta = compute_delta(80, 100, prev)
        serialized = json.dumps({"score_delta": delta["score_delta"]})
        assert "-0.0" not in serialized

    def test_previous_date_preserved(self):
        prev = self._make_previous(70, 100)
        delta = compute_delta(80, 100, prev)
        assert delta["previous_date"] == "2026-04-27"


class TestFormatDeltaLine:
    def _make_delta(self, score_delta: float, prev_score: float, curr_score: float) -> dict:
        return {
            "score_delta": score_delta,
            "previous_score": prev_score,
            "current_score": curr_score,
            "previous_date": "2026-04-27",
            "previous_failed": 0,
            "previous_passed": 0,
            "previous_total": 0,
        }

    def test_improvement_shows_up_arrow(self):
        delta = self._make_delta(0.1, 0.7, 0.8)
        line = format_delta_line(delta)
        assert "↑" in line
        assert "+10%" in line

    def test_regression_shows_down_arrow(self):
        delta = self._make_delta(-0.1, 0.8, 0.7)
        line = format_delta_line(delta)
        assert "↓" in line
        assert "-10%" in line

    def test_no_change_shows_arrow(self):
        delta = self._make_delta(0.0, 0.8, 0.8)
        line = format_delta_line(delta)
        assert "→" in line
        assert "no change" in line

    def test_includes_previous_date(self):
        delta = self._make_delta(0.05, 0.75, 0.8)
        line = format_delta_line(delta)
        assert "2026-04-27" in line


class TestListHistory:
    def test_empty_when_no_history(self, tmp_path):
        results = list_history("my_agent:fn", base_dir=tmp_path)
        assert results == []

    def test_returns_records_newest_first(self, tmp_path):
        ts_old = time.time() - 200
        ts_new = time.time() - 100
        save_scan_result(
            "my_agent:fn",
            passed=60,
            failed=40,
            errors=0,
            total=100,
            elapsed=1.0,
            timestamp=ts_old,
            base_dir=tmp_path,
        )
        save_scan_result(
            "my_agent:fn",
            passed=70,
            failed=30,
            errors=0,
            total=100,
            elapsed=1.0,
            timestamp=ts_new,
            base_dir=tmp_path,
        )
        results = list_history("my_agent:fn", base_dir=tmp_path)
        assert len(results) == 2
        # Newest first
        assert results[0]["summary"]["passed"] == 70
        assert results[1]["summary"]["passed"] == 60

    def test_limit_respected(self, tmp_path):
        for i in range(5):
            save_scan_result(
                "my_agent:fn",
                passed=i * 10,
                failed=100 - i * 10,
                errors=0,
                total=100,
                elapsed=1.0,
                timestamp=time.time() - (5 - i) * 10,
                base_dir=tmp_path,
            )
        results = list_history("my_agent:fn", limit=3, base_dir=tmp_path)
        assert len(results) == 3


class TestHistoryCli:
    def test_no_history_shows_message(self, tmp_path):
        from click.testing import CliRunner

        from checkagent.cli.history_cmd import history_cmd

        runner = CliRunner()
        result = runner.invoke(history_cmd, ["my_agent:fn", "--dir", str(tmp_path)])
        assert result.exit_code == 0
        assert "No scan history" in result.output

    def test_shows_table_with_history(self, tmp_path):
        from click.testing import CliRunner

        from checkagent.cli.history_cmd import history_cmd

        save_scan_result(
            "my_agent:fn",
            passed=80,
            failed=20,
            errors=0,
            total=100,
            elapsed=1.0,
            base_dir=tmp_path,
        )
        runner = CliRunner()
        result = runner.invoke(history_cmd, ["my_agent:fn", "--dir", str(tmp_path)])
        assert result.exit_code == 0
        assert "80%" in result.output
        assert "Scan history" in result.output

    def test_limit_flag(self, tmp_path):
        from click.testing import CliRunner

        from checkagent.cli.history_cmd import history_cmd

        for i in range(5):
            save_scan_result(
                "my_agent:fn",
                passed=i * 20,
                failed=100 - i * 20,
                errors=0,
                total=100,
                elapsed=1.0,
                timestamp=time.time() - (5 - i) * 10,
                base_dir=tmp_path,
            )
        runner = CliRunner()
        result = runner.invoke(
            history_cmd, ["my_agent:fn", "--dir", str(tmp_path), "--limit", "2"]
        )
        assert result.exit_code == 0
        assert "2 scan(s)" in result.output
