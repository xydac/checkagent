"""Tests for the checkagent dashboard command."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from checkagent.cli.dashboard_cmd import dashboard_cmd


def _write_history(tmp_path: Path, target: str, scores: list[float]) -> None:
    """Write fake scan history for a target under tmp_path/.checkagent/history/."""
    from checkagent.cli.history import _target_id

    tdir = tmp_path / ".checkagent" / "history" / _target_id(target)
    tdir.mkdir(parents=True, exist_ok=True)

    for i, score in enumerate(scores):
        ts = 1_700_000_000 + i * 3600
        passed = int(score * 35)
        failed = 35 - passed
        record = {
            "target": target,
            "timestamp": float(ts),
            "date": "2026-01-01",
            "time": "00:00:00 UTC",
            "summary": {
                "total": 35,
                "passed": passed,
                "failed": failed,
                "errors": 0,
                "score": round(score, 4),
                "elapsed_seconds": 1.0,
            },
        }
        fname = f"20260101-{i:06d}.json"
        (tdir / fname).write_text(json.dumps(record), encoding="utf-8")

    # latest.json = last score
    latest = {
        "target": target,
        "timestamp": float(1_700_000_000 + (len(scores) - 1) * 3600),
        "date": "2026-01-01",
        "time": "00:00:00 UTC",
        "summary": {
            "total": 35,
            "passed": int(scores[-1] * 35),
            "failed": 35 - int(scores[-1] * 35),
            "errors": 0,
            "score": round(scores[-1], 4),
            "elapsed_seconds": 1.0,
        },
    }
    (tdir / "latest.json").write_text(json.dumps(latest), encoding="utf-8")


class TestDashboardNoHistory:
    def test_empty_history_shows_message(self, tmp_path: Path) -> None:
        runner = CliRunner()
        result = runner.invoke(dashboard_cmd, ["--dir", str(tmp_path)])
        assert result.exit_code == 0
        assert "No scan history" in result.output

    def test_empty_history_json(self, tmp_path: Path) -> None:
        runner = CliRunner()
        result = runner.invoke(dashboard_cmd, ["--dir", str(tmp_path), "--json"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["agents"] == []
        assert data["total"] == 0


class TestDashboardWithHistory:
    def test_shows_all_agents(self, tmp_path: Path) -> None:
        _write_history(tmp_path, "agent_a:fn", [0.8])
        _write_history(tmp_path, "agent_b:fn", [0.5])
        runner = CliRunner()
        result = runner.invoke(dashboard_cmd, ["--dir", str(tmp_path)])
        assert result.exit_code == 0
        assert "agent_a:fn" in result.output
        assert "agent_b:fn" in result.output

    def test_json_output(self, tmp_path: Path) -> None:
        _write_history(tmp_path, "my_agent:fn", [0.6, 0.7, 0.8])
        runner = CliRunner()
        result = runner.invoke(dashboard_cmd, ["--dir", str(tmp_path), "--json"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["total"] == 1
        assert data["showing"] == 1
        assert data["agents"][0]["target"] == "my_agent:fn"
        assert data["agents"][0]["score"] == pytest.approx(0.8, abs=0.01)
        assert data["agents"][0]["scans"] == 3

    def test_lowest_score_first(self, tmp_path: Path) -> None:
        _write_history(tmp_path, "good_agent:fn", [0.9])
        _write_history(tmp_path, "bad_agent:fn", [0.2])
        runner = CliRunner()
        result = runner.invoke(dashboard_cmd, ["--dir", str(tmp_path), "--json"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        targets = [a["target"] for a in data["agents"]]
        assert targets[0] == "bad_agent:fn"
        assert targets[1] == "good_agent:fn"

    def test_top_limits_display(self, tmp_path: Path) -> None:
        for i in range(5):
            _write_history(tmp_path, f"agent_{i}:fn", [float(i) / 10.0 + 0.1])
        runner = CliRunner()
        result = runner.invoke(
            dashboard_cmd, ["--dir", str(tmp_path), "--json", "--top", "3"]
        )
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["total"] == 5
        assert data["showing"] == 3
        assert len(data["agents"]) == 3

    def test_trend_up_shown_in_table(self, tmp_path: Path) -> None:
        _write_history(tmp_path, "improving:fn", [0.4, 0.8])
        runner = CliRunner()
        result = runner.invoke(dashboard_cmd, ["--dir", str(tmp_path)])
        assert result.exit_code == 0
        # trend column should contain an up arrow for improving agent
        assert "↑" in result.output

    def test_trend_down_shown_in_table(self, tmp_path: Path) -> None:
        _write_history(tmp_path, "regressing:fn", [0.8, 0.4])
        runner = CliRunner()
        result = runner.invoke(dashboard_cmd, ["--dir", str(tmp_path)])
        assert result.exit_code == 0
        assert "↓" in result.output
