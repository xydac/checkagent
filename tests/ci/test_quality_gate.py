"""Tests for quality gate evaluation (F5.3)."""

from __future__ import annotations

import pytest

from checkagent.ci.quality_gate import (
    GateResult,
    GateVerdict,
    QualityGateReport,
    evaluate_gate,
    evaluate_gates,
    scores_to_dict,
)
from checkagent.core.config import QualityGateEntry
from checkagent.core.types import Score


class TestEvaluateGate:
    """Tests for single gate evaluation."""

    def test_passes_when_value_meets_min(self):
        gate = QualityGateEntry(min=0.90, on_fail="block")
        result = evaluate_gate("accuracy", 0.95, gate)
        assert result.verdict == GateVerdict.PASSED
        assert result.metric == "accuracy"
        assert result.actual == 0.95

    def test_passes_when_value_equals_min(self):
        gate = QualityGateEntry(min=0.90, on_fail="block")
        result = evaluate_gate("accuracy", 0.90, gate)
        assert result.verdict == GateVerdict.PASSED

    def test_blocks_when_value_below_min(self):
        gate = QualityGateEntry(min=0.90, on_fail="block")
        result = evaluate_gate("accuracy", 0.85, gate)
        assert result.verdict == GateVerdict.BLOCKED
        assert result.actual == 0.85
        assert result.threshold == 0.90
        assert "failed" in result.message

    def test_warns_when_value_below_min(self):
        gate = QualityGateEntry(min=0.90, on_fail="warn")
        result = evaluate_gate("accuracy", 0.85, gate)
        assert result.verdict == GateVerdict.WARNED

    def test_ignores_when_value_below_min(self):
        gate = QualityGateEntry(min=0.90, on_fail="ignore")
        result = evaluate_gate("accuracy", 0.85, gate)
        assert result.verdict == GateVerdict.SKIPPED

    def test_passes_when_value_below_max(self):
        gate = QualityGateEntry(max=0.50, on_fail="block")
        result = evaluate_gate("cost", 0.30, gate)
        assert result.verdict == GateVerdict.PASSED

    def test_passes_when_value_equals_max(self):
        gate = QualityGateEntry(max=0.50, on_fail="block")
        result = evaluate_gate("cost", 0.50, gate)
        assert result.verdict == GateVerdict.PASSED

    def test_blocks_when_value_above_max(self):
        gate = QualityGateEntry(max=0.50, on_fail="block")
        result = evaluate_gate("cost", 0.75, gate)
        assert result.verdict == GateVerdict.BLOCKED
        assert result.direction == "max"

    def test_both_min_and_max_pass(self):
        gate = QualityGateEntry(min=0.50, max=1.00, on_fail="block")
        result = evaluate_gate("score", 0.75, gate)
        assert result.verdict == GateVerdict.PASSED

    def test_both_min_and_max_fail_min(self):
        gate = QualityGateEntry(min=0.50, max=1.00, on_fail="block")
        result = evaluate_gate("score", 0.30, gate)
        assert result.verdict == GateVerdict.BLOCKED

    def test_both_min_and_max_fail_max(self):
        gate = QualityGateEntry(min=0.50, max=1.00, on_fail="warn")
        result = evaluate_gate("score", 1.50, gate)
        assert result.verdict == GateVerdict.WARNED


class TestEvaluateGates:
    """Tests for multi-gate evaluation."""

    def test_all_pass(self):
        gates = {
            "accuracy": QualityGateEntry(min=0.90, on_fail="block"),
            "efficiency": QualityGateEntry(min=0.70, on_fail="warn"),
        }
        scores = {"accuracy": 0.95, "efficiency": 0.80}
        report = evaluate_gates(scores, gates)
        assert report.passed
        assert not report.has_warnings
        assert len(report.results) == 2

    def test_one_blocked(self):
        gates = {
            "accuracy": QualityGateEntry(min=0.90, on_fail="block"),
            "efficiency": QualityGateEntry(min=0.70, on_fail="warn"),
        }
        scores = {"accuracy": 0.50, "efficiency": 0.80}
        report = evaluate_gates(scores, gates)
        assert not report.passed
        assert len(report.blocked_gates) == 1
        assert report.blocked_gates[0].metric == "accuracy"

    def test_one_warned(self):
        gates = {
            "accuracy": QualityGateEntry(min=0.90, on_fail="block"),
            "efficiency": QualityGateEntry(min=0.70, on_fail="warn"),
        }
        scores = {"accuracy": 0.95, "efficiency": 0.50}
        report = evaluate_gates(scores, gates)
        assert report.passed  # warnings don't block
        assert report.has_warnings
        assert len(report.warned_gates) == 1

    def test_missing_metric_skipped(self):
        gates = {
            "accuracy": QualityGateEntry(min=0.90, on_fail="block"),
        }
        scores: dict[str, float] = {}
        report = evaluate_gates(scores, gates)
        assert report.passed  # skipped doesn't block
        assert len(report.results) == 1
        assert report.results[0].verdict == GateVerdict.SKIPPED

    def test_empty_gates(self):
        report = evaluate_gates({"accuracy": 0.95}, {})
        assert report.passed
        assert len(report.results) == 0


class TestQualityGateReport:
    """Tests for QualityGateReport properties."""

    def test_passed_gates(self):
        report = QualityGateReport(
            results=[
                GateResult(metric="a", verdict=GateVerdict.PASSED),
                GateResult(metric="b", verdict=GateVerdict.WARNED),
                GateResult(metric="c", verdict=GateVerdict.BLOCKED),
            ]
        )
        assert len(report.passed_gates) == 1
        assert len(report.warned_gates) == 1
        assert len(report.blocked_gates) == 1
        assert not report.passed


class TestScoresToDict:
    """Tests for Score list to dict conversion."""

    def test_converts_scores(self):
        scores = [
            Score(name="accuracy", value=0.95),
            Score(name="cost", value=0.30),
        ]
        d = scores_to_dict(scores)
        assert d == {"accuracy": 0.95, "cost": 0.30}

    def test_empty_list(self):
        assert scores_to_dict([]) == {}
