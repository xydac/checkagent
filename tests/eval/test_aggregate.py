"""Tests for aggregate scoring and regression detection."""

import json

import pytest

from checkagent.eval.aggregate import (
    AggregateResult,
    RegressionResult,
    RunSummary,
    StepStats,
    aggregate_scores,
    compute_step_stats,
    detect_regressions,
)


class TestAggregateScores:
    """Tests for aggregate_scores()."""

    def test_single_metric(self):
        scores = [
            ("task_completion", 1.0, True),
            ("task_completion", 0.5, False),
            ("task_completion", 0.75, True),
        ]
        result = aggregate_scores(scores)
        assert "task_completion" in result
        agg = result["task_completion"]
        assert agg.count == 3
        assert agg.mean == pytest.approx(0.75)
        assert agg.median == 0.75
        assert agg.min_value == 0.5
        assert agg.max_value == 1.0

    def test_multiple_metrics(self):
        scores = [
            ("task_completion", 1.0, True),
            ("tool_correctness", 0.8, True),
            ("task_completion", 0.5, False),
            ("tool_correctness", 0.6, True),
        ]
        result = aggregate_scores(scores)
        assert len(result) == 2
        assert result["task_completion"].count == 2
        assert result["tool_correctness"].count == 2

    def test_pass_rate(self):
        scores = [
            ("m", 1.0, True),
            ("m", 0.5, True),
            ("m", 0.0, False),
        ]
        result = aggregate_scores(scores)
        assert result["m"].pass_rate == pytest.approx(2 / 3)

    def test_pass_rate_none_when_no_flags(self):
        scores = [
            ("m", 0.5, None),
            ("m", 0.8, None),
        ]
        result = aggregate_scores(scores)
        assert result["m"].pass_rate is None

    def test_single_value_stdev_zero(self):
        scores = [("m", 0.5, True)]
        result = aggregate_scores(scores)
        assert result["m"].stdev == 0.0

    def test_empty_scores(self):
        result = aggregate_scores([])
        assert result == {}

    def test_to_dict(self):
        scores = [("m", 1.0, True), ("m", 0.5, False)]
        result = aggregate_scores(scores)
        d = result["m"].to_dict()
        assert d["metric_name"] == "m"
        assert "mean" in d
        assert "pass_rate" in d


class TestComputeStepStats:
    """Tests for compute_step_stats()."""

    def test_basic(self):
        stats = compute_step_stats([3, 5, 7, 2, 10])
        assert stats.count == 5
        assert stats.min_steps == 2
        assert stats.max_steps == 10
        assert stats.mean == pytest.approx(5.4)

    def test_p50_p95(self):
        # 20 values from 1-20
        counts = list(range(1, 21))
        stats = compute_step_stats(counts)
        assert stats.p50 == 10  # median of 1..20
        assert stats.p95 >= 19

    def test_single_value(self):
        stats = compute_step_stats([5])
        assert stats.p50 == 5
        assert stats.p95 == 5
        assert stats.mean == 5.0

    def test_empty(self):
        stats = compute_step_stats([])
        assert stats.count == 0
        assert stats.mean == 0.0

    def test_to_dict(self):
        stats = compute_step_stats([3, 5])
        d = stats.to_dict()
        assert "p50" in d
        assert "p95" in d
        assert "mean" in d


class TestDetectRegressions:
    """Tests for detect_regressions()."""

    def _make_agg(self, name: str, mean: float) -> AggregateResult:
        return AggregateResult(
            metric_name=name,
            count=10,
            mean=mean,
            median=mean,
            stdev=0.1,
            min_value=mean - 0.1,
            max_value=mean + 0.1,
        )

    def test_no_regression(self):
        current = {"m": self._make_agg("m", 0.9)}
        baseline = {"m": self._make_agg("m", 0.9)}
        results = detect_regressions(current, baseline)
        assert len(results) == 1
        assert not results[0].regressed

    def test_improvement_not_regression(self):
        current = {"m": self._make_agg("m", 0.95)}
        baseline = {"m": self._make_agg("m", 0.85)}
        results = detect_regressions(current, baseline)
        assert not results[0].regressed
        assert results[0].delta == pytest.approx(0.10)

    def test_small_drop_within_threshold(self):
        current = {"m": self._make_agg("m", 0.88)}
        baseline = {"m": self._make_agg("m", 0.90)}
        results = detect_regressions(current, baseline, threshold=0.05)
        assert not results[0].regressed  # -0.02 within 0.05

    def test_large_drop_is_regression(self):
        current = {"m": self._make_agg("m", 0.70)}
        baseline = {"m": self._make_agg("m", 0.90)}
        results = detect_regressions(current, baseline, threshold=0.05)
        assert results[0].regressed
        assert results[0].delta == pytest.approx(-0.20)

    def test_missing_baseline_metric_skipped(self):
        current = {"m1": self._make_agg("m1", 0.9), "m2": self._make_agg("m2", 0.8)}
        baseline = {"m1": self._make_agg("m1", 0.9)}
        results = detect_regressions(current, baseline)
        assert len(results) == 1
        assert results[0].metric_name == "m1"

    def test_custom_threshold(self):
        current = {"m": self._make_agg("m", 0.85)}
        baseline = {"m": self._make_agg("m", 0.90)}
        # -0.05 drop, threshold=0.10 → not regressed
        results = detect_regressions(current, baseline, threshold=0.10)
        assert not results[0].regressed

    def test_to_dict(self):
        current = {"m": self._make_agg("m", 0.70)}
        baseline = {"m": self._make_agg("m", 0.90)}
        results = detect_regressions(current, baseline)
        d = results[0].to_dict()
        assert d["regressed"] is True
        assert "delta" in d


class TestRunSummary:
    """Tests for RunSummary."""

    def test_has_regressions_false(self):
        summary = RunSummary()
        assert not summary.has_regressions

    def test_has_regressions_true(self):
        summary = RunSummary(
            regressions=[
                RegressionResult("m", 0.7, 0.9, -0.2, True, 0.05),
            ]
        )
        assert summary.has_regressions

    def test_to_dict(self):
        summary = RunSummary(
            aggregates={
                "m": AggregateResult("m", 10, 0.8, 0.8, 0.1, 0.5, 1.0, 0.9)
            },
            step_stats=StepStats(10, 5.0, 5, 9, 2, 10),
            total_cost=0.0123,
        )
        d = summary.to_dict()
        assert "aggregates" in d
        assert "step_stats" in d
        assert d["total_cost"] == 0.0123

    def test_save_and_load(self, tmp_path):
        summary = RunSummary(
            aggregates={
                "task_completion": AggregateResult(
                    "task_completion", 5, 0.8, 0.85, 0.1, 0.5, 1.0, 0.8
                ),
            },
            step_stats=StepStats(5, 4.2, 4, 7, 2, 8),
            total_cost=0.05,
        )
        path = tmp_path / "summary.json"
        summary.save(path)

        loaded = RunSummary.load(path)
        assert loaded.aggregates["task_completion"].mean == pytest.approx(0.8)
        assert loaded.step_stats is not None
        assert loaded.step_stats.p50 == 4
        assert loaded.total_cost == 0.05

    def test_save_creates_parent_dirs(self, tmp_path):
        path = tmp_path / "nested" / "dir" / "summary.json"
        summary = RunSummary()
        summary.save(path)
        assert path.exists()

    def test_load_minimal(self, tmp_path):
        path = tmp_path / "minimal.json"
        path.write_text(json.dumps({"aggregates": {}}))
        loaded = RunSummary.load(path)
        assert loaded.aggregates == {}
        assert loaded.step_stats is None
        assert loaded.total_cost is None

    def test_save_load_preserves_regressions(self, tmp_path):
        """F-035: regressions must survive save/load round-trip."""
        summary = RunSummary(
            aggregates={
                "task_completion": AggregateResult(
                    "task_completion", 5, 0.8, 0.85, 0.1, 0.5, 1.0, 0.8
                ),
            },
            regressions=[
                RegressionResult(
                    metric_name="task_completion",
                    current=0.6,
                    baseline=0.8,
                    delta=-0.2,
                    regressed=True,
                    threshold=0.1,
                ),
            ],
        )
        path = tmp_path / "summary.json"
        summary.save(path)

        loaded = RunSummary.load(path)
        assert len(loaded.regressions) == 1
        r = loaded.regressions[0]
        assert r.metric_name == "task_completion"
        assert r.regressed is True
        assert r.current == pytest.approx(0.6)
        assert r.baseline == pytest.approx(0.8)
        assert r.delta == pytest.approx(-0.2)
        assert r.threshold == pytest.approx(0.1)
