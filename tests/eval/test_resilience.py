"""Tests for the resilience scoring module."""

from __future__ import annotations

import pytest

from checkagent.core.types import AgentRun, Score, Step, ToolCall
from checkagent.eval.resilience import ResilienceProfile, ScenarioResult

# ---- from_scores tests ----


class TestFromScores:
    def test_perfect_resilience(self):
        """Agent performs identically under faults."""
        profile = ResilienceProfile.from_scores(
            baseline={"task_completion": 1.0, "tool_correctness": 1.0},
            scenarios={
                "timeout": {"task_completion": 1.0, "tool_correctness": 1.0},
            },
        )
        assert profile.overall == 1.0
        assert profile.scenario_results["timeout"].resilience == 1.0

    def test_total_failure(self):
        """Agent completely fails under faults."""
        profile = ResilienceProfile.from_scores(
            baseline={"task_completion": 1.0, "tool_correctness": 1.0},
            scenarios={
                "timeout": {"task_completion": 0.0, "tool_correctness": 0.0},
            },
        )
        assert profile.overall == 0.0
        assert profile.scenario_results["timeout"].resilience == 0.0

    def test_partial_degradation(self):
        """Agent partially degrades — resilience is between 0 and 1."""
        profile = ResilienceProfile.from_scores(
            baseline={"task_completion": 1.0, "tool_correctness": 1.0},
            scenarios={
                "rate_limit": {"task_completion": 0.5, "tool_correctness": 0.8},
            },
        )
        # Resilience = mean(0.5/1.0, 0.8/1.0) = mean(0.5, 0.8) = 0.65
        assert profile.overall == pytest.approx(0.65)

    def test_multiple_scenarios(self):
        """Overall resilience is mean across scenarios."""
        profile = ResilienceProfile.from_scores(
            baseline={"tc": 1.0},
            scenarios={
                "timeout": {"tc": 0.8},
                "rate_limit": {"tc": 0.4},
            },
        )
        # timeout resilience = 0.8, rate_limit = 0.4, mean = 0.6
        assert profile.overall == pytest.approx(0.6)

    def test_degradation_values(self):
        """Degradation is faulted - baseline (negative means worse)."""
        profile = ResilienceProfile.from_scores(
            baseline={"tc": 0.9, "tools": 1.0},
            scenarios={
                "fault": {"tc": 0.6, "tools": 0.5},
            },
        )
        deg = profile.scenario_results["fault"].degradation
        assert deg["tc"] == pytest.approx(-0.3)
        assert deg["tools"] == pytest.approx(-0.5)

    def test_zero_baseline_no_crash(self):
        """Zero baseline score doesn't cause division by zero."""
        profile = ResilienceProfile.from_scores(
            baseline={"tc": 0.0},
            scenarios={"fault": {"tc": 0.0}},
        )
        assert profile.overall == 1.0  # both zero = no degradation

    def test_zero_baseline_positive_faulted(self):
        """Faulted score better than zero baseline — resilience is 1.0."""
        profile = ResilienceProfile.from_scores(
            baseline={"tc": 0.0},
            scenarios={"fault": {"tc": 0.5}},
        )
        assert profile.overall == 1.0

    def test_faulted_score_better_than_baseline_capped(self):
        """Improvement under fault is capped at 1.0 resilience."""
        profile = ResilienceProfile.from_scores(
            baseline={"tc": 0.5},
            scenarios={"fault": {"tc": 0.9}},
        )
        # 0.9/0.5 = 1.8 but capped at 1.0
        assert profile.overall == 1.0

    def test_missing_faulted_metric_treated_as_zero(self):
        """If a metric is missing from faulted scores, assume 0."""
        profile = ResilienceProfile.from_scores(
            baseline={"tc": 1.0, "tools": 1.0},
            scenarios={"fault": {"tc": 0.8}},  # tools missing
        )
        result = profile.scenario_results["fault"]
        assert result.scores.get("tools") is None
        assert result.degradation["tools"] == pytest.approx(-1.0)
        # Resilience = mean(0.8, 0.0) = 0.4
        assert result.resilience == pytest.approx(0.4)

    def test_no_scenarios(self):
        """No fault scenarios means perfect resilience (nothing to degrade)."""
        profile = ResilienceProfile.from_scores(
            baseline={"tc": 0.9},
            scenarios={},
        )
        assert profile.overall == 1.0
        assert profile.worst_scenario is None
        assert profile.best_scenario is None


class TestProperties:
    def _make_profile(self):
        return ResilienceProfile.from_scores(
            baseline={"tc": 1.0, "tools": 0.8, "efficiency": 0.9},
            scenarios={
                "timeout": {"tc": 0.3, "tools": 0.7, "efficiency": 0.8},
                "rate_limit": {"tc": 0.8, "tools": 0.6, "efficiency": 0.9},
                "server_error": {"tc": 0.0, "tools": 0.0, "efficiency": 0.0},
            },
        )

    def test_worst_scenario(self):
        profile = self._make_profile()
        assert profile.worst_scenario == "server_error"

    def test_best_scenario(self):
        profile = self._make_profile()
        assert profile.best_scenario == "rate_limit"

    def test_weakest_metric(self):
        """Weakest metric has the most total degradation across scenarios."""
        profile = self._make_profile()
        # tc: (0.3-1.0) + (0.8-1.0) + (0.0-1.0) = -0.7 + -0.2 + -1.0 = -1.9
        # tools: (0.7-0.8) + (0.6-0.8) + (0.0-0.8) = -0.1 + -0.2 + -0.8 = -1.1
        # efficiency: (0.8-0.9) + (0.9-0.9) + (0.0-0.9) = -0.1 + 0.0 + -0.9 = -1.0
        assert profile.weakest_metric == "tc"

    def test_most_resilient_metric(self):
        profile = self._make_profile()
        assert profile.most_resilient_metric == "efficiency"

    def test_empty_profile_properties(self):
        profile = ResilienceProfile(baseline={})
        assert profile.overall == 1.0
        assert profile.worst_scenario is None
        assert profile.best_scenario is None
        assert profile.weakest_metric is None
        assert profile.most_resilient_metric is None


class TestToDict:
    def test_serialization_roundtrip(self):
        profile = ResilienceProfile.from_scores(
            baseline={"tc": 1.0},
            scenarios={"fault": {"tc": 0.7}},
        )
        d = profile.to_dict()
        assert d["overall_resilience"] == pytest.approx(0.7)
        assert d["baseline"]["tc"] == pytest.approx(1.0)
        assert d["worst_scenario"] == "fault"
        assert "fault" in d["scenarios"]
        assert d["scenarios"]["fault"]["resilience"] == pytest.approx(0.7)
        assert d["scenarios"]["fault"]["degradation"]["tc"] == pytest.approx(-0.3)


# ---- from_runs tests ----


def _make_run(output: str, tools: list[str] | None = None, error: str | None = None) -> AgentRun:
    """Helper to create AgentRun objects for testing."""
    tool_calls = [ToolCall(name=t, arguments={}) for t in (tools or [])]
    return AgentRun(
        input="test query",
        steps=[Step(step_index=0, output_text=output, tool_calls=tool_calls)],
        final_output=output,
        error=error,
    )


def _tc_metric(run: AgentRun) -> Score:
    """Simple task completion: 1.0 if succeeded, 0.0 if error."""
    return Score(name="tc", value=1.0 if run.succeeded else 0.0)


def _tool_metric(run: AgentRun) -> Score:
    """Simple tool metric: 1.0 if any tools called, 0.0 otherwise."""
    return Score(name="tools", value=1.0 if run.tool_calls else 0.0)


class TestFromRuns:
    def test_basic_from_runs(self):
        baseline = [_make_run("ok", tools=["search"]), _make_run("ok", tools=["search"])]
        faulted = {"timeout": [_make_run("ok", tools=["search"], error="timeout")]}

        profile = ResilienceProfile.from_runs(
            baseline_runs=baseline,
            faulted_runs=faulted,
            metrics={"tc": _tc_metric, "tools": _tool_metric},
        )
        # Baseline: tc=1.0, tools=1.0
        # Faulted: tc=0.0 (error), tools=1.0
        assert profile.baseline["tc"] == pytest.approx(1.0)
        assert profile.scenario_results["timeout"].resilience == pytest.approx(0.5)

    def test_from_runs_multiple_scenarios(self):
        baseline = [_make_run("ok")]
        faulted = {
            "s1": [_make_run("ok")],  # No degradation
            "s2": [_make_run("", error="fail")],  # Total failure
        }
        profile = ResilienceProfile.from_runs(
            baseline_runs=baseline,
            faulted_runs=faulted,
            metrics={"tc": _tc_metric},
        )
        assert profile.scenario_results["s1"].resilience == 1.0
        assert profile.scenario_results["s2"].resilience == 0.0
        assert profile.overall == pytest.approx(0.5)

    def test_from_runs_empty_baseline(self):
        profile = ResilienceProfile.from_runs(
            baseline_runs=[],
            faulted_runs={"s1": [_make_run("ok")]},
            metrics={"tc": _tc_metric},
        )
        # Empty baseline → all zeros
        assert profile.baseline["tc"] == 0.0

    def test_from_runs_averaging(self):
        """Scores are averaged across multiple runs."""
        baseline = [
            _make_run("ok"),  # tc=1.0
            _make_run("", error="e"),  # tc=0.0
        ]
        profile = ResilienceProfile.from_runs(
            baseline_runs=baseline,
            faulted_runs={},
            metrics={"tc": _tc_metric},
        )
        assert profile.baseline["tc"] == pytest.approx(0.5)


class TestScenarioResult:
    def test_scenario_result_fields(self):
        r = ScenarioResult(
            scenario="test",
            scores={"tc": 0.7},
            degradation={"tc": -0.3},
            resilience=0.7,
        )
        assert r.scenario == "test"
        assert r.resilience == 0.7
