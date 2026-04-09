"""Tests for built-in evaluation metrics."""

from __future__ import annotations

import pytest

from checkagent.core.types import AgentInput, AgentRun, Step, ToolCall
from checkagent.eval.metrics import (
    step_efficiency,
    task_completion,
    tool_correctness,
    trajectory_match,
)

# ── Helpers ──────────────────────────────────────────────────────────────

def _make_run(
    *,
    output: str | None = "done",
    error: str | None = None,
    tools: list[str] | None = None,
    steps: int = 1,
) -> AgentRun:
    """Build a minimal AgentRun for testing."""
    step_list = []
    for i in range(steps):
        tool_calls = []
        if tools and i == 0:
            tool_calls = [ToolCall(name=t) for t in tools]
        step_list.append(Step(step_index=i, tool_calls=tool_calls))

    return AgentRun(
        input=AgentInput(query="test"),
        steps=step_list,
        final_output=output,
        error=error,
    )


def _make_run_with_trajectory(tool_names: list[str]) -> AgentRun:
    """Build an AgentRun with one tool call per step."""
    steps = []
    for i, name in enumerate(tool_names):
        steps.append(Step(
            step_index=i,
            tool_calls=[ToolCall(name=name)],
        ))
    return AgentRun(
        input=AgentInput(query="test"),
        steps=steps,
        final_output="done",
    )


# ── TaskCompletion ───────────────────────────────────────────────────────

class TestTaskCompletion:
    def test_successful_run_no_checks(self):
        score = task_completion(_make_run())
        assert score.value == 1.0
        assert score.passed is True

    def test_failed_run_no_checks(self):
        score = task_completion(_make_run(error="boom"))
        assert score.value == 0.0
        assert score.passed is False

    def test_output_contains_all_match(self):
        run = _make_run(output="Refund initiated, expect 3-5 business days")
        score = task_completion(
            run, expected_output_contains=["refund initiated", "3-5 business days"]
        )
        assert score.value == 1.0

    def test_output_contains_partial_match(self):
        run = _make_run(output="Refund initiated")
        score = task_completion(
            run, expected_output_contains=["refund initiated", "3-5 business days"]
        )
        # 2 out of 3 checks pass (no_error + 1 substring)
        assert 0.0 < score.value < 1.0

    def test_output_contains_case_insensitive(self):
        run = _make_run(output="REFUND INITIATED")
        score = task_completion(run, expected_output_contains=["refund initiated"])
        assert score.value == 1.0

    def test_output_equals_match(self):
        score = task_completion(_make_run(output="yes"), expected_output_equals="yes")
        assert score.value == 1.0

    def test_output_equals_mismatch(self):
        score = task_completion(_make_run(output="no"), expected_output_equals="yes")
        assert score.value == 0.5  # no_error passes, equals fails

    def test_skip_error_check(self):
        run = _make_run(error="boom", output="yes")
        score = task_completion(
            run, expected_output_equals="yes", check_no_error=False
        )
        assert score.value == 1.0

    def test_custom_threshold(self):
        run = _make_run(output="partial")
        score = task_completion(
            run,
            expected_output_contains=["partial", "missing"],
            threshold=0.5,
        )
        # 2/3 checks pass (no_error + "partial"), "missing" fails
        assert score.passed is True

    def test_none_output_contains(self):
        run = _make_run(output=None)
        score = task_completion(run, expected_output_contains=["something"])
        assert score.value < 1.0

    def test_none_output_does_not_equal_empty_string(self):
        """F-031: None output must not match expected_output_equals=''."""
        run = _make_run(output=None)
        score = task_completion(run, expected_output_equals="", check_no_error=False)
        assert score.passed is False
        assert score.value == 0.0

    def test_none_output_does_not_equal_any_string(self):
        """F-031: None output must not match any expected string."""
        run = _make_run(output=None)
        score = task_completion(run, expected_output_equals="hello", check_no_error=False)
        assert score.passed is False

    def test_empty_string_output_equals_empty_string(self):
        """Actual empty string output should match expected_output_equals=''."""
        run = _make_run(output="")
        score = task_completion(run, expected_output_equals="", check_no_error=False)
        assert score.value == 1.0
        assert score.passed is True

    def test_none_output_fails_all_contains_checks(self):
        """F-031: None output should fail every contains check."""
        run = _make_run(output=None)
        score = task_completion(
            run, expected_output_contains=["a", "b"], check_no_error=False
        )
        assert score.value == 0.0


# ── ToolCorrectness ─────────────────────────────────────────────────────

class TestToolCorrectness:
    def test_perfect_match(self):
        run = _make_run(tools=["search", "submit"])
        score = tool_correctness(run, expected_tools=["search", "submit"])
        assert score.value == 1.0
        assert score.metadata["precision"] == 1.0
        assert score.metadata["recall"] == 1.0

    def test_extra_tools(self):
        run = _make_run(tools=["search", "submit", "log"])
        score = tool_correctness(run, expected_tools=["search", "submit"])
        assert score.metadata["precision"] == pytest.approx(2 / 3)
        assert score.metadata["recall"] == 1.0

    def test_missing_tools(self):
        run = _make_run(tools=["search"])
        score = tool_correctness(run, expected_tools=["search", "submit"])
        assert score.metadata["precision"] == 1.0
        assert score.metadata["recall"] == 0.5

    def test_no_overlap(self):
        run = _make_run(tools=["foo"])
        score = tool_correctness(run, expected_tools=["bar"])
        assert score.value == 0.0

    def test_empty_both(self):
        run = _make_run(tools=[])
        score = tool_correctness(run, expected_tools=[])
        assert score.value == 1.0

    def test_no_tools_called(self):
        run = _make_run(tools=[])
        score = tool_correctness(run, expected_tools=["search"])
        assert score.value == 0.0

    def test_metadata_fields(self):
        run = _make_run(tools=["a", "b", "c"])
        score = tool_correctness(run, expected_tools=["b", "d"])
        assert "a" in score.metadata["false_positives"]
        assert "d" in score.metadata["false_negatives"]
        assert "b" in score.metadata["true_positives"]

    def test_threshold_pass(self):
        run = _make_run(tools=["search", "submit"])
        score = tool_correctness(
            run, expected_tools=["search", "submit"], threshold=0.8
        )
        assert score.passed is True

    def test_threshold_fail(self):
        run = _make_run(tools=["foo"])
        score = tool_correctness(
            run, expected_tools=["bar"], threshold=0.5
        )
        assert score.passed is False


# ── StepEfficiency ───────────────────────────────────────────────────────

class TestStepEfficiency:
    def test_optimal(self):
        run = _make_run(steps=3)
        score = step_efficiency(run, optimal_steps=3)
        assert score.value == 1.0

    def test_fewer_than_optimal(self):
        run = _make_run(steps=2)
        score = step_efficiency(run, optimal_steps=3)
        # Capped at 1.0
        assert score.value == 1.0

    def test_more_than_optimal(self):
        run = _make_run(steps=6)
        score = step_efficiency(run, optimal_steps=3)
        assert score.value == pytest.approx(0.5)

    def test_no_steps(self):
        run = _make_run(steps=0)
        score = step_efficiency(run, optimal_steps=3)
        assert score.value == 0.0

    def test_threshold(self):
        run = _make_run(steps=4)
        score = step_efficiency(run, optimal_steps=3, threshold=0.5)
        assert score.value == pytest.approx(0.75)
        assert score.passed is True

    def test_metadata(self):
        run = _make_run(steps=5)
        score = step_efficiency(run, optimal_steps=3)
        assert score.metadata["actual_steps"] == 5
        assert score.metadata["optimal_steps"] == 3


# ── TrajectoryMatch ──────────────────────────────────────────────────────

class TestTrajectoryMatch:
    def test_strict_exact_match(self):
        run = _make_run_with_trajectory(["search", "submit", "notify"])
        score = trajectory_match(
            run,
            expected_trajectory=["search", "submit", "notify"],
            mode="strict",
        )
        assert score.value == 1.0

    def test_strict_wrong_order(self):
        run = _make_run_with_trajectory(["submit", "search", "notify"])
        score = trajectory_match(
            run,
            expected_trajectory=["search", "submit", "notify"],
            mode="strict",
        )
        assert score.value == 0.0

    def test_strict_extra_tools(self):
        run = _make_run_with_trajectory(["search", "log", "submit"])
        score = trajectory_match(
            run,
            expected_trajectory=["search", "submit"],
            mode="strict",
        )
        assert score.value == 0.0

    def test_ordered_with_extras(self):
        run = _make_run_with_trajectory(["search", "log", "submit", "notify"])
        score = trajectory_match(
            run,
            expected_trajectory=["search", "submit"],
            mode="ordered",
        )
        assert score.value == 1.0

    def test_ordered_missing_tool(self):
        run = _make_run_with_trajectory(["search", "notify"])
        score = trajectory_match(
            run,
            expected_trajectory=["search", "submit", "notify"],
            mode="ordered",
        )
        # "search" matches at idx 0, "submit" not found, "notify" not found
        # (idx already past end after scanning for "submit")
        assert score.value == pytest.approx(1 / 3)

    def test_ordered_skipped_middle(self):
        run = _make_run_with_trajectory(["search", "notify", "done"])
        score = trajectory_match(
            run,
            expected_trajectory=["search", "notify"],
            mode="ordered",
        )
        assert score.value == 1.0

    def test_ordered_wrong_order(self):
        run = _make_run_with_trajectory(["submit", "search"])
        score = trajectory_match(
            run,
            expected_trajectory=["search", "submit"],
            mode="ordered",
        )
        # Only "submit" matches (after "search" is consumed at wrong position)
        assert score.value < 1.0

    def test_unordered_all_present(self):
        run = _make_run_with_trajectory(["notify", "search", "submit"])
        score = trajectory_match(
            run,
            expected_trajectory=["search", "submit", "notify"],
            mode="unordered",
        )
        assert score.value == 1.0

    def test_unordered_partial(self):
        run = _make_run_with_trajectory(["search"])
        score = trajectory_match(
            run,
            expected_trajectory=["search", "submit"],
            mode="unordered",
        )
        assert score.value == 0.5

    def test_empty_expected(self):
        run = _make_run_with_trajectory(["search"])
        score = trajectory_match(run, expected_trajectory=[], mode="strict")
        assert score.value == 0.0  # strict: actual != expected

    def test_empty_expected_ordered(self):
        run = _make_run_with_trajectory(["search"])
        score = trajectory_match(run, expected_trajectory=[], mode="ordered")
        assert score.value == 1.0

    def test_invalid_mode(self):
        run = _make_run_with_trajectory([])
        with pytest.raises(ValueError, match="Invalid mode"):
            trajectory_match(run, expected_trajectory=[], mode="fuzzy")

    def test_metadata_includes_trajectories(self):
        run = _make_run_with_trajectory(["a", "b"])
        score = trajectory_match(
            run, expected_trajectory=["a", "b"], mode="strict"
        )
        assert score.metadata["mode"] == "strict"
        assert score.metadata["actual_trajectory"] == ["a", "b"]
        assert score.metadata["expected_trajectory"] == ["a", "b"]

    def test_threshold_pass(self):
        run = _make_run_with_trajectory(["search", "submit"])
        score = trajectory_match(
            run,
            expected_trajectory=["search", "submit"],
            mode="strict",
            threshold=0.5,
        )
        assert score.passed is True
