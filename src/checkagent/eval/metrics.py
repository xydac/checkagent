"""Built-in evaluation metrics for agent runs.

Provides deterministic metrics for evaluating agent performance:
- TaskCompletion: Did the agent achieve the stated goal?
- ToolCorrectness: Were the right tools called? (precision, recall, F1)
- StepEfficiency: How many steps vs. optimal? (ratio)
- TrajectoryMatch: Does the step sequence match expected?

Requirements: F3.1
"""

from __future__ import annotations

from checkagent.core.types import AgentRun, Score


def task_completion(
    run: AgentRun,
    *,
    expected_output_contains: list[str] | None = None,
    expected_output_equals: str | None = None,
    check_no_error: bool = True,
    threshold: float = 1.0,
) -> Score:
    """Score task completion based on output content and success.

    Computes a score from 0.0 to 1.0 based on:
    - Whether the run completed without errors (if check_no_error=True)
    - Whether the output contains expected substrings
    - Whether the output exactly matches expected value

    Args:
        run: The agent run to evaluate.
        expected_output_contains: Substrings that must appear in the output.
        expected_output_equals: Exact expected output string.
        check_no_error: Whether to check that the run had no errors.
        threshold: Score threshold for pass/fail (default 1.0).

    Returns:
        Score with value between 0.0 and 1.0.
    """
    checks: list[bool] = []

    if check_no_error:
        checks.append(run.succeeded)

    if expected_output_equals is not None:
        if run.final_output is None:
            checks.append(False)
        else:
            checks.append(str(run.final_output) == expected_output_equals)

    if expected_output_contains:
        if run.final_output is None:
            for _substring in expected_output_contains:
                checks.append(False)
        else:
            output_lower = str(run.final_output).lower()
            for substring in expected_output_contains:
                checks.append(substring.lower() in output_lower)

    value = (1.0 if run.succeeded else 0.0) if not checks else sum(checks) / len(checks)

    return Score(
        name="task_completion",
        value=value,
        threshold=threshold,
        reason=f"{sum(checks)}/{len(checks)} checks passed",
        metadata={"checks": checks},
    )


def tool_correctness(
    run: AgentRun,
    *,
    expected_tools: list[str],
    threshold: float = 0.5,
) -> Score:
    """Score tool usage with precision, recall, and F1.

    Compares the set of tools actually called against the expected set.

    Args:
        run: The agent run to evaluate.
        expected_tools: List of tool names that should have been called.
        threshold: F1 score threshold for pass/fail (default 0.5).

    Returns:
        Score with F1 value and precision/recall in metadata.
    """
    actual_tools = {tc.name for tc in run.tool_calls}
    expected_set = set(expected_tools)

    if not expected_set and not actual_tools:
        return Score(
            name="tool_correctness",
            value=1.0,
            threshold=threshold,
            reason="No tools expected or called",
            metadata={"precision": 1.0, "recall": 1.0, "f1": 1.0},
        )

    true_positives = actual_tools & expected_set
    tp = len(true_positives)

    precision = tp / len(actual_tools) if actual_tools else 0.0
    recall = tp / len(expected_set) if expected_set else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return Score(
        name="tool_correctness",
        value=f1,
        threshold=threshold,
        reason=(
            f"P={precision:.2f} R={recall:.2f} F1={f1:.2f} "
            f"(called={sorted(actual_tools)}, expected={sorted(expected_set)})"
        ),
        metadata={
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "actual_tools": sorted(actual_tools),
            "expected_tools": sorted(expected_set),
            "true_positives": sorted(true_positives),
            "false_positives": sorted(actual_tools - expected_set),
            "false_negatives": sorted(expected_set - actual_tools),
        },
    )


def step_efficiency(
    run: AgentRun,
    *,
    optimal_steps: int,
    threshold: float = 0.5,
) -> Score:
    """Score step efficiency as ratio of optimal to actual steps.

    A score of 1.0 means the agent used the optimal number of steps.
    Scores decrease as the agent uses more steps than optimal.

    Args:
        run: The agent run to evaluate.
        optimal_steps: The minimum number of steps needed.
        threshold: Efficiency ratio threshold for pass/fail (default 0.5).

    Returns:
        Score with efficiency ratio.
    """
    actual_steps = len(run.steps)
    if actual_steps == 0:
        return Score(
            name="step_efficiency",
            value=0.0,
            threshold=threshold,
            reason="No steps taken",
            metadata={"actual_steps": 0, "optimal_steps": optimal_steps},
        )

    ratio = min(optimal_steps / actual_steps, 1.0)

    return Score(
        name="step_efficiency",
        value=ratio,
        threshold=threshold,
        reason=f"{actual_steps} steps taken, {optimal_steps} optimal (ratio={ratio:.2f})",
        metadata={
            "actual_steps": actual_steps,
            "optimal_steps": optimal_steps,
            "ratio": ratio,
        },
    )


def trajectory_match(
    run: AgentRun,
    *,
    expected_trajectory: list[str],
    mode: str = "ordered",
    threshold: float = 1.0,
) -> Score:
    """Score whether the agent's tool call sequence matches expected.

    Supports three matching modes:
    - "strict": Exact sequence match (same tools in same order, same count)
    - "ordered": Expected tools appear in order (allows extra tools between)
    - "unordered": All expected tools were called (any order)

    Args:
        run: The agent run to evaluate.
        expected_trajectory: Ordered list of expected tool names.
        mode: Matching mode — "strict", "ordered", or "unordered".
        threshold: Score threshold for pass/fail (default 1.0).

    Returns:
        Score with match value.

    Raises:
        ValueError: If mode is not one of "strict", "ordered", "unordered".
    """
    if mode not in ("strict", "ordered", "unordered"):
        raise ValueError(f"Invalid mode: {mode!r}. Must be 'strict', 'ordered', or 'unordered'.")

    actual_trajectory = [tc.name for tc in run.tool_calls]

    if mode == "strict":
        match = actual_trajectory == expected_trajectory
        value = 1.0 if match else 0.0
        reason = "exact match" if match else (
            f"expected {expected_trajectory}, got {actual_trajectory}"
        )
    elif mode == "ordered":
        # Check that expected tools appear in order within actual
        matched = 0
        idx = 0
        for expected_tool in expected_trajectory:
            while idx < len(actual_trajectory):
                if actual_trajectory[idx] == expected_tool:
                    matched += 1
                    idx += 1
                    break
                idx += 1
        value = matched / len(expected_trajectory) if expected_trajectory else 1.0
        reason = f"{matched}/{len(expected_trajectory)} tools matched in order"
    else:  # unordered
        expected_set = set(expected_trajectory)
        actual_set = set(actual_trajectory)
        matched = len(expected_set & actual_set)
        value = matched / len(expected_set) if expected_set else 1.0
        reason = f"{matched}/{len(expected_set)} expected tools found"

    return Score(
        name="trajectory_match",
        value=value,
        threshold=threshold,
        reason=reason,
        metadata={
            "mode": mode,
            "actual_trajectory": actual_trajectory,
            "expected_trajectory": expected_trajectory,
        },
    )
