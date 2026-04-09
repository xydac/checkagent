"""Eval layer — evaluation metrics and assertions for AI agents (Layer 3)."""

from checkagent.eval.aggregate import (
    AggregateResult,
    RunSummary,
    StepStats,
    aggregate_scores,
    compute_step_stats,
    detect_regressions,
)
from checkagent.eval.assertions import (
    StructuredAssertionError,
    assert_json_schema,
    assert_output_matches,
    assert_output_schema,
    assert_tool_called,
)
from checkagent.eval.evaluator import Evaluator, EvaluatorRegistry
from checkagent.eval.metrics import (
    step_efficiency,
    task_completion,
    tool_correctness,
    trajectory_match,
)
from checkagent.eval.resilience import ResilienceProfile, ScenarioResult

__all__ = [
    "AggregateResult",
    "Evaluator",
    "EvaluatorRegistry",
    "ResilienceProfile",
    "RunSummary",
    "ScenarioResult",
    "StepStats",
    "StructuredAssertionError",
    "aggregate_scores",
    "assert_json_schema",
    "assert_output_matches",
    "assert_output_schema",
    "assert_tool_called",
    "compute_step_stats",
    "detect_regressions",
    "step_efficiency",
    "task_completion",
    "tool_correctness",
    "trajectory_match",
]
