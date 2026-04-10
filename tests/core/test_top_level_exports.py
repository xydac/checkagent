"""Verify that commonly-used symbols are importable from the top-level ``checkagent`` package.

Covers F-020 (eval metrics), F-021 (safety evaluators), F-048 (judge types),
F-053 (multi-judge), F-086 (adapters).
"""


def test_core_types():
    from checkagent import AgentInput, AgentRun, Score, Step, StreamEvent, ToolCall

    for cls in (AgentInput, AgentRun, Score, Step, StreamEvent, ToolCall):
        assert cls is not None


def test_eval_metrics():
    from checkagent import (
        step_efficiency,
        task_completion,
        tool_correctness,
        trajectory_match,
    )

    for fn in (step_efficiency, task_completion, tool_correctness, trajectory_match):
        assert callable(fn)


def test_eval_aggregate():
    from checkagent import (
        AggregateResult,
        RunSummary,
        StepStats,
        aggregate_scores,
        compute_step_stats,
        detect_regressions,
    )

    for item in (AggregateResult, RunSummary, StepStats):
        assert item is not None
    for fn in (aggregate_scores, compute_step_stats, detect_regressions):
        assert callable(fn)


def test_eval_evaluator():
    from checkagent import Evaluator, EvaluatorRegistry

    assert Evaluator is not None
    assert EvaluatorRegistry is not None


def test_eval_resilience():
    from checkagent import ResilienceProfile, ScenarioResult

    assert ResilienceProfile is not None
    assert ScenarioResult is not None


def test_safety_evaluators():
    from checkagent import (
        PIILeakageScanner,
        PromptInjectionDetector,
        RefusalComplianceChecker,
        SafetyEvaluator,
        SafetyFinding,
        SafetyResult,
        SystemPromptLeakDetector,
        ToolCallBoundaryValidator,
    )

    for cls in (
        PIILeakageScanner,
        PromptInjectionDetector,
        RefusalComplianceChecker,
        SafetyEvaluator,
        SafetyFinding,
        SafetyResult,
        SystemPromptLeakDetector,
        ToolCallBoundaryValidator,
    ):
        assert cls is not None


def test_safety_taxonomy():
    from checkagent import SafetyCategory, Severity

    assert SafetyCategory is not None
    assert Severity is not None


def test_probes():
    from checkagent import Probe, ProbeSet

    assert Probe is not None
    assert ProbeSet is not None


def test_judge_types():
    from checkagent import (
        ConsensusVerdict,
        Criterion,
        JudgeScore,
        JudgeVerdict,
        Rubric,
        RubricJudge,
        Verdict,
        compute_verdict,
        multi_judge_evaluate,
    )

    for item in (
        ConsensusVerdict,
        Criterion,
        JudgeScore,
        JudgeVerdict,
        Rubric,
        RubricJudge,
        Verdict,
    ):
        assert item is not None
    for fn in (compute_verdict, multi_judge_evaluate):
        assert callable(fn)


def test_adapters_lazy_load():
    """Framework adapters are lazy-loaded to avoid import errors for optional deps."""
    import checkagent

    # These should be in __all__ even though they are lazy
    assert "LangChainAdapter" in checkagent.__all__
    assert "OpenAIAgentsAdapter" in checkagent.__all__
    assert "PydanticAIAdapter" in checkagent.__all__
    assert "AnthropicAdapter" in checkagent.__all__
    assert "CrewAIAdapter" in checkagent.__all__


def test_cost_types():
    from checkagent import (
        BudgetExceededError,
        CostBreakdown,
        CostReport,
        CostTracker,
        calculate_run_cost,
    )

    for cls in (BudgetExceededError, CostBreakdown, CostReport, CostTracker):
        assert cls is not None
    assert callable(calculate_run_cost)


def test_all_list_is_sorted_by_section():
    """__all__ should contain all directly imported symbols."""
    import checkagent

    # Every name in __all__ should be resolvable
    for name in checkagent.__all__:
        assert hasattr(checkagent, name), f"{name} in __all__ but not accessible"
