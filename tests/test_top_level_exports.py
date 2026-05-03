"""Tests that all public names are importable from the top-level checkagent package."""

import checkagent


def test_core_types_importable():
    """Core types are directly importable from checkagent."""
    for name in [
        "AgentInput",
        "AgentRun",
        "Score",
        "Step",
        "StreamEvent",
        "StreamEventType",
        "ToolCall",
    ]:
        assert hasattr(checkagent, name), f"checkagent.{name} missing"


def test_mock_types_importable():
    """Mock layer types are directly importable from checkagent."""
    for name in ["MockLLM", "MockTool", "MockMCPServer", "MatchMode", "FaultInjector"]:
        assert hasattr(checkagent, name), f"checkagent.{name} missing"


def test_assertion_helpers_importable():
    """Assertion helpers are directly importable from checkagent."""
    for name in [
        "assert_tool_called",
        "assert_output_schema",
        "assert_output_matches",
        "assert_json_schema",
        "StructuredAssertionError",
    ]:
        assert hasattr(checkagent, name), f"checkagent.{name} missing"


def test_adapters_importable_from_top_level():
    """All adapters are importable from top-level checkagent (F-086)."""
    for name in [
        "GenericAdapter",
        "wrap",
        "AnthropicAdapter",
        "CrewAIAdapter",
        "LangChainAdapter",
        "OpenAIAgentsAdapter",
        "PydanticAIAdapter",
    ]:
        obj = getattr(checkagent, name)
        assert obj is not None, f"checkagent.{name} resolved to None"


def test_adapters_importable_via_from_import():
    """Adapters work with from-import syntax."""
    from checkagent import (
        AnthropicAdapter,
        CrewAIAdapter,
        GenericAdapter,
        LangChainAdapter,
        OpenAIAgentsAdapter,
        PydanticAIAdapter,
        wrap,
    )

    assert GenericAdapter is not None
    assert wrap is not None
    assert PydanticAIAdapter is not None
    assert LangChainAdapter is not None
    assert AnthropicAdapter is not None
    assert OpenAIAgentsAdapter is not None
    assert CrewAIAdapter is not None


def test_datasets_importable():
    """Dataset types are importable from top-level (F-063, F-067)."""
    from checkagent import EvalCase, GoldenDataset, load_cases, load_dataset

    assert GoldenDataset is not None
    assert EvalCase is not None
    assert callable(load_dataset)
    assert callable(load_cases)


def test_judge_importable():
    """Judge types are importable from top-level (F-063)."""
    from checkagent import Criterion, JudgeScore, RubricJudge

    assert RubricJudge is not None
    assert Criterion is not None
    assert JudgeScore is not None


def test_replay_importable():
    """Replay types are importable from top-level."""
    from checkagent import Cassette, ReplayEngine

    assert Cassette is not None
    assert ReplayEngine is not None


def test_multiagent_importable():
    """Multi-agent types are importable from top-level (F-068, F-071)."""
    from checkagent import HandoffType, MultiAgentTrace

    assert MultiAgentTrace is not None
    assert HandoffType is not None


def test_ci_importable():
    """CI types are importable from top-level (F-030)."""
    from checkagent import QualityGateEntry, TestRunSummary

    assert QualityGateEntry is not None
    assert TestRunSummary is not None


def test_safety_importable():
    """Safety types are importable from top-level."""
    from checkagent import ProbeSet

    assert ProbeSet is not None


def test_check_behavioral_compliance_top_level():
    """check_behavioral_compliance is importable from top-level checkagent (fixes F-117)."""
    from checkagent import check_behavioral_compliance

    assert callable(check_behavioral_compliance)


def test_all_list_matches_actual_exports():
    """__all__ contains every name we expect to be public."""
    expected = {
        "GenericAdapter",
        "wrap",
        "AnthropicAdapter",
        "CrewAIAdapter",
        "LangChainAdapter",
        "OpenAIAgentsAdapter",
        "PydanticAIAdapter",
        "MockLLM",
        "MockTool",
        "AgentRun",
        "GoldenDataset",
        "EvalCase",
        "load_dataset",
        "load_cases",
        "RubricJudge",
        "Criterion",
        "JudgeScore",
        "Cassette",
        "ReplayEngine",
        "MultiAgentTrace",
        "HandoffType",
        "QualityGateEntry",
        "TestRunSummary",
        "ProbeSet",
        "check_behavioral_compliance",
    }
    missing = expected - set(checkagent.__all__)
    assert not missing, f"Missing from __all__: {missing}"
