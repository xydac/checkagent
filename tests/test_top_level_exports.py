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
    }
    missing = expected - set(checkagent.__all__)
    assert not missing, f"Missing from __all__: {missing}"
