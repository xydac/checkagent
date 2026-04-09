"""CheckAgent — The open-source testing framework for AI agents."""

__version__ = "0.0.1a1"

from checkagent.adapters.generic import GenericAdapter, wrap
from checkagent.ci.quality_gate import QualityGateEntry
from checkagent.ci.reporter import TestRunSummary
from checkagent.conversation.session import Conversation, Turn
from checkagent.core.config import CheckAgentConfig, load_config
from checkagent.core.cost import (
    BudgetExceededError,
    CostBreakdown,
    CostReport,
    CostTracker,
    calculate_run_cost,
)
from checkagent.core.types import (
    AgentInput,
    AgentRun,
    Score,
    Step,
    StreamEvent,
    StreamEventType,
    ToolCall,
)
from checkagent.datasets import GoldenDataset, load_cases, load_dataset
from checkagent.datasets.schema import EvalCase
from checkagent.eval.assertions import (
    StructuredAssertionError,
    assert_json_schema,
    assert_output_matches,
    assert_output_schema,
    assert_tool_called,
)
from checkagent.judge import Criterion, JudgeScore, RubricJudge
from checkagent.mock.fault import FaultInjector
from checkagent.mock.llm import MatchMode, MockLLM
from checkagent.mock.mcp import MockMCPServer
from checkagent.mock.tool import MockTool, literal
from checkagent.multiagent import HandoffType, MultiAgentTrace
from checkagent.replay import Cassette, ReplayEngine
from checkagent.safety.probes import ProbeSet
from checkagent.streaming.collector import StreamCollector

_LAZY_ADAPTER_IMPORTS: dict[str, tuple[str, str]] = {
    "AnthropicAdapter": ("checkagent.adapters.anthropic", "AnthropicAdapter"),
    "CrewAIAdapter": ("checkagent.adapters.crewai", "CrewAIAdapter"),
    "LangChainAdapter": ("checkagent.adapters.langchain", "LangChainAdapter"),
    "OpenAIAgentsAdapter": ("checkagent.adapters.openai_agents", "OpenAIAgentsAdapter"),
    "PydanticAIAdapter": ("checkagent.adapters.pydantic_ai", "PydanticAIAdapter"),
}


def __getattr__(name: str):
    """Lazy-load framework adapters to avoid import errors for optional deps."""
    if name in _LAZY_ADAPTER_IMPORTS:
        import importlib

        module_path, attr = _LAZY_ADAPTER_IMPORTS[name]
        mod = importlib.import_module(module_path)
        return getattr(mod, attr)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "__version__",
    "AgentInput",
    "AgentRun",
    "AnthropicAdapter",
    "BudgetExceededError",
    "Cassette",
    "CheckAgentConfig",
    "Conversation",
    "CostBreakdown",
    "CostReport",
    "CostTracker",
    "CrewAIAdapter",
    "Criterion",
    "EvalCase",
    "FaultInjector",
    "GenericAdapter",
    "GoldenDataset",
    "HandoffType",
    "JudgeScore",
    "LangChainAdapter",
    "MatchMode",
    "MockLLM",
    "MockMCPServer",
    "MockTool",
    "MultiAgentTrace",
    "OpenAIAgentsAdapter",
    "ProbeSet",
    "PydanticAIAdapter",
    "QualityGateEntry",
    "ReplayEngine",
    "RubricJudge",
    "Score",
    "Step",
    "StreamCollector",
    "StreamEvent",
    "StreamEventType",
    "StructuredAssertionError",
    "TestRunSummary",
    "ToolCall",
    "Turn",
    "assert_json_schema",
    "assert_output_matches",
    "assert_output_schema",
    "assert_tool_called",
    "calculate_run_cost",
    "literal",
    "load_cases",
    "load_config",
    "load_dataset",
    "wrap",
]
