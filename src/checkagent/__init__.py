"""CheckAgent — The open-source testing framework for AI agents."""

__version__ = "0.1.0"

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
from checkagent.judge import (
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
from checkagent.mock.fault import FaultInjector
from checkagent.mock.llm import MatchMode, MockLLM
from checkagent.mock.mcp import MockMCPServer
from checkagent.mock.tool import MockTool, literal
from checkagent.multiagent import HandoffType, MultiAgentTrace
from checkagent.replay import Cassette, ReplayEngine
from checkagent.safety.evaluator import SafetyEvaluator, SafetyFinding, SafetyResult
from checkagent.safety.injection import PromptInjectionDetector
from checkagent.safety.pii import PIILeakageScanner
from checkagent.safety.probes import Probe, ProbeSet
from checkagent.safety.refusal import RefusalComplianceChecker
from checkagent.safety.system_prompt import SystemPromptLeakDetector
from checkagent.safety.taxonomy import SafetyCategory, Severity
from checkagent.safety.tool_boundary import ToolCallBoundaryValidator
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
    # Core types
    "AgentInput",
    "AgentRun",
    "Score",
    "Step",
    "StreamEvent",
    "StreamEventType",
    "ToolCall",
    # Adapters
    "AnthropicAdapter",
    "CrewAIAdapter",
    "GenericAdapter",
    "LangChainAdapter",
    "OpenAIAgentsAdapter",
    "PydanticAIAdapter",
    # Config
    "CheckAgentConfig",
    "load_config",
    # Cost
    "BudgetExceededError",
    "CostBreakdown",
    "CostReport",
    "CostTracker",
    "calculate_run_cost",
    # Conversation
    "Conversation",
    "Turn",
    # Datasets
    "EvalCase",
    "GoldenDataset",
    "load_cases",
    "load_dataset",
    # Eval — assertions
    "StructuredAssertionError",
    "assert_json_schema",
    "assert_output_matches",
    "assert_output_schema",
    "assert_tool_called",
    # Eval — metrics
    "step_efficiency",
    "task_completion",
    "tool_correctness",
    "trajectory_match",
    # Eval — aggregate
    "AggregateResult",
    "RunSummary",
    "StepStats",
    "aggregate_scores",
    "compute_step_stats",
    "detect_regressions",
    # Eval — evaluator
    "Evaluator",
    "EvaluatorRegistry",
    # Eval — resilience
    "ResilienceProfile",
    "ScenarioResult",
    # Judge
    "ConsensusVerdict",
    "Criterion",
    "JudgeScore",
    "JudgeVerdict",
    "Rubric",
    "RubricJudge",
    "Verdict",
    "compute_verdict",
    "multi_judge_evaluate",
    # Mock
    "FaultInjector",
    "MatchMode",
    "MockLLM",
    "MockMCPServer",
    "MockTool",
    "literal",
    # Multi-agent
    "HandoffType",
    "MultiAgentTrace",
    # Replay
    "Cassette",
    "ReplayEngine",
    # Safety
    "PIILeakageScanner",
    "Probe",
    "ProbeSet",
    "PromptInjectionDetector",
    "RefusalComplianceChecker",
    "SafetyCategory",
    "SafetyEvaluator",
    "SafetyFinding",
    "SafetyResult",
    "Severity",
    "SystemPromptLeakDetector",
    "ToolCallBoundaryValidator",
    # Streaming
    "StreamCollector",
    # CI
    "QualityGateEntry",
    "TestRunSummary",
    # Helpers
    "wrap",
]
