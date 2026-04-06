"""Core module — types, adapter protocol, and plugin infrastructure."""

from checkagent.core.adapter import AgentAdapter
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
    HandoffType,
    Score,
    Step,
    StreamEvent,
    StreamEventType,
    ToolCall,
)

__all__ = [
    "AgentAdapter",
    "AgentInput",
    "AgentRun",
    "BudgetExceededError",
    "CostBreakdown",
    "CostReport",
    "CostTracker",
    "HandoffType",
    "Score",
    "Step",
    "StreamEvent",
    "StreamEventType",
    "ToolCall",
    "calculate_run_cost",
]
