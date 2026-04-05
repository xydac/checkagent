"""Core module — types, adapter protocol, and plugin infrastructure."""

from checkagent.core.adapter import AgentAdapter
from checkagent.core.types import (
    AgentInput,
    AgentRun,
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
    "Score",
    "Step",
    "StreamEvent",
    "StreamEventType",
    "ToolCall",
]
