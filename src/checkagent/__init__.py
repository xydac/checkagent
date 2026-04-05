"""CheckAgent — The open-source testing framework for AI agents."""

__version__ = "0.0.1a1"

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
    "__version__",
    "AgentInput",
    "AgentRun",
    "Score",
    "Step",
    "StreamEvent",
    "StreamEventType",
    "ToolCall",
]
