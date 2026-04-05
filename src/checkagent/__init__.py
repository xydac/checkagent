"""CheckAgent — The open-source testing framework for AI agents."""

__version__ = "0.0.1a1"

from checkagent.core.config import CheckAgentConfig, load_config
from checkagent.mock.llm import MatchMode, MockLLM
from checkagent.mock.tool import MockTool
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
    "CheckAgentConfig",
    "MatchMode",
    "MockLLM",
    "MockTool",
    "Score",
    "Step",
    "StreamEvent",
    "StreamEventType",
    "ToolCall",
    "load_config",
]
