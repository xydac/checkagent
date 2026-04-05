"""CheckAgent — The open-source testing framework for AI agents."""

__version__ = "0.0.1a1"

from checkagent.conversation.session import Conversation, Turn
from checkagent.core.config import CheckAgentConfig, load_config
from checkagent.eval.assertions import (
    StructuredAssertionError,
    assert_json_schema,
    assert_output_matches,
    assert_output_schema,
    assert_tool_called,
)
from checkagent.mock.fault import FaultInjector
from checkagent.mock.llm import MatchMode, MockLLM
from checkagent.mock.mcp import MockMCPServer
from checkagent.mock.tool import MockTool
from checkagent.streaming.collector import StreamCollector
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
    "Conversation",
    "FaultInjector",
    "MatchMode",
    "MockLLM",
    "MockMCPServer",
    "MockTool",
    "Score",
    "Step",
    "StreamCollector",
    "StreamEvent",
    "StreamEventType",
    "StructuredAssertionError",
    "ToolCall",
    "Turn",
    "assert_json_schema",
    "assert_output_matches",
    "assert_output_schema",
    "assert_tool_called",
    "load_config",
]
