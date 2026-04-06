"""Core data types for CheckAgent.

These types represent the universal vocabulary for agent execution traces.
Every adapter converts framework-specific data into these types.
"""

from __future__ import annotations

import time
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class ToolCall(BaseModel):
    """A single tool/function invocation within an agent step."""

    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    result: Any = None
    error: str | None = None
    duration_ms: float | None = None

    @property
    def succeeded(self) -> bool:
        return self.error is None


class Step(BaseModel):
    """A single reasoning/action step in an agent run."""

    step_index: int = 0
    input_text: str | None = None
    output_text: str | None = None
    tool_calls: list[ToolCall] = Field(default_factory=list)
    model: str | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    duration_ms: float | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def total_tokens(self) -> int | None:
        if self.prompt_tokens is not None and self.completion_tokens is not None:
            return self.prompt_tokens + self.completion_tokens
        return None


class AgentInput(BaseModel):
    """Input to an agent run."""

    query: str
    context: dict[str, Any] = Field(default_factory=dict)
    conversation_history: list[dict[str, str]] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class AgentRun(BaseModel):
    """Complete execution trace of an agent run."""

    input: AgentInput
    steps: list[Step] = Field(default_factory=list)
    final_output: Any = None
    error: str | None = None
    duration_ms: float | None = None
    total_prompt_tokens: int | None = None
    total_completion_tokens: int | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    # Multi-agent fields (optional — single-agent runs leave these as None)
    run_id: str | None = None
    agent_id: str | None = None
    agent_name: str | None = None
    parent_run_id: str | None = None

    @property
    def total_tokens(self) -> int | None:
        if self.total_prompt_tokens is not None and self.total_completion_tokens is not None:
            return self.total_prompt_tokens + self.total_completion_tokens
        return None

    @property
    def succeeded(self) -> bool:
        return self.error is None

    @property
    def tool_calls(self) -> list[ToolCall]:
        """All tool calls across all steps, flattened."""
        return [tc for step in self.steps for tc in step.tool_calls]

    def tool_was_called(self, name: str) -> bool:
        return any(tc.name == name for tc in self.tool_calls)

    def get_tool_calls(self, name: str) -> list[ToolCall]:
        return [tc for tc in self.tool_calls if tc.name == name]


class HandoffType(str, Enum):
    """Types of agent-to-agent handoff."""

    DELEGATION = "delegation"  # Parent spawns child, waits for result
    RELAY = "relay"  # Agent passes control entirely to next agent
    BROADCAST = "broadcast"  # Agent sends to multiple agents in parallel


class StreamEventType(str, Enum):
    """Types of streaming events."""

    TEXT_DELTA = "text_delta"
    TOOL_CALL_START = "tool_call_start"
    TOOL_CALL_DELTA = "tool_call_delta"
    TOOL_CALL_END = "tool_call_end"
    TOOL_RESULT = "tool_result"
    STEP_START = "step_start"
    STEP_END = "step_end"
    RUN_START = "run_start"
    RUN_END = "run_end"
    HANDOFF = "handoff"
    ERROR = "error"


class StreamEvent(BaseModel):
    """A single streaming event from an agent execution."""

    event_type: StreamEventType
    data: Any = None
    timestamp: float = Field(default_factory=time.time)
    step_index: int | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class Score(BaseModel):
    """An evaluation score for an agent run."""

    name: str
    value: float = Field(ge=0.0, le=1.0)
    passed: bool | None = None
    threshold: float | None = Field(default=None, ge=0.0, le=1.0)
    reason: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    def model_post_init(self, __context: Any) -> None:
        if self.passed is None and self.threshold is not None:
            self.passed = self.value >= self.threshold
