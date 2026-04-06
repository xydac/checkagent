"""Tests for core data types."""

import pytest
from pydantic import ValidationError

from checkagent.core.types import (
    AgentInput,
    AgentRun,
    Score,
    Step,
    StreamEvent,
    StreamEventType,
    ToolCall,
)


class TestToolCall:
    def test_minimal(self):
        tc = ToolCall(name="search")
        assert tc.name == "search"
        assert tc.arguments == {}
        assert tc.result is None
        assert tc.error is None
        assert tc.succeeded is True

    def test_with_error(self):
        tc = ToolCall(name="search", error="timeout")
        assert tc.succeeded is False

    def test_with_arguments_and_result(self):
        tc = ToolCall(
            name="search",
            arguments={"query": "hello"},
            result={"hits": 3},
            duration_ms=42.5,
        )
        assert tc.arguments["query"] == "hello"
        assert tc.result == {"hits": 3}
        assert tc.duration_ms == 42.5


class TestStep:
    def test_minimal(self):
        step = Step()
        assert step.step_index == 0
        assert step.tool_calls == []
        assert step.total_tokens is None

    def test_token_counting(self):
        step = Step(prompt_tokens=100, completion_tokens=50)
        assert step.total_tokens == 150

    def test_partial_tokens_returns_none(self):
        step = Step(prompt_tokens=100)
        assert step.total_tokens is None

    def test_with_tool_calls(self):
        step = Step(
            step_index=1,
            tool_calls=[
                ToolCall(name="search", arguments={"q": "test"}),
                ToolCall(name="write", arguments={"text": "hi"}),
            ],
        )
        assert len(step.tool_calls) == 2
        assert step.tool_calls[0].name == "search"


class TestAgentInput:
    def test_minimal(self):
        inp = AgentInput(query="hello")
        assert inp.query == "hello"
        assert inp.context == {}
        assert inp.conversation_history == []

    def test_with_context(self):
        inp = AgentInput(query="test", context={"user_id": "abc"})
        assert inp.context["user_id"] == "abc"


class TestAgentRun:
    def test_minimal(self):
        run = AgentRun(input=AgentInput(query="hi"))
        assert run.succeeded is True
        assert run.final_output is None
        assert run.tool_calls == []

    def test_with_error(self):
        run = AgentRun(input=AgentInput(query="hi"), error="boom")
        assert run.succeeded is False

    def test_tool_calls_flattened(self):
        run = AgentRun(
            input=AgentInput(query="test"),
            steps=[
                Step(tool_calls=[ToolCall(name="a"), ToolCall(name="b")]),
                Step(tool_calls=[ToolCall(name="c")]),
            ],
        )
        assert len(run.tool_calls) == 3
        assert [tc.name for tc in run.tool_calls] == ["a", "b", "c"]

    def test_tool_was_called(self):
        run = AgentRun(
            input=AgentInput(query="test"),
            steps=[Step(tool_calls=[ToolCall(name="search")])],
        )
        assert run.tool_was_called("search") is True
        assert run.tool_was_called("delete") is False

    def test_get_tool_calls(self):
        run = AgentRun(
            input=AgentInput(query="test"),
            steps=[
                Step(tool_calls=[ToolCall(name="search"), ToolCall(name="search")]),
                Step(tool_calls=[ToolCall(name="write")]),
            ],
        )
        assert len(run.get_tool_calls("search")) == 2
        assert len(run.get_tool_calls("write")) == 1
        assert len(run.get_tool_calls("delete")) == 0

    def test_total_tokens(self):
        run = AgentRun(
            input=AgentInput(query="test"),
            total_prompt_tokens=200,
            total_completion_tokens=100,
        )
        assert run.total_tokens == 300


class TestStreamEvent:
    def test_creation(self):
        event = StreamEvent(event_type=StreamEventType.TEXT_DELTA, data="hello")
        assert event.event_type == StreamEventType.TEXT_DELTA
        assert event.data == "hello"
        assert event.timestamp > 0

    def test_all_event_types_exist(self):
        expected = {
            "text_delta", "tool_call_start", "tool_call_delta", "tool_call_end",
            "tool_result", "step_start", "step_end", "run_start", "run_end",
            "handoff", "error",
        }
        actual = {e.value for e in StreamEventType}
        assert actual == expected


class TestScore:
    def test_basic(self):
        score = Score(name="accuracy", value=0.85)
        assert score.value == 0.85
        assert score.passed is None

    def test_auto_pass_above_threshold(self):
        score = Score(name="accuracy", value=0.85, threshold=0.8)
        assert score.passed is True

    def test_auto_fail_below_threshold(self):
        score = Score(name="accuracy", value=0.7, threshold=0.8)
        assert score.passed is False

    def test_explicit_passed_overrides(self):
        score = Score(name="accuracy", value=0.7, threshold=0.8, passed=True)
        assert score.passed is True

    def test_value_bounds(self):
        with pytest.raises(ValidationError):
            Score(name="bad", value=1.5)
        with pytest.raises(ValidationError):
            Score(name="bad", value=-0.1)

    def test_threshold_bounds(self):
        with pytest.raises(ValidationError):
            Score(name="bad", value=0.5, threshold=1.5)
