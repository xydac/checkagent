"""Tests for MockTool — schema validation, call recording, and assertions."""

from __future__ import annotations

import pytest

from checkagent.mock.tool import (
    MockTool,
    MockToolError,
    ToolCallRecord,
    ToolExecutionError,
    ToolNotFoundError,
    ToolSchema,
    ToolValidationError,
)


# --- ToolSchema validation ---


class TestToolSchema:
    def test_validate_required_field_present(self):
        schema = ToolSchema(
            name="test",
            parameters={"properties": {"q": {"type": "string"}}, "required": ["q"]},
        )
        assert schema.validate_args({"q": "hello"}) == []

    def test_validate_required_field_missing(self):
        schema = ToolSchema(
            name="test",
            parameters={"properties": {"q": {"type": "string"}}, "required": ["q"]},
        )
        errors = schema.validate_args({})
        assert len(errors) == 1
        assert "Missing required argument: q" in errors[0]

    def test_validate_type_string(self):
        schema = ToolSchema(
            name="test",
            parameters={"properties": {"q": {"type": "string"}}},
        )
        assert schema.validate_args({"q": "hello"}) == []
        errors = schema.validate_args({"q": 42})
        assert len(errors) == 1
        assert "expected string" in errors[0]

    def test_validate_type_integer(self):
        schema = ToolSchema(
            name="test",
            parameters={"properties": {"n": {"type": "integer"}}},
        )
        assert schema.validate_args({"n": 5}) == []
        errors = schema.validate_args({"n": "five"})
        assert "expected integer" in errors[0]

    def test_validate_type_boolean_not_integer(self):
        """bool is subclass of int in Python — schema validation handles this."""
        schema = ToolSchema(
            name="test",
            parameters={"properties": {"n": {"type": "integer"}}},
        )
        errors = schema.validate_args({"n": True})
        assert len(errors) == 1

    def test_validate_type_number(self):
        schema = ToolSchema(
            name="test",
            parameters={"properties": {"x": {"type": "number"}}},
        )
        assert schema.validate_args({"x": 3.14}) == []
        assert schema.validate_args({"x": 42}) == []

    def test_validate_type_array(self):
        schema = ToolSchema(
            name="test",
            parameters={"properties": {"items": {"type": "array"}}},
        )
        assert schema.validate_args({"items": [1, 2, 3]}) == []
        errors = schema.validate_args({"items": "not a list"})
        assert "expected array" in errors[0]

    def test_validate_type_object(self):
        schema = ToolSchema(
            name="test",
            parameters={"properties": {"data": {"type": "object"}}},
        )
        assert schema.validate_args({"data": {"a": 1}}) == []

    def test_validate_additional_properties_false(self):
        schema = ToolSchema(
            name="test",
            parameters={
                "properties": {"q": {"type": "string"}},
                "additionalProperties": False,
            },
        )
        errors = schema.validate_args({"q": "ok", "extra": "bad"})
        assert len(errors) == 1
        assert "Unexpected argument: extra" in errors[0]

    def test_validate_additional_properties_allowed_by_default(self):
        schema = ToolSchema(
            name="test",
            parameters={"properties": {"q": {"type": "string"}}},
        )
        assert schema.validate_args({"q": "ok", "extra": "fine"}) == []

    def test_validate_multiple_errors(self):
        schema = ToolSchema(
            name="test",
            parameters={
                "properties": {"q": {"type": "string"}, "n": {"type": "integer"}},
                "required": ["q", "n"],
            },
        )
        errors = schema.validate_args({})
        assert len(errors) == 2


# --- MockTool basics ---


class TestMockToolBasics:
    @pytest.mark.asyncio
    async def test_register_and_call(self):
        tool = MockTool()
        tool.register("greet", response="Hello!")
        result = await tool.call("greet")
        assert result == "Hello!"

    @pytest.mark.asyncio
    async def test_call_with_arguments(self):
        tool = MockTool()
        tool.register("add", response=42)
        result = await tool.call("add", {"a": 20, "b": 22})
        assert result == 42

    @pytest.mark.asyncio
    async def test_call_unknown_tool_raises(self):
        tool = MockTool()
        with pytest.raises(ToolNotFoundError) as exc_info:
            await tool.call("nonexistent")
        assert "nonexistent" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_call_unknown_tool_with_default_response(self):
        tool = MockTool(default_response="fallback")
        result = await tool.call("anything")
        assert result == "fallback"

    @pytest.mark.asyncio
    async def test_dict_response(self):
        tool = MockTool()
        tool.register("weather", response={"temp": 72, "unit": "F"})
        result = await tool.call("weather", {"city": "NYC"})
        assert result == {"temp": 72, "unit": "F"}

    @pytest.mark.asyncio
    async def test_none_response(self):
        tool = MockTool()
        tool.register("noop", response=None)
        result = await tool.call("noop")
        assert result is None

    def test_sync_call(self):
        tool = MockTool()
        tool.register("greet", response="Hi")
        result = tool.call_sync("greet")
        assert result == "Hi"

    def test_chaining(self):
        tool = MockTool()
        result = tool.register("a", response=1).register("b", response=2)
        assert result is tool
        assert tool.registered_tools == ["a", "b"]


# --- Schema validation ---


class TestMockToolValidation:
    @pytest.mark.asyncio
    async def test_valid_args_pass(self):
        tool = MockTool()
        tool.register(
            "search",
            response={"results": []},
            schema={"properties": {"query": {"type": "string"}}, "required": ["query"]},
        )
        result = await tool.call("search", {"query": "hello"})
        assert result == {"results": []}

    @pytest.mark.asyncio
    async def test_missing_required_raises(self):
        tool = MockTool()
        tool.register(
            "search",
            response={"results": []},
            schema={"properties": {"query": {"type": "string"}}, "required": ["query"]},
        )
        with pytest.raises(ToolValidationError) as exc_info:
            await tool.call("search", {})
        assert "query" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_wrong_type_raises(self):
        tool = MockTool()
        tool.register(
            "search",
            response={"results": []},
            schema={"properties": {"query": {"type": "string"}}},
        )
        with pytest.raises(ToolValidationError):
            await tool.call("search", {"query": 123})

    @pytest.mark.asyncio
    async def test_non_strict_records_errors_but_succeeds(self):
        tool = MockTool(strict_validation=False)
        tool.register(
            "search",
            response="ok",
            schema={"properties": {"query": {"type": "string"}}, "required": ["query"]},
        )
        result = await tool.call("search", {})
        assert result == "ok"
        assert tool.last_call is not None
        assert len(tool.last_call.validation_errors) > 0

    @pytest.mark.asyncio
    async def test_validation_error_is_recorded(self):
        tool = MockTool()
        tool.register(
            "search",
            response="ok",
            schema={"properties": {"query": {"type": "string"}}, "required": ["query"]},
        )
        with pytest.raises(ToolValidationError):
            await tool.call("search", {})
        assert tool.call_count == 1
        assert tool.last_call is not None
        assert tool.last_call.error is not None


# --- Error simulation ---


class TestMockToolErrors:
    @pytest.mark.asyncio
    async def test_configured_error_raises(self):
        tool = MockTool()
        tool.register("fail_tool", error="Connection timeout")
        with pytest.raises(ToolExecutionError) as exc_info:
            await tool.call("fail_tool")
        assert "Connection timeout" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_configured_error_recorded(self):
        tool = MockTool()
        tool.register("fail_tool", error="timeout")
        with pytest.raises(ToolExecutionError):
            await tool.call("fail_tool")
        assert tool.last_call is not None
        assert tool.last_call.error == "timeout"


# --- Sequential responses ---


class TestMockToolSequences:
    @pytest.mark.asyncio
    async def test_sequential_responses(self):
        tool = MockTool()
        tool.register("roll", response=[4, 2, 6])
        assert await tool.call("roll") == 4
        assert await tool.call("roll") == 2
        assert await tool.call("roll") == 6

    @pytest.mark.asyncio
    async def test_sequential_responses_cycle(self):
        tool = MockTool()
        tool.register("flip", response=["heads", "tails"])
        assert await tool.call("flip") == "heads"
        assert await tool.call("flip") == "tails"
        assert await tool.call("flip") == "heads"  # cycles

    def test_sync_sequential(self):
        tool = MockTool()
        tool.register("count", response=[1, 2, 3])
        assert tool.call_sync("count") == 1
        assert tool.call_sync("count") == 2
        assert tool.call_sync("count") == 3


# --- Call recording ---


class TestMockToolRecording:
    @pytest.mark.asyncio
    async def test_call_count(self):
        tool = MockTool()
        tool.register("a", response=1)
        await tool.call("a")
        await tool.call("a")
        assert tool.call_count == 2

    @pytest.mark.asyncio
    async def test_calls_list(self):
        tool = MockTool()
        tool.register("a", response=1)
        tool.register("b", response=2)
        await tool.call("a", {"x": 1})
        await tool.call("b", {"y": 2})
        assert len(tool.calls) == 2
        assert tool.calls[0].tool_name == "a"
        assert tool.calls[0].arguments == {"x": 1}
        assert tool.calls[1].tool_name == "b"

    @pytest.mark.asyncio
    async def test_last_call(self):
        tool = MockTool()
        tool.register("a", response=1)
        assert tool.last_call is None
        await tool.call("a", {"x": 1})
        assert tool.last_call is not None
        assert tool.last_call.tool_name == "a"

    @pytest.mark.asyncio
    async def test_get_calls_for(self):
        tool = MockTool()
        tool.register("a", response=1)
        tool.register("b", response=2)
        await tool.call("a")
        await tool.call("b")
        await tool.call("a")
        assert len(tool.get_calls_for("a")) == 2
        assert len(tool.get_calls_for("b")) == 1

    @pytest.mark.asyncio
    async def test_was_called(self):
        tool = MockTool()
        tool.register("a", response=1)
        tool.register("b", response=2)
        await tool.call("a")
        assert tool.was_called("a") is True
        assert tool.was_called("b") is False

    @pytest.mark.asyncio
    async def test_calls_returns_copy(self):
        tool = MockTool()
        tool.register("a", response=1)
        await tool.call("a")
        calls = tool.calls
        calls.clear()
        assert tool.call_count == 1  # original unchanged


# --- Assertion helpers ---


class TestAssertionHelpers:
    @pytest.mark.asyncio
    async def test_assert_tool_called_passes(self):
        tool = MockTool()
        tool.register("a", response=1)
        await tool.call("a")
        tool.assert_tool_called("a")  # should not raise

    @pytest.mark.asyncio
    async def test_assert_tool_called_fails_descriptively(self):
        tool = MockTool()
        tool.register("a", response=1)
        tool.register("b", response=2)
        await tool.call("a")
        with pytest.raises(AssertionError, match="never called"):
            tool.assert_tool_called("b")

    @pytest.mark.asyncio
    async def test_assert_tool_called_with_times(self):
        tool = MockTool()
        tool.register("a", response=1)
        await tool.call("a")
        await tool.call("a")
        tool.assert_tool_called("a", times=2)

    @pytest.mark.asyncio
    async def test_assert_tool_called_times_mismatch(self):
        tool = MockTool()
        tool.register("a", response=1)
        await tool.call("a")
        with pytest.raises(AssertionError, match="1 time"):
            tool.assert_tool_called("a", times=3)

    @pytest.mark.asyncio
    async def test_assert_tool_called_with_args(self):
        tool = MockTool()
        tool.register("search", response="ok")
        await tool.call("search", {"query": "hello"})
        tool.assert_tool_called("search", with_args={"query": "hello"})

    @pytest.mark.asyncio
    async def test_assert_tool_called_with_args_mismatch(self):
        tool = MockTool()
        tool.register("search", response="ok")
        await tool.call("search", {"query": "hello"})
        with pytest.raises(AssertionError, match="never called with"):
            tool.assert_tool_called("search", with_args={"query": "world"})

    @pytest.mark.asyncio
    async def test_assert_tool_not_called_passes(self):
        tool = MockTool()
        tool.register("a", response=1)
        tool.assert_tool_not_called("a")  # should not raise

    @pytest.mark.asyncio
    async def test_assert_tool_not_called_fails(self):
        tool = MockTool()
        tool.register("a", response=1)
        await tool.call("a")
        with pytest.raises(AssertionError, match="1 time"):
            tool.assert_tool_not_called("a")

    @pytest.mark.asyncio
    async def test_assert_no_calls_lists_called_tools(self):
        """Error message lists what WAS called when expected tool wasn't."""
        tool = MockTool()
        tool.register("x", response=1)
        await tool.call("x")
        with pytest.raises(AssertionError, match="x"):
            tool.assert_tool_called("y")


# --- Reset ---


class TestMockToolReset:
    @pytest.mark.asyncio
    async def test_reset_clears_calls_and_counters(self):
        tool = MockTool()
        tool.register("a", response=[1, 2])
        await tool.call("a")
        tool.reset()
        assert tool.call_count == 0
        assert await tool.call("a") == 1  # counter reset

    @pytest.mark.asyncio
    async def test_reset_calls_preserves_counters(self):
        tool = MockTool()
        tool.register("a", response=[1, 2])
        await tool.call("a")  # returns 1
        tool.reset_calls()
        assert tool.call_count == 0
        assert await tool.call("a") == 2  # counter preserved


# --- Exception hierarchy ---


class TestExceptions:
    def test_all_errors_are_mock_tool_error(self):
        assert issubclass(ToolNotFoundError, MockToolError)
        assert issubclass(ToolValidationError, MockToolError)
        assert issubclass(ToolExecutionError, MockToolError)

    def test_validation_error_has_errors_list(self):
        err = ToolValidationError("test", ["error1", "error2"])
        assert err.validation_errors == ["error1", "error2"]
        assert err.tool_name == "test"
