"""Tests for structured output assertions (F1.8, F12.1-F12.5)."""

from __future__ import annotations

import pytest
from pydantic import BaseModel, Field

from checkagent.core.types import AgentInput, AgentRun, Step, ToolCall
from checkagent.eval.assertions import (
    StructuredAssertionError,
    assert_json_schema,
    assert_output_matches,
    assert_output_schema,
    assert_tool_called,
)

# -- Test models --


class BookingResult(BaseModel):
    confirmed: bool
    event_id: str
    message: str


class UserProfile(BaseModel):
    name: str
    age: int = Field(ge=0)
    email: str | None = None


class NestedModel(BaseModel):
    user: UserProfile
    tags: list[str] = Field(default_factory=list)


# -- Helpers --


def _make_run(output: object) -> AgentRun:
    return AgentRun(
        input=AgentInput(query="test"),
        final_output=output,
    )


def _make_run_with_tools(*tool_calls: ToolCall) -> AgentRun:
    return AgentRun(
        input=AgentInput(query="test"),
        steps=[Step(step_index=0, tool_calls=list(tool_calls))],
        final_output="done",
    )


# ============================================================
# assert_output_schema (F12.1)
# ============================================================


class TestAssertOutputSchema:
    def test_dict_output_validates(self):
        run = _make_run({"confirmed": True, "event_id": "e-123", "message": "ok"})
        result = assert_output_schema(run, BookingResult)
        assert isinstance(result, BookingResult)
        assert result.confirmed is True
        assert result.event_id == "e-123"

    def test_json_string_output_validates(self):
        import json

        data = {"confirmed": False, "event_id": "e-456", "message": "cancelled"}
        run = _make_run(json.dumps(data))
        result = assert_output_schema(run, BookingResult)
        assert result.confirmed is False

    def test_pydantic_instance_passthrough(self):
        booking = BookingResult(confirmed=True, event_id="e-1", message="hi")
        run = _make_run(booking)
        result = assert_output_schema(run, BookingResult)
        assert result is booking

    def test_pydantic_cross_model_conversion(self):
        """A different BaseModel instance gets dumped and re-validated."""

        class AltBooking(BaseModel):
            confirmed: bool
            event_id: str
            message: str

        alt = AltBooking(confirmed=True, event_id="e-2", message="alt")
        run = _make_run(alt)
        result = assert_output_schema(run, BookingResult)
        assert isinstance(result, BookingResult)

    def test_missing_required_field_raises(self):
        run = _make_run({"confirmed": True})
        with pytest.raises(StructuredAssertionError, match="event_id"):
            assert_output_schema(run, BookingResult)

    def test_wrong_type_raises(self):
        run = _make_run({"confirmed": "not_bool", "event_id": "e-1", "message": "ok"})
        # Pydantic coerces "not_bool" to bool in lax mode, so use strict
        with pytest.raises(StructuredAssertionError, match="BookingResult"):
            assert_output_schema(run, BookingResult, strict=True)

    def test_invalid_json_string_raises(self):
        run = _make_run("this is not json")
        with pytest.raises(StructuredAssertionError, match="not valid JSON"):
            assert_output_schema(run, BookingResult)

    def test_unsupported_type_raises(self):
        run = _make_run(42)
        with pytest.raises(StructuredAssertionError, match="cannot be validated"):
            assert_output_schema(run, BookingResult)

    def test_nested_model_validates(self):
        data = {
            "user": {"name": "Alice", "age": 30},
            "tags": ["admin"],
        }
        run = _make_run(data)
        result = assert_output_schema(run, NestedModel)
        assert result.user.name == "Alice"
        assert result.tags == ["admin"]

    def test_nested_validation_error_shows_path(self):
        data = {"user": {"name": "Alice", "age": -5}, "tags": []}
        run = _make_run(data)
        with pytest.raises(StructuredAssertionError, match="age"):
            assert_output_schema(run, NestedModel)

    def test_raw_value_without_agentrun(self):
        """Can pass a raw dict instead of an AgentRun."""
        data = {"confirmed": True, "event_id": "e-1", "message": "ok"}
        result = assert_output_schema(data, BookingResult)
        assert isinstance(result, BookingResult)

    def test_error_details_populated(self):
        run = _make_run({"confirmed": True})
        with pytest.raises(StructuredAssertionError) as exc_info:
            assert_output_schema(run, BookingResult)
        assert exc_info.value.details is not None
        assert "errors" in exc_info.value.details


# ============================================================
# assert_output_matches (F12.2)
# ============================================================


class TestAssertOutputMatches:
    def test_exact_dict_match(self):
        run = _make_run({"status": "ok", "count": 3})
        assert_output_matches(run, {"status": "ok", "count": 3})

    def test_partial_match_ignores_extra_keys(self):
        run = _make_run({"status": "ok", "count": 3, "extra": "ignored"})
        assert_output_matches(run, {"status": "ok"})

    def test_mismatch_raises(self):
        run = _make_run({"status": "error", "count": 3})
        with pytest.raises(StructuredAssertionError, match="status"):
            assert_output_matches(run, {"status": "ok"})

    def test_missing_key_raises(self):
        run = _make_run({"status": "ok"})
        with pytest.raises(StructuredAssertionError, match="missing"):
            assert_output_matches(run, {"count": 5})

    def test_nested_dict_match(self):
        run = _make_run({"data": {"inner": "value"}, "meta": "x"})
        assert_output_matches(run, {"data": {"inner": "value"}})

    def test_nested_dict_mismatch(self):
        run = _make_run({"data": {"inner": "wrong"}})
        with pytest.raises(StructuredAssertionError, match="inner"):
            assert_output_matches(run, {"data": {"inner": "expected"}})

    def test_json_string_parsed(self):
        import json

        run = _make_run(json.dumps({"status": "ok"}))
        assert_output_matches(run, {"status": "ok"})

    def test_pydantic_model_dumped(self):
        booking = BookingResult(confirmed=True, event_id="e-1", message="hi")
        run = _make_run(booking)
        assert_output_matches(run, {"confirmed": True, "event_id": "e-1"})

    def test_scalar_match(self):
        run = _make_run("hello")
        assert_output_matches(run, "hello")

    def test_scalar_mismatch(self):
        run = _make_run("hello")
        with pytest.raises(StructuredAssertionError, match="does not match"):
            assert_output_matches(run, "world")

    def test_raw_value_without_agentrun(self):
        assert_output_matches({"a": 1, "b": 2}, {"a": 1})

    def test_dirty_equals_integration(self):
        """Test with dirty-equals matchers for fuzzy matching."""
        from dirty_equals import IsPositiveInt, IsStr

        run = _make_run({"name": "Alice", "age": 30, "role": "admin"})
        assert_output_matches(run, {"name": IsStr, "age": IsPositiveInt})

    def test_dirty_equals_mismatch(self):
        from dirty_equals import IsPositiveInt

        run = _make_run({"count": -5})
        with pytest.raises(StructuredAssertionError, match="count"):
            assert_output_matches(run, {"count": IsPositiveInt})


# ============================================================
# assert_json_schema (F12.3)
# ============================================================


SIMPLE_SCHEMA = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "age": {"type": "integer", "minimum": 0},
    },
    "required": ["name", "age"],
}


_has_jsonschema = True
try:
    import jsonschema as _jsonschema  # noqa: F401
except ImportError:
    _has_jsonschema = False


@pytest.mark.skipif(not _has_jsonschema, reason="jsonschema not installed")
class TestAssertJsonSchema:
    def test_valid_dict(self):
        assert_json_schema({"name": "Alice", "age": 30}, SIMPLE_SCHEMA)

    def test_valid_json_string(self):
        import json

        assert_json_schema(json.dumps({"name": "Bob", "age": 25}), SIMPLE_SCHEMA)

    def test_missing_required_field(self):
        with pytest.raises(StructuredAssertionError, match="required"):
            assert_json_schema({"name": "Alice"}, SIMPLE_SCHEMA)

    def test_wrong_type(self):
        with pytest.raises(StructuredAssertionError, match="type"):
            assert_json_schema({"name": "Alice", "age": "thirty"}, SIMPLE_SCHEMA)

    def test_minimum_violation(self):
        with pytest.raises(StructuredAssertionError, match="minimum"):
            assert_json_schema({"name": "Alice", "age": -1}, SIMPLE_SCHEMA)

    def test_invalid_json_string(self):
        with pytest.raises(StructuredAssertionError, match="not valid JSON"):
            assert_json_schema("not json", SIMPLE_SCHEMA)

    def test_pydantic_model_dumped(self):
        profile = UserProfile(name="Alice", age=30)
        assert_json_schema(profile, SIMPLE_SCHEMA)

    def test_error_shows_path(self):
        nested_schema = {
            "type": "object",
            "properties": {
                "data": {
                    "type": "object",
                    "properties": {"value": {"type": "integer"}},
                    "required": ["value"],
                }
            },
            "required": ["data"],
        }
        with pytest.raises(StructuredAssertionError, match="value"):
            assert_json_schema({"data": {"value": "not_int"}}, nested_schema)

    def test_error_details_populated(self):
        with pytest.raises(StructuredAssertionError) as exc_info:
            assert_json_schema({"name": 123, "age": 30}, SIMPLE_SCHEMA)
        assert exc_info.value.details is not None
        assert "message" in exc_info.value.details

    def test_array_schema(self):
        schema = {"type": "array", "items": {"type": "string"}, "minItems": 1}
        assert_json_schema(["a", "b"], schema)

    def test_array_schema_violation(self):
        schema = {"type": "array", "items": {"type": "string"}, "minItems": 1}
        with pytest.raises(StructuredAssertionError):
            assert_json_schema([], schema)


# ============================================================
# assert_tool_called (F12.4)
# ============================================================


class TestAssertToolCalled:
    def test_tool_found(self):
        tc = ToolCall(name="search", arguments={"query": "hello"})
        run = _make_run_with_tools(tc)
        result = assert_tool_called(run, "search")
        assert result.name == "search"

    def test_tool_not_found_raises(self):
        tc = ToolCall(name="search", arguments={"query": "hello"})
        run = _make_run_with_tools(tc)
        with pytest.raises(StructuredAssertionError, match="never called"):
            assert_tool_called(run, "create_event")

    def test_tool_not_found_shows_available(self):
        tc = ToolCall(name="search", arguments={})
        run = _make_run_with_tools(tc)
        with pytest.raises(StructuredAssertionError, match="search"):
            assert_tool_called(run, "missing")

    def test_argument_exact_match(self):
        tc = ToolCall(name="search", arguments={"query": "hello", "limit": 10})
        run = _make_run_with_tools(tc)
        result = assert_tool_called(run, "search", query="hello")
        assert result.arguments["limit"] == 10

    def test_argument_mismatch_raises(self):
        tc = ToolCall(name="search", arguments={"query": "hello"})
        run = _make_run_with_tools(tc)
        with pytest.raises(StructuredAssertionError, match="query"):
            assert_tool_called(run, "search", query="world")

    def test_missing_argument_raises(self):
        tc = ToolCall(name="search", arguments={"query": "hello"})
        run = _make_run_with_tools(tc)
        with pytest.raises(StructuredAssertionError, match="missing"):
            assert_tool_called(run, "search", limit=10)

    def test_multiple_calls_any_match(self):
        tc1 = ToolCall(name="search", arguments={"query": "first"})
        tc2 = ToolCall(name="search", arguments={"query": "second"})
        run = _make_run_with_tools(tc1, tc2)
        result = assert_tool_called(run, "search", query="second")
        assert result.arguments["query"] == "second"

    def test_call_index_specific(self):
        tc1 = ToolCall(name="search", arguments={"query": "first"})
        tc2 = ToolCall(name="search", arguments={"query": "second"})
        run = _make_run_with_tools(tc1, tc2)
        result = assert_tool_called(run, "search", call_index=1, query="second")
        assert result.arguments["query"] == "second"

    def test_call_index_mismatch_raises(self):
        tc1 = ToolCall(name="search", arguments={"query": "first"})
        tc2 = ToolCall(name="search", arguments={"query": "second"})
        run = _make_run_with_tools(tc1, tc2)
        with pytest.raises(StructuredAssertionError, match="query"):
            assert_tool_called(run, "search", call_index=0, query="second")

    def test_call_index_out_of_range(self):
        tc = ToolCall(name="search", arguments={"query": "hello"})
        run = _make_run_with_tools(tc)
        with pytest.raises(StructuredAssertionError, match="1 time"):
            assert_tool_called(run, "search", call_index=5)

    def test_no_tool_calls_at_all(self):
        run = AgentRun(input=AgentInput(query="test"), final_output="done")
        with pytest.raises(StructuredAssertionError, match="never called"):
            assert_tool_called(run, "anything")

    def test_dirty_equals_argument_matching(self):
        from dirty_equals import IsPositiveInt, IsStr

        tc = ToolCall(
            name="create_event",
            arguments={"title": "Meeting", "duration": 60, "attendees": ["alice"]},
        )
        run = _make_run_with_tools(tc)
        result = assert_tool_called(
            run, "create_event", title=IsStr, duration=IsPositiveInt
        )
        assert result.arguments["attendees"] == ["alice"]

    def test_dirty_equals_argument_mismatch(self):
        from dirty_equals import IsPositiveInt

        tc = ToolCall(name="create_event", arguments={"duration": -5})
        run = _make_run_with_tools(tc)
        with pytest.raises(StructuredAssertionError, match="duration"):
            assert_tool_called(run, "create_event", duration=IsPositiveInt)


# ============================================================
# Deep diff on failure (F12.5)
# ============================================================


class TestDeepDiff:
    def test_schema_error_is_structured(self):
        run = _make_run({"confirmed": True, "event_id": 123, "message": "ok"})
        with pytest.raises(StructuredAssertionError) as exc_info:
            assert_output_schema(run, BookingResult, strict=True)
        assert exc_info.value.details is not None

    def test_match_error_shows_field_path(self):
        run = _make_run({"data": {"inner": {"deep": "wrong"}}})
        with pytest.raises(StructuredAssertionError, match="deep"):
            assert_output_matches(run, {"data": {"inner": {"deep": "expected"}}})

    def test_tool_error_shows_actual_calls(self):
        tc = ToolCall(name="search", arguments={"query": "actual"})
        run = _make_run_with_tools(tc)
        with pytest.raises(StructuredAssertionError) as exc_info:
            assert_tool_called(run, "search", query="expected")
        assert "actual_calls" in exc_info.value.details


# ============================================================
# StructuredAssertionError
# ============================================================


class TestStructuredAssertionError:
    def test_is_assertion_error(self):
        err = StructuredAssertionError("msg")
        assert isinstance(err, AssertionError)

    def test_has_details(self):
        err = StructuredAssertionError("msg", details={"key": "val"})
        assert err.details == {"key": "val"}

    def test_details_default_none(self):
        err = StructuredAssertionError("msg")
        assert err.details is None
