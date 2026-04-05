"""Structured output assertions for agent test results.

Provides rich assertion helpers for validating agent outputs against
Pydantic models, JSON schemas, and partial/fuzzy patterns. Produces
detailed failure messages showing exactly which fields diverged.

Requirements: F1.8, F12.1-F12.5
"""

from __future__ import annotations

import contextlib
import json
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

from checkagent.core.types import AgentRun, ToolCall

T = TypeVar("T", bound=BaseModel)


class StructuredAssertionError(AssertionError):
    """Rich assertion error with structured diff information."""

    def __init__(self, message: str, *, details: Any = None) -> None:
        self.details = details
        super().__init__(message)


def _extract_output(result: AgentRun | Any) -> Any:
    """Extract the output value from an AgentRun or pass through raw values."""
    if isinstance(result, AgentRun):
        return result.final_output
    return result


def _format_deep_diff(expected: Any, actual: Any) -> str:
    """Format a deep diff between expected and actual values.

    Uses deepdiff if available, falls back to basic comparison.
    """
    try:
        from deepdiff import DeepDiff

        diff = DeepDiff(expected, actual, verbose_level=2)
        if not diff:
            return ""
        lines = []
        if "values_changed" in diff:
            for path, change in diff["values_changed"].items():
                lines.append(
                    f"  {path}: {change['old_value']!r} → {change['new_value']!r}"
                )
        if "dictionary_item_added" in diff:
            for path in diff["dictionary_item_added"]:
                lines.append(f"  {path}: <missing> → (unexpected key present)")
        if "dictionary_item_removed" in diff:
            for path in diff["dictionary_item_removed"]:
                lines.append(f"  {path}: (expected key missing)")
        if "type_changes" in diff:
            for path, change in diff["type_changes"].items():
                lines.append(
                    f"  {path}: type {change['old_type'].__name__}"
                    f" → {change['new_type'].__name__}"
                )
        if "iterable_item_added" in diff:
            for path, val in diff["iterable_item_added"].items():
                lines.append(f"  {path}: <missing> → {val!r}")
        if "iterable_item_removed" in diff:
            for path, val in diff["iterable_item_removed"].items():
                lines.append(f"  {path}: {val!r} → <missing>")
        return "\n".join(lines) if lines else str(diff)
    except ImportError:
        return f"  expected: {expected!r}\n  actual:   {actual!r}"


def assert_output_schema(
    result: AgentRun | Any,
    model: type[T],
    *,
    strict: bool = False,
) -> T:
    """Validate that the agent's output parses into the given Pydantic model.

    Args:
        result: An AgentRun (uses final_output) or a raw value.
        model: The Pydantic model class to validate against.
        strict: If True, use Pydantic's strict mode (no coercion).

    Returns:
        The validated Pydantic model instance.

    Raises:
        StructuredAssertionError: If validation fails, with field-level detail.
    """
    output = _extract_output(result)

    if isinstance(output, model):
        return output

    # Try to parse the output
    try:
        if isinstance(output, str):
            # Try JSON parsing first
            try:
                data = json.loads(output)
            except (json.JSONDecodeError, TypeError) as err:
                raise StructuredAssertionError(
                    f"Output is a string that is not valid JSON.\n"
                    f"  output: {output!r}\n"
                    f"  expected: {model.__name__}",
                    details={"output": output, "model": model.__name__},
                ) from err
        elif isinstance(output, dict):
            data = output
        elif isinstance(output, BaseModel):
            data = output.model_dump()
        else:
            raise StructuredAssertionError(
                f"Output type {type(output).__name__} cannot be validated "
                f"against {model.__name__}.\n"
                f"  expected types: dict, str (JSON), or {model.__name__}\n"
                f"  got: {type(output).__name__} = {output!r}",
                details={"output": output, "model": model.__name__},
            )

        if strict:
            return model.model_validate(data, strict=True)
        return model.model_validate(data)

    except ValidationError as e:
        # Format Pydantic errors with field-level detail
        error_lines = []
        for err in e.errors():
            loc = " → ".join(str(x) for x in err["loc"]) if err["loc"] else "(root)"
            error_lines.append(f"  {loc}: {err['msg']} (type={err['type']})")

        raise StructuredAssertionError(
            f"Output does not match {model.__name__}:\n"
            + "\n".join(error_lines),
            details={"errors": e.errors(), "output": data, "model": model.__name__},
        ) from e


def assert_output_matches(
    result: AgentRun | Any,
    pattern: dict[str, Any] | Any,
) -> None:
    """Assert output matches a partial/fuzzy pattern.

    Works with plain values for exact matching, or with dirty-equals
    matchers for flexible structural matching.

    Args:
        result: An AgentRun (uses final_output) or a raw value.
        pattern: A dict of field patterns (supports dirty-equals matchers),
                 or any value for direct comparison.

    Raises:
        StructuredAssertionError: If any field doesn't match the pattern.
    """
    output = _extract_output(result)

    if isinstance(output, str):
        with contextlib.suppress(json.JSONDecodeError, TypeError):
            output = json.loads(output)

    if isinstance(output, BaseModel):
        output = output.model_dump()

    if isinstance(pattern, dict) and isinstance(output, dict):
        _match_dict(output, pattern, path="output")
    elif output != pattern:
        diff = _format_deep_diff(pattern, output)
        raise StructuredAssertionError(
            f"Output does not match pattern:\n{diff}",
            details={"expected": pattern, "actual": output},
        )


def _match_dict(actual: dict, pattern: dict, path: str) -> None:
    """Recursively match a dict against a pattern dict.

    Only keys present in the pattern are checked (partial matching).
    """
    mismatches = []

    for key, expected_value in pattern.items():
        if key not in actual:
            mismatches.append(f"  {path}.{key}: key missing from output")
            continue

        actual_value = actual[key]

        if isinstance(expected_value, dict) and isinstance(actual_value, dict):
            try:
                _match_dict(actual_value, expected_value, path=f"{path}.{key}")
            except StructuredAssertionError as e:
                mismatches.append(str(e))
                continue
        elif actual_value != expected_value:
            mismatches.append(
                f"  {path}.{key}: {actual_value!r} does not match {expected_value!r}"
            )

    if mismatches:
        raise StructuredAssertionError(
            "Output does not match pattern:\n" + "\n".join(mismatches),
            details={"path": path, "pattern": pattern, "actual": actual},
        )


def assert_json_schema(
    output: Any,
    schema: dict[str, Any],
) -> None:
    """Validate output against a JSON Schema.

    Args:
        output: The value to validate (dict, list, or JSON string).
        schema: A JSON Schema dict.

    Raises:
        StructuredAssertionError: If validation fails.
        ImportError: If jsonschema is not installed.
    """
    try:
        import jsonschema
    except ImportError as err:
        raise ImportError(
            "jsonschema is required for assert_json_schema. "
            "Install it with: pip install jsonschema"
        ) from err

    if isinstance(output, str):
        try:
            output = json.loads(output)
        except (json.JSONDecodeError, TypeError) as e:
            raise StructuredAssertionError(
                f"Output is not valid JSON: {e}",
                details={"output": output},
            ) from e

    if isinstance(output, BaseModel):
        output = output.model_dump()

    try:
        jsonschema.validate(instance=output, schema=schema)
    except jsonschema.ValidationError as e:
        path = " → ".join(str(x) for x in e.absolute_path) if e.absolute_path else "(root)"
        raise StructuredAssertionError(
            f"JSON Schema validation failed at {path}:\n"
            f"  {e.message}\n"
            f"  schema path: {' → '.join(str(x) for x in e.absolute_schema_path)}",
            details={
                "message": e.message,
                "path": list(e.absolute_path),
                "schema_path": list(e.absolute_schema_path),
                "instance": output,
            },
        ) from e


def assert_tool_called(
    result: AgentRun,
    tool_name: str,
    *,
    call_index: int | None = None,
    **expected_args: Any,
) -> ToolCall:
    """Assert a tool was called with specific argument patterns.

    Args:
        result: The AgentRun to inspect.
        tool_name: Name of the tool that should have been called.
        call_index: If given, check a specific call (0-based). Otherwise
                    checks that at least one call matches all patterns.
        **expected_args: Argument patterns to match. Values can be exact
                        matches or dirty-equals matchers.

    Returns:
        The matching ToolCall.

    Raises:
        StructuredAssertionError: If no matching call is found.
    """
    calls = result.get_tool_calls(tool_name)

    if not calls:
        all_names = sorted({tc.name for tc in result.tool_calls})
        raise StructuredAssertionError(
            f"Tool '{tool_name}' was never called.\n"
            f"  tools called: {all_names or '(none)'}",
            details={"tool_name": tool_name, "available": all_names},
        )

    if call_index is not None:
        if call_index >= len(calls):
            raise StructuredAssertionError(
                f"Tool '{tool_name}' was called {len(calls)} time(s), "
                f"but call_index={call_index} requested.",
                details={"tool_name": tool_name, "call_count": len(calls)},
            )
        calls_to_check = [calls[call_index]]
    else:
        calls_to_check = calls

    if not expected_args:
        return calls_to_check[0]

    # Check each candidate call against expected args
    best_mismatch: list[str] = []
    for call in calls_to_check:
        mismatches = []
        for key, expected in expected_args.items():
            if key not in call.arguments:
                mismatches.append(f"  {key}: missing from arguments")
            elif call.arguments[key] != expected:
                mismatches.append(
                    f"  {key}: {call.arguments[key]!r} does not match {expected!r}"
                )
        if not mismatches:
            return call
        if not best_mismatch or len(mismatches) < len(best_mismatch):
            best_mismatch = mismatches

    if call_index is not None:
        prefix = f"Tool '{tool_name}' call[{call_index}] arguments don't match:"
    else:
        prefix = (
            f"Tool '{tool_name}' was called {len(calls)} time(s), "
            f"but no call matched all expected arguments:"
        )

    raise StructuredAssertionError(
        f"{prefix}\n" + "\n".join(best_mismatch),
        details={
            "tool_name": tool_name,
            "expected_args": expected_args,
            "actual_calls": [c.arguments for c in calls],
        },
    )
