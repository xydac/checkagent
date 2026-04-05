"""MockTool — configurable tool executor with schema validation and call recording.

Intercepts tool/function calls and returns configured responses.
Validates tool call arguments against JSON Schema definitions.
Records all tool call attempts for assertion in tests.

Implements F1.2 from the PRD.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ToolSchema(BaseModel):
    """JSON Schema definition for a tool's parameters."""

    name: str
    description: str = ""
    parameters: dict[str, Any] = Field(default_factory=dict)

    def validate_args(self, arguments: dict[str, Any]) -> list[str]:
        """Validate arguments against this schema. Returns list of errors."""
        errors: list[str] = []
        props = self.parameters.get("properties", {})
        required = self.parameters.get("required", [])

        # Check required fields
        for field in required:
            if field not in arguments:
                errors.append(f"Missing required argument: {field}")

        # Check types of provided fields
        for key, value in arguments.items():
            if key not in props:
                if not self.parameters.get("additionalProperties", True):
                    errors.append(f"Unexpected argument: {key}")
                continue
            expected_type = props[key].get("type")
            if expected_type and not _type_matches(value, expected_type):
                errors.append(
                    f"Argument '{key}': expected {expected_type}, "
                    f"got {type(value).__name__}"
                )

        return errors


def _type_matches(value: Any, json_type: str) -> bool:
    """Check if a Python value matches a JSON Schema type."""
    type_map: dict[str, tuple[type, ...]] = {
        "string": (str,),
        "number": (int, float),
        "integer": (int,),
        "boolean": (bool,),
        "array": (list,),
        "object": (dict,),
        "null": (type(None),),
    }
    # bool is subclass of int in Python — check bool first
    if json_type == "integer" and isinstance(value, bool):
        return False
    if json_type == "number" and isinstance(value, bool):
        return False
    expected = type_map.get(json_type)
    if expected is None:
        return True  # unknown type, don't reject
    return isinstance(value, expected)


class ToolCallRecord(BaseModel):
    """A recorded tool call for assertion/inspection."""

    tool_name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    result: Any = None
    error: str | None = None
    validation_errors: list[str] = Field(default_factory=list)


class MockTool:
    """A mock tool executor that returns preconfigured responses.

    Register tools with their schemas and responses. When called, validates
    arguments against the schema, records the call, and returns the
    configured response.

    Usage::

        tool = MockTool()
        tool.register("get_weather", response={"temp": 72, "unit": "F"})
        result = await tool.call("get_weather", {"city": "NYC"})
        assert result == {"temp": 72, "unit": "F"}
        assert tool.call_count == 1

    With schema validation::

        tool.register(
            "search",
            response={"results": []},
            schema={"properties": {"query": {"type": "string"}}, "required": ["query"]},
        )
        result = await tool.call("search", {"query": "hello"})  # OK
        result = await tool.call("search", {})  # raises ToolValidationError

    Sequential responses::

        tool.register("roll_dice", response=[4, 2, 6])
        # First call returns 4, second returns 2, third returns 6, then cycles
    """

    def __init__(
        self,
        *,
        strict_validation: bool = True,
        default_response: Any = None,
    ) -> None:
        self.strict_validation = strict_validation
        self.default_response = default_response
        self._tools: dict[str, _RegisteredTool] = {}
        self._calls: list[ToolCallRecord] = []

    def register(
        self,
        name: str,
        *,
        response: Any = None,
        error: str | None = None,
        schema: dict[str, Any] | None = None,
        description: str = "",
    ) -> MockTool:
        """Register a tool with its response and optional schema. Returns self for chaining."""
        tool_schema = None
        if schema is not None:
            tool_schema = ToolSchema(
                name=name, description=description, parameters=schema
            )
        self._tools[name] = _RegisteredTool(
            name=name,
            response=response,
            error=error,
            schema=tool_schema,
            description=description,
        )
        return self

    async def call(self, name: str, arguments: dict[str, Any] | None = None) -> Any:
        """Call a registered tool with the given arguments."""
        return self._do_call(name, arguments or {})

    def call_sync(self, name: str, arguments: dict[str, Any] | None = None) -> Any:
        """Synchronous version of call for non-async agents."""
        return self._do_call(name, arguments or {})

    def _do_call(self, name: str, arguments: dict[str, Any]) -> Any:
        """Internal: execute a tool call with validation and recording."""
        registered = self._tools.get(name)

        if registered is None:
            # Unknown tool
            record = ToolCallRecord(
                tool_name=name,
                arguments=arguments,
                error=f"Unknown tool: {name}",
            )
            self._calls.append(record)
            if self.default_response is not None:
                record.result = self.default_response
                record.error = None
                return self.default_response
            raise ToolNotFoundError(name)

        # Validate arguments against schema
        validation_errors: list[str] = []
        if registered.schema is not None:
            validation_errors = registered.schema.validate_args(arguments)
            if validation_errors and self.strict_validation:
                record = ToolCallRecord(
                    tool_name=name,
                    arguments=arguments,
                    validation_errors=validation_errors,
                    error=f"Validation failed: {'; '.join(validation_errors)}",
                )
                self._calls.append(record)
                raise ToolValidationError(name, validation_errors)

        # Return configured error
        if registered.error is not None:
            record = ToolCallRecord(
                tool_name=name,
                arguments=arguments,
                error=registered.error,
                validation_errors=validation_errors,
            )
            self._calls.append(record)
            raise ToolExecutionError(name, registered.error)

        # Get response
        result = registered.get_response()
        record = ToolCallRecord(
            tool_name=name,
            arguments=arguments,
            result=result,
            validation_errors=validation_errors,
        )
        self._calls.append(record)
        return result

    # --- Inspection / assertion helpers ---

    @property
    def calls(self) -> list[ToolCallRecord]:
        """All recorded calls."""
        return list(self._calls)

    @property
    def call_count(self) -> int:
        """Total number of calls made."""
        return len(self._calls)

    @property
    def last_call(self) -> ToolCallRecord | None:
        """The most recent call, or None if no calls have been made."""
        return self._calls[-1] if self._calls else None

    def get_calls_for(self, tool_name: str) -> list[ToolCallRecord]:
        """Get all calls for a specific tool."""
        return [c for c in self._calls if c.tool_name == tool_name]

    def was_called(self, tool_name: str) -> bool:
        """Check if a tool was called at least once."""
        return any(c.tool_name == tool_name for c in self._calls)

    def assert_tool_called(
        self,
        tool_name: str,
        *,
        times: int | None = None,
        with_args: dict[str, Any] | None = None,
    ) -> None:
        """Assert that a tool was called, optionally checking count and arguments.

        Raises AssertionError with a descriptive message on failure.
        """
        matching = self.get_calls_for(tool_name)
        if not matching:
            called_tools = sorted({c.tool_name for c in self._calls})
            raise AssertionError(
                f"Tool '{tool_name}' was never called. "
                f"Called tools: {called_tools or '(none)'}"
            )
        if times is not None and len(matching) != times:
            raise AssertionError(
                f"Tool '{tool_name}' was called {len(matching)} time(s), "
                f"expected {times}"
            )
        if with_args is not None:
            for key, expected_value in with_args.items():
                if not any(c.arguments.get(key) == expected_value for c in matching):
                    actual_values = [c.arguments.get(key) for c in matching]
                    raise AssertionError(
                        f"Tool '{tool_name}' was never called with "
                        f"{key}={expected_value!r}. "
                        f"Actual values for '{key}': {actual_values}"
                    )

    def assert_tool_not_called(self, tool_name: str) -> None:
        """Assert that a tool was never called.

        Raises AssertionError with a descriptive message on failure.
        """
        matching = self.get_calls_for(tool_name)
        if matching:
            raise AssertionError(
                f"Tool '{tool_name}' was called {len(matching)} time(s), "
                f"expected 0"
            )

    def reset(self) -> None:
        """Clear all recorded calls and reset response sequence counters."""
        self._calls.clear()
        for tool in self._tools.values():
            tool._call_count = 0

    def reset_calls(self) -> None:
        """Clear recorded calls but keep response sequence counters."""
        self._calls.clear()

    @property
    def registered_tools(self) -> list[str]:
        """Names of all registered tools."""
        return list(self._tools.keys())


class _RegisteredTool:
    """Internal: a registered tool with its response configuration."""

    def __init__(
        self,
        name: str,
        response: Any,
        error: str | None,
        schema: ToolSchema | None,
        description: str,
    ) -> None:
        self.name = name
        self.response = response
        self.error = error
        self.schema = schema
        self.description = description
        self._call_count = 0

    def get_response(self) -> Any:
        """Get the next response, cycling through sequences."""
        if isinstance(self.response, list):
            idx = self._call_count % len(self.response)
            self._call_count += 1
            return self.response[idx]
        self._call_count += 1
        return self.response


# --- Exceptions ---


class MockToolError(Exception):
    """Base exception for MockTool errors."""

    def __init__(self, tool_name: str, message: str) -> None:
        self.tool_name = tool_name
        super().__init__(f"MockTool({tool_name}): {message}")


class ToolNotFoundError(MockToolError):
    """Raised when calling an unregistered tool."""

    def __init__(self, tool_name: str) -> None:
        super().__init__(tool_name, "tool not registered")


class ToolValidationError(MockToolError):
    """Raised when tool arguments fail schema validation."""

    def __init__(self, tool_name: str, errors: list[str]) -> None:
        self.validation_errors = errors
        super().__init__(tool_name, f"validation failed: {'; '.join(errors)}")


class ToolExecutionError(MockToolError):
    """Raised when a tool is configured to return an error."""

    def __init__(self, tool_name: str, error: str) -> None:
        super().__init__(tool_name, error)
