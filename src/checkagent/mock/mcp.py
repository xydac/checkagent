"""MockMCPServer — MCP protocol-level mock server for testing MCP-aware agents.

Simulates a Model Context Protocol server that handles JSON-RPC 2.0 messages.
Agents can register tools, call them, and list available tools through the
standard MCP protocol interface.

Implements F1.2 (MCP-aware) and F1.5 (ap_mock_mcp_server fixture) from the PRD.
"""

from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel, Field


class MCPToolDef(BaseModel):
    """Definition of a tool exposed by the MCP server."""

    name: str
    description: str = ""
    input_schema: dict[str, Any] = Field(default_factory=lambda: {"type": "object", "properties": {}})


class MCPCallRecord(BaseModel):
    """A recorded MCP tool call for assertion/inspection."""

    tool_name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    result: Any = None
    error: dict[str, Any] | None = None
    is_error: bool = False


class MockMCPServer:
    """A mock MCP server that responds to JSON-RPC 2.0 messages.

    Supports the core MCP protocol methods:
    - ``initialize`` — server handshake
    - ``notifications/initialized`` — client acknowledgment (no response)
    - ``tools/list`` — enumerate registered tools
    - ``tools/call`` — invoke a registered tool

    Usage::

        server = MockMCPServer(name="test-server")
        server.register_tool("get_weather", response={"temp": 72})

        # JSON-RPC message handling
        resp = await server.handle_message({
            "jsonrpc": "2.0", "id": 1,
            "method": "tools/call",
            "params": {"name": "get_weather", "arguments": {"city": "NYC"}}
        })
        assert resp["result"]["content"][0]["text"] == '{"temp": 72}'

    Batch and raw string handling::

        resp = await server.handle_raw('{"jsonrpc":"2.0","id":1,"method":"tools/list"}')

    Call recording and assertions::

        server.assert_tool_called("get_weather")
        assert server.call_count == 1
    """

    MCP_VERSION = "2024-11-05"

    def __init__(
        self,
        *,
        name: str = "mock-mcp-server",
        version: str = "1.0.0",
    ) -> None:
        self.name = name
        self.version = version
        self._tools: dict[str, _RegisteredMCPTool] = {}
        self._calls: list[MCPCallRecord] = []
        self._initialized = False

    def register_tool(
        self,
        name: str,
        *,
        response: Any = None,
        error: str | None = None,
        description: str = "",
        input_schema: dict[str, Any] | None = None,
    ) -> MockMCPServer:
        """Register a tool that the server exposes. Returns self for chaining."""
        schema = input_schema or {"type": "object", "properties": {}}
        self._tools[name] = _RegisteredMCPTool(
            definition=MCPToolDef(
                name=name,
                description=description,
                input_schema=schema,
            ),
            response=response,
            error=error,
        )
        return self

    async def handle_message(self, message: dict[str, Any]) -> dict[str, Any] | None:
        """Handle a single JSON-RPC 2.0 message. Returns response or None for notifications."""
        method = message.get("method", "")
        msg_id = message.get("id")
        params = message.get("params", {})

        # Notifications (no id) don't get responses
        if msg_id is None:
            if method == "notifications/initialized":
                self._initialized = True
            return None

        handler = self._method_handlers.get(method)
        if handler is None:
            return _error_response(msg_id, -32601, f"Method not found: {method}")

        return handler(self, msg_id, params)

    async def handle_raw(self, raw: str) -> str:
        """Handle a raw JSON-RPC string. Returns JSON string response."""
        try:
            message = json.loads(raw)
        except json.JSONDecodeError as e:
            return json.dumps(_error_response(None, -32700, f"Parse error: {e}"))

        result = await self.handle_message(message)
        if result is None:
            return ""
        return json.dumps(result)

    # --- Protocol method handlers ---

    def _handle_initialize(self, msg_id: Any, params: dict[str, Any]) -> dict[str, Any]:
        """Handle initialize handshake."""
        return {
            "jsonrpc": "2.0",
            "id": msg_id,
            "result": {
                "protocolVersion": self.MCP_VERSION,
                "capabilities": {"tools": {"listChanged": True}},
                "serverInfo": {"name": self.name, "version": self.version},
            },
        }

    def _handle_tools_list(self, msg_id: Any, params: dict[str, Any]) -> dict[str, Any]:
        """Handle tools/list — return all registered tools."""
        tools = []
        for reg in self._tools.values():
            tools.append({
                "name": reg.definition.name,
                "description": reg.definition.description,
                "inputSchema": reg.definition.input_schema,
            })
        return {
            "jsonrpc": "2.0",
            "id": msg_id,
            "result": {"tools": tools},
        }

    def _handle_tools_call(self, msg_id: Any, params: dict[str, Any]) -> dict[str, Any]:
        """Handle tools/call — invoke a registered tool."""
        tool_name = params.get("name", "")
        arguments = params.get("arguments", {})

        registered = self._tools.get(tool_name)
        if registered is None:
            record = MCPCallRecord(
                tool_name=tool_name,
                arguments=arguments,
                error={"code": -32602, "message": f"Unknown tool: {tool_name}"},
                is_error=True,
            )
            self._calls.append(record)
            return {
                "jsonrpc": "2.0",
                "id": msg_id,
                "result": {
                    "content": [{"type": "text", "text": f"Unknown tool: {tool_name}"}],
                    "isError": True,
                },
            }

        # Configured error
        if registered.error is not None:
            record = MCPCallRecord(
                tool_name=tool_name,
                arguments=arguments,
                error={"message": registered.error},
                is_error=True,
            )
            self._calls.append(record)
            return {
                "jsonrpc": "2.0",
                "id": msg_id,
                "result": {
                    "content": [{"type": "text", "text": registered.error}],
                    "isError": True,
                },
            }

        # Success
        result = registered.get_response()
        result_text = json.dumps(result) if not isinstance(result, str) else result
        record = MCPCallRecord(
            tool_name=tool_name,
            arguments=arguments,
            result=result,
        )
        self._calls.append(record)
        return {
            "jsonrpc": "2.0",
            "id": msg_id,
            "result": {
                "content": [{"type": "text", "text": result_text}],
            },
        }

    _method_handlers: dict[str, Any] = {
        "initialize": _handle_initialize,
        "tools/list": _handle_tools_list,
        "tools/call": _handle_tools_call,
    }

    # --- Inspection / assertion helpers ---

    @property
    def calls(self) -> list[MCPCallRecord]:
        """All recorded tool calls."""
        return list(self._calls)

    @property
    def call_count(self) -> int:
        """Total number of tool calls made."""
        return len(self._calls)

    @property
    def last_call(self) -> MCPCallRecord | None:
        """Most recent tool call, or None."""
        return self._calls[-1] if self._calls else None

    def get_calls_for(self, tool_name: str) -> list[MCPCallRecord]:
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
        """Assert a tool was called, optionally checking count and arguments."""
        matching = self.get_calls_for(tool_name)
        if not matching:
            called = sorted({c.tool_name for c in self._calls})
            raise AssertionError(
                f"Tool '{tool_name}' was never called. "
                f"Called tools: {called or '(none)'}"
            )
        if times is not None and len(matching) != times:
            raise AssertionError(
                f"Tool '{tool_name}' was called {len(matching)} time(s), "
                f"expected {times}"
            )
        if with_args is not None:
            for key, expected in with_args.items():
                if not any(c.arguments.get(key) == expected for c in matching):
                    actual = [c.arguments.get(key) for c in matching]
                    raise AssertionError(
                        f"Tool '{tool_name}' was never called with "
                        f"{key}={expected!r}. Actual values: {actual}"
                    )

    def assert_tool_not_called(self, tool_name: str) -> None:
        """Assert that a tool was never called."""
        matching = self.get_calls_for(tool_name)
        if matching:
            raise AssertionError(
                f"Tool '{tool_name}' was called {len(matching)} time(s), expected 0"
            )

    @property
    def registered_tools(self) -> list[str]:
        """Names of all registered tools."""
        return list(self._tools.keys())

    @property
    def tool_definitions(self) -> list[MCPToolDef]:
        """All tool definitions."""
        return [t.definition for t in self._tools.values()]

    def reset(self) -> None:
        """Clear all recorded calls and reset sequence counters."""
        self._calls.clear()
        for tool in self._tools.values():
            tool._call_count = 0

    def reset_calls(self) -> None:
        """Clear recorded calls but keep response sequence counters."""
        self._calls.clear()


class _RegisteredMCPTool:
    """Internal: a registered tool with response configuration."""

    def __init__(
        self,
        definition: MCPToolDef,
        response: Any,
        error: str | None,
    ) -> None:
        self.definition = definition
        self.response = response
        self.error = error
        self._call_count = 0

    def get_response(self) -> Any:
        """Get the next response, cycling through sequences."""
        if isinstance(self.response, list) and len(self.response) > 0:
            idx = self._call_count % len(self.response)
            self._call_count += 1
            return self.response[idx]
        self._call_count += 1
        return self.response


def _error_response(msg_id: Any, code: int, message: str) -> dict[str, Any]:
    """Build a JSON-RPC error response."""
    return {
        "jsonrpc": "2.0",
        "id": msg_id,
        "error": {"code": code, "message": message},
    }
