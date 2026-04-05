"""Tests for MockMCPServer — MCP protocol-level mock server."""

from __future__ import annotations

import json

import pytest

from checkagent.mock.mcp import MCPCallRecord, MCPToolDef, MockMCPServer


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------

class TestInitialize:
    """Test MCP initialize handshake."""

    @pytest.mark.asyncio
    async def test_initialize_returns_server_info(self):
        server = MockMCPServer(name="my-server", version="2.0.0")
        resp = await server.handle_message({
            "jsonrpc": "2.0", "id": 1, "method": "initialize",
            "params": {"protocolVersion": "2024-11-05", "capabilities": {}},
        })
        assert resp["id"] == 1
        result = resp["result"]
        assert result["serverInfo"]["name"] == "my-server"
        assert result["serverInfo"]["version"] == "2.0.0"
        assert result["protocolVersion"] == "2024-11-05"
        assert "tools" in result["capabilities"]

    @pytest.mark.asyncio
    async def test_initialized_notification_no_response(self):
        server = MockMCPServer()
        resp = await server.handle_message({
            "jsonrpc": "2.0", "method": "notifications/initialized",
        })
        assert resp is None
        assert server._initialized is True

    @pytest.mark.asyncio
    async def test_default_server_name(self):
        server = MockMCPServer()
        resp = await server.handle_message({
            "jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {},
        })
        assert resp["result"]["serverInfo"]["name"] == "mock-mcp-server"


# ---------------------------------------------------------------------------
# Tool Registration
# ---------------------------------------------------------------------------

class TestToolRegistration:
    """Test tool registration and listing."""

    def test_register_tool_basic(self):
        server = MockMCPServer()
        server.register_tool("search", response={"results": []})
        assert "search" in server.registered_tools

    def test_register_tool_chaining(self):
        server = MockMCPServer()
        result = server.register_tool("a", response=1).register_tool("b", response=2)
        assert result is server
        assert server.registered_tools == ["a", "b"]

    def test_register_tool_with_schema(self):
        server = MockMCPServer()
        server.register_tool(
            "get_weather",
            response={"temp": 72},
            description="Get weather for a city",
            input_schema={
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        )
        defs = server.tool_definitions
        assert len(defs) == 1
        assert defs[0].name == "get_weather"
        assert defs[0].description == "Get weather for a city"
        assert "city" in defs[0].input_schema["properties"]

    @pytest.mark.asyncio
    async def test_tools_list_empty(self):
        server = MockMCPServer()
        resp = await server.handle_message({
            "jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {},
        })
        assert resp["result"]["tools"] == []

    @pytest.mark.asyncio
    async def test_tools_list_returns_all_tools(self):
        server = MockMCPServer()
        server.register_tool("search", response=[], description="Search things")
        server.register_tool("calculate", response=0, description="Do math")
        resp = await server.handle_message({
            "jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {},
        })
        tools = resp["result"]["tools"]
        assert len(tools) == 2
        names = {t["name"] for t in tools}
        assert names == {"search", "calculate"}

    @pytest.mark.asyncio
    async def test_tools_list_includes_schema(self):
        server = MockMCPServer()
        schema = {"type": "object", "properties": {"q": {"type": "string"}}}
        server.register_tool("search", response=[], input_schema=schema)
        resp = await server.handle_message({
            "jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {},
        })
        tool = resp["result"]["tools"][0]
        assert tool["inputSchema"] == schema


# ---------------------------------------------------------------------------
# Tool Calls
# ---------------------------------------------------------------------------

class TestToolCalls:
    """Test tools/call method handling."""

    @pytest.mark.asyncio
    async def test_call_returns_response(self):
        server = MockMCPServer()
        server.register_tool("get_weather", response={"temp": 72, "unit": "F"})
        resp = await server.handle_message({
            "jsonrpc": "2.0", "id": 1, "method": "tools/call",
            "params": {"name": "get_weather", "arguments": {"city": "NYC"}},
        })
        content = resp["result"]["content"]
        assert len(content) == 1
        assert content[0]["type"] == "text"
        assert json.loads(content[0]["text"]) == {"temp": 72, "unit": "F"}
        assert "isError" not in resp["result"]

    @pytest.mark.asyncio
    async def test_call_string_response(self):
        server = MockMCPServer()
        server.register_tool("echo", response="hello world")
        resp = await server.handle_message({
            "jsonrpc": "2.0", "id": 1, "method": "tools/call",
            "params": {"name": "echo", "arguments": {}},
        })
        assert resp["result"]["content"][0]["text"] == "hello world"

    @pytest.mark.asyncio
    async def test_call_unknown_tool(self):
        server = MockMCPServer()
        resp = await server.handle_message({
            "jsonrpc": "2.0", "id": 1, "method": "tools/call",
            "params": {"name": "nonexistent", "arguments": {}},
        })
        assert resp["result"]["isError"] is True
        assert "Unknown tool" in resp["result"]["content"][0]["text"]

    @pytest.mark.asyncio
    async def test_call_configured_error(self):
        server = MockMCPServer()
        server.register_tool("broken", error="Service unavailable")
        resp = await server.handle_message({
            "jsonrpc": "2.0", "id": 1, "method": "tools/call",
            "params": {"name": "broken", "arguments": {}},
        })
        assert resp["result"]["isError"] is True
        assert resp["result"]["content"][0]["text"] == "Service unavailable"

    @pytest.mark.asyncio
    async def test_call_sequential_responses(self):
        server = MockMCPServer()
        server.register_tool("counter", response=[1, 2, 3])
        results = []
        for i in range(5):
            resp = await server.handle_message({
                "jsonrpc": "2.0", "id": i, "method": "tools/call",
                "params": {"name": "counter", "arguments": {}},
            })
            results.append(json.loads(resp["result"]["content"][0]["text"]))
        assert results == [1, 2, 3, 1, 2]  # cycles

    @pytest.mark.asyncio
    async def test_call_with_no_arguments(self):
        server = MockMCPServer()
        server.register_tool("ping", response="pong")
        resp = await server.handle_message({
            "jsonrpc": "2.0", "id": 1, "method": "tools/call",
            "params": {"name": "ping"},
        })
        assert resp["result"]["content"][0]["text"] == "pong"

    @pytest.mark.asyncio
    async def test_call_none_response(self):
        server = MockMCPServer()
        server.register_tool("noop", response=None)
        resp = await server.handle_message({
            "jsonrpc": "2.0", "id": 1, "method": "tools/call",
            "params": {"name": "noop", "arguments": {}},
        })
        assert resp["result"]["content"][0]["text"] == "null"


# ---------------------------------------------------------------------------
# Call Recording
# ---------------------------------------------------------------------------

class TestCallRecording:
    """Test call recording and inspection."""

    @pytest.mark.asyncio
    async def test_records_successful_call(self):
        server = MockMCPServer()
        server.register_tool("search", response={"results": []})
        await server.handle_message({
            "jsonrpc": "2.0", "id": 1, "method": "tools/call",
            "params": {"name": "search", "arguments": {"q": "test"}},
        })
        assert server.call_count == 1
        assert server.last_call.tool_name == "search"
        assert server.last_call.arguments == {"q": "test"}
        assert server.last_call.result == {"results": []}
        assert server.last_call.is_error is False

    @pytest.mark.asyncio
    async def test_records_error_call(self):
        server = MockMCPServer()
        server.register_tool("broken", error="fail")
        await server.handle_message({
            "jsonrpc": "2.0", "id": 1, "method": "tools/call",
            "params": {"name": "broken", "arguments": {}},
        })
        assert server.last_call.is_error is True
        assert server.last_call.error is not None

    @pytest.mark.asyncio
    async def test_records_unknown_tool_call(self):
        server = MockMCPServer()
        await server.handle_message({
            "jsonrpc": "2.0", "id": 1, "method": "tools/call",
            "params": {"name": "missing", "arguments": {"x": 1}},
        })
        assert server.call_count == 1
        assert server.last_call.tool_name == "missing"
        assert server.last_call.is_error is True

    @pytest.mark.asyncio
    async def test_get_calls_for_tool(self):
        server = MockMCPServer()
        server.register_tool("a", response="A")
        server.register_tool("b", response="B")
        await server.handle_message({
            "jsonrpc": "2.0", "id": 1, "method": "tools/call",
            "params": {"name": "a", "arguments": {}},
        })
        await server.handle_message({
            "jsonrpc": "2.0", "id": 2, "method": "tools/call",
            "params": {"name": "b", "arguments": {}},
        })
        await server.handle_message({
            "jsonrpc": "2.0", "id": 3, "method": "tools/call",
            "params": {"name": "a", "arguments": {}},
        })
        assert len(server.get_calls_for("a")) == 2
        assert len(server.get_calls_for("b")) == 1

    def test_no_calls_initially(self):
        server = MockMCPServer()
        assert server.call_count == 0
        assert server.last_call is None
        assert server.calls == []


# ---------------------------------------------------------------------------
# Assertions
# ---------------------------------------------------------------------------

class TestAssertions:
    """Test assertion helpers."""

    @pytest.mark.asyncio
    async def test_assert_tool_called(self):
        server = MockMCPServer()
        server.register_tool("search", response=[])
        await server.handle_message({
            "jsonrpc": "2.0", "id": 1, "method": "tools/call",
            "params": {"name": "search", "arguments": {"q": "test"}},
        })
        server.assert_tool_called("search")

    @pytest.mark.asyncio
    async def test_assert_tool_called_fails(self):
        server = MockMCPServer()
        with pytest.raises(AssertionError, match="never called"):
            server.assert_tool_called("search")

    @pytest.mark.asyncio
    async def test_assert_tool_called_times(self):
        server = MockMCPServer()
        server.register_tool("ping", response="pong")
        for i in range(3):
            await server.handle_message({
                "jsonrpc": "2.0", "id": i, "method": "tools/call",
                "params": {"name": "ping", "arguments": {}},
            })
        server.assert_tool_called("ping", times=3)

    @pytest.mark.asyncio
    async def test_assert_tool_called_times_fails(self):
        server = MockMCPServer()
        server.register_tool("ping", response="pong")
        await server.handle_message({
            "jsonrpc": "2.0", "id": 1, "method": "tools/call",
            "params": {"name": "ping", "arguments": {}},
        })
        with pytest.raises(AssertionError, match="1 time"):
            server.assert_tool_called("ping", times=5)

    @pytest.mark.asyncio
    async def test_assert_tool_called_with_args(self):
        server = MockMCPServer()
        server.register_tool("search", response=[])
        await server.handle_message({
            "jsonrpc": "2.0", "id": 1, "method": "tools/call",
            "params": {"name": "search", "arguments": {"q": "hello"}},
        })
        server.assert_tool_called("search", with_args={"q": "hello"})

    @pytest.mark.asyncio
    async def test_assert_tool_called_with_args_fails(self):
        server = MockMCPServer()
        server.register_tool("search", response=[])
        await server.handle_message({
            "jsonrpc": "2.0", "id": 1, "method": "tools/call",
            "params": {"name": "search", "arguments": {"q": "hello"}},
        })
        with pytest.raises(AssertionError, match="never called with"):
            server.assert_tool_called("search", with_args={"q": "goodbye"})

    @pytest.mark.asyncio
    async def test_assert_tool_not_called(self):
        server = MockMCPServer()
        server.assert_tool_not_called("search")

    @pytest.mark.asyncio
    async def test_assert_tool_not_called_fails(self):
        server = MockMCPServer()
        server.register_tool("search", response=[])
        await server.handle_message({
            "jsonrpc": "2.0", "id": 1, "method": "tools/call",
            "params": {"name": "search", "arguments": {}},
        })
        with pytest.raises(AssertionError, match="1 time"):
            server.assert_tool_not_called("search")

    @pytest.mark.asyncio
    async def test_was_called(self):
        server = MockMCPServer()
        server.register_tool("search", response=[])
        assert not server.was_called("search")
        await server.handle_message({
            "jsonrpc": "2.0", "id": 1, "method": "tools/call",
            "params": {"name": "search", "arguments": {}},
        })
        assert server.was_called("search")


# ---------------------------------------------------------------------------
# Raw Message Handling
# ---------------------------------------------------------------------------

class TestRawHandling:
    """Test raw JSON string handling."""

    @pytest.mark.asyncio
    async def test_handle_raw_valid(self):
        server = MockMCPServer()
        server.register_tool("ping", response="pong")
        raw = json.dumps({
            "jsonrpc": "2.0", "id": 1, "method": "tools/call",
            "params": {"name": "ping", "arguments": {}},
        })
        resp_str = await server.handle_raw(raw)
        resp = json.loads(resp_str)
        assert resp["result"]["content"][0]["text"] == "pong"

    @pytest.mark.asyncio
    async def test_handle_raw_invalid_json(self):
        server = MockMCPServer()
        resp_str = await server.handle_raw("not json")
        resp = json.loads(resp_str)
        assert resp["error"]["code"] == -32700

    @pytest.mark.asyncio
    async def test_handle_raw_notification(self):
        server = MockMCPServer()
        raw = json.dumps({
            "jsonrpc": "2.0", "method": "notifications/initialized",
        })
        resp_str = await server.handle_raw(raw)
        assert resp_str == ""


# ---------------------------------------------------------------------------
# Error Handling
# ---------------------------------------------------------------------------

class TestErrorHandling:
    """Test protocol-level error handling."""

    @pytest.mark.asyncio
    async def test_unknown_method(self):
        server = MockMCPServer()
        resp = await server.handle_message({
            "jsonrpc": "2.0", "id": 1, "method": "unknown/method", "params": {},
        })
        assert resp["error"]["code"] == -32601
        assert "Method not found" in resp["error"]["message"]


# ---------------------------------------------------------------------------
# Reset
# ---------------------------------------------------------------------------

class TestReset:
    """Test reset functionality."""

    @pytest.mark.asyncio
    async def test_reset_clears_calls(self):
        server = MockMCPServer()
        server.register_tool("ping", response="pong")
        await server.handle_message({
            "jsonrpc": "2.0", "id": 1, "method": "tools/call",
            "params": {"name": "ping", "arguments": {}},
        })
        assert server.call_count == 1
        server.reset()
        assert server.call_count == 0
        assert server.calls == []

    @pytest.mark.asyncio
    async def test_reset_resets_sequences(self):
        server = MockMCPServer()
        server.register_tool("counter", response=[1, 2, 3])
        await server.handle_message({
            "jsonrpc": "2.0", "id": 1, "method": "tools/call",
            "params": {"name": "counter", "arguments": {}},
        })
        server.reset()
        resp = await server.handle_message({
            "jsonrpc": "2.0", "id": 2, "method": "tools/call",
            "params": {"name": "counter", "arguments": {}},
        })
        assert json.loads(resp["result"]["content"][0]["text"]) == 1

    @pytest.mark.asyncio
    async def test_reset_calls_keeps_sequences(self):
        server = MockMCPServer()
        server.register_tool("counter", response=[1, 2, 3])
        await server.handle_message({
            "jsonrpc": "2.0", "id": 1, "method": "tools/call",
            "params": {"name": "counter", "arguments": {}},
        })
        server.reset_calls()
        assert server.call_count == 0
        resp = await server.handle_message({
            "jsonrpc": "2.0", "id": 2, "method": "tools/call",
            "params": {"name": "counter", "arguments": {}},
        })
        # Sequence continues from where it left off
        assert json.loads(resp["result"]["content"][0]["text"]) == 2


# ---------------------------------------------------------------------------
# Fixture Integration
# ---------------------------------------------------------------------------

class TestFixture:
    """Test the ap_mock_mcp_server fixture."""

    def test_fixture_provides_fresh_instance(self, ap_mock_mcp_server):
        assert isinstance(ap_mock_mcp_server, MockMCPServer)
        assert ap_mock_mcp_server.call_count == 0
        assert ap_mock_mcp_server.registered_tools == []

    @pytest.mark.asyncio
    async def test_fixture_is_functional(self, ap_mock_mcp_server):
        ap_mock_mcp_server.register_tool("test", response="ok")
        resp = await ap_mock_mcp_server.handle_message({
            "jsonrpc": "2.0", "id": 1, "method": "tools/call",
            "params": {"name": "test", "arguments": {}},
        })
        assert resp["result"]["content"][0]["text"] == "ok"
        ap_mock_mcp_server.assert_tool_called("test")
