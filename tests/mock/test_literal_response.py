"""Tests for literal() wrapper — returns list values as-is without cycling."""

import pytest

from checkagent.mock.tool import MockTool, literal


class TestMockToolLiteral:
    """MockTool returns literal list values without cycling."""

    @pytest.mark.asyncio
    async def test_list_without_literal_cycles(self):
        """Default behavior: lists cycle through elements."""
        tool = MockTool()
        tool.register("search", response=["doc1", "doc2", "doc3"])
        assert await tool.call("search") == "doc1"
        assert await tool.call("search") == "doc2"
        assert await tool.call("search") == "doc3"
        assert await tool.call("search") == "doc1"  # cycles

    @pytest.mark.asyncio
    async def test_literal_list_returns_whole_list(self):
        """literal() wrapping returns the full list every call."""
        tool = MockTool()
        tool.register("search", response=literal(["doc1", "doc2", "doc3"]))
        result = await tool.call("search")
        assert result == ["doc1", "doc2", "doc3"]
        # Second call returns the same list
        assert await tool.call("search") == ["doc1", "doc2", "doc3"]

    @pytest.mark.asyncio
    async def test_literal_empty_list(self):
        tool = MockTool()
        tool.register("search", response=literal([]))
        assert await tool.call("search") == []

    @pytest.mark.asyncio
    async def test_literal_nested_list(self):
        tool = MockTool()
        tool.register("search", response=literal([["a", "b"], ["c"]]))
        assert await tool.call("search") == [["a", "b"], ["c"]]

    @pytest.mark.asyncio
    async def test_literal_none(self):
        tool = MockTool()
        tool.register("search", response=literal(None))
        assert await tool.call("search") is None

    @pytest.mark.asyncio
    async def test_literal_dict(self):
        """literal() works with non-list values too (no-op but explicit)."""
        tool = MockTool()
        tool.register("search", response=literal({"key": "value"}))
        assert await tool.call("search") == {"key": "value"}

    @pytest.mark.asyncio
    async def test_literal_with_fluent_api(self):
        tool = MockTool()
        tool.on_call("search").respond(literal(["doc1", "doc2"]))
        assert await tool.call("search") == ["doc1", "doc2"]
        assert await tool.call("search") == ["doc1", "doc2"]

    def test_literal_with_sync_call(self):
        tool = MockTool()
        tool.register("search", response=literal(["doc1", "doc2"]))
        assert tool.call_sync("search") == ["doc1", "doc2"]

    def test_literal_repr(self):
        lr = literal(["a", "b"])
        assert repr(lr) == "literal(['a', 'b'])"

    @pytest.mark.asyncio
    async def test_literal_call_count_increments(self):
        """literal() responses still track call counts."""
        tool = MockTool()
        tool.register("search", response=literal(["doc1"]))
        await tool.call("search")
        await tool.call("search")
        assert tool.call_count == 2


class TestLiteralImport:
    """literal() is importable from expected locations."""

    def test_import_from_mock(self):
        from checkagent.mock import literal as lit
        assert callable(lit)

    def test_import_from_top_level(self):
        from checkagent import literal as lit
        assert callable(lit)

    def test_import_from_mock_tool(self):
        from checkagent.mock.tool import literal as lit
        assert callable(lit)
