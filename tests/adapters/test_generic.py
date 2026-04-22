"""Tests for the GenericAdapter and @wrap decorator."""

import asyncio
from unittest.mock import MagicMock, patch

import pytest

from checkagent.adapters.generic import (
    GenericAdapter,
    _non_callable_error,
    _try_framework_adapter,
    wrap,
)
from checkagent.core.adapter import AgentAdapter
from checkagent.core.types import AgentInput, StreamEventType


class TestGenericAdapterSync:
    async def test_wrap_sync_function(self):
        def my_agent(query: str) -> str:
            return f"Answer: {query}"

        adapter = wrap(my_agent)
        result = await adapter.run(AgentInput(query="hello"))

        assert result.succeeded is True
        assert result.final_output == "Answer: hello"
        assert result.duration_ms is not None
        assert result.duration_ms >= 0
        assert len(result.steps) == 1
        assert result.steps[0].output_text == "Answer: hello"

    async def test_sync_function_with_kwargs(self):
        def my_agent(query: str, **kwargs: object) -> str:
            user = kwargs.get("user", "anon")
            return f"{user}: {query}"

        adapter = wrap(my_agent)
        result = await adapter.run(AgentInput(query="hi", context={"user": "alice"}))

        assert result.final_output == "alice: hi"

    async def test_sync_function_error_handling(self):
        def bad_agent(query: str) -> str:
            raise ValueError("something broke")

        adapter = wrap(bad_agent)
        result = await adapter.run(AgentInput(query="test"))

        assert result.succeeded is False
        assert "ValueError: something broke" in result.error  # type: ignore[operator]
        assert result.final_output is None
        assert len(result.steps) == 0


class TestGenericAdapterAsync:
    async def test_wrap_async_function(self):
        async def my_agent(query: str) -> str:
            await asyncio.sleep(0.001)
            return f"Async: {query}"

        adapter = wrap(my_agent)
        result = await adapter.run(AgentInput(query="world"))

        assert result.succeeded is True
        assert result.final_output == "Async: world"

    async def test_async_function_with_kwargs(self):
        async def my_agent(query: str, **kwargs: object) -> str:
            return f"{kwargs.get('mode', 'default')}: {query}"

        adapter = wrap(my_agent)
        result = await adapter.run(AgentInput(query="test", context={"mode": "fast"}))

        assert result.final_output == "fast: test"

    async def test_async_function_error_handling(self):
        async def bad_agent(query: str) -> str:
            raise RuntimeError("async boom")

        adapter = wrap(bad_agent)
        result = await adapter.run(AgentInput(query="test"))

        assert result.succeeded is False
        assert "RuntimeError: async boom" in result.error  # type: ignore[operator]


class TestGenericAdapterStream:
    async def test_stream_fallback(self):
        def simple(query: str) -> str:
            return "done"

        adapter = wrap(simple)
        events = []
        async for event in adapter.run_stream(AgentInput(query="go")):
            events.append(event)

        assert len(events) == 3
        assert events[0].event_type == StreamEventType.RUN_START
        assert events[1].event_type == StreamEventType.TEXT_DELTA
        assert events[1].data == "done"
        assert events[2].event_type == StreamEventType.RUN_END

    async def test_stream_error(self):
        def bad(query: str) -> str:
            raise ValueError("nope")

        adapter = wrap(bad)
        events = []
        async for event in adapter.run_stream(AgentInput(query="go")):
            events.append(event)

        assert len(events) == 3
        assert events[0].event_type == StreamEventType.RUN_START
        assert events[1].event_type == StreamEventType.ERROR
        assert events[2].event_type == StreamEventType.RUN_END


class TestWrapDecorator:
    def test_wrap_as_decorator(self):
        @wrap
        def my_agent(query: str) -> str:
            return query

        assert isinstance(my_agent, GenericAdapter)

    def test_wrap_as_function(self):
        def my_agent(query: str) -> str:
            return query

        adapter = wrap(my_agent)
        assert isinstance(adapter, GenericAdapter)

    def test_adapter_protocol_compliance(self):
        @wrap
        def my_agent(query: str) -> str:
            return query

        assert isinstance(my_agent, AgentAdapter)


class TestGenericAdapterNonStringReturn:
    async def test_returns_dict(self):
        def my_agent(query: str) -> dict:
            return {"answer": query, "confidence": 0.95}

        adapter = wrap(my_agent)
        result = await adapter.run(AgentInput(query="test"))

        assert result.final_output == {"answer": "test", "confidence": 0.95}

    async def test_returns_int(self):
        def math_agent(query: str) -> int:
            return 42

        adapter = wrap(math_agent)
        result = await adapter.run(AgentInput(query="what is 6*7"))

        assert result.final_output == 42


class TestF112WrapNonCallableError:
    def test_plain_non_callable_raises_type_error(self):
        class NotAnAgent:
            pass

        with pytest.raises(TypeError) as exc_info:
            wrap(NotAnAgent())

        msg = str(exc_info.value)
        assert "wrap() requires a callable" in msg
        assert "NotAnAgent" in msg

    def test_error_message_lists_adapters(self):
        with pytest.raises(TypeError) as exc_info:
            wrap(object())

        msg = str(exc_info.value)
        assert "PydanticAIAdapter" in msg
        assert "LangChainAdapter" in msg
        assert "CrewAIAdapter" in msg
        assert "OpenAIAgentsAdapter" in msg

    def test_error_message_suggests_lambda(self):
        with pytest.raises(TypeError) as exc_info:
            wrap(42)

        assert "lambda" in str(exc_info.value)

    def test_generic_adapter_non_callable_raises(self):
        with pytest.raises(TypeError):
            GenericAdapter(42)  # type: ignore[arg-type]

    def test_callable_still_works_after_check(self):
        def fn(q: str) -> str:
            return q

        adapter = wrap(fn)
        assert isinstance(adapter, GenericAdapter)

    def test_non_callable_error_includes_type_name(self):
        class MyCustomAgent:
            pass

        obj = MyCustomAgent()
        err = _non_callable_error(obj)
        assert "MyCustomAgent" in str(err)


class TestF112AutoFrameworkDetection:
    def test_pydantic_ai_agent_auto_detected(self):
        """wrap() routes non-callable pydantic_ai objects to PydanticAIAdapter."""
        class FakePydanticAgent:
            pass

        FakePydanticAgent.__module__ = "pydantic_ai.agent"
        obj = FakePydanticAgent()

        mock_adapter_instance = MagicMock()
        with patch("checkagent.adapters.generic._try_framework_adapter") as mock_detect:
            mock_detect.return_value = mock_adapter_instance
            result = wrap(obj)

        mock_detect.assert_called_once_with(obj)
        assert result is mock_adapter_instance

    def test_try_framework_adapter_returns_none_for_unknown(self):
        class UnknownFrameworkAgent:
            pass

        result = _try_framework_adapter(UnknownFrameworkAgent())
        assert result is None

    def test_try_framework_adapter_returns_none_for_callable(self):
        def fn(q: str) -> str:
            return q

        result = _try_framework_adapter(fn)
        assert result is None

    def test_pydantic_ai_module_prefix_matches(self):
        """Objects from pydantic_ai.* should be detected."""
        class FakeAgent:
            pass
        FakeAgent.__module__ = "pydantic_ai.agent"

        obj = FakeAgent()

        mock_adapter = MagicMock()
        with patch("importlib.import_module") as mock_import:
            mock_module = MagicMock()
            mock_module.PydanticAIAdapter = MagicMock(return_value=mock_adapter)
            mock_import.return_value = mock_module
            result = _try_framework_adapter(obj)

        assert result is mock_adapter
        mock_import.assert_called_once_with("checkagent.adapters.pydantic_ai")

    def test_langchain_module_prefix_matches(self):
        """Objects from langchain_core.* should be detected (underscore variant)."""
        class FakeChain:
            pass
        FakeChain.__module__ = "langchain_core.runnables.base"

        obj = FakeChain()

        mock_adapter = MagicMock()
        with patch("importlib.import_module") as mock_import:
            mock_module = MagicMock()
            mock_module.LangChainAdapter = MagicMock(return_value=mock_adapter)
            mock_import.return_value = mock_module
            result = _try_framework_adapter(obj)

        assert result is mock_adapter

    def test_crewai_module_prefix_matches(self):
        class FakeCrew:
            pass
        FakeCrew.__module__ = "crewai.crew"

        obj = FakeCrew()

        with patch("importlib.import_module") as mock_import:
            mock_module = MagicMock()
            mock_module.CrewAIAdapter = MagicMock(return_value=MagicMock())
            mock_import.return_value = mock_module
            result = _try_framework_adapter(obj)

        assert result is not None
        mock_import.assert_called_once_with("checkagent.adapters.crewai")

    def test_openai_agents_module_prefix_matches(self):
        class FakeOAAgent:
            pass
        FakeOAAgent.__module__ = "agents.agent"

        obj = FakeOAAgent()

        with patch("importlib.import_module") as mock_import:
            mock_module = MagicMock()
            mock_module.OpenAIAgentsAdapter = MagicMock(return_value=MagicMock())
            mock_import.return_value = mock_module
            result = _try_framework_adapter(obj)

        assert result is not None
        mock_import.assert_called_once_with("checkagent.adapters.openai_agents")

    def test_exact_module_name_matches(self):
        """Module exactly equal to prefix (not just starting with) also matches."""
        class FakeAgent:
            pass
        FakeAgent.__module__ = "pydantic_ai"

        obj = FakeAgent()

        with patch("importlib.import_module") as mock_import:
            mock_module = MagicMock()
            mock_module.PydanticAIAdapter = MagicMock(return_value=MagicMock())
            mock_import.return_value = mock_module
            result = _try_framework_adapter(obj)

        assert result is not None


class TestToolBoundaryTopLevelExport:
    def test_tool_boundary_importable_from_checkagent(self):
        import checkagent
        assert hasattr(checkagent, "ToolBoundary")

    def test_tool_boundary_in_all(self):
        import checkagent
        assert "ToolBoundary" in checkagent.__all__

    def test_tool_boundary_is_correct_class(self):
        import checkagent
        from checkagent.safety.tool_boundary import ToolBoundary as ToolBoundaryDirect
        assert checkagent.ToolBoundary is ToolBoundaryDirect

    def test_tool_boundary_instantiable_from_top_level(self):
        from checkagent import ToolBoundary
        boundary = ToolBoundary(allowed_tools={"search", "read_file"})
        assert "search" in boundary.allowed_tools
