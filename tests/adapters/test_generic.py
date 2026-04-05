"""Tests for the GenericAdapter and @wrap decorator."""

import asyncio

from checkagent.adapters.generic import GenericAdapter, wrap
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
        assert result.duration_ms > 0
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
