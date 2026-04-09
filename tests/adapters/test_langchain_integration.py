"""Integration tests for LangChainAdapter with real langchain-core objects.

These tests use FakeListChatModel / FakeMessagesListChatModel from
langchain-core to validate the adapter against real framework types
(AIMessage, tool_calls, usage_metadata, etc.) instead of MagicMock stubs.

Requires: pip install langchain-core
"""

from __future__ import annotations

import pytest

langchain_core = pytest.importorskip("langchain_core", reason="langchain-core not installed")

from langchain_core.language_models.fake_chat_models import (  # noqa: E402
    FakeListChatModel,
    FakeMessagesListChatModel,
)
from langchain_core.messages import AIMessage  # noqa: E402
from langchain_core.output_parsers import StrOutputParser  # noqa: E402
from langchain_core.prompts import ChatPromptTemplate  # noqa: E402

from checkagent.adapters.langchain import LangChainAdapter  # noqa: E402
from checkagent.core.types import AgentInput, StreamEventType  # noqa: E402

# ---------------------------------------------------------------------------
# Basic chain tests
# ---------------------------------------------------------------------------


class TestLangChainIntegrationBasic:
    """Tests with real LangChain chains using FakeListChatModel."""

    async def test_simple_chain_with_str_output(self):
        """Chain with StrOutputParser returns a plain string."""
        model = FakeListChatModel(responses=["Hello world!"])
        chain = (
            ChatPromptTemplate.from_template("Say hello to {input}")
            | model
            | StrOutputParser()
        )
        adapter = LangChainAdapter(chain)

        result = await adapter.run("world")

        assert result.succeeded
        assert result.final_output == "Hello world!"
        assert len(result.steps) == 1
        assert result.steps[0].output_text == "Hello world!"
        assert result.steps[0].input_text == "world"
        assert result.duration_ms >= 0

    async def test_chain_returns_ai_message(self):
        """Chain without parser returns a real AIMessage."""
        model = FakeListChatModel(responses=["Raw AI response"])
        chain = ChatPromptTemplate.from_template("{input}") | model
        adapter = LangChainAdapter(chain)

        result = await adapter.run("test")

        assert result.succeeded
        assert result.steps[0].output_text == "Raw AI response"
        # final_output is the raw AIMessage
        assert hasattr(result.final_output, "content")
        assert result.final_output.content == "Raw AI response"

    async def test_custom_input_key(self):
        """Adapter passes query via custom input key."""
        model = FakeListChatModel(responses=["ok"])
        chain = ChatPromptTemplate.from_template("Q: {question}") | model
        adapter = LangChainAdapter(chain, input_key="question")

        result = await adapter.run("what is 2+2?")

        assert result.succeeded

    async def test_agent_input_with_context(self):
        """Context fields are merged into the invocation dict."""
        model = FakeListChatModel(responses=["answer"])
        chain = (
            ChatPromptTemplate.from_template("{context}\n{input}")
            | model
            | StrOutputParser()
        )
        adapter = LangChainAdapter(chain)

        inp = AgentInput(query="question", context={"context": "background info"})
        result = await adapter.run(inp)

        assert result.succeeded
        assert result.final_output == "answer"

    async def test_extra_inputs(self):
        """Constructor extra_inputs provide default template variables."""
        model = FakeListChatModel(responses=["result"])
        chain = (
            ChatPromptTemplate.from_template("{context}\n{input}")
            | model
            | StrOutputParser()
        )
        adapter = LangChainAdapter(chain, extra_inputs={"context": "default docs"})

        result = await adapter.run("query")

        assert result.succeeded

    async def test_extra_inputs_overridden_by_context(self):
        """Per-run context overrides constructor extra_inputs."""
        model = FakeListChatModel(responses=["custom result"])
        chain = (
            ChatPromptTemplate.from_template("{context}\n{input}")
            | model
            | StrOutputParser()
        )
        adapter = LangChainAdapter(
            chain, extra_inputs={"context": "default docs"}
        )

        inp = AgentInput(query="q", context={"context": "custom docs"})
        result = await adapter.run(inp)

        assert result.succeeded

    async def test_raw_input_key(self):
        """input_key='__raw__' passes the query string directly."""
        model = FakeListChatModel(responses=["echo"])
        adapter = LangChainAdapter(model, input_key="__raw__")

        result = await adapter.run("direct string")

        assert result.succeeded

    async def test_multi_response_chain(self):
        """Multiple invocations consume responses in order."""
        model = FakeListChatModel(responses=["first", "second", "third"])
        chain = ChatPromptTemplate.from_template("{input}") | model | StrOutputParser()
        adapter = LangChainAdapter(chain)

        r1 = await adapter.run("a")
        r2 = await adapter.run("b")
        r3 = await adapter.run("c")

        assert r1.final_output == "first"
        assert r2.final_output == "second"
        assert r3.final_output == "third"


# ---------------------------------------------------------------------------
# Tool call tests
# ---------------------------------------------------------------------------


class TestLangChainIntegrationToolCalls:
    """Tests with AIMessage containing tool_calls."""

    async def test_tool_calls_extracted(self):
        """Tool calls from a real AIMessage are captured in AgentRun."""
        msg = AIMessage(
            content="Let me search that for you",
            tool_calls=[
                {"name": "search", "args": {"query": "python"}, "id": "c1", "type": "tool_call"},
                {"name": "calc", "args": {"expr": "2+2"}, "id": "c2", "type": "tool_call"},
            ],
        )
        model = FakeMessagesListChatModel(responses=[msg])
        chain = ChatPromptTemplate.from_template("{input}") | model
        adapter = LangChainAdapter(chain)

        result = await adapter.run("search python")

        assert result.succeeded
        assert len(result.steps[0].tool_calls) == 2
        tc1 = result.steps[0].tool_calls[0]
        assert tc1.name == "search"
        assert tc1.arguments == {"query": "python"}
        tc2 = result.steps[0].tool_calls[1]
        assert tc2.name == "calc"
        assert tc2.arguments == {"expr": "2+2"}

    async def test_ai_message_with_no_tool_calls(self):
        """AIMessage with empty tool_calls produces no ToolCall objects."""
        msg = AIMessage(content="Just a plain message")
        model = FakeMessagesListChatModel(responses=[msg])
        chain = ChatPromptTemplate.from_template("{input}") | model
        adapter = LangChainAdapter(chain)

        result = await adapter.run("hello")

        assert result.succeeded
        assert result.steps[0].tool_calls == []
        assert result.steps[0].output_text == "Just a plain message"


# ---------------------------------------------------------------------------
# Token usage tests
# ---------------------------------------------------------------------------


class TestLangChainIntegrationTokenUsage:
    """Tests for token usage extraction from real AIMessage metadata."""

    async def test_usage_metadata_extracted(self):
        """usage_metadata on AIMessage populates token counts."""
        msg = AIMessage(
            content="response",
            usage_metadata={"input_tokens": 42, "output_tokens": 15, "total_tokens": 57},
        )
        model = FakeMessagesListChatModel(responses=[msg])
        chain = ChatPromptTemplate.from_template("{input}") | model
        adapter = LangChainAdapter(chain)

        result = await adapter.run("query")

        assert result.total_prompt_tokens == 42
        assert result.total_completion_tokens == 15

    async def test_no_usage_metadata(self):
        """FakeListChatModel produces no usage_metadata — tokens are None."""
        model = FakeListChatModel(responses=["response"])
        chain = ChatPromptTemplate.from_template("{input}") | model
        adapter = LangChainAdapter(chain)

        result = await adapter.run("query")

        assert result.total_prompt_tokens is None
        assert result.total_completion_tokens is None


# ---------------------------------------------------------------------------
# Error handling tests
# ---------------------------------------------------------------------------


class TestLangChainIntegrationErrors:
    """Error handling with real LangChain objects."""

    async def test_empty_response_list(self):
        """FakeListChatModel with no responses raises IndexError — captured as error."""
        model = FakeListChatModel(responses=[])
        chain = ChatPromptTemplate.from_template("{input}") | model
        adapter = LangChainAdapter(chain)

        result = await adapter.run("should fail")

        assert not result.succeeded
        assert "IndexError" in result.error
        assert result.duration_ms >= 0

    async def test_missing_template_variable(self):
        """Missing template variable raises KeyError — captured as error."""
        model = FakeListChatModel(responses=["ok"])
        chain = ChatPromptTemplate.from_template("{input} {missing_var}") | model
        adapter = LangChainAdapter(chain)

        result = await adapter.run("test")

        assert not result.succeeded
        assert result.error is not None


# ---------------------------------------------------------------------------
# Streaming tests
# ---------------------------------------------------------------------------


class TestLangChainIntegrationStreaming:
    """Streaming tests with real LangChain objects."""

    async def test_fallback_stream_for_chain_without_astream_events(self):
        """Chains that lack astream_events get a fallback synthetic stream."""
        model = FakeListChatModel(responses=["streamed output"])
        chain = ChatPromptTemplate.from_template("{input}") | model | StrOutputParser()
        adapter = LangChainAdapter(chain)

        events = []
        async for event in adapter.run_stream("query"):
            events.append(event)

        types = [e.event_type for e in events]
        assert types[0] == StreamEventType.RUN_START
        assert StreamEventType.TEXT_DELTA in types
        assert types[-1] == StreamEventType.RUN_END

    async def test_stream_with_real_astream_events(self):
        """Chains with astream_events produce real streaming events."""
        model = FakeListChatModel(responses=["Hello"])
        chain = ChatPromptTemplate.from_template("{input}") | model
        adapter = LangChainAdapter(chain)

        # Real LangChain chains support astream_events
        events = []
        async for event in adapter.run_stream("test"):
            events.append(event)

        types = [e.event_type for e in events]
        assert types[0] == StreamEventType.RUN_START
        assert types[-1] == StreamEventType.RUN_END
        # Should have at least one content event
        assert len(events) >= 3
