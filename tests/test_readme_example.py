"""Verify all README examples are copy-pasteable and pass."""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from checkagent import (
    AgentInput,
    AgentRun,
    GenericAdapter,
    PromptInjectionDetector,
    Step,
    ToolCall,
    assert_output_matches,
    assert_output_schema,
    assert_tool_called,
)


# ── Agent under test (replace with your real agent) ────────────────────
async def booking_agent(query, *, llm, tools):
    """Minimal agent that books meetings via tool calls."""
    plan = await llm.complete(query)
    event = await tools.call("create_event", {"title": "Meeting"})
    return AgentRun(
        input=AgentInput(query=query),
        steps=[Step(output_text=plan, tool_calls=[
            ToolCall(name="create_event", arguments={"title": "Meeting"}, result=event),
        ])],
        final_output=event,
    )


# ── Test ───────────────────────────────────────────────────────────────
@pytest.mark.agent_test(layer="mock")
async def test_booking(ca_mock_llm, ca_mock_tool):
    ca_mock_llm.on_input(contains="book").respond("Booking your meeting now.")
    ca_mock_tool.on_call("create_event").respond(
        {"confirmed": True, "event_id": "evt-123"}
    )

    result = await booking_agent(
        "Book a meeting", llm=ca_mock_llm, tools=ca_mock_tool
    )

    assert_tool_called(result, "create_event", title="Meeting")
    assert result.final_output["confirmed"] is True


# ── README: Fault injection example ──────────────────────────────────
async def fault_agent(query, *, llm, tools):
    """Agent that may fail when tools fault."""
    try:
        result = await tools.call("search", {"q": query})
        return AgentRun(
            input=AgentInput(query=query),
            steps=[Step(output_text=str(result))],
            final_output=result,
        )
    except Exception as e:
        return AgentRun(
            input=AgentInput(query=query),
            steps=[],
            final_output=None,
            error=str(e),
        )


@pytest.mark.agent_test(layer="mock")
async def test_agent_handles_timeout(ca_mock_llm, ca_mock_tool, ca_fault):
    ca_fault.on_tool("search").timeout(seconds=5.0)
    ca_mock_tool.register("search")
    ca_mock_tool.attach_faults(ca_fault)
    ca_mock_llm.on_input(contains="search").respond("Searching...")

    result = await fault_agent("Find docs", llm=ca_mock_llm, tools=ca_mock_tool)
    assert result.error is not None


# ── README: Structured output assertions ─────────────────────────────
class BookingResponse(BaseModel):
    confirmed: bool
    event_id: str


@pytest.mark.agent_test(layer="mock")
async def test_output_structure():
    result = AgentRun(
        input=AgentInput(query="book meeting"),
        steps=[Step(output_text="done")],
        final_output={"confirmed": True, "event_id": "evt-123"},
    )
    assert_output_schema(result, BookingResponse)
    assert_output_matches(result, {"confirmed": True})


# ── README: Safety testing ───────────────────────────────────────────
@pytest.mark.agent_test(layer="eval")
async def test_no_prompt_injection():
    detector = PromptInjectionDetector()
    safety = detector.evaluate("Here's the weather forecast for tomorrow.")
    assert safety.passed, f"Found {safety.finding_count} injection(s)"


# ── README: GenericAdapter ───────────────────────────────────────────
async def my_agent_function(query: str) -> str:
    return f"Response to: {query}"


@pytest.mark.agent_test(layer="mock")
async def test_generic_adapter():
    adapter = GenericAdapter(my_agent_function)
    result = await adapter.run("Hello")
    assert result.final_output is not None
