"""Verify the README example is copy-pasteable and passes."""

from __future__ import annotations

import pytest

from checkagent import AgentInput, AgentRun, Step, ToolCall, assert_tool_called


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
