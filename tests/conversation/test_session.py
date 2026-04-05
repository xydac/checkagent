"""Tests for the multi-turn conversation session (F10.1, F10.3, F10.4)."""

from __future__ import annotations

import pytest

from checkagent.conversation.session import Conversation
from checkagent.core.types import AgentInput, AgentRun, Step, ToolCall

# --- Helpers ---


def make_agent_fn(
    response: str = "Mock response",
    tool_calls: list[ToolCall] | None = None,
    steps: list[Step] | None = None,
    prompt_tokens: int | None = None,
    completion_tokens: int | None = None,
):
    """Create a mock agent function that returns predictable AgentRuns."""

    async def agent_fn(input: AgentInput) -> AgentRun:
        if steps is not None:
            run_steps = steps
        else:
            run_steps = [
                Step(
                    step_index=0,
                    input_text=input.query,
                    output_text=response,
                    tool_calls=tool_calls or [],
                )
            ]
        return AgentRun(
            input=input,
            steps=run_steps,
            final_output=response,
            total_prompt_tokens=prompt_tokens,
            total_completion_tokens=completion_tokens,
        )

    return agent_fn


def make_echo_agent():
    """Agent that echoes input and includes conversation history in output."""

    async def agent_fn(input: AgentInput) -> AgentRun:
        # Build output that references conversation history
        parts = [f"You said: {input.query}"]
        if input.conversation_history:
            last_user = [
                m["content"]
                for m in input.conversation_history
                if m["role"] == "user"
            ]
            if last_user:
                parts.append(f"Previously you said: {last_user[-1]}")

        output = ". ".join(parts)
        return AgentRun(
            input=input,
            steps=[Step(step_index=0, input_text=input.query, output_text=output)],
            final_output=output,
        )

    return agent_fn


def make_tool_agent():
    """Agent that makes tool calls based on keywords in input."""

    async def agent_fn(input: AgentInput) -> AgentRun:
        calls = []
        if "order" in input.query.lower():
            calls.append(ToolCall(name="lookup_order", arguments={"query": input.query}))
        if "refund" in input.query.lower():
            calls.append(ToolCall(name="initiate_refund", arguments={}))
        if "?" in input.query:
            calls.append(ToolCall(name="ask_clarification", arguments={}))

        return AgentRun(
            input=input,
            steps=[Step(step_index=0, tool_calls=calls, output_text="Done")],
            final_output="Done",
        )

    return agent_fn


# --- Test: Basic conversation flow ---


class TestConversationBasics:
    async def test_single_turn(self):
        conv = Conversation(make_agent_fn("Hello!"))
        result = await conv.say("Hi")
        assert result.final_output == "Hello!"
        assert conv.total_turns == 1

    async def test_multiple_turns(self):
        conv = Conversation(make_agent_fn("Response"))
        await conv.say("Turn 1")
        await conv.say("Turn 2")
        await conv.say("Turn 3")
        assert conv.total_turns == 3

    async def test_say_returns_agent_run(self):
        conv = Conversation(make_agent_fn("Result"))
        result = await conv.say("Test")
        assert isinstance(result, AgentRun)
        assert result.final_output == "Result"

    async def test_empty_conversation(self):
        conv = Conversation(make_agent_fn())
        assert conv.total_turns == 0
        assert conv.last_turn is None
        assert conv.last_result is None
        assert conv.total_tool_calls == 0
        assert conv.total_steps == 0

    async def test_last_turn(self):
        conv = Conversation(make_agent_fn("Reply"))
        await conv.say("First")
        await conv.say("Second")
        assert conv.last_turn is not None
        assert conv.last_turn.input_text == "Second"
        assert conv.last_turn.index == 1

    async def test_last_result(self):
        conv = Conversation(make_agent_fn("Reply"))
        await conv.say("Test")
        assert conv.last_result is not None
        assert conv.last_result.final_output == "Reply"


# --- Test: Conversation history accumulation ---


class TestHistoryAccumulation:
    async def test_first_turn_has_empty_history(self):
        """First turn should receive no conversation history."""
        received_inputs: list[AgentInput] = []

        async def capture_fn(input: AgentInput) -> AgentRun:
            received_inputs.append(input)
            return AgentRun(input=input, final_output="ok")

        conv = Conversation(capture_fn)
        await conv.say("Hello")
        assert received_inputs[0].conversation_history == []

    async def test_second_turn_has_first_turn_history(self):
        received_inputs: list[AgentInput] = []

        async def capture_fn(input: AgentInput) -> AgentRun:
            received_inputs.append(input)
            return AgentRun(input=input, final_output="Response")

        conv = Conversation(capture_fn)
        await conv.say("Hello")
        await conv.say("How are you?")

        history = received_inputs[1].conversation_history
        assert len(history) == 2
        assert history[0] == {"role": "user", "content": "Hello"}
        assert history[1] == {"role": "assistant", "content": "Response"}

    async def test_three_turns_accumulate_history(self):
        received_inputs: list[AgentInput] = []

        async def capture_fn(input: AgentInput) -> AgentRun:
            received_inputs.append(input)
            return AgentRun(input=input, final_output=f"Reply to: {input.query}")

        conv = Conversation(capture_fn)
        await conv.say("A")
        await conv.say("B")
        await conv.say("C")

        history = received_inputs[2].conversation_history
        assert len(history) == 4  # 2 turns * 2 messages each
        assert history[0] == {"role": "user", "content": "A"}
        assert history[1] == {"role": "assistant", "content": "Reply to: A"}
        assert history[2] == {"role": "user", "content": "B"}
        assert history[3] == {"role": "assistant", "content": "Reply to: B"}

    async def test_none_output_omitted_from_history(self):
        """If agent returns None output, no assistant message is added."""
        received_inputs: list[AgentInput] = []

        async def none_agent(input: AgentInput) -> AgentRun:
            received_inputs.append(input)
            return AgentRun(input=input, final_output=None)

        conv = Conversation(none_agent)
        await conv.say("Hello")
        await conv.say("Again")

        history = received_inputs[1].conversation_history
        # Only user message, no assistant message since output was None
        assert len(history) == 1
        assert history[0] == {"role": "user", "content": "Hello"}


# --- Test: Turn object ---


class TestTurn:
    async def test_turn_index(self):
        conv = Conversation(make_agent_fn("R"))
        await conv.say("A")
        await conv.say("B")
        assert conv.turns[0].index == 0
        assert conv.turns[1].index == 1

    async def test_turn_input_text(self):
        conv = Conversation(make_agent_fn("R"))
        await conv.say("My input")
        assert conv.turns[0].input_text == "My input"

    async def test_turn_output_text(self):
        conv = Conversation(make_agent_fn("My output"))
        await conv.say("Test")
        assert conv.turns[0].output_text == "My output"

    async def test_turn_output_text_none(self):
        async def none_agent(input: AgentInput) -> AgentRun:
            return AgentRun(input=input, final_output=None)

        conv = Conversation(none_agent)
        await conv.say("Test")
        assert conv.turns[0].output_text is None

    async def test_turn_tool_calls(self):
        calls = [ToolCall(name="search", arguments={"q": "test"})]
        conv = Conversation(make_agent_fn("R", tool_calls=calls))
        await conv.say("Test")
        assert len(conv.turns[0].tool_calls) == 1
        assert conv.turns[0].tool_calls[0].name == "search"

    async def test_get_turn(self):
        conv = Conversation(make_agent_fn("R"))
        await conv.say("A")
        await conv.say("B")
        turn = conv.get_turn(1)
        assert turn.input_text == "B"

    async def test_get_turn_out_of_range(self):
        conv = Conversation(make_agent_fn("R"))
        await conv.say("A")
        with pytest.raises(IndexError):
            conv.get_turn(5)


# --- Test: Tool call tracking ---


class TestToolCallTracking:
    async def test_total_tool_calls(self):
        conv = Conversation(make_tool_agent())
        await conv.say("Check my order?")  # lookup_order + ask_clarification
        await conv.say("Process refund")  # initiate_refund
        assert conv.total_tool_calls == 3

    async def test_all_tool_calls(self):
        conv = Conversation(make_tool_agent())
        await conv.say("order?")
        await conv.say("refund")
        names = [tc.name for tc in conv.all_tool_calls]
        assert "lookup_order" in names
        assert "ask_clarification" in names
        assert "initiate_refund" in names

    async def test_tool_was_called(self):
        conv = Conversation(make_tool_agent())
        await conv.say("Check my order")
        assert conv.tool_was_called("lookup_order")
        assert not conv.tool_was_called("initiate_refund")

    async def test_tool_was_called_in_turn(self):
        conv = Conversation(make_tool_agent())
        await conv.say("Check my order")
        await conv.say("Process refund")
        assert conv.tool_was_called_in_turn(0, "lookup_order")
        assert not conv.tool_was_called_in_turn(0, "initiate_refund")
        assert conv.tool_was_called_in_turn(1, "initiate_refund")
        assert not conv.tool_was_called_in_turn(1, "lookup_order")


# --- Test: Token / metrics tracking (F10.4) ---


class TestConversationMetrics:
    async def test_total_tokens_both_set(self):
        conv = Conversation(
            make_agent_fn("R", prompt_tokens=100, completion_tokens=50)
        )
        await conv.say("A")
        await conv.say("B")
        assert conv.total_prompt_tokens == 200
        assert conv.total_completion_tokens == 100
        assert conv.total_tokens == 300

    async def test_total_tokens_none_when_missing(self):
        conv = Conversation(make_agent_fn("R"))  # no token counts
        await conv.say("A")
        assert conv.total_prompt_tokens is None
        assert conv.total_completion_tokens is None
        assert conv.total_tokens is None

    async def test_total_steps(self):
        steps = [
            Step(step_index=0, output_text="thinking"),
            Step(step_index=1, output_text="acting"),
        ]
        conv = Conversation(make_agent_fn("R", steps=steps))
        await conv.say("A")
        await conv.say("B")
        assert conv.total_steps == 4  # 2 steps * 2 turns


# --- Test: Context references (F10.3) ---


class TestContextReferences:
    async def test_references_earlier_input(self):
        conv = Conversation(make_echo_agent())
        await conv.say("Hello world")
        await conv.say("What was my first message?")
        # Turn 1 output includes "Previously you said: Hello world"
        assert conv.context_references(turn=1, references_turn=0)

    async def test_no_reference(self):
        call_count = 0

        async def static_agent(input: AgentInput) -> AgentRun:
            nonlocal call_count
            call_count += 1
            output = f"Response number {call_count}"
            return AgentRun(
                input=input,
                steps=[Step(step_index=0, output_text=output)],
                final_output=output,
            )

        conv = Conversation(static_agent)
        await conv.say("Hello world")
        await conv.say("Something else")
        # Turn 1 output ("Response number 2") doesn't reference
        # turn 0's input ("Hello world") or output ("Response number 1")
        assert not conv.context_references(turn=1, references_turn=0)

    async def test_references_turn_must_be_before(self):
        conv = Conversation(make_agent_fn("R"))
        await conv.say("A")
        await conv.say("B")
        with pytest.raises(ValueError, match="must be before"):
            conv.context_references(turn=0, references_turn=1)

    async def test_references_same_turn_raises(self):
        conv = Conversation(make_agent_fn("R"))
        await conv.say("A")
        with pytest.raises(ValueError, match="must be before"):
            conv.context_references(turn=0, references_turn=0)

    async def test_references_out_of_range(self):
        conv = Conversation(make_agent_fn("R"))
        await conv.say("A")
        with pytest.raises(IndexError):
            conv.context_references(turn=5, references_turn=0)

    async def test_references_none_output_returns_false(self):
        async def none_agent(input: AgentInput) -> AgentRun:
            return AgentRun(input=input, final_output=None)

        conv = Conversation(none_agent)
        await conv.say("A")
        await conv.say("B")
        assert not conv.context_references(turn=1, references_turn=0)


# --- Test: Context and metadata passthrough ---


class TestContextPassthrough:
    async def test_say_with_context(self):
        received: list[AgentInput] = []

        async def capture(input: AgentInput) -> AgentRun:
            received.append(input)
            return AgentRun(input=input, final_output="ok")

        conv = Conversation(capture)
        await conv.say("Test", context={"user_id": "123"})
        assert received[0].context == {"user_id": "123"}

    async def test_say_with_metadata(self):
        received: list[AgentInput] = []

        async def capture(input: AgentInput) -> AgentRun:
            received.append(input)
            return AgentRun(input=input, final_output="ok")

        conv = Conversation(capture)
        await conv.say("Test", metadata={"turn_tag": "greeting"})
        assert received[0].metadata == {"turn_tag": "greeting"}


# --- Test: Reset ---


class TestReset:
    async def test_reset_clears_turns(self):
        conv = Conversation(make_agent_fn("R"))
        await conv.say("A")
        await conv.say("B")
        assert conv.total_turns == 2
        conv.reset()
        assert conv.total_turns == 0
        assert conv.turns == []
        assert conv.last_turn is None

    async def test_reset_then_new_turn_has_empty_history(self):
        received: list[AgentInput] = []

        async def capture(input: AgentInput) -> AgentRun:
            received.append(input)
            return AgentRun(input=input, final_output="ok")

        conv = Conversation(capture)
        await conv.say("First")
        conv.reset()
        await conv.say("After reset")
        assert received[1].conversation_history == []


# --- Test: ap_conversation fixture ---


class TestFixture:
    def test_fixture_returns_conversation_class(self, ap_conversation):
        """ap_conversation fixture returns the Conversation class."""
        assert ap_conversation is Conversation

    async def test_fixture_creates_working_conversation(self, ap_conversation):
        """Can create a working Conversation from the fixture."""
        conv = ap_conversation(make_agent_fn("Fixture works!"))
        result = await conv.say("Hello")
        assert result.final_output == "Fixture works!"
        assert conv.total_turns == 1
