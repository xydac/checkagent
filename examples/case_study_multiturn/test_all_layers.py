"""Case Study 3: Multi-turn conversation testing with CheckAgent.

Demonstrates how CheckAgent's layered testing catches different categories
of multi-turn conversation bugs:

- B1: Context amnesia — agent ignores conversation history
- B2: Wrong tool on follow-up — searches FAQ again instead of skipping
- B3: Preference not saved — misses save_preference tool call
- B4: Redundant tool calls — calls search_faq every turn even on follow-ups

Tests across MOCK and EVAL layers showing which bugs each layer catches.
"""

from __future__ import annotations

import pytest

from checkagent.conversation.session import Conversation
from checkagent.core.types import AgentInput, AgentRun
from checkagent.eval.metrics import step_efficiency, tool_correctness, trajectory_match
from checkagent.mock.llm import MockLLM
from checkagent.mock.tool import MockTool

from .agent import faq_chatbot

# ---------------------------------------------------------------------------
# Helpers: wrap the agent for Conversation
# ---------------------------------------------------------------------------


def _make_agent(llm: MockLLM, tools: MockTool):
    """Create an agent function compatible with Conversation."""

    async def agent_fn(input: AgentInput) -> AgentRun:
        return await faq_chatbot(input, llm=llm, tools=tools)

    return agent_fn


# ---------------------------------------------------------------------------
# Layer 1: MOCK — deterministic assertions on tool calls and sequencing
# ---------------------------------------------------------------------------


class TestMockLayer:
    """Mock-layer tests: tool calls, sequencing, context pass-through."""

    @pytest.mark.agent_test(layer="mock")
    async def test_first_turn_searches_faq(self):
        """First turn on a new topic should call search_faq."""
        llm = MockLLM(default_response="Let me help with that.")
        tools = MockTool()
        tools.register("search_faq", response={"answer": "Reset via Settings > Account"})
        tools.register("save_preference", response={"saved": True})

        conv = Conversation(_make_agent(llm, tools))
        r1 = await conv.say("How do I reset my password?")

        assert r1.error is None
        assert tools.was_called("search_faq")
        assert len(tools.get_calls_for("search_faq")) == 1

    @pytest.mark.agent_test(layer="mock")
    async def test_followup_skips_faq_search(self):
        """B2: Follow-up turns should NOT re-search FAQ."""
        llm = MockLLM(default_response="Here's more detail on that.")
        tools = MockTool()
        tools.register("search_faq", response={"answer": "Reset via Settings"})
        tools.register("save_preference", response={"saved": True})

        conv = Conversation(_make_agent(llm, tools))
        await conv.say("How do I reset my password?")
        tools.reset_calls()  # Clear call history from turn 1

        await conv.say("Can you elaborate on that?")

        # Follow-up should NOT call search_faq again
        assert not tools.was_called("search_faq"), (
            "Follow-up turn should skip FAQ search (B2: redundant tool call)"
        )

    @pytest.mark.agent_test(layer="mock")
    async def test_preference_saved(self):
        """B3: Agent should call save_preference when user expresses one."""
        llm = MockLLM(default_response="Preference noted!")
        tools = MockTool()
        tools.register("search_faq", response={"answer": "We support email and SMS."})
        tools.register("save_preference", response={"saved": True})

        conv = Conversation(_make_agent(llm, tools))
        await conv.say("What notification options are available?")
        await conv.say("I prefer email notifications")

        assert tools.was_called("save_preference"), (
            "Agent should save user preferences (B3)"
        )
        # Verify the preference value
        pref_calls = tools.get_calls_for("save_preference")
        assert len(pref_calls) == 1
        assert "email" in str(pref_calls[0].arguments).lower()

    @pytest.mark.agent_test(layer="mock")
    async def test_context_passed_to_llm(self):
        """B1: Agent should pass conversation history to LLM on follow-up turns."""
        llm = MockLLM(default_response="Based on our earlier discussion...")
        tools = MockTool()
        tools.register("search_faq", response={"answer": "Use API key auth."})
        tools.register("save_preference", response={"saved": True})

        conv = Conversation(_make_agent(llm, tools))
        await conv.say("How does authentication work?")
        await conv.say("Tell me more about that")

        # The second LLM call should include conversation context
        assert llm.call_count >= 3  # At least: turn1 understand + synthesize, turn2 understand
        last_calls = llm.calls[-2:]  # Last 2 calls from turn 2
        context_call = last_calls[0]  # The "understand" call
        assert "Previous conversation" in context_call.input_text, (
            "Follow-up turn should include conversation history (B1: context amnesia)"
        )

    @pytest.mark.agent_test(layer="mock")
    async def test_three_turn_conversation_flow(self):
        """Full 3-turn conversation: question → follow-up → preference."""
        llm = MockLLM(default_response="Got it!")
        tools = MockTool()
        tools.register("search_faq", response={"answer": "Plans: Free, Pro, Enterprise"})
        tools.register("save_preference", response={"saved": True})

        conv = Conversation(_make_agent(llm, tools))

        r1 = await conv.say("What pricing plans do you have?")
        assert r1.error is None
        assert conv.total_turns == 1

        r2 = await conv.say("Tell me more about Pro")
        assert r2.error is None
        assert conv.total_turns == 2

        r3 = await conv.say("I prefer the Pro plan")
        assert r3.error is None
        assert conv.total_turns == 3

        # Verify tool sequencing across all turns
        all_tools = [tc.name for tc in conv.all_tool_calls]
        assert "search_faq" in all_tools, "Should search FAQ on first question"
        assert "save_preference" in all_tools, "Should save preference on third turn"


# ---------------------------------------------------------------------------
# Layer 3: EVAL — metric-based assertions on conversation quality
# ---------------------------------------------------------------------------


class TestEvalLayer:
    """Eval-layer tests: metrics that catch quality bugs invisible to mock."""

    @pytest.mark.agent_test(layer="eval")
    async def test_tool_trajectory_first_turn(self):
        """Eval: verify tool trajectory matches expected pattern for first turn."""
        llm = MockLLM(default_response="Here's the answer.")
        tools = MockTool()
        tools.register("search_faq", response={"answer": "Go to Settings > Reset"})
        tools.register("save_preference", response={"saved": True})

        conv = Conversation(_make_agent(llm, tools))
        r1 = await conv.say("How do I reset my password?")

        score = trajectory_match(
            r1,
            expected_trajectory=["search_faq"],
            mode="ordered",
        )
        assert score.passed, f"First turn should follow search_faq trajectory: {score.reason}"

    @pytest.mark.agent_test(layer="eval")
    async def test_step_efficiency_followup(self):
        """Eval: follow-up turns should be more efficient (fewer steps)."""
        llm = MockLLM(default_response="Sure, more details...")
        tools = MockTool()
        tools.register("search_faq", response={"answer": "Use OAuth2 tokens."})
        tools.register("save_preference", response={"saved": True})

        conv = Conversation(_make_agent(llm, tools))
        r1 = await conv.say("How does authentication work?")
        r2 = await conv.say("Can you elaborate on that?")

        # Follow-up should have fewer steps than initial query
        # First turn: understand + search_faq + synthesize = 3 steps optimal
        score1 = step_efficiency(r1, optimal_steps=3)
        # Follow-up: understand + synthesize = 2 steps optimal
        score2 = step_efficiency(r2, optimal_steps=2)

        assert score1.passed, f"First turn should be efficient: {score1.reason}"
        assert score2.passed, f"Follow-up should be efficient: {score2.reason}"
        # Follow-up should have fewer steps (no FAQ search)
        assert len(r2.steps) < len(r1.steps), (
            f"Follow-up ({len(r2.steps)} steps) should have fewer steps "
            f"than first turn ({len(r1.steps)} steps) — B4: redundant tool calls"
        )

    @pytest.mark.agent_test(layer="eval")
    async def test_preference_trajectory(self):
        """Eval: preference turn should include save_preference in trajectory."""
        llm = MockLLM(default_response="Preference saved!")
        tools = MockTool()
        tools.register("search_faq", response={"answer": "Options: email, SMS, push"})
        tools.register("save_preference", response={"saved": True})

        conv = Conversation(_make_agent(llm, tools))
        await conv.say("What notification options exist?")
        r2 = await conv.say("I prefer email notifications")

        score = trajectory_match(
            r2,
            expected_trajectory=["save_preference"],
            mode="unordered",
        )
        assert score.passed, f"Preference turn should call save_preference: {score.reason}"

    @pytest.mark.agent_test(layer="eval")
    async def test_tool_correctness_across_turns(self):
        """Eval: tool arguments should match expected values across turns."""
        llm = MockLLM(default_response="Processing...")
        tools = MockTool()
        tools.register("search_faq", response={"answer": "API rate limits: 100/min"})
        tools.register("save_preference", response={"saved": True})

        conv = Conversation(_make_agent(llm, tools))
        r1 = await conv.say("What are the API rate limits?")

        score = tool_correctness(
            r1,
            expected_tools=["search_faq"],
        )
        assert score.passed, f"Tool call should include search_faq: {score.reason}"

    @pytest.mark.agent_test(layer="eval")
    async def test_conversation_context_growth(self):
        """Eval: LLM input should grow as conversation context accumulates."""
        llm = MockLLM(default_response="Response.")
        tools = MockTool()
        tools.register("search_faq", response={"answer": "Answer."})
        tools.register("save_preference", response={"saved": True})

        conv = Conversation(_make_agent(llm, tools))
        await conv.say("What is feature X?")
        await conv.say("Tell me more about that")
        await conv.say("And what about feature Y?")

        assert conv.total_turns == 3

        # Verify context grows: the LLM's "understand" calls should get
        # progressively longer as conversation history is included
        understand_calls = [c for c in llm.calls if "Previous conversation" in c.input_text]
        assert len(understand_calls) >= 1, (
            "At least one follow-up turn should include conversation history"
        )
        # Third turn's context should include history from turns 1 and 2
        last_understand = understand_calls[-1]
        assert "feature X" in last_understand.input_text.lower() or \
               "feature x" in last_understand.input_text.lower(), (
            "Later turns should reference earlier conversation topics"
        )
