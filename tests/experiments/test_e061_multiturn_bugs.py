"""E-061: Multi-turn bug detection across testing layers.

Tests 4 seeded multi-turn conversation bugs across mock and eval layers
to determine which layers catch which bug categories. Produces a bug
detection matrix for paper Section 6 (Case Study 3).

Bug types:
- B1: Context amnesia — agent ignores conversation history on follow-ups
- B2: Redundant tool calls — re-searches FAQ on every turn
- B3: Missing preference save — ignores user preference expression
- B4: Conversation drift — final output doesn't reference earlier context
"""

from __future__ import annotations

import time

import pytest

from checkagent.conversation.session import Conversation
from checkagent.core.types import AgentInput, AgentRun, Step, ToolCall
from checkagent.eval.metrics import trajectory_match
from checkagent.mock.llm import MockLLM
from checkagent.mock.tool import MockTool

# ---------------------------------------------------------------------------
# Agent variants: correct + 4 buggy versions
# ---------------------------------------------------------------------------


async def _correct_agent(
    input: AgentInput, *, llm: object, tools: object
) -> AgentRun:
    """Correct multi-turn agent: uses history, avoids redundant calls."""
    steps: list[Step] = []
    query = input.query
    history = input.conversation_history

    # Build context-aware prompt
    prompt = query
    if history:
        recent = history[-4:]
        prompt = (
            "Previous conversation:\n"
            + "\n".join(f"{m['role']}: {m['content']}" for m in recent)
            + f"\n\nCurrent question: {query}"
        )

    response = await llm.complete(prompt)
    steps.append(Step(input_text=prompt, output_text=response))

    # Only search FAQ on new topics
    is_followup = any(
        s in query.lower()
        for s in ["tell me more", "elaborate", "what about", "you mentioned"]
    )
    if not is_followup:
        result = await tools.call("search_faq", {"query": query})
        steps.append(Step(
            input_text=f"search_faq({query})",
            output_text=str(result),
            tool_calls=[ToolCall(name="search_faq", arguments={"query": query}, result=result)],
        ))

    # Save preference if expressed
    if any(s in query.lower() for s in ["i prefer", "i like", "i want"]):
        pref = query.split("prefer")[-1].strip() if "prefer" in query.lower() else query
        result = await tools.call("save_preference", {"preference": pref})
        steps.append(Step(
            input_text=f"save_preference({pref})",
            output_text=str(result),
            tool_calls=[
                ToolCall(name="save_preference", arguments={"preference": pref}, result=result)
            ],
        ))

    final = await llm.complete(f"Synthesize: {' '.join(s.output_text for s in steps)}")
    steps.append(Step(input_text="synthesize", output_text=final))

    return AgentRun(input=input, steps=steps, final_output=final)


async def _buggy_amnesia(
    input: AgentInput, *, llm: object, tools: object
) -> AgentRun:
    """B1: Ignores conversation history — always treats input as standalone."""
    steps: list[Step] = []
    query = input.query
    # BUG: Never looks at conversation_history

    response = await llm.complete(query)  # No context prefix
    steps.append(Step(input_text=query, output_text=response))

    result = await tools.call("search_faq", {"query": query})
    steps.append(Step(
        input_text=f"search_faq({query})",
        output_text=str(result),
        tool_calls=[ToolCall(name="search_faq", arguments={"query": query}, result=result)],
    ))

    final = await llm.complete(f"Synthesize: {' '.join(s.output_text for s in steps)}")
    steps.append(Step(input_text="synthesize", output_text=final))

    return AgentRun(input=input, steps=steps, final_output=final)


async def _buggy_redundant(
    input: AgentInput, *, llm: object, tools: object
) -> AgentRun:
    """B2: Always calls search_faq, even on follow-ups."""
    steps: list[Step] = []
    query = input.query
    history = input.conversation_history

    prompt = query
    if history:
        recent = history[-4:]
        prompt = (
            "Previous conversation:\n"
            + "\n".join(f"{m['role']}: {m['content']}" for m in recent)
            + f"\n\nCurrent question: {query}"
        )

    response = await llm.complete(prompt)
    steps.append(Step(input_text=prompt, output_text=response))

    # BUG: Always searches FAQ, even on follow-ups
    result = await tools.call("search_faq", {"query": query})
    steps.append(Step(
        input_text=f"search_faq({query})",
        output_text=str(result),
        tool_calls=[ToolCall(name="search_faq", arguments={"query": query}, result=result)],
    ))

    if any(s in query.lower() for s in ["i prefer", "i like", "i want"]):
        pref = query.split("prefer")[-1].strip() if "prefer" in query.lower() else query
        result = await tools.call("save_preference", {"preference": pref})
        steps.append(Step(
            input_text=f"save_preference({pref})",
            output_text=str(result),
            tool_calls=[
                ToolCall(name="save_preference", arguments={"preference": pref}, result=result)
            ],
        ))

    final = await llm.complete(f"Synthesize: {' '.join(s.output_text for s in steps)}")
    steps.append(Step(input_text="synthesize", output_text=final))

    return AgentRun(input=input, steps=steps, final_output=final)


async def _buggy_no_pref(
    input: AgentInput, *, llm: object, tools: object
) -> AgentRun:
    """B3: Never calls save_preference."""
    steps: list[Step] = []
    query = input.query
    history = input.conversation_history

    prompt = query
    if history:
        recent = history[-4:]
        prompt = (
            "Previous conversation:\n"
            + "\n".join(f"{m['role']}: {m['content']}" for m in recent)
            + f"\n\nCurrent question: {query}"
        )

    response = await llm.complete(prompt)
    steps.append(Step(input_text=prompt, output_text=response))

    is_followup = any(
        s in query.lower()
        for s in ["tell me more", "elaborate", "what about", "you mentioned"]
    )
    if not is_followup:
        result = await tools.call("search_faq", {"query": query})
        steps.append(Step(
            input_text=f"search_faq({query})",
            output_text=str(result),
            tool_calls=[ToolCall(name="search_faq", arguments={"query": query}, result=result)],
        ))

    # BUG: Never saves preferences — ignores "I prefer" signals

    final = await llm.complete(f"Synthesize: {' '.join(s.output_text for s in steps)}")
    steps.append(Step(input_text="synthesize", output_text=final))

    return AgentRun(input=input, steps=steps, final_output=final)


async def _buggy_drift(
    input: AgentInput, *, llm: object, tools: object
) -> AgentRun:
    """B4: Includes history in prompt but output doesn't reference it."""
    steps: list[Step] = []
    query = input.query
    history = input.conversation_history

    prompt = query
    if history:
        recent = history[-4:]
        prompt = (
            "Previous conversation:\n"
            + "\n".join(f"{m['role']}: {m['content']}" for m in recent)
            + f"\n\nCurrent question: {query}"
        )

    response = await llm.complete(prompt)
    steps.append(Step(input_text=prompt, output_text=response))

    is_followup = any(
        s in query.lower()
        for s in ["tell me more", "elaborate", "what about", "you mentioned"]
    )
    if not is_followup:
        result = await tools.call("search_faq", {"query": query})
        steps.append(Step(
            input_text=f"search_faq({query})",
            output_text=str(result),
            tool_calls=[ToolCall(name="search_faq", arguments={"query": query}, result=result)],
        ))

    if any(s in query.lower() for s in ["i prefer", "i like", "i want"]):
        pref = query.split("prefer")[-1].strip() if "prefer" in query.lower() else query
        result = await tools.call("save_preference", {"preference": pref})
        steps.append(Step(
            input_text=f"save_preference({pref})",
            output_text=str(result),
            tool_calls=[
                ToolCall(name="save_preference", arguments={"preference": pref}, result=result)
            ],
        ))

    # BUG: Final synthesis ignores all context, gives generic response
    final = "I hope that helps! Let me know if you have any other questions."
    steps.append(Step(input_text="synthesize", output_text=final))

    return AgentRun(input=input, steps=steps, final_output=final)


# ---------------------------------------------------------------------------
# Bug detection matrix
# ---------------------------------------------------------------------------

AGENTS = {
    "correct": _correct_agent,
    "B1_amnesia": _buggy_amnesia,
    "B2_redundant": _buggy_redundant,
    "B3_no_pref": _buggy_no_pref,
    "B4_drift": _buggy_drift,
}


def _setup():
    """Create shared MockLLM and MockTool."""
    llm = MockLLM(default_response="You can reset your password in Settings > Account.")
    tools = MockTool()
    tools.register("search_faq", response={"answer": "You can reset via Settings > Account."})
    tools.register("save_preference", response={"saved": True})
    return llm, tools


def _make_conv(agent_fn, llm, tools):
    """Wrap agent function for use with Conversation."""
    async def fn(input: AgentInput) -> AgentRun:
        return await agent_fn(input, llm=llm, tools=tools)
    return Conversation(fn)


# ---------------------------------------------------------------------------
# Tests: run each bug detector against all agents
# ---------------------------------------------------------------------------

class TestMultiturnBugDetection:
    """Systematic bug detection across mock and eval layers."""

    @pytest.mark.agent_test(layer="mock")
    async def test_mock_context_history_check(self):
        """MOCK detector: does the LLM receive conversation history?"""
        results = {}
        for name, agent_fn in AGENTS.items():
            llm, tools = _setup()
            conv = _make_conv(agent_fn, llm, tools)
            await conv.say("How do I reset my password?")
            turn1_count = llm.call_count
            await conv.say("Tell me more about that")

            # Check if second turn's LLM call includes history
            turn2_calls = llm.calls[turn1_count:]
            has_history = any("Previous conversation" in c.input_text for c in turn2_calls)
            results[name] = has_history

        # B1 (amnesia) should fail this check
        assert results["correct"] is True, "Correct agent should pass history"
        assert results["B1_amnesia"] is False, "B1 should be caught: no history"
        assert results["B2_redundant"] is True, "B2 passes history (different bug)"
        assert results["B3_no_pref"] is True, "B3 passes history (different bug)"
        assert results["B4_drift"] is True, "B4 passes history (different bug)"

    @pytest.mark.agent_test(layer="mock")
    async def test_mock_redundant_tool_check(self):
        """MOCK detector: does the agent avoid redundant FAQ searches on follow-ups?"""
        results = {}
        for name, agent_fn in AGENTS.items():
            llm, tools = _setup()
            conv = _make_conv(agent_fn, llm, tools)
            await conv.say("How do I reset my password?")
            tools.reset_calls()
            await conv.say("Tell me more about that")

            # Follow-up should NOT call search_faq
            results[name] = not tools.was_called("search_faq")

        assert results["correct"] is True
        assert results["B1_amnesia"] is False, "B1: always searches (amnesia)"
        assert results["B2_redundant"] is False, "B2 caught: redundant search"
        assert results["B3_no_pref"] is True, "B3 passes (different bug)"
        assert results["B4_drift"] is True, "B4 passes (different bug)"

    @pytest.mark.agent_test(layer="mock")
    async def test_mock_preference_save_check(self):
        """MOCK detector: does the agent save preferences?"""
        results = {}
        for name, agent_fn in AGENTS.items():
            llm, tools = _setup()
            conv = _make_conv(agent_fn, llm, tools)
            await conv.say("What options are available?")
            await conv.say("I prefer email notifications")

            results[name] = tools.was_called("save_preference")

        assert results["correct"] is True
        assert results["B1_amnesia"] is False, "B1 caught: amnesia means no pref detection"
        assert results["B2_redundant"] is True, "B2 saves prefs (different bug)"
        assert results["B3_no_pref"] is False, "B3 caught: never saves"
        assert results["B4_drift"] is True, "B4 saves prefs (different bug)"

    @pytest.mark.agent_test(layer="eval")
    async def test_eval_step_efficiency_followup(self):
        """EVAL detector: are follow-up turns more efficient than first turns?"""
        results = {}
        for name, agent_fn in AGENTS.items():
            llm, tools = _setup()
            conv = _make_conv(agent_fn, llm, tools)
            r1 = await conv.say("How do I reset my password?")
            r2 = await conv.say("Tell me more about that")

            # Follow-up should have fewer steps than first turn
            results[name] = len(r2.steps) < len(r1.steps)

        assert results["correct"] is True
        assert results["B1_amnesia"] is False, "B1 caught: same steps every turn"
        assert results["B2_redundant"] is False, "B2 caught: same steps (redundant search)"
        assert results["B3_no_pref"] is True, "B3 passes (different bug)"
        assert results["B4_drift"] is True, "B4 passes (different bug)"

    @pytest.mark.agent_test(layer="eval")
    async def test_eval_trajectory_preference_turn(self):
        """EVAL detector: does preference turn include save_preference?"""
        results = {}
        for name, agent_fn in AGENTS.items():
            llm, tools = _setup()
            conv = _make_conv(agent_fn, llm, tools)
            await conv.say("What options are available?")
            r2 = await conv.say("I prefer email notifications")

            score = trajectory_match(
                r2, expected_trajectory=["save_preference"], mode="unordered"
            )
            results[name] = score.passed

        assert results["correct"] is True
        # B1 doesn't detect preferences because it ignores context
        # (but it still might match "i prefer" literally)
        assert results["B2_redundant"] is True, "B2 passes (saves prefs)"
        assert results["B3_no_pref"] is False, "B3 caught: missing save_preference"
        assert results["B4_drift"] is True, "B4 passes (saves prefs)"

    @pytest.mark.agent_test(layer="eval")
    async def test_eval_context_reference(self):
        """EVAL detector: does the conversation reference earlier context?"""
        results = {}
        for name, agent_fn in AGENTS.items():
            llm, tools = _setup()
            conv = _make_conv(agent_fn, llm, tools)
            await conv.say("How do I reset my password?")
            await conv.say("Tell me more about that")
            await conv.say("What about two-factor authentication?")

            # Check if turn 2's output references turn 0
            # For this test, check if the LLM was given earlier context
            calls = llm.calls
            turn3_calls = [c for c in calls if "two-factor" in c.input_text.lower()
                           or "Previous conversation" in c.input_text]
            has_cross_ref = any(
                "password" in c.input_text.lower() or "reset" in c.input_text.lower()
                for c in turn3_calls
            )
            results[name] = has_cross_ref

        assert results["correct"] is True
        assert results["B1_amnesia"] is False, "B1 caught: no cross-turn references"
        assert results["B2_redundant"] is True, "B2 has context (different bug)"
        assert results["B3_no_pref"] is True, "B3 has context (different bug)"
        assert results["B4_drift"] is True, "B4 has context (bug is in output, not input)"


class TestMultiturnBugMatrix:
    """Synthesize the bug detection matrix for the paper."""

    @pytest.mark.agent_test(layer="mock")
    async def test_generate_detection_matrix(self):
        """Generate the full bug detection matrix across layers and detectors."""
        start = time.perf_counter()

        # Define detectors and which layer they belong to
        detectors = {
            "mock:history": self._check_history,
            "mock:no_redundant": self._check_no_redundant,
            "mock:pref_saved": self._check_pref_saved,
            "eval:step_efficiency": self._check_step_efficiency,
            "eval:trajectory": self._check_trajectory,
            "eval:context_ref": self._check_context_ref,
            "eval:output_relevance": self._check_output_relevance,
        }

        matrix: dict[str, dict[str, bool]] = {}
        for agent_name, agent_fn in AGENTS.items():
            if agent_name == "correct":
                continue
            matrix[agent_name] = {}
            for det_name, det_fn in detectors.items():
                caught = await det_fn(agent_fn)
                matrix[agent_name][det_name] = caught

        elapsed_ms = (time.perf_counter() - start) * 1000

        # Print the matrix for research logging
        print("\n=== Multi-Turn Bug Detection Matrix ===")
        print(f"{'Bug':<16} | {'mock:hist':>10} {'mock:redn':>10} {'mock:pref':>10} "
              f"{'eval:step':>10} {'eval:traj':>10} {'eval:ctx':>10} {'eval:rel':>10}")
        print("-" * 82)
        for bug, results in matrix.items():
            vals = [
                "Y" if results.get(d) else "N"
                for d in detectors
            ]
            print(f"{bug:<16} | {'  '.join(f'{v:>8}' for v in vals)}")

        # Summary statistics
        for bug, results in matrix.items():
            caught_count = sum(1 for v in results.values() if v)
            total = len(results)
            print(f"{bug}: caught by {caught_count}/{total} detectors")

        print(f"\nTotal execution time: {elapsed_ms:.1f}ms")
        print(f"Time per agent×detector: {elapsed_ms / (4 * 6):.2f}ms")

        # Verify each bug is caught by at least one detector
        for bug_name, results in matrix.items():
            assert any(results.values()), (
                f"Bug {bug_name} not caught by any detector — test gap!"
            )

    # --- Detector implementations ---

    async def _check_history(self, agent_fn) -> bool:
        llm, tools = _setup()
        conv = _make_conv(agent_fn, llm, tools)
        await conv.say("How do I reset my password?")
        turn1_count = llm.call_count
        await conv.say("Tell me more about that")
        turn2_calls = llm.calls[turn1_count:]
        has_history = any("Previous conversation" in c.input_text for c in turn2_calls)
        return not has_history  # True if bug detected (no history = bug)

    async def _check_no_redundant(self, agent_fn) -> bool:
        llm, tools = _setup()
        conv = _make_conv(agent_fn, llm, tools)
        await conv.say("How do I reset my password?")
        tools.reset_calls()
        await conv.say("Tell me more about that")
        return tools.was_called("search_faq")  # True if bug detected (redundant call)

    async def _check_pref_saved(self, agent_fn) -> bool:
        llm, tools = _setup()
        conv = _make_conv(agent_fn, llm, tools)
        await conv.say("What options are available?")
        await conv.say("I prefer email notifications")
        return not tools.was_called("save_preference")  # True if bug (not saved)

    async def _check_step_efficiency(self, agent_fn) -> bool:
        llm, tools = _setup()
        conv = _make_conv(agent_fn, llm, tools)
        r1 = await conv.say("How do I reset my password?")
        r2 = await conv.say("Tell me more about that")
        return len(r2.steps) >= len(r1.steps)  # True if bug (not more efficient)

    async def _check_trajectory(self, agent_fn) -> bool:
        llm, tools = _setup()
        conv = _make_conv(agent_fn, llm, tools)
        await conv.say("What options are available?")
        r2 = await conv.say("I prefer email notifications")
        score = trajectory_match(r2, expected_trajectory=["save_preference"], mode="unordered")
        return not score.passed  # True if bug (missing save_preference)

    async def _check_context_ref(self, agent_fn) -> bool:
        llm, tools = _setup()
        conv = _make_conv(agent_fn, llm, tools)
        await conv.say("How do I reset my password?")
        await conv.say("Tell me more about that")
        turn2_count = llm.call_count
        await conv.say("What about two-factor authentication?")
        turn3_calls = llm.calls[turn2_count:]
        has_cross_ref = any(
            "password" in c.input_text.lower() or "reset" in c.input_text.lower()
            for c in turn3_calls
        )
        return not has_cross_ref  # True if bug (no cross-reference)

    async def _check_output_relevance(self, agent_fn) -> bool:
        """Check if the final output is relevant to the conversation topic."""
        llm, tools = _setup()
        conv = _make_conv(agent_fn, llm, tools)
        await conv.say("How do I reset my password?")
        r2 = await conv.say("Tell me more about that")

        # The follow-up response should contain topic-relevant content
        output = str(r2.final_output).lower()
        # A generic response won't mention the topic
        topic_relevant = any(
            word in output
            for word in ["reset", "password", "settings", "account"]
        )
        return not topic_relevant  # True if bug (generic/irrelevant output)
