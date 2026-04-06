"""Tests for credit assignment heuristics."""

from checkagent.core.types import AgentInput, AgentRun, Step
from checkagent.multiagent.credit import (
    BlameResult,
    BlameStrategy,
    assign_blame,
    assign_blame_ensemble,
    top_blamed_agent,
)
from checkagent.multiagent.trace import MultiAgentTrace


def _run(
    agent_id: str,
    run_id: str | None = None,
    parent_run_id: str | None = None,
    error: str | None = None,
    steps: int = 1,
    prompt_tokens: int | None = 50,
    completion_tokens: int | None = 30,
    agent_name: str | None = None,
) -> AgentRun:
    return AgentRun(
        input=AgentInput(query="test"),
        steps=[Step(step_index=i) for i in range(steps)],
        final_output=None if error else "ok",
        error=error,
        duration_ms=100.0,
        total_prompt_tokens=prompt_tokens,
        total_completion_tokens=completion_tokens,
        run_id=run_id,
        agent_id=agent_id,
        agent_name=agent_name or agent_id,
        parent_run_id=parent_run_id,
    )


class TestFirstError:
    def test_blames_first_failing_agent(self):
        trace = MultiAgentTrace()
        trace.add_run(_run("a", error=None))
        trace.add_run(_run("b", error="timeout"))
        trace.add_run(_run("c", error="crash"))
        result = assign_blame(trace, BlameStrategy.FIRST_ERROR)
        assert result is not None
        assert result.agent_id == "b"
        assert result.strategy == BlameStrategy.FIRST_ERROR
        assert result.confidence == 0.8

    def test_returns_none_when_all_succeed(self):
        trace = MultiAgentTrace()
        trace.add_run(_run("a"))
        trace.add_run(_run("b"))
        assert assign_blame(trace, BlameStrategy.FIRST_ERROR) is None

    def test_returns_none_for_empty_trace(self):
        trace = MultiAgentTrace()
        assert assign_blame(trace, BlameStrategy.FIRST_ERROR) is None


class TestLastAgent:
    def test_blames_last_failing_agent(self):
        trace = MultiAgentTrace()
        trace.add_run(_run("a", error="err1"))
        trace.add_run(_run("b", error="err2"))
        result = assign_blame(trace, BlameStrategy.LAST_AGENT)
        assert result is not None
        assert result.agent_id == "b"

    def test_returns_none_when_all_succeed(self):
        trace = MultiAgentTrace()
        trace.add_run(_run("a"))
        assert assign_blame(trace, BlameStrategy.LAST_AGENT) is None


class TestMostSteps:
    def test_blames_agent_with_most_steps(self):
        trace = MultiAgentTrace()
        trace.add_run(_run("a", error="stuck", steps=10))
        trace.add_run(_run("b", error="crash", steps=2))
        result = assign_blame(trace, BlameStrategy.MOST_STEPS)
        assert result is not None
        assert result.agent_id == "a"
        assert "10" in result.reason

    def test_returns_none_when_all_succeed(self):
        trace = MultiAgentTrace()
        trace.add_run(_run("a", steps=5))
        assert assign_blame(trace, BlameStrategy.MOST_STEPS) is None


class TestHighestCost:
    def test_blames_highest_token_agent(self):
        trace = MultiAgentTrace()
        trace.add_run(_run("a", error="err", prompt_tokens=100, completion_tokens=50))
        trace.add_run(_run("b", error="err", prompt_tokens=500, completion_tokens=200))
        result = assign_blame(trace, BlameStrategy.HIGHEST_COST)
        assert result is not None
        assert result.agent_id == "b"
        assert "700" in result.reason

    def test_skips_agents_without_tokens(self):
        trace = MultiAgentTrace()
        trace.add_run(_run("a", error="err", prompt_tokens=None, completion_tokens=None))
        trace.add_run(_run("b", error="err", prompt_tokens=100, completion_tokens=50))
        result = assign_blame(trace, BlameStrategy.HIGHEST_COST)
        assert result is not None
        assert result.agent_id == "b"

    def test_returns_none_when_no_token_data(self):
        trace = MultiAgentTrace()
        trace.add_run(_run("a", error="err", prompt_tokens=None, completion_tokens=None))
        assert assign_blame(trace, BlameStrategy.HIGHEST_COST) is None


class TestLeafErrors:
    def test_blames_leaf_agent(self):
        trace = MultiAgentTrace()
        trace.add_run(_run("orch", run_id="r1", error="child failed"))
        trace.add_run(_run("worker", run_id="r2", parent_run_id="r1", error="timeout"))
        result = assign_blame(trace, BlameStrategy.LEAF_ERRORS)
        assert result is not None
        assert result.agent_id == "worker"
        assert result.confidence == 0.85

    def test_parent_not_blamed_when_leaf_fails(self):
        trace = MultiAgentTrace()
        # r1 is a parent of r2, so r1 is not a leaf
        trace.add_run(_run("orch", run_id="r1", error="propagated"))
        trace.add_run(_run("worker", run_id="r2", parent_run_id="r1", error="root cause"))
        result = assign_blame(trace, BlameStrategy.LEAF_ERRORS)
        assert result.agent_id == "worker"

    def test_returns_none_when_no_leaf_errors(self):
        trace = MultiAgentTrace()
        trace.add_run(_run("a", run_id="r1"))
        trace.add_run(_run("b", run_id="r2", parent_run_id="r1"))
        assert assign_blame(trace, BlameStrategy.LEAF_ERRORS) is None

    def test_handoff_topology_excludes_delegators(self):
        """F-069: LEAF_ERRORS should use handoff edges, not just parent_run_id."""
        from checkagent.multiagent.trace import Handoff

        trace = MultiAgentTrace()
        # A delegates to B via handoff (no parent_run_id set)
        trace.add_run(_run("A", error="propagated"))
        trace.add_run(_run("B", error="root cause"))
        trace.add_handoff(Handoff(from_agent_id="A", to_agent_id="B"))

        result = assign_blame(trace, BlameStrategy.LEAF_ERRORS)
        assert result is not None
        # B is the leaf (no outgoing handoffs), not A
        assert result.agent_id == "B"

    def test_handoff_chain_blames_final_leaf(self):
        """In A→B→C chain via handoffs, only C is a leaf."""
        from checkagent.multiagent.trace import Handoff

        trace = MultiAgentTrace()
        trace.add_run(_run("A", error="propagated"))
        trace.add_run(_run("B", error="propagated"))
        trace.add_run(_run("C", error="root cause"))
        trace.add_handoff(Handoff(from_agent_id="A", to_agent_id="B"))
        trace.add_handoff(Handoff(from_agent_id="B", to_agent_id="C"))

        result = assign_blame(trace, BlameStrategy.LEAF_ERRORS)
        assert result is not None
        assert result.agent_id == "C"

    def test_mixed_parent_and_handoff_topology(self):
        """Both parent_run_id and handoff edges should exclude non-leaves."""
        from checkagent.multiagent.trace import Handoff

        trace = MultiAgentTrace()
        # A is parent of B via parent_run_id
        trace.add_run(_run("A", run_id="r1", error="propagated"))
        trace.add_run(_run("B", run_id="r2", parent_run_id="r1", error="delegated"))
        # B delegates to C via handoff
        trace.add_run(_run("C", error="root cause"))
        trace.add_handoff(Handoff(from_agent_id="B", to_agent_id="C"))

        result = assign_blame(trace, BlameStrategy.LEAF_ERRORS)
        assert result is not None
        # Only C is a leaf (A is parent via run_id, B delegates via handoff)
        assert result.agent_id == "C"


class TestMissingAgentId:
    """F-070: Warn when agent_id is missing on all runs."""

    def test_warns_when_no_agent_ids(self):
        import warnings as w

        trace = MultiAgentTrace()
        trace.add_run(AgentRun(
            input=AgentInput(query="test"),
            steps=[Step(step_index=0)],
            error="fail",
        ))
        with w.catch_warnings(record=True) as caught:
            w.simplefilter("always")
            result = assign_blame(trace, BlameStrategy.FIRST_ERROR)
        assert result is None
        assert len(caught) == 1
        assert "agent_id" in str(caught[0].message)

    def test_no_warning_when_agent_ids_present(self):
        import warnings as w

        trace = MultiAgentTrace()
        trace.add_run(_run("a", error="err"))
        with w.catch_warnings(record=True) as caught:
            w.simplefilter("always")
            result = assign_blame(trace, BlameStrategy.FIRST_ERROR)
        assert result is not None
        assert len(caught) == 0


class TestEnsemble:
    def test_returns_all_strategy_results(self):
        trace = MultiAgentTrace()
        trace.add_run(_run("a", run_id="r1", error="err", steps=5,
                           prompt_tokens=100, completion_tokens=50))
        results = assign_blame_ensemble(trace)
        # All strategies should blame "a" since it's the only failing agent
        assert len(results) >= 3  # At least first_error, last_agent, most_steps
        assert all(r.agent_id == "a" for r in results)

    def test_with_specific_strategies(self):
        trace = MultiAgentTrace()
        trace.add_run(_run("a", error="err"))
        results = assign_blame_ensemble(
            trace, strategies=[BlameStrategy.FIRST_ERROR, BlameStrategy.LAST_AGENT]
        )
        assert len(results) == 2

    def test_empty_trace(self):
        trace = MultiAgentTrace()
        results = assign_blame_ensemble(trace)
        assert results == []


class TestTopBlamedAgent:
    def test_unanimous_blame(self):
        trace = MultiAgentTrace()
        trace.add_run(_run("a", run_id="r1", error="err", steps=5,
                           prompt_tokens=100, completion_tokens=50))
        result = top_blamed_agent(trace)
        assert result is not None
        assert result.agent_id == "a"
        assert result.confidence == 1.0  # All strategies agree

    def test_split_blame(self):
        trace = MultiAgentTrace()
        # "a" errors first with fewer steps, "b" errors second with more steps
        trace.add_run(_run("a", run_id="r1", error="err1", steps=1,
                           prompt_tokens=10, completion_tokens=5))
        trace.add_run(_run("b", run_id="r2", error="err2", steps=10,
                           prompt_tokens=500, completion_tokens=200))
        result = top_blamed_agent(trace)
        assert result is not None
        # "b" should win — most_steps, highest_cost, last_agent all point to "b"
        assert result.agent_id == "b"

    def test_returns_none_on_success(self):
        trace = MultiAgentTrace()
        trace.add_run(_run("a"))
        assert top_blamed_agent(trace) is None

    def test_includes_agent_name(self):
        trace = MultiAgentTrace()
        trace.add_run(_run("a", agent_name="Research Agent", error="err"))
        result = top_blamed_agent(trace)
        assert result.agent_name == "Research Agent"

    def test_reason_shows_vote_count(self):
        trace = MultiAgentTrace()
        trace.add_run(_run("a", error="err", prompt_tokens=100, completion_tokens=50))
        result = top_blamed_agent(trace)
        assert "strategies" in result.reason


class TestBlameResult:
    def test_fields(self):
        r = BlameResult(
            agent_id="a",
            agent_name="Agent A",
            strategy=BlameStrategy.FIRST_ERROR,
            confidence=0.8,
            reason="test",
            run_id="r1",
        )
        assert r.agent_id == "a"
        assert r.confidence == 0.8

    def test_confidence_bounds(self):
        import pytest
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            BlameResult(
                agent_id="a", strategy=BlameStrategy.FIRST_ERROR,
                confidence=1.5, reason="test",
            )
