"""Credit assignment heuristics for multi-agent traces.

Given a failed multi-agent execution, these heuristics help identify
which agent or step most likely caused the failure. Provides multiple
strategies that can be combined for robust attribution.
"""

from __future__ import annotations

import warnings
from enum import Enum

from pydantic import BaseModel, Field

from checkagent.multiagent.trace import MultiAgentTrace


class BlameStrategy(str, Enum):
    """Strategy for assigning blame in a multi-agent failure."""

    FIRST_ERROR = "first_error"  # Blame the first agent that errored
    LAST_AGENT = "last_agent"  # Blame the last agent to execute
    MOST_STEPS = "most_steps"  # Blame the agent that took the most steps
    HIGHEST_COST = "highest_cost"  # Blame the agent that used the most tokens
    LEAF_ERRORS = "leaf_errors"  # Blame leaf agents (no children) that errored


class BlameResult(BaseModel):
    """Result of credit/blame assignment."""

    agent_id: str
    agent_name: str | None = None
    strategy: BlameStrategy
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str
    run_id: str | None = None


def assign_blame(
    trace: MultiAgentTrace,
    strategy: BlameStrategy = BlameStrategy.FIRST_ERROR,
) -> BlameResult | None:
    """Assign blame for a multi-agent failure using the given strategy.

    Returns None if no blame can be assigned (e.g., all agents succeeded).
    """
    if not trace.runs:
        return None

    # Warn if runs exist but none have agent_id set (F-070)
    if trace.runs and not any(r.agent_id for r in trace.runs):
        warnings.warn(
            "No runs have agent_id set — blame assignment requires agent_id "
            "on AgentRun instances. All strategies will return None.",
            UserWarning,
            stacklevel=2,
        )
        return None

    if strategy == BlameStrategy.FIRST_ERROR:
        return _blame_first_error(trace)
    elif strategy == BlameStrategy.LAST_AGENT:
        return _blame_last_agent(trace)
    elif strategy == BlameStrategy.MOST_STEPS:
        return _blame_most_steps(trace)
    elif strategy == BlameStrategy.HIGHEST_COST:
        return _blame_highest_cost(trace)
    elif strategy == BlameStrategy.LEAF_ERRORS:
        return _blame_leaf_errors(trace)
    return None


def assign_blame_ensemble(
    trace: MultiAgentTrace,
    strategies: list[BlameStrategy] | None = None,
) -> list[BlameResult]:
    """Run multiple blame strategies and return all results.

    When multiple strategies agree on the same agent, confidence is higher.
    """
    if strategies is None:
        strategies = list(BlameStrategy)

    results = []
    for strategy in strategies:
        result = assign_blame(trace, strategy)
        if result is not None:
            results.append(result)
    return results


def top_blamed_agent(
    trace: MultiAgentTrace,
    strategies: list[BlameStrategy] | None = None,
) -> BlameResult | None:
    """Run ensemble blame and return the agent blamed by the most strategies.

    Breaks ties by highest total confidence score.
    """
    results = assign_blame_ensemble(trace, strategies)
    if not results:
        return None

    # Count votes per agent_id
    votes: dict[str, list[BlameResult]] = {}
    for r in results:
        votes.setdefault(r.agent_id, []).append(r)

    # Sort by (vote count desc, total confidence desc)
    ranked = sorted(
        votes.items(),
        key=lambda item: (len(item[1]), sum(r.confidence for r in item[1])),
        reverse=True,
    )

    agent_id, agent_results = ranked[0]
    vote_count = len(agent_results)
    total_strategies = len(results)
    confidence = vote_count / total_strategies if total_strategies > 0 else 0.0

    return BlameResult(
        agent_id=agent_id,
        agent_name=agent_results[0].agent_name,
        strategy=BlameStrategy.FIRST_ERROR,  # Ensemble result
        confidence=confidence,
        reason=f"Blamed by {vote_count}/{total_strategies} strategies",
        run_id=agent_results[0].run_id,
    )


def _blame_first_error(trace: MultiAgentTrace) -> BlameResult | None:
    """Blame the first agent in execution order that has an error."""
    for run in trace.runs:
        if not run.succeeded and run.agent_id:
            return BlameResult(
                agent_id=run.agent_id,
                agent_name=run.agent_name,
                strategy=BlameStrategy.FIRST_ERROR,
                confidence=0.8,
                reason=f"First agent to error: {run.error}",
                run_id=run.run_id,
            )
    return None


def _blame_last_agent(trace: MultiAgentTrace) -> BlameResult | None:
    """Blame the last agent to execute (by position in runs list)."""
    failed = [r for r in trace.runs if not r.succeeded and r.agent_id]
    if not failed:
        return None
    last = failed[-1]
    return BlameResult(
        agent_id=last.agent_id,
        agent_name=last.agent_name,
        strategy=BlameStrategy.LAST_AGENT,
        confidence=0.6,
        reason=f"Last agent to fail: {last.error}",
        run_id=last.run_id,
    )


def _blame_most_steps(trace: MultiAgentTrace) -> BlameResult | None:
    """Blame the failed agent that took the most steps (likely stuck/looping)."""
    failed = [r for r in trace.runs if not r.succeeded and r.agent_id]
    if not failed:
        return None
    worst = max(failed, key=lambda r: len(r.steps))
    return BlameResult(
        agent_id=worst.agent_id,
        agent_name=worst.agent_name,
        strategy=BlameStrategy.MOST_STEPS,
        confidence=0.7,
        reason=f"Failed agent with most steps ({len(worst.steps)}): {worst.error}",
        run_id=worst.run_id,
    )


def _blame_highest_cost(trace: MultiAgentTrace) -> BlameResult | None:
    """Blame the failed agent that consumed the most tokens."""
    failed = [
        r for r in trace.runs
        if not r.succeeded and r.agent_id and r.total_tokens is not None
    ]
    if not failed:
        return None
    worst = max(failed, key=lambda r: r.total_tokens or 0)
    return BlameResult(
        agent_id=worst.agent_id,
        agent_name=worst.agent_name,
        strategy=BlameStrategy.HIGHEST_COST,
        confidence=0.6,
        reason=f"Failed agent with highest token usage ({worst.total_tokens}): {worst.error}",
        run_id=worst.run_id,
    )


def _blame_leaf_errors(trace: MultiAgentTrace) -> BlameResult | None:
    """Blame leaf agents (no children) that errored.

    In delegation patterns, leaf failures propagate up.
    The actual root cause is usually at the leaves.
    """
    # Find run_ids that are parents (via parent_run_id links)
    parent_run_ids = {r.parent_run_id for r in trace.runs if r.parent_run_id}
    # Find agent_ids that delegate to other agents (via handoff edges)
    delegating_agent_ids = {h.from_agent_id for h in trace.handoffs}

    # A run is a leaf if:
    # 1. Its run_id is not referenced as parent_run_id by any other run, AND
    # 2. Its agent_id is not a source of any handoff
    leaves = [
        r for r in trace.runs
        if not r.succeeded
        and r.agent_id
        and (r.run_id is None or r.run_id not in parent_run_ids)
        and r.agent_id not in delegating_agent_ids
    ]
    if not leaves:
        return None
    # If multiple leaves failed, pick the first
    leaf = leaves[0]
    return BlameResult(
        agent_id=leaf.agent_id,
        agent_name=leaf.agent_name,
        strategy=BlameStrategy.LEAF_ERRORS,
        confidence=0.85,
        reason=f"Leaf agent error (no children): {leaf.error}",
        run_id=leaf.run_id,
    )
