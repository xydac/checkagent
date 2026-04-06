"""Multi-agent trace container and handoff analysis.

A MultiAgentTrace groups related AgentRun objects from a multi-agent system,
tracks handoffs between agents, and provides analysis utilities for testing
agent coordination patterns.
"""

from __future__ import annotations

import warnings

from pydantic import BaseModel, Field

from checkagent.core.types import AgentRun, HandoffType


class Handoff(BaseModel):
    """A recorded handoff between two agents."""

    from_agent_id: str
    to_agent_id: str
    handoff_type: HandoffType = HandoffType.DELEGATION
    from_run_id: str | None = None
    to_run_id: str | None = None
    input_summary: str | None = None
    latency_ms: float | None = None


class MultiAgentTrace(BaseModel):
    """Container for a multi-agent execution trace.

    Holds all AgentRun objects from a coordinated multi-agent execution
    and provides methods to query agent relationships, handoffs, and
    aggregate metrics.
    """

    runs: list[AgentRun] = Field(default_factory=list)
    handoffs: list[Handoff] = Field(default_factory=list)
    trace_id: str | None = None

    def add_run(self, run: AgentRun) -> MultiAgentTrace:
        """Add an agent run to the trace. Returns self for chaining."""
        self.runs.append(run)
        return self

    def add_handoff(self, handoff: Handoff) -> MultiAgentTrace:
        """Record a handoff between agents. Returns self for chaining."""
        self.handoffs.append(handoff)
        return self

    @property
    def agent_ids(self) -> list[str]:
        """Unique agent IDs in execution order."""
        seen: set[str] = set()
        result: list[str] = []
        for run in self.runs:
            if run.agent_id and run.agent_id not in seen:
                seen.add(run.agent_id)
                result.append(run.agent_id)
        return result

    @property
    def root_runs(self) -> list[AgentRun]:
        """Runs with no parent (top-level orchestrators)."""
        return [r for r in self.runs if r.parent_run_id is None]

    def get_runs_by_agent(self, agent_id: str) -> list[AgentRun]:
        """Get all runs for a specific agent."""
        return [r for r in self.runs if r.agent_id == agent_id]

    def get_children(self, run_id: str) -> list[AgentRun]:
        """Get child runs spawned by a given run.

        Note: this method matches against ``parent_run_id``, so the argument
        must be a *run* ID (``AgentRun.run_id``), not an *agent* ID.  If you
        need to look up runs by agent, use :meth:`get_runs_by_agent`.
        """
        children = [r for r in self.runs if r.parent_run_id == run_id]
        if not children:
            # Check if the caller accidentally passed an agent_id
            run_ids = {r.run_id for r in self.runs if r.run_id}
            agent_ids = {r.agent_id for r in self.runs if r.agent_id}
            if run_id in agent_ids and run_id not in run_ids:
                warnings.warn(
                    f"get_children() received '{run_id}' which matches an "
                    f"agent_id, not a run_id. This method filters by "
                    f"parent_run_id. Did you mean "
                    f"get_runs_by_agent('{run_id}')? "
                    f"Available run_ids: {sorted(run_ids)}",
                    UserWarning,
                    stacklevel=2,
                )
        return children

    def get_handoffs_from(self, agent_id: str) -> list[Handoff]:
        """Get all handoffs originating from an agent."""
        return [h for h in self.handoffs if h.from_agent_id == agent_id]

    def get_handoffs_to(self, agent_id: str) -> list[Handoff]:
        """Get all handoffs targeting an agent."""
        return [h for h in self.handoffs if h.to_agent_id == agent_id]

    @property
    def total_duration_ms(self) -> float | None:
        """Sum of all run durations."""
        durations = [r.duration_ms for r in self.runs if r.duration_ms is not None]
        return sum(durations) if durations else None

    @property
    def total_tokens(self) -> int | None:
        """Sum of tokens across all runs."""
        totals = [r.total_tokens for r in self.runs if r.total_tokens is not None]
        return sum(totals) if totals else None

    @property
    def total_steps(self) -> int:
        """Total steps across all runs."""
        return sum(len(r.steps) for r in self.runs)

    @property
    def failed_runs(self) -> list[AgentRun]:
        """Runs that ended with an error."""
        return [r for r in self.runs if not r.succeeded]

    @property
    def succeeded(self) -> bool:
        """True if all runs completed without error."""
        return all(r.succeeded for r in self.runs)

    def handoff_chain(self) -> list[str]:
        """Return ordered list of agent IDs following the handoff chain.

        Follows handoffs from the first handoff's source through each
        subsequent target. Useful for asserting on delegation order.
        """
        if not self.handoffs:
            return []
        chain = [self.handoffs[0].from_agent_id]
        for h in self.handoffs:
            chain.append(h.to_agent_id)
        return chain

    def detect_handoffs(self) -> list[Handoff]:
        """Auto-detect handoffs from parent-child relationships.

        Scans runs for parent_run_id links and creates Handoff records.
        Returns detected handoffs **without** mutating ``self.handoffs``.
        To persist detected handoffs, use :meth:`apply_detected_handoffs`.
        """
        # Index runs by run_id for fast lookup
        run_map: dict[str, AgentRun] = {}
        for run in self.runs:
            if run.run_id:
                run_map[run.run_id] = run

        detected: list[Handoff] = []
        for run in self.runs:
            if run.parent_run_id and run.parent_run_id in run_map:
                parent = run_map[run.parent_run_id]
                if parent.agent_id and run.agent_id:
                    handoff = Handoff(
                        from_agent_id=parent.agent_id,
                        to_agent_id=run.agent_id,
                        handoff_type=HandoffType.DELEGATION,
                        from_run_id=parent.run_id,
                        to_run_id=run.run_id,
                        input_summary=run.input.query[:100] if run.input.query else None,
                    )
                    detected.append(handoff)

        return detected

    def apply_detected_handoffs(self) -> list[Handoff]:
        """Detect handoffs from parent-child links and append them to ``self.handoffs``.

        Unlike :meth:`detect_handoffs`, this method **does** mutate
        ``self.handoffs``. Safe to call multiple times — deduplicates by
        checking existing (from_agent_id, to_agent_id) pairs.

        Returns the newly added handoffs.
        """
        existing = {(h.from_agent_id, h.to_agent_id) for h in self.handoffs}
        detected = self.detect_handoffs()
        added: list[Handoff] = []
        for h in detected:
            key = (h.from_agent_id, h.to_agent_id)
            if key not in existing:
                self.handoffs.append(h)
                existing.add(key)
                added.append(h)
        return added
