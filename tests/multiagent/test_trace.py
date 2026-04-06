"""Tests for multi-agent trace container and handoff analysis."""

from checkagent.core.types import AgentInput, AgentRun, HandoffType, Step
from checkagent.multiagent.trace import Handoff, MultiAgentTrace


def _make_run(
    agent_id: str = "agent-1",
    agent_name: str = "Agent 1",
    run_id: str | None = None,
    parent_run_id: str | None = None,
    query: str = "hello",
    final_output: str = "done",
    error: str | None = None,
    duration_ms: float | None = 100.0,
    prompt_tokens: int | None = 50,
    completion_tokens: int | None = 30,
    steps: list[Step] | None = None,
) -> AgentRun:
    return AgentRun(
        input=AgentInput(query=query),
        steps=steps or [Step(step_index=0, output_text="step")],
        final_output=final_output,
        error=error,
        duration_ms=duration_ms,
        total_prompt_tokens=prompt_tokens,
        total_completion_tokens=completion_tokens,
        run_id=run_id,
        agent_id=agent_id,
        agent_name=agent_name,
        parent_run_id=parent_run_id,
    )


class TestMultiAgentTraceBasics:
    def test_empty_trace(self):
        trace = MultiAgentTrace()
        assert trace.runs == []
        assert trace.handoffs == []
        assert trace.agent_ids == []
        assert trace.root_runs == []
        assert trace.total_duration_ms is None
        assert trace.total_tokens is None
        assert trace.total_steps == 0
        assert trace.succeeded is True

    def test_add_run(self):
        trace = MultiAgentTrace()
        run = _make_run()
        trace.add_run(run)
        assert len(trace.runs) == 1
        assert trace.runs[0].agent_id == "agent-1"

    def test_agent_ids_preserves_order(self):
        trace = MultiAgentTrace()
        trace.add_run(_make_run(agent_id="b"))
        trace.add_run(_make_run(agent_id="a"))
        trace.add_run(_make_run(agent_id="b"))  # duplicate
        assert trace.agent_ids == ["b", "a"]

    def test_root_runs(self):
        trace = MultiAgentTrace()
        trace.add_run(_make_run(run_id="r1", parent_run_id=None))
        trace.add_run(_make_run(run_id="r2", parent_run_id="r1"))
        assert len(trace.root_runs) == 1
        assert trace.root_runs[0].run_id == "r1"

    def test_get_runs_by_agent(self):
        trace = MultiAgentTrace()
        trace.add_run(_make_run(agent_id="a", run_id="r1"))
        trace.add_run(_make_run(agent_id="b", run_id="r2"))
        trace.add_run(_make_run(agent_id="a", run_id="r3"))
        runs = trace.get_runs_by_agent("a")
        assert len(runs) == 2
        assert {r.run_id for r in runs} == {"r1", "r3"}

    def test_get_children(self):
        trace = MultiAgentTrace()
        trace.add_run(_make_run(run_id="parent", agent_id="orchestrator"))
        trace.add_run(_make_run(run_id="child1", parent_run_id="parent", agent_id="worker1"))
        trace.add_run(_make_run(run_id="child2", parent_run_id="parent", agent_id="worker2"))
        trace.add_run(_make_run(run_id="unrelated", agent_id="other"))
        children = trace.get_children("parent")
        assert len(children) == 2
        assert {c.run_id for c in children} == {"child1", "child2"}


class TestMultiAgentMetrics:
    def test_total_duration(self):
        trace = MultiAgentTrace()
        trace.add_run(_make_run(duration_ms=100.0))
        trace.add_run(_make_run(duration_ms=200.0))
        assert trace.total_duration_ms == 300.0

    def test_total_duration_with_none(self):
        trace = MultiAgentTrace()
        trace.add_run(_make_run(duration_ms=100.0))
        trace.add_run(_make_run(duration_ms=None))
        assert trace.total_duration_ms == 100.0

    def test_total_tokens(self):
        trace = MultiAgentTrace()
        trace.add_run(_make_run(prompt_tokens=50, completion_tokens=30))
        trace.add_run(_make_run(prompt_tokens=100, completion_tokens=60))
        assert trace.total_tokens == 240

    def test_total_tokens_none(self):
        trace = MultiAgentTrace()
        trace.add_run(_make_run(prompt_tokens=None, completion_tokens=None))
        assert trace.total_tokens is None

    def test_total_steps(self):
        trace = MultiAgentTrace()
        trace.add_run(_make_run(steps=[Step(step_index=0), Step(step_index=1)]))
        trace.add_run(_make_run(steps=[Step(step_index=0)]))
        assert trace.total_steps == 3

    def test_failed_runs(self):
        trace = MultiAgentTrace()
        trace.add_run(_make_run(error=None))
        trace.add_run(_make_run(error="timeout"))
        assert len(trace.failed_runs) == 1
        assert not trace.succeeded

    def test_all_succeeded(self):
        trace = MultiAgentTrace()
        trace.add_run(_make_run(error=None))
        trace.add_run(_make_run(error=None))
        assert trace.succeeded


class TestHandoffs:
    def test_add_handoff(self):
        trace = MultiAgentTrace()
        h = Handoff(from_agent_id="a", to_agent_id="b")
        trace.add_handoff(h)
        assert len(trace.handoffs) == 1

    def test_get_handoffs_from(self):
        trace = MultiAgentTrace()
        trace.add_handoff(Handoff(from_agent_id="a", to_agent_id="b"))
        trace.add_handoff(Handoff(from_agent_id="a", to_agent_id="c"))
        trace.add_handoff(Handoff(from_agent_id="b", to_agent_id="c"))
        assert len(trace.get_handoffs_from("a")) == 2

    def test_get_handoffs_to(self):
        trace = MultiAgentTrace()
        trace.add_handoff(Handoff(from_agent_id="a", to_agent_id="c"))
        trace.add_handoff(Handoff(from_agent_id="b", to_agent_id="c"))
        assert len(trace.get_handoffs_to("c")) == 2

    def test_handoff_type_default(self):
        h = Handoff(from_agent_id="a", to_agent_id="b")
        assert h.handoff_type == HandoffType.DELEGATION

    def test_handoff_type_relay(self):
        h = Handoff(from_agent_id="a", to_agent_id="b", handoff_type=HandoffType.RELAY)
        assert h.handoff_type == HandoffType.RELAY

    def test_handoff_chain(self):
        trace = MultiAgentTrace()
        trace.add_handoff(Handoff(from_agent_id="orchestrator", to_agent_id="researcher"))
        trace.add_handoff(Handoff(from_agent_id="researcher", to_agent_id="writer"))
        chain = trace.handoff_chain()
        assert chain == ["orchestrator", "researcher", "writer"]

    def test_handoff_chain_empty(self):
        trace = MultiAgentTrace()
        assert trace.handoff_chain() == []

    def test_handoff_with_latency(self):
        h = Handoff(from_agent_id="a", to_agent_id="b", latency_ms=15.5)
        assert h.latency_ms == 15.5


class TestDetectHandoffs:
    def test_detect_from_parent_child(self):
        trace = MultiAgentTrace()
        trace.add_run(_make_run(run_id="r1", agent_id="orchestrator"))
        trace.add_run(_make_run(run_id="r2", agent_id="worker", parent_run_id="r1"))
        detected = trace.detect_handoffs()
        assert len(detected) == 1
        assert detected[0].from_agent_id == "orchestrator"
        assert detected[0].to_agent_id == "worker"
        assert detected[0].from_run_id == "r1"
        assert detected[0].to_run_id == "r2"
        assert detected[0].handoff_type == HandoffType.DELEGATION
        # detect_handoffs() is read-only — does NOT mutate trace.handoffs
        assert len(trace.handoffs) == 0

    def test_detect_multiple_children(self):
        trace = MultiAgentTrace()
        trace.add_run(_make_run(run_id="r1", agent_id="orch"))
        trace.add_run(_make_run(run_id="r2", agent_id="w1", parent_run_id="r1"))
        trace.add_run(_make_run(run_id="r3", agent_id="w2", parent_run_id="r1"))
        detected = trace.detect_handoffs()
        assert len(detected) == 2
        assert {d.to_agent_id for d in detected} == {"w1", "w2"}

    def test_detect_no_handoffs_without_parent(self):
        trace = MultiAgentTrace()
        trace.add_run(_make_run(run_id="r1", agent_id="a"))
        trace.add_run(_make_run(run_id="r2", agent_id="b"))
        detected = trace.detect_handoffs()
        assert len(detected) == 0

    def test_detect_skips_missing_agent_id(self):
        trace = MultiAgentTrace()
        trace.add_run(_make_run(run_id="r1", agent_id="orch"))
        # Child with no agent_id set
        run = _make_run(run_id="r2", parent_run_id="r1")
        run.agent_id = None
        trace.add_run(run)
        detected = trace.detect_handoffs()
        assert len(detected) == 0

    def test_detect_skips_orphan_parent_ref(self):
        trace = MultiAgentTrace()
        # Child references non-existent parent
        trace.add_run(_make_run(run_id="r2", agent_id="w", parent_run_id="r1"))
        detected = trace.detect_handoffs()
        assert len(detected) == 0

    def test_detect_input_summary_truncated(self):
        trace = MultiAgentTrace()
        trace.add_run(_make_run(run_id="r1", agent_id="orch"))
        long_query = "x" * 200
        trace.add_run(_make_run(run_id="r2", agent_id="w", parent_run_id="r1", query=long_query))
        detected = trace.detect_handoffs()
        assert len(detected[0].input_summary) == 100

    def test_detect_deep_chain(self):
        trace = MultiAgentTrace()
        trace.add_run(_make_run(run_id="r1", agent_id="a"))
        trace.add_run(_make_run(run_id="r2", agent_id="b", parent_run_id="r1"))
        trace.add_run(_make_run(run_id="r3", agent_id="c", parent_run_id="r2"))
        detected = trace.detect_handoffs()
        assert len(detected) == 2
        assert detected[0].from_agent_id == "b" or detected[0].from_agent_id == "a"

    def test_detect_handoffs_is_read_only(self):
        """F-076: detect_handoffs() must not mutate trace.handoffs."""
        trace = MultiAgentTrace()
        trace.add_run(_make_run(run_id="r1", agent_id="orch"))
        trace.add_run(_make_run(run_id="r2", agent_id="w", parent_run_id="r1"))

        detected1 = trace.detect_handoffs()
        assert len(detected1) == 1
        assert len(trace.handoffs) == 0  # not mutated

        # Calling twice does NOT duplicate
        detected2 = trace.detect_handoffs()
        assert len(detected2) == 1
        assert len(trace.handoffs) == 0

    def test_apply_detected_handoffs_persists(self):
        """apply_detected_handoffs() should mutate trace.handoffs."""
        trace = MultiAgentTrace()
        trace.add_run(_make_run(run_id="r1", agent_id="orch"))
        trace.add_run(_make_run(run_id="r2", agent_id="w", parent_run_id="r1"))

        added = trace.apply_detected_handoffs()
        assert len(added) == 1
        assert len(trace.handoffs) == 1
        assert trace.handoffs[0].from_agent_id == "orch"

    def test_apply_detected_handoffs_deduplicates(self):
        """Calling apply_detected_handoffs() twice should not duplicate."""
        trace = MultiAgentTrace()
        trace.add_run(_make_run(run_id="r1", agent_id="orch"))
        trace.add_run(_make_run(run_id="r2", agent_id="w", parent_run_id="r1"))

        trace.apply_detected_handoffs()
        added2 = trace.apply_detected_handoffs()
        assert len(added2) == 0
        assert len(trace.handoffs) == 1


class TestBuilderChaining:
    """F-074: add_run() and add_handoff() return self for chaining."""

    def test_add_run_returns_self(self):
        trace = MultiAgentTrace()
        run = _make_run(run_id="r1", agent_id="a")
        result = trace.add_run(run)
        assert result is trace

    def test_add_handoff_returns_self(self):
        trace = MultiAgentTrace()
        h = Handoff(from_agent_id="a", to_agent_id="b")
        result = trace.add_handoff(h)
        assert result is trace

    def test_chaining_add_run(self):
        trace = MultiAgentTrace()
        trace.add_run(_make_run(run_id="r1", agent_id="a")).add_run(
            _make_run(run_id="r2", agent_id="b")
        )
        assert len(trace.runs) == 2

    def test_chaining_mixed(self):
        trace = MultiAgentTrace()
        trace.add_run(_make_run(run_id="r1", agent_id="a")).add_run(
            _make_run(run_id="r2", agent_id="b")
        ).add_handoff(Handoff(from_agent_id="a", to_agent_id="b"))
        assert len(trace.runs) == 2
        assert len(trace.handoffs) == 1


class TestAgentRunMultiAgentFields:
    """Test the new multi-agent fields on AgentRun."""

    def test_default_none(self):
        run = AgentRun(input=AgentInput(query="test"))
        assert run.run_id is None
        assert run.agent_id is None
        assert run.agent_name is None
        assert run.parent_run_id is None

    def test_set_fields(self):
        run = AgentRun(
            input=AgentInput(query="test"),
            run_id="r1",
            agent_id="agent-a",
            agent_name="Research Agent",
            parent_run_id="r0",
        )
        assert run.run_id == "r1"
        assert run.agent_id == "agent-a"
        assert run.agent_name == "Research Agent"
        assert run.parent_run_id == "r0"

    def test_serialization_roundtrip(self):
        run = AgentRun(
            input=AgentInput(query="test"),
            run_id="r1",
            agent_id="agent-a",
            agent_name="Test Agent",
            parent_run_id="r0",
        )
        data = run.model_dump()
        restored = AgentRun.model_validate(data)
        assert restored.run_id == "r1"
        assert restored.agent_id == "agent-a"
        assert restored.parent_run_id == "r0"

    def test_backward_compatible(self):
        """Existing code that doesn't set multi-agent fields still works."""
        data = {
            "input": {"query": "hello"},
            "steps": [],
            "final_output": "world",
        }
        run = AgentRun.model_validate(data)
        assert run.run_id is None
        assert run.agent_id is None
        assert run.succeeded


class TestHandoffType:
    def test_values(self):
        assert HandoffType.DELEGATION == "delegation"
        assert HandoffType.RELAY == "relay"
        assert HandoffType.BROADCAST == "broadcast"

    def test_from_string(self):
        assert HandoffType("delegation") == HandoffType.DELEGATION


class TestMultiAgentTraceId:
    def test_trace_id(self):
        trace = MultiAgentTrace(trace_id="trace-123")
        assert trace.trace_id == "trace-123"

    def test_trace_id_default_none(self):
        trace = MultiAgentTrace()
        assert trace.trace_id is None


class TestGetChildrenWarning:
    """F-073: get_children() should warn when passed an agent_id instead of run_id."""

    def test_warns_when_agent_id_passed(self):
        """Passing an agent_id to get_children() emits a helpful warning."""
        import warnings

        trace = MultiAgentTrace()
        trace.add_run(
            _make_run(agent_id="orchestrator", run_id="run-001")
        ).add_run(
            _make_run(
                agent_id="worker", run_id="run-002", parent_run_id="run-001"
            )
        )

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = trace.get_children("orchestrator")

        assert result == []
        assert len(w) == 1
        assert "agent_id" in str(w[0].message)
        assert "get_runs_by_agent" in str(w[0].message)
        assert "orchestrator" in str(w[0].message)

    def test_no_warning_when_run_id_passed(self):
        """Passing a valid run_id does not emit a warning."""
        import warnings

        trace = MultiAgentTrace()
        trace.add_run(
            _make_run(agent_id="orchestrator", run_id="run-001")
        ).add_run(
            _make_run(
                agent_id="worker", run_id="run-002", parent_run_id="run-001"
            )
        )

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = trace.get_children("run-001")

        assert len(result) == 1
        assert result[0].agent_id == "worker"
        assert len(w) == 0

    def test_no_warning_for_unknown_id(self):
        """Passing an ID that matches neither agent_id nor run_id does not warn."""
        import warnings

        trace = MultiAgentTrace()
        trace.add_run(_make_run(agent_id="orchestrator", run_id="run-001"))

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = trace.get_children("nonexistent")

        assert result == []
        assert len(w) == 0

    def test_no_warning_when_id_is_both_agent_and_run_id(self):
        """If the ID matches both agent_id and run_id, don't warn (ambiguous but valid)."""
        import warnings

        trace = MultiAgentTrace()
        # Edge case: agent_id == run_id
        trace.add_run(_make_run(agent_id="shared-id", run_id="shared-id"))

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = trace.get_children("shared-id")

        assert result == []
        assert len(w) == 0

    def test_warning_includes_available_run_ids(self):
        """The warning message lists available run_ids for discoverability."""
        import warnings

        trace = MultiAgentTrace()
        trace.add_run(
            _make_run(agent_id="orch", run_id="run-001")
        ).add_run(
            _make_run(agent_id="worker", run_id="run-002", parent_run_id="run-001")
        )

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            trace.get_children("orch")

        assert "run-001" in str(w[0].message)
        assert "run-002" in str(w[0].message)


class TestMultiagentNamespace:
    def test_handoff_type_importable_from_multiagent(self):
        """F-071: HandoffType should be accessible from checkagent.multiagent."""
        from checkagent.multiagent import HandoffType

        assert HandoffType.DELEGATION == "delegation"
        assert HandoffType.RELAY == "relay"
        assert HandoffType.BROADCAST == "broadcast"
