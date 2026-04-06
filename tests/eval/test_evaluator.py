"""Tests for custom evaluator plugin interface."""

import pytest

from checkagent.core.types import AgentInput, AgentRun, Score, Step, ToolCall
from checkagent.datasets.schema import EvalCase
from checkagent.eval.evaluator import Evaluator, EvaluatorRegistry


class PoliteEvaluator(Evaluator):
    """Test evaluator that checks if output is polite."""

    name = "politeness"

    def score(self, run: AgentRun, expected: EvalCase) -> Score:
        output = str(run.final_output or "").lower()
        polite = any(w in output for w in ["please", "thank", "sorry"])
        return Score(name=self.name, value=1.0 if polite else 0.0, threshold=0.5)


class ToolCountEvaluator(Evaluator):
    """Test evaluator that checks tool call count against max_steps."""

    name = "tool_count"

    def score(self, run: AgentRun, expected: EvalCase) -> Score:
        n_tools = len(run.tool_calls)
        max_steps = expected.max_steps or 10
        value = min(1.0, max_steps / max(n_tools, 1))
        return Score(name=self.name, value=value, threshold=0.8)


def _make_run(output: str = "Thank you!", tool_names: list[str] | None = None) -> AgentRun:
    steps = []
    if tool_names:
        steps.append(
            Step(
                step_index=0,
                tool_calls=[ToolCall(name=t, arguments={}) for t in tool_names],
            )
        )
    return AgentRun(
        input=AgentInput(query="test"),
        steps=steps,
        final_output=output,
    )


def _make_case(**kwargs) -> EvalCase:
    defaults = {"id": "test-001", "input": "test query"}
    defaults.update(kwargs)
    return EvalCase(**defaults)


class TestEvaluator:
    """Tests for the Evaluator base class."""

    def test_cannot_instantiate_abstract(self):
        with pytest.raises(TypeError):
            Evaluator()  # type: ignore[abstract]

    def test_concrete_evaluator(self):
        ev = PoliteEvaluator()
        assert ev.name == "politeness"

    def test_score_polite(self):
        ev = PoliteEvaluator()
        run = _make_run("Thank you for your help!")
        case = _make_case()
        score = ev.score(run, case)
        assert score.value == 1.0
        assert score.passed is True

    def test_score_not_polite(self):
        ev = PoliteEvaluator()
        run = _make_run("Here is the answer.")
        case = _make_case()
        score = ev.score(run, case)
        assert score.value == 0.0
        assert score.passed is False

    def test_repr(self):
        ev = PoliteEvaluator()
        assert "PoliteEvaluator" in repr(ev)
        assert "politeness" in repr(ev)


class TestEvaluatorRegistry:
    """Tests for EvaluatorRegistry."""

    def test_register_and_lookup(self):
        reg = EvaluatorRegistry()
        reg.register(PoliteEvaluator())
        assert "politeness" in reg
        assert len(reg) == 1

    def test_unregister(self):
        reg = EvaluatorRegistry()
        reg.register(PoliteEvaluator())
        reg.unregister("politeness")
        assert "politeness" not in reg
        assert len(reg) == 0

    def test_unregister_missing_is_noop(self):
        reg = EvaluatorRegistry()
        reg.unregister("nonexistent")  # no error

    def test_evaluators_property(self):
        reg = EvaluatorRegistry()
        reg.register(PoliteEvaluator())
        reg.register(ToolCountEvaluator())
        evs = reg.evaluators
        assert len(evs) == 2
        assert "politeness" in evs
        assert "tool_count" in evs

    def test_score_all(self):
        reg = EvaluatorRegistry()
        reg.register(PoliteEvaluator())
        reg.register(ToolCountEvaluator())

        run = _make_run("Thank you!", tool_names=["search"])
        case = _make_case(max_steps=5)

        scores = reg.score_all(run, case)
        assert len(scores) == 2
        assert scores["politeness"].value == 1.0
        assert scores["tool_count"].value > 0

    def test_score_all_empty_registry(self):
        reg = EvaluatorRegistry()
        run = _make_run()
        case = _make_case()
        scores = reg.score_all(run, case)
        assert scores == {}

    def test_register_replaces_same_name(self):
        reg = EvaluatorRegistry()
        ev1 = PoliteEvaluator()
        ev2 = PoliteEvaluator()
        reg.register(ev1)
        reg.register(ev2)
        assert len(reg) == 1
        assert reg.evaluators["politeness"] is ev2

    def test_discover_entry_points_empty(self):
        """Discover with no installed plugins returns 0."""
        reg = EvaluatorRegistry()
        count = reg.discover_entry_points(group="checkagent.test_nonexistent")
        assert count == 0
        assert len(reg) == 0
