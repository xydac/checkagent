"""Tests for token tracking and cost reporting (F7.1, F7.2)."""

from __future__ import annotations

import pytest

from checkagent.core.config import BudgetConfig, ProviderPricing
from checkagent.core.cost import (
    BUILTIN_PRICING,
    BudgetExceededError,
    CostReport,
    CostTracker,
    ModelCost,
    calculate_run_cost,
    calculate_step_cost,
    get_pricing,
)
from checkagent.core.types import AgentInput, AgentRun, Step

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_run(
    steps: list[tuple[str | None, int | None, int | None]],
) -> AgentRun:
    """Create an AgentRun from a list of (model, prompt_tokens, completion_tokens)."""
    return AgentRun(
        input=AgentInput(query="test"),
        steps=[
            Step(
                step_index=i,
                model=model,
                prompt_tokens=pt,
                completion_tokens=ct,
            )
            for i, (model, pt, ct) in enumerate(steps)
        ],
    )


# ---------------------------------------------------------------------------
# Built-in pricing table
# ---------------------------------------------------------------------------


class TestBuiltinPricing:
    def test_common_models_present(self):
        for model in ["gpt-4o", "gpt-4o-mini", "claude-sonnet", "claude-opus"]:
            assert model in BUILTIN_PRICING

    def test_pricing_values_positive(self):
        for model, pricing in BUILTIN_PRICING.items():
            assert pricing.input >= 0, f"{model} input pricing negative"
            assert pricing.output >= 0, f"{model} output pricing negative"


# ---------------------------------------------------------------------------
# get_pricing
# ---------------------------------------------------------------------------


class TestGetPricing:
    def test_builtin_lookup(self):
        p = get_pricing("gpt-4o")
        assert p is not None
        assert p.input == 2.50

    def test_unknown_model_returns_none(self):
        assert get_pricing("nonexistent-model-xyz") is None

    def test_override_takes_precedence(self):
        overrides = {"gpt-4o": ProviderPricing(input=99.0, output=99.0)}
        p = get_pricing("gpt-4o", overrides)
        assert p is not None
        assert p.input == 99.0

    def test_override_for_custom_model(self):
        overrides = {"my-model": ProviderPricing(input=1.0, output=2.0)}
        p = get_pricing("my-model", overrides)
        assert p is not None
        assert p.output == 2.0


# ---------------------------------------------------------------------------
# calculate_step_cost
# ---------------------------------------------------------------------------


class TestCalculateStepCost:
    def test_basic_cost(self):
        step = Step(prompt_tokens=1000, completion_tokens=500)
        pricing = ProviderPricing(input=3.00, output=15.00)
        cost = calculate_step_cost(step, pricing)
        # 1000 * 3.00/1M + 500 * 15.00/1M = 0.003 + 0.0075 = 0.0105
        assert abs(cost - 0.0105) < 1e-9

    def test_zero_tokens(self):
        step = Step(prompt_tokens=0, completion_tokens=0)
        pricing = ProviderPricing(input=3.00, output=15.00)
        assert calculate_step_cost(step, pricing) == 0.0

    def test_none_tokens_treated_as_zero(self):
        step = Step()  # tokens are None
        pricing = ProviderPricing(input=3.00, output=15.00)
        assert calculate_step_cost(step, pricing) == 0.0

    def test_only_input_tokens(self):
        step = Step(prompt_tokens=1_000_000, completion_tokens=None)
        pricing = ProviderPricing(input=3.00, output=15.00)
        assert abs(calculate_step_cost(step, pricing) - 3.00) < 1e-9


# ---------------------------------------------------------------------------
# calculate_run_cost
# ---------------------------------------------------------------------------


class TestCalculateRunCost:
    def test_single_model_run(self):
        run = _make_run([("gpt-4o", 1000, 500), ("gpt-4o", 2000, 1000)])
        bd = calculate_run_cost(run)
        assert bd.total_input_tokens == 3000
        assert bd.total_output_tokens == 1500
        assert bd.total_tokens == 4500
        assert bd.unpriced_steps == 0
        assert "gpt-4o" in bd.per_model
        assert bd.total_cost > 0

    def test_multi_model_run(self):
        run = _make_run([
            ("gpt-4o", 1000, 500),
            ("claude-sonnet", 2000, 1000),
        ])
        bd = calculate_run_cost(run)
        assert len(bd.per_model) == 2
        assert "gpt-4o" in bd.per_model
        assert "claude-sonnet" in bd.per_model

    def test_unknown_model_counted_as_unpriced(self):
        run = _make_run([("unknown-model", 1000, 500)])
        bd = calculate_run_cost(run)
        assert bd.unpriced_steps == 1
        assert bd.total_cost == 0.0
        # Tokens still counted
        assert bd.total_input_tokens == 1000

    def test_default_pricing_fallback(self):
        run = _make_run([("unknown-model", 1000, 500)])
        fallback = ProviderPricing(input=1.00, output=5.00)
        bd = calculate_run_cost(run, default_pricing=fallback)
        assert bd.unpriced_steps == 0
        assert bd.total_cost > 0

    def test_pricing_overrides(self):
        overrides = {"custom": ProviderPricing(input=10.0, output=20.0)}
        run = _make_run([("custom", 1_000_000, 1_000_000)])
        bd = calculate_run_cost(run, pricing_overrides=overrides)
        # 1M * 10/1M + 1M * 20/1M = 30.0
        assert abs(bd.total_cost - 30.0) < 1e-9

    def test_empty_run(self):
        run = AgentRun(input=AgentInput(query="test"))
        bd = calculate_run_cost(run)
        assert bd.total_cost == 0.0
        assert bd.total_tokens == 0

    def test_none_model_uses_unknown(self):
        run = _make_run([(None, 100, 50)])
        bd = calculate_run_cost(run)
        assert bd.unpriced_steps == 1

    def test_to_dict(self):
        run = _make_run([("gpt-4o", 1000, 500)])
        bd = calculate_run_cost(run)
        d = bd.to_dict()
        assert "total_cost_usd" in d
        assert "per_model" in d
        assert isinstance(d["per_model"], dict)


# ---------------------------------------------------------------------------
# ModelCost
# ---------------------------------------------------------------------------


class TestModelCost:
    def test_total_tokens(self):
        mc = ModelCost(model="gpt-4o", input_tokens=100, output_tokens=50)
        assert mc.total_tokens == 150

    def test_to_dict(self):
        mc = ModelCost(model="gpt-4o", input_tokens=100, output_tokens=50, cost=0.001)
        d = mc.to_dict()
        assert d["model"] == "gpt-4o"
        assert d["total_tokens"] == 150
        assert d["cost_usd"] == 0.001


# ---------------------------------------------------------------------------
# CostTracker
# ---------------------------------------------------------------------------


class TestCostTracker:
    def test_record_and_accumulate(self):
        tracker = CostTracker()
        run1 = _make_run([("gpt-4o", 1000, 500)])
        run2 = _make_run([("gpt-4o", 2000, 1000)])
        tracker.record(run1)
        tracker.record(run2)
        assert tracker.run_count == 2
        assert tracker.total_cost > 0
        assert tracker.total_tokens > 0

    def test_empty_tracker(self):
        tracker = CostTracker()
        assert tracker.total_cost == 0.0
        assert tracker.total_tokens == 0
        assert tracker.run_count == 0

    def test_per_test_budget_ok(self):
        budget = BudgetConfig(per_test=1.00)
        tracker = CostTracker(budget=budget)
        run = _make_run([("gpt-4o-mini", 100, 50)])  # very cheap
        bd = tracker.record(run)
        tracker.check_test_budget(bd)  # should not raise

    def test_per_test_budget_exceeded(self):
        budget = BudgetConfig(per_test=0.000001)
        tracker = CostTracker(budget=budget)
        run = _make_run([("gpt-4o", 1_000_000, 1_000_000)])
        bd = tracker.record(run)
        with pytest.raises(BudgetExceededError) as exc_info:
            tracker.check_test_budget(bd)
        assert exc_info.value.limit_name == "per_test"

    def test_suite_budget_exceeded(self):
        budget = BudgetConfig(per_suite=0.000001)
        tracker = CostTracker(budget=budget)
        run = _make_run([("gpt-4o", 1_000_000, 1_000_000)])
        tracker.record(run)
        with pytest.raises(BudgetExceededError) as exc_info:
            tracker.check_suite_budget()
        assert exc_info.value.limit_name == "per_suite"

    def test_ci_budget_exceeded(self):
        budget = BudgetConfig(per_ci_run=0.000001)
        tracker = CostTracker(budget=budget)
        run = _make_run([("gpt-4o", 1_000_000, 1_000_000)])
        tracker.record(run)
        with pytest.raises(BudgetExceededError) as exc_info:
            tracker.check_ci_budget()
        assert exc_info.value.limit_name == "per_ci_run"

    def test_no_budget_never_raises(self):
        tracker = CostTracker()  # no budget set
        run = _make_run([("gpt-4o", 10_000_000, 10_000_000)])
        bd = tracker.record(run)
        tracker.check_test_budget(bd)
        tracker.check_suite_budget()
        tracker.check_ci_budget()

    def test_with_pricing_overrides(self):
        overrides = {"my-llm": ProviderPricing(input=100.0, output=200.0)}
        tracker = CostTracker(pricing_overrides=overrides)
        run = _make_run([("my-llm", 1_000_000, 1_000_000)])
        bd = tracker.record(run)
        # 1M * 100/1M + 1M * 200/1M = 300.0
        assert abs(bd.total_cost - 300.0) < 1e-6

    def test_with_default_pricing(self):
        fallback = ProviderPricing(input=1.0, output=2.0)
        tracker = CostTracker(default_pricing=fallback)
        run = _make_run([("unknown", 1_000_000, 1_000_000)])
        bd = tracker.record(run)
        assert abs(bd.total_cost - 3.0) < 1e-6

    def test_runs_list_is_copy(self):
        tracker = CostTracker()
        run = _make_run([("gpt-4o", 100, 50)])
        tracker.record(run)
        runs = tracker.runs
        runs.clear()
        assert tracker.run_count == 1


# ---------------------------------------------------------------------------
# CostTracker.summary / CostReport
# ---------------------------------------------------------------------------


class TestCostReport:
    def test_summary_aggregates(self):
        tracker = CostTracker()
        tracker.record(_make_run([("gpt-4o", 1000, 500)]))
        tracker.record(_make_run([("gpt-4o", 2000, 1000), ("claude-sonnet", 500, 250)]))

        report = tracker.summary()
        assert report.run_count == 2
        assert report.total_input_tokens == 3500
        assert report.total_output_tokens == 1750
        assert len(report.per_model) == 2
        assert report.total_cost > 0

    def test_avg_cost_per_run(self):
        tracker = CostTracker()
        tracker.record(_make_run([("gpt-4o", 1000, 500)]))
        tracker.record(_make_run([("gpt-4o", 1000, 500)]))
        report = tracker.summary()
        assert abs(report.avg_cost_per_run - report.total_cost / 2) < 1e-9

    def test_avg_cost_empty(self):
        report = CostReport()
        assert report.avg_cost_per_run == 0.0

    def test_budget_utilization(self):
        budget = BudgetConfig(per_suite=10.0)
        tracker = CostTracker(budget=budget)
        tracker.record(_make_run([("gpt-4o", 1_000_000, 1_000_000)]))
        report = tracker.summary()
        util = report.budget_utilization()
        assert "per_suite" in util
        assert util["per_suite"] is not None
        assert util["per_suite"] > 0

    def test_budget_utilization_no_budget(self):
        report = CostReport()
        assert report.budget_utilization() == {}

    def test_to_dict(self):
        tracker = CostTracker()
        tracker.record(_make_run([("gpt-4o", 1000, 500)]))
        report = tracker.summary()
        d = report.to_dict()
        assert "run_count" in d
        assert "total_cost_usd" in d
        assert "avg_cost_per_run_usd" in d
        assert "budget_utilization" in d
        assert "per_model" in d

    def test_unpriced_steps_accumulated(self):
        tracker = CostTracker()
        tracker.record(_make_run([("unknown1", 100, 50)]))
        tracker.record(_make_run([("unknown2", 200, 100)]))
        report = tracker.summary()
        assert report.unpriced_steps == 2


# ---------------------------------------------------------------------------
# BudgetExceededError
# ---------------------------------------------------------------------------


class TestBudgetExceededError:
    def test_attributes(self):
        err = BudgetExceededError("per_test", 0.10, 0.25)
        assert err.limit_name == "per_test"
        assert err.limit_usd == 0.10
        assert err.actual_usd == 0.25
        assert "per_test" in str(err)
        assert "$0.10" in str(err)
