"""Token tracking and cost reporting for CheckAgent.

Provides cost calculation from token usage, built-in pricing tables
for common models, cumulative tracking with budget enforcement, and
cost reporting/summaries.

Implements PRD requirements F7.1 (Token Tracking) and F7.2 (Budget Limits).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from checkagent.core.config import BudgetConfig, ProviderPricing
from checkagent.core.types import AgentRun, Step


# ---------------------------------------------------------------------------
# Built-in pricing table (per 1M tokens, USD)
# Updated as of 2025-05 — users can override via checkagent.yml providers
# ---------------------------------------------------------------------------

BUILTIN_PRICING: dict[str, ProviderPricing] = {
    # OpenAI
    "gpt-4o": ProviderPricing(input=2.50, output=10.00),
    "gpt-4o-mini": ProviderPricing(input=0.15, output=0.60),
    "gpt-4-turbo": ProviderPricing(input=10.00, output=30.00),
    "gpt-4": ProviderPricing(input=30.00, output=60.00),
    "gpt-3.5-turbo": ProviderPricing(input=0.50, output=1.50),
    "o1": ProviderPricing(input=15.00, output=60.00),
    "o1-mini": ProviderPricing(input=3.00, output=12.00),
    "o3-mini": ProviderPricing(input=1.10, output=4.40),
    # Anthropic
    "claude-opus-4-20250514": ProviderPricing(input=15.00, output=75.00),
    "claude-sonnet-4-20250514": ProviderPricing(input=3.00, output=15.00),
    "claude-haiku-3-5-20241022": ProviderPricing(input=0.80, output=4.00),
    # Short aliases
    "claude-opus": ProviderPricing(input=15.00, output=75.00),
    "claude-sonnet": ProviderPricing(input=3.00, output=15.00),
    "claude-haiku": ProviderPricing(input=0.80, output=4.00),
    # Google
    "gemini-2.0-flash": ProviderPricing(input=0.10, output=0.40),
    "gemini-2.5-pro": ProviderPricing(input=1.25, output=10.00),
}


def get_pricing(model: str, overrides: dict[str, ProviderPricing] | None = None) -> ProviderPricing | None:
    """Look up pricing for a model, checking overrides first then built-ins."""
    if overrides and model in overrides:
        return overrides[model]
    return BUILTIN_PRICING.get(model)


def calculate_step_cost(step: Step, pricing: ProviderPricing) -> float:
    """Calculate the USD cost of a single step given pricing."""
    cost = 0.0
    if step.prompt_tokens is not None:
        cost += step.prompt_tokens * pricing.input / 1_000_000
    if step.completion_tokens is not None:
        cost += step.completion_tokens * pricing.output / 1_000_000
    return cost


def calculate_run_cost(
    run: AgentRun,
    pricing_overrides: dict[str, ProviderPricing] | None = None,
    default_pricing: ProviderPricing | None = None,
) -> CostBreakdown:
    """Calculate cost breakdown for an entire agent run.

    Pricing resolution order per step:
    1. Step's model in pricing_overrides
    2. Step's model in BUILTIN_PRICING
    3. default_pricing (fallback)

    If no pricing is found for a step, it is counted as unpriced.
    """
    total_input_tokens = 0
    total_output_tokens = 0
    total_cost = 0.0
    per_model: dict[str, ModelCost] = {}
    unpriced_steps = 0

    for step in run.steps:
        model = step.model or "__unknown__"
        pricing = get_pricing(model, pricing_overrides) or default_pricing

        input_t = step.prompt_tokens or 0
        output_t = step.completion_tokens or 0
        total_input_tokens += input_t
        total_output_tokens += output_t

        if pricing is None:
            unpriced_steps += 1
            continue

        step_cost = calculate_step_cost(step, pricing)
        total_cost += step_cost

        if model not in per_model:
            per_model[model] = ModelCost(model=model)
        per_model[model].input_tokens += input_t
        per_model[model].output_tokens += output_t
        per_model[model].cost += step_cost

    return CostBreakdown(
        total_input_tokens=total_input_tokens,
        total_output_tokens=total_output_tokens,
        total_cost=total_cost,
        per_model=per_model,
        unpriced_steps=unpriced_steps,
    )


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class ModelCost:
    """Cost breakdown for a single model."""

    model: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    cost: float = 0.0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "cost_usd": round(self.cost, 6),
        }


@dataclass
class CostBreakdown:
    """Full cost breakdown for a single agent run."""

    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cost: float = 0.0
    per_model: dict[str, ModelCost] = field(default_factory=dict)
    unpriced_steps: int = 0

    @property
    def total_tokens(self) -> int:
        return self.total_input_tokens + self.total_output_tokens

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_tokens": self.total_tokens,
            "total_cost_usd": round(self.total_cost, 6),
            "unpriced_steps": self.unpriced_steps,
            "per_model": {k: v.to_dict() for k, v in self.per_model.items()},
        }


# ---------------------------------------------------------------------------
# Budget checking
# ---------------------------------------------------------------------------


class BudgetExceededError(Exception):
    """Raised when a cost budget limit is exceeded."""

    def __init__(self, limit_name: str, limit_usd: float, actual_usd: float) -> None:
        self.limit_name = limit_name
        self.limit_usd = limit_usd
        self.actual_usd = actual_usd
        super().__init__(
            f"Budget exceeded: {limit_name} limit is ${limit_usd:.4f}, "
            f"actual cost is ${actual_usd:.4f}"
        )


# ---------------------------------------------------------------------------
# CostTracker — cumulative tracking with budget enforcement
# ---------------------------------------------------------------------------


class CostTracker:
    """Tracks cumulative costs across multiple agent runs with budget enforcement.

    Usage:
        tracker = CostTracker(budget=config.budget, pricing_overrides=...)
        # After each test run:
        breakdown = tracker.record(run)
        # Check budget:
        tracker.check_budget()  # raises BudgetExceededError if over
    """

    def __init__(
        self,
        budget: BudgetConfig | None = None,
        pricing_overrides: dict[str, ProviderPricing] | None = None,
        default_pricing: ProviderPricing | None = None,
    ) -> None:
        self.budget = budget or BudgetConfig()
        self.pricing_overrides = pricing_overrides
        self.default_pricing = default_pricing
        self._runs: list[CostBreakdown] = []

    @property
    def total_cost(self) -> float:
        return sum(r.total_cost for r in self._runs)

    @property
    def total_tokens(self) -> int:
        return sum(r.total_tokens for r in self._runs)

    @property
    def run_count(self) -> int:
        return len(self._runs)

    @property
    def runs(self) -> list[CostBreakdown]:
        return list(self._runs)

    def record(self, run: AgentRun) -> CostBreakdown:
        """Calculate cost for a run and add it to the cumulative tracker."""
        breakdown = calculate_run_cost(
            run,
            pricing_overrides=self.pricing_overrides,
            default_pricing=self.default_pricing,
        )
        self._runs.append(breakdown)
        return breakdown

    def check_test_budget(self, breakdown: CostBreakdown) -> None:
        """Check if a single test's cost exceeds the per-test budget.

        Raises BudgetExceededError if over limit.
        """
        if self.budget.per_test is not None and breakdown.total_cost > self.budget.per_test:
            raise BudgetExceededError("per_test", self.budget.per_test, breakdown.total_cost)

    def check_suite_budget(self) -> None:
        """Check if cumulative cost exceeds the per-suite budget.

        Raises BudgetExceededError if over limit.
        """
        if self.budget.per_suite is not None and self.total_cost > self.budget.per_suite:
            raise BudgetExceededError("per_suite", self.budget.per_suite, self.total_cost)

    def check_ci_budget(self) -> None:
        """Check if cumulative cost exceeds the per-CI-run budget.

        Raises BudgetExceededError if over limit.
        """
        if self.budget.per_ci_run is not None and self.total_cost > self.budget.per_ci_run:
            raise BudgetExceededError("per_ci_run", self.budget.per_ci_run, self.total_cost)

    def summary(self) -> CostReport:
        """Generate a summary report of all tracked runs."""
        merged_models: dict[str, ModelCost] = {}
        total_unpriced = 0

        for breakdown in self._runs:
            total_unpriced += breakdown.unpriced_steps
            for model, mc in breakdown.per_model.items():
                if model not in merged_models:
                    merged_models[model] = ModelCost(model=model)
                merged_models[model].input_tokens += mc.input_tokens
                merged_models[model].output_tokens += mc.output_tokens
                merged_models[model].cost += mc.cost

        return CostReport(
            run_count=self.run_count,
            total_input_tokens=sum(r.total_input_tokens for r in self._runs),
            total_output_tokens=sum(r.total_output_tokens for r in self._runs),
            total_cost=self.total_cost,
            per_model=merged_models,
            unpriced_steps=total_unpriced,
            budget=self.budget,
        )


# ---------------------------------------------------------------------------
# CostReport
# ---------------------------------------------------------------------------


@dataclass
class CostReport:
    """Summary cost report across multiple runs."""

    run_count: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cost: float = 0.0
    per_model: dict[str, ModelCost] = field(default_factory=dict)
    unpriced_steps: int = 0
    budget: BudgetConfig = field(default_factory=BudgetConfig)

    @property
    def total_tokens(self) -> int:
        return self.total_input_tokens + self.total_output_tokens

    @property
    def avg_cost_per_run(self) -> float:
        if self.run_count == 0:
            return 0.0
        return self.total_cost / self.run_count

    def budget_utilization(self) -> dict[str, float | None]:
        """Return budget utilization as fractions (0.0-1.0+) for each limit."""
        result: dict[str, float | None] = {}
        if self.budget.per_suite is not None and self.budget.per_suite > 0:
            result["per_suite"] = self.total_cost / self.budget.per_suite
        if self.budget.per_ci_run is not None and self.budget.per_ci_run > 0:
            result["per_ci_run"] = self.total_cost / self.budget.per_ci_run
        return result

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_count": self.run_count,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_tokens": self.total_tokens,
            "total_cost_usd": round(self.total_cost, 6),
            "avg_cost_per_run_usd": round(self.avg_cost_per_run, 6),
            "unpriced_steps": self.unpriced_steps,
            "per_model": {k: v.to_dict() for k, v in self.per_model.items()},
            "budget_utilization": self.budget_utilization(),
        }
