"""Agent resilience scoring — measure how gracefully agents degrade under faults.

Combines fault injection results with evaluation metrics to produce a
resilience profile: a quantitative answer to "how does my agent handle
failures?"

Novel capability: no existing agent testing framework offers systematic
resilience measurement. This module bridges CheckAgent's fault injection
layer (Layer 1) with evaluation metrics (Layer 3).

Requirements: F14.1, F3.1
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from checkagent.core.types import AgentRun, Score


@dataclass
class ScenarioResult:
    """Metrics for a single fault scenario."""

    scenario: str
    scores: dict[str, float]
    degradation: dict[str, float]  # metric -> (faulted - baseline), negative = worse
    resilience: float  # 0-1, how much of baseline performance is retained


@dataclass
class ResilienceProfile:
    """Complete resilience analysis across multiple fault scenarios.

    A resilience score of 1.0 means the agent performs identically under
    faults as under normal conditions. A score of 0.0 means total failure.

    Usage::

        profile = ResilienceProfile.from_scores(
            baseline={"task_completion": 0.95, "tool_correctness": 1.0},
            scenarios={
                "llm_rate_limit": {"task_completion": 0.6, "tool_correctness": 0.8},
                "tool_timeout": {"task_completion": 0.3, "tool_correctness": 0.5},
            },
        )
        assert profile.overall >= 0.7
        print(profile.worst_scenario)   # "tool_timeout"
        print(profile.weakest_metric)   # "task_completion"
    """

    baseline: dict[str, float]
    scenario_results: dict[str, ScenarioResult] = field(default_factory=dict)

    @property
    def overall(self) -> float:
        """Overall resilience score (mean across all scenarios)."""
        if not self.scenario_results:
            return 1.0
        return _mean([r.resilience for r in self.scenario_results.values()])

    @property
    def worst_scenario(self) -> str | None:
        """The fault scenario that caused the most degradation."""
        if not self.scenario_results:
            return None
        return min(self.scenario_results, key=lambda s: self.scenario_results[s].resilience)

    @property
    def best_scenario(self) -> str | None:
        """The fault scenario the agent handled best."""
        if not self.scenario_results:
            return None
        return max(self.scenario_results, key=lambda s: self.scenario_results[s].resilience)

    @property
    def weakest_metric(self) -> str | None:
        """The metric that degrades the most across all scenarios."""
        if not self.scenario_results or not self.baseline:
            return None
        metric_degradations: dict[str, float] = {}
        for result in self.scenario_results.values():
            for metric, deg in result.degradation.items():
                metric_degradations.setdefault(metric, 0.0)
                metric_degradations[metric] += deg
        if not metric_degradations:
            return None
        return min(metric_degradations, key=metric_degradations.get)  # type: ignore[arg-type]

    @property
    def most_resilient_metric(self) -> str | None:
        """The metric that degrades the least across all scenarios."""
        if not self.scenario_results or not self.baseline:
            return None
        metric_degradations: dict[str, float] = {}
        for result in self.scenario_results.values():
            for metric, deg in result.degradation.items():
                metric_degradations.setdefault(metric, 0.0)
                metric_degradations[metric] += deg
        if not metric_degradations:
            return None
        return max(metric_degradations, key=metric_degradations.get)  # type: ignore[arg-type]

    @classmethod
    def from_scores(
        cls,
        baseline: dict[str, float],
        scenarios: dict[str, dict[str, float]],
    ) -> ResilienceProfile:
        """Build a resilience profile from pre-computed metric scores.

        Args:
            baseline: Metric name -> score under normal conditions.
            scenarios: Scenario name -> {metric name -> score under fault}.

        Returns:
            ResilienceProfile with per-scenario analysis.
        """
        profile = cls(baseline=dict(baseline))
        for scenario_name, faulted_scores in scenarios.items():
            result = _compute_scenario(scenario_name, baseline, faulted_scores)
            profile.scenario_results[scenario_name] = result
        return profile

    @classmethod
    def from_runs(
        cls,
        baseline_runs: list[AgentRun],
        faulted_runs: dict[str, list[AgentRun]],
        metrics: dict[str, Callable[[AgentRun], Score]],
    ) -> ResilienceProfile:
        """Build a resilience profile from AgentRun objects.

        Evaluates each run against provided metrics and computes resilience.

        Args:
            baseline_runs: Agent runs under normal conditions.
            faulted_runs: Scenario name -> runs under that fault condition.
            metrics: Metric name -> callable(AgentRun) -> Score.

        Returns:
            ResilienceProfile with per-scenario analysis.
        """
        baseline_scores = _average_scores(baseline_runs, metrics)
        scenarios: dict[str, dict[str, float]] = {}
        for scenario_name, runs in faulted_runs.items():
            scenarios[scenario_name] = _average_scores(runs, metrics)
        return cls.from_scores(baseline=baseline_scores, scenarios=scenarios)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dict for JSON output."""
        return {
            "overall_resilience": round(self.overall, 4),
            "baseline": {k: round(v, 4) for k, v in self.baseline.items()},
            "worst_scenario": self.worst_scenario,
            "weakest_metric": self.weakest_metric,
            "scenarios": {
                name: {
                    "resilience": round(r.resilience, 4),
                    "scores": {k: round(v, 4) for k, v in r.scores.items()},
                    "degradation": {k: round(v, 4) for k, v in r.degradation.items()},
                }
                for name, r in self.scenario_results.items()
            },
        }


def _compute_scenario(
    name: str,
    baseline: dict[str, float],
    faulted: dict[str, float],
) -> ScenarioResult:
    """Compute resilience for a single fault scenario."""
    degradation: dict[str, float] = {}
    retention_ratios: list[float] = []

    for metric, baseline_val in baseline.items():
        faulted_val = faulted.get(metric, 0.0)
        degradation[metric] = faulted_val - baseline_val

        if baseline_val > 0:
            # Retention: what fraction of baseline performance is kept?
            # Clamp to [0, 1] — improvement under fault is capped at 1.0
            retention = min(faulted_val / baseline_val, 1.0)
        elif faulted_val == 0:
            # Both zero — no degradation
            retention = 1.0
        else:
            # Baseline was 0 but faulted is positive — no degradation
            retention = 1.0

        retention_ratios.append(retention)

    resilience = _mean(retention_ratios) if retention_ratios else 1.0

    return ScenarioResult(
        scenario=name,
        scores=dict(faulted),
        degradation=degradation,
        resilience=resilience,
    )


def _average_scores(
    runs: list[AgentRun],
    metrics: dict[str, Callable[[AgentRun], Score]],
) -> dict[str, float]:
    """Compute mean score for each metric across runs."""
    if not runs:
        return {name: 0.0 for name in metrics}

    totals: dict[str, float] = {name: 0.0 for name in metrics}
    for run in runs:
        for name, metric_fn in metrics.items():
            score = metric_fn(run)
            totals[name] += score.value
    return {name: total / len(runs) for name, total in totals.items()}


def _mean(values: list[float]) -> float:
    """Simple mean, returns 0.0 for empty list."""
    if not values:
        return 0.0
    return sum(values) / len(values)
