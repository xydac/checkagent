"""Quality gate evaluation for CI/CD integration.

Evaluates test scores against configurable thresholds and produces
pass/fail/warn verdicts. Implements PRD requirement F5.3.

Quality gates are configured in checkagent.yml:

    quality_gates:
      task_completion: { min: 0.90, on_fail: "block" }
      tool_correctness: { min: 0.95, on_fail: "block" }
      step_efficiency: { min: 0.70, on_fail: "warn" }
      cost_per_run: { max: 0.50, on_fail: "block" }
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from checkagent.core.config import QualityGateEntry
from checkagent.core.types import Score


class GateVerdict(str, Enum):
    """Outcome of a quality gate check."""

    PASSED = "passed"
    WARNED = "warned"
    BLOCKED = "blocked"
    SKIPPED = "skipped"


@dataclass
class GateResult:
    """Result of evaluating a single quality gate."""

    metric: str
    verdict: GateVerdict
    actual: float | None = None
    threshold: float | None = None
    direction: str = "min"  # "min" means actual >= threshold, "max" means actual <= threshold
    message: str = ""


@dataclass
class QualityGateReport:
    """Aggregate result of all quality gate evaluations."""

    results: list[GateResult] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        """True if no gates are blocked."""
        return not any(r.verdict == GateVerdict.BLOCKED for r in self.results)

    @property
    def has_warnings(self) -> bool:
        return any(r.verdict == GateVerdict.WARNED for r in self.results)

    @property
    def blocked_gates(self) -> list[GateResult]:
        return [r for r in self.results if r.verdict == GateVerdict.BLOCKED]

    @property
    def warned_gates(self) -> list[GateResult]:
        return [r for r in self.results if r.verdict == GateVerdict.WARNED]

    @property
    def passed_gates(self) -> list[GateResult]:
        return [r for r in self.results if r.verdict == GateVerdict.PASSED]


def evaluate_gate(
    metric: str,
    value: float,
    gate: QualityGateEntry,
) -> GateResult:
    """Evaluate a single metric value against a quality gate entry.

    If the gate has a `min`, the value must be >= min.
    If the gate has a `max`, the value must be <= max.
    If both are set, both conditions must hold.
    """
    failed = False
    direction = "min"
    threshold = gate.min

    if gate.min is not None and value < gate.min:
        failed = True
        direction = "min"
        threshold = gate.min

    if gate.max is not None and value > gate.max:
        failed = True
        direction = "max"
        threshold = gate.max

    if failed:
        if gate.on_fail == "ignore":
            verdict = GateVerdict.SKIPPED
        elif gate.on_fail == "warn":
            verdict = GateVerdict.WARNED
        else:
            verdict = GateVerdict.BLOCKED

        msg = _failure_message(metric, value, gate)
        return GateResult(
            metric=metric,
            verdict=verdict,
            actual=value,
            threshold=threshold,
            direction=direction,
            message=msg,
        )

    return GateResult(
        metric=metric,
        verdict=GateVerdict.PASSED,
        actual=value,
        threshold=threshold or gate.max,
        direction=direction,
    )


def evaluate_gates(
    scores: dict[str, float],
    gates: dict[str, QualityGateEntry],
) -> QualityGateReport:
    """Evaluate multiple metric scores against their quality gates.

    Args:
        scores: Mapping of metric name to its computed value.
        gates: Mapping of metric name to its gate configuration.

    Returns:
        QualityGateReport with a result per configured gate.
    """
    results: list[GateResult] = []

    for metric, gate in gates.items():
        if metric not in scores:
            results.append(
                GateResult(
                    metric=metric,
                    verdict=GateVerdict.SKIPPED,
                    message=f"Metric '{metric}' not found in scores",
                )
            )
            continue

        results.append(evaluate_gate(metric, scores[metric], gate))

    return QualityGateReport(results=results)


def scores_to_dict(scores: list[Score]) -> dict[str, float]:
    """Convert a list of Score objects to a {name: value} dict for gate evaluation."""
    return {s.name: s.value for s in scores}


def _failure_message(metric: str, value: float, gate: QualityGateEntry) -> str:
    """Build a human-readable failure message."""
    parts = [f"Quality gate '{metric}' failed:"]
    if gate.min is not None and value < gate.min:
        parts.append(f"value {value:.4f} < min {gate.min:.4f}")
    if gate.max is not None and value > gate.max:
        parts.append(f"value {value:.4f} > max {gate.max:.4f}")
    parts.append(f"(action: {gate.on_fail})")
    return " ".join(parts)
