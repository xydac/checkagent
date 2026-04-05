"""Base class for safety evaluators.

Implements F11.5 from the PRD — the SafetyEvaluator protocol that all
built-in and community safety evaluators implement.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from checkagent.safety.taxonomy import SafetyCategory, Severity


@dataclass
class SafetyFinding:
    """A single safety issue detected by an evaluator."""

    category: SafetyCategory
    severity: Severity
    description: str
    evidence: str = ""
    probe: str = ""


@dataclass
class SafetyResult:
    """Aggregated result from a safety evaluator."""

    passed: bool
    findings: list[SafetyFinding] = field(default_factory=list)
    evaluator: str = ""
    details: dict[str, object] = field(default_factory=dict)

    @property
    def finding_count(self) -> int:
        return len(self.findings)

    def findings_by_severity(self, severity: Severity) -> list[SafetyFinding]:
        return [f for f in self.findings if f.severity == severity]


class SafetyEvaluator:
    """Base class for safety evaluators.

    Subclass and override :meth:`evaluate` to implement a custom evaluator.
    Community evaluators register via the ``checkagent.safety`` entry point.
    """

    name: str = "base"
    category: SafetyCategory = SafetyCategory.PROMPT_INJECTION

    def evaluate(self, text: str) -> SafetyResult:
        """Evaluate *text* for safety issues.

        Returns a :class:`SafetyResult` with pass/fail and any findings.
        """
        raise NotImplementedError
