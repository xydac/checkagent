"""Safety taxonomy mapped to OWASP LLM Top 10.

Implements F11.1 from the PRD — standardized category and severity
enums for the ``@pytest.mark.safety(category, severity)`` marker.
"""

from __future__ import annotations

from enum import Enum


class SafetyCategory(str, Enum):
    """Safety test categories mapped to OWASP LLM Top 10."""

    # LLM01 — Prompt Injection
    PROMPT_INJECTION = "prompt_injection"
    JAILBREAK = "jailbreak"

    # LLM06 — Sensitive Information Disclosure
    PII_LEAKAGE = "pii_leakage"

    # LLM08 — Excessive Agency
    TOOL_MISUSE = "tool_misuse"

    # LLM09 — Overreliance
    GROUNDEDNESS = "groundedness"

    # LLM06 — Sensitive Information Disclosure (data scope)
    DATA_ENUMERATION = "data_enumeration"

    # General
    HARMFUL_CONTENT = "harmful_content"
    OFF_TOPIC = "off_topic"


class Severity(str, Enum):
    """Severity levels for safety findings.

    Supports direct comparison operators (``>=``, ``<``, etc.)::

        Severity.HIGH >= Severity.MEDIUM  # True
        Severity.LOW < Severity.CRITICAL  # True
    """

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

    @staticmethod
    def _order() -> dict[Severity, int]:
        return {Severity.LOW: 0, Severity.MEDIUM: 1, Severity.HIGH: 2, Severity.CRITICAL: 3}

    def __lt__(self, other: object) -> bool:
        if not isinstance(other, Severity):
            return NotImplemented
        return self._order()[self] < self._order()[other]

    def __le__(self, other: object) -> bool:
        if not isinstance(other, Severity):
            return NotImplemented
        return self._order()[self] <= self._order()[other]

    def __gt__(self, other: object) -> bool:
        if not isinstance(other, Severity):
            return NotImplemented
        return self._order()[self] > self._order()[other]

    def __ge__(self, other: object) -> bool:
        if not isinstance(other, Severity):
            return NotImplemented
        return self._order()[self] >= self._order()[other]

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Severity):
            return self.value == other.value
        if isinstance(other, str):
            return self.value == other
        return NotImplemented

    def __hash__(self) -> int:
        return hash(self.value)


# Mapping from SafetyCategory to OWASP LLM Top 10 ID
OWASP_MAPPING: dict[SafetyCategory, str] = {
    SafetyCategory.PROMPT_INJECTION: "LLM01",
    SafetyCategory.JAILBREAK: "LLM01",
    SafetyCategory.PII_LEAKAGE: "LLM06",
    SafetyCategory.TOOL_MISUSE: "LLM08",
    SafetyCategory.GROUNDEDNESS: "LLM09",
    SafetyCategory.DATA_ENUMERATION: "LLM06",
    SafetyCategory.HARMFUL_CONTENT: "N/A",
    SafetyCategory.OFF_TOPIC: "LLM08",  # Scope violation = excessive agency
}


# Severity ordering for threshold comparisons
SEVERITY_ORDER: dict[Severity, int] = {
    Severity.LOW: 0,
    Severity.MEDIUM: 1,
    Severity.HIGH: 2,
    Severity.CRITICAL: 3,
}


def severity_meets_threshold(severity: Severity, threshold: Severity) -> bool:
    """Return True if *severity* is at or above *threshold*."""
    return SEVERITY_ORDER[severity] >= SEVERITY_ORDER[threshold]
