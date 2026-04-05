"""Safety testing for AI agents — taxonomy, evaluators, and detection.

Implements F11.1 (safety taxonomy), F11.2 (built-in evaluators),
and F11.5 (custom evaluator base class) from the PRD.
"""

from checkagent.safety.evaluator import SafetyEvaluator, SafetyFinding, SafetyResult
from checkagent.safety.injection import PromptInjectionDetector
from checkagent.safety.pii import PIILeakageScanner
from checkagent.safety.taxonomy import (
    OWASP_MAPPING,
    SEVERITY_ORDER,
    SafetyCategory,
    Severity,
    severity_meets_threshold,
)

__all__ = [
    "OWASP_MAPPING",
    "PIILeakageScanner",
    "PromptInjectionDetector",
    "SEVERITY_ORDER",
    "SafetyCategory",
    "SafetyEvaluator",
    "SafetyFinding",
    "SafetyResult",
    "Severity",
    "severity_meets_threshold",
]
