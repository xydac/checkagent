"""Safety testing for AI agents — taxonomy, evaluators, detection, probes, and compliance.

Implements F11.1 (safety taxonomy), F11.2 (built-in evaluators),
F11.3 (attack probe library), F11.4 (compliance reports), and F11.5 (custom evaluator base class).
"""

from checkagent.safety.compliance import (
    ComplianceReport,
    EU_AI_ACT_MAPPING,
    generate_compliance_report,
    render_compliance_html,
    render_compliance_json,
    render_compliance_markdown,
)
from checkagent.safety.evaluator import SafetyEvaluator, SafetyFinding, SafetyResult
from checkagent.safety.injection import PromptInjectionDetector
from checkagent.safety.pii import PIILeakageScanner
from checkagent.safety.probes import Probe, ProbeSet
from checkagent.safety.probes import injection as probes_injection
from checkagent.safety.probes import jailbreak as probes_jailbreak
from checkagent.safety.probes import pii as probes_pii
from checkagent.safety.probes import scope as probes_scope
from checkagent.safety.refusal import RefusalComplianceChecker
from checkagent.safety.system_prompt import SystemPromptLeakDetector
from checkagent.safety.taxonomy import (
    OWASP_MAPPING,
    SEVERITY_ORDER,
    SafetyCategory,
    Severity,
    severity_meets_threshold,
)
from checkagent.safety.tool_boundary import ToolBoundary, ToolCallBoundaryValidator

# Namespace for ``from checkagent.safety import probes``
from checkagent.safety import probes  # noqa: E402

__all__ = [
    "ComplianceReport",
    "EU_AI_ACT_MAPPING",
    "OWASP_MAPPING",
    "PIILeakageScanner",
    "Probe",
    "ProbeSet",
    "PromptInjectionDetector",
    "RefusalComplianceChecker",
    "SEVERITY_ORDER",
    "SafetyCategory",
    "SafetyEvaluator",
    "SafetyFinding",
    "SafetyResult",
    "Severity",
    "SystemPromptLeakDetector",
    "ToolBoundary",
    "ToolCallBoundaryValidator",
    "probes",
    "probes_injection",
    "probes_jailbreak",
    "probes_pii",
    "probes_scope",
    "generate_compliance_report",
    "render_compliance_html",
    "render_compliance_json",
    "render_compliance_markdown",
    "severity_meets_threshold",
]
