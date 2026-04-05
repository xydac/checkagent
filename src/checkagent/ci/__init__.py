"""CI/CD integration for CheckAgent.

Provides quality gate evaluation, PR comment generation, and
GitHub Action / GitLab CI support.
"""

from checkagent.ci.quality_gate import (
    GateResult,
    GateVerdict,
    QualityGateReport,
    evaluate_gate,
    evaluate_gates,
    scores_to_dict,
)
from checkagent.ci.junit_xml import (
    JUnitProperty,
    JUnitTestCase,
    JUnitTestSuite,
    from_quality_gate_report,
    from_run_summary,
    render_junit_xml,
)
from checkagent.ci.reporter import RunSummary, generate_pr_comment

__all__ = [
    "GateResult",
    "GateVerdict",
    "JUnitProperty",
    "JUnitTestCase",
    "JUnitTestSuite",
    "QualityGateReport",
    "RunSummary",
    "evaluate_gate",
    "evaluate_gates",
    "from_quality_gate_report",
    "from_run_summary",
    "generate_pr_comment",
    "render_junit_xml",
    "scores_to_dict",
]
