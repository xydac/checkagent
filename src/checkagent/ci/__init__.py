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
from checkagent.ci.reporter import RunSummary, generate_pr_comment

__all__ = [
    "GateResult",
    "GateVerdict",
    "QualityGateReport",
    "RunSummary",
    "evaluate_gate",
    "evaluate_gates",
    "generate_pr_comment",
    "scores_to_dict",
]
