"""Compliance report generation for audit-ready safety documentation.

Implements F11.4 from the PRD — ``checkagent report --compliance`` generates
timestamped, version-tagged reports with per-category pass/fail counts,
attack resistance rates, and regulatory mapping (EU AI Act, OWASP LLM Top 10).

Exportable as JSON, HTML, or Markdown.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from checkagent.safety.evaluator import SafetyFinding, SafetyResult
from checkagent.safety.taxonomy import (
    OWASP_MAPPING,
    SafetyCategory,
    Severity,
    severity_meets_threshold,
)


# EU AI Act article mapping for safety categories
EU_AI_ACT_MAPPING: dict[SafetyCategory, list[str]] = {
    SafetyCategory.PROMPT_INJECTION: ["Article 9 (Risk Management)", "Article 15 (Accuracy)"],
    SafetyCategory.JAILBREAK: ["Article 9 (Risk Management)", "Article 15 (Robustness)"],
    SafetyCategory.PII_LEAKAGE: ["Article 10 (Data Governance)", "Article 15 (Accuracy)"],
    SafetyCategory.TOOL_MISUSE: ["Article 9 (Risk Management)", "Article 14 (Human Oversight)"],
    SafetyCategory.GROUNDEDNESS: ["Article 15 (Accuracy)"],
    SafetyCategory.HARMFUL_CONTENT: ["Article 9 (Risk Management)"],
    SafetyCategory.OFF_TOPIC: ["Article 15 (Robustness)"],
}


@dataclass
class CategorySummary:
    """Per-category aggregation of safety test results."""

    category: SafetyCategory
    total: int = 0
    passed: int = 0
    failed: int = 0
    findings_by_severity: dict[Severity, int] = field(default_factory=dict)

    @property
    def resistance_rate(self) -> float:
        """Fraction of tests that passed (agent resisted the attack)."""
        if self.total == 0:
            return 0.0
        return self.passed / self.total

    @property
    def owasp_id(self) -> str:
        return OWASP_MAPPING.get(self.category, "N/A")

    @property
    def eu_ai_act_articles(self) -> list[str]:
        return EU_AI_ACT_MAPPING.get(self.category, [])


@dataclass
class ComplianceReport:
    """Audit-ready compliance report from safety test results.

    Attributes:
        agent_version: Version string of the agent under test.
        model_version: LLM model identifier used during testing.
        timestamp: When the report was generated (UTC).
        categories: Per-category summaries.
        total_tests: Total number of safety tests run.
        total_passed: Total tests that passed.
        total_findings: Total safety findings across all evaluators.
    """

    agent_version: str = ""
    model_version: str = ""
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    categories: dict[SafetyCategory, CategorySummary] = field(default_factory=dict)
    total_tests: int = 0
    total_passed: int = 0
    total_findings: int = 0

    @property
    def total_failed(self) -> int:
        return self.total_tests - self.total_passed

    @property
    def overall_resistance_rate(self) -> float:
        if self.total_tests == 0:
            return 0.0
        return self.total_passed / self.total_tests

    @property
    def has_critical_findings(self) -> bool:
        """True if any category has critical-severity findings."""
        for summary in self.categories.values():
            if summary.findings_by_severity.get(Severity.CRITICAL, 0) > 0:
                return True
        return False

    def to_dict(self) -> dict[str, Any]:
        """Serialize the report to a JSON-compatible dict."""
        return {
            "report_type": "checkagent_compliance",
            "schema_version": "1.0",
            "agent_version": self.agent_version,
            "model_version": self.model_version,
            "timestamp": self.timestamp.isoformat(),
            "summary": {
                "total_tests": self.total_tests,
                "total_passed": self.total_passed,
                "total_failed": self.total_failed,
                "total_findings": self.total_findings,
                "overall_resistance_rate": round(self.overall_resistance_rate, 4),
                "has_critical_findings": self.has_critical_findings,
            },
            "categories": {
                cat.value: {
                    "total": s.total,
                    "passed": s.passed,
                    "failed": s.failed,
                    "resistance_rate": round(s.resistance_rate, 4),
                    "owasp_id": s.owasp_id,
                    "eu_ai_act_articles": s.eu_ai_act_articles,
                    "findings_by_severity": {
                        sev.value: count
                        for sev, count in s.findings_by_severity.items()
                    },
                }
                for cat, s in sorted(
                    self.categories.items(), key=lambda x: x[0].value
                )
            },
        }


def generate_compliance_report(
    results: list[SafetyResult],
    *,
    agent_version: str = "",
    model_version: str = "",
    timestamp: datetime | None = None,
) -> ComplianceReport:
    """Build a :class:`ComplianceReport` from a list of evaluator results.

    Each :class:`SafetyResult` represents one evaluator run (which may contain
    multiple findings). A result with ``passed=True`` counts as a passed test;
    ``passed=False`` counts as failed.

    Parameters:
        results: Safety evaluator results to aggregate.
        agent_version: Version string for the agent under test.
        model_version: LLM model identifier.
        timestamp: Override for report timestamp (defaults to now UTC).
    """
    ts = timestamp or datetime.now(timezone.utc)
    categories: dict[SafetyCategory, CategorySummary] = {}
    total_findings = 0

    for result in results:
        # Determine the primary category from findings, or use a default
        if result.findings:
            cat = result.findings[0].category
        else:
            # No findings — infer from evaluator name or default
            cat = SafetyCategory.PROMPT_INJECTION

        if cat not in categories:
            categories[cat] = CategorySummary(category=cat)

        summary = categories[cat]
        summary.total += 1

        if result.passed:
            summary.passed += 1
        else:
            summary.failed += 1

        # Count findings by severity
        for finding in result.findings:
            total_findings += 1
            sev = finding.severity
            summary.findings_by_severity[sev] = (
                summary.findings_by_severity.get(sev, 0) + 1
            )
            # If this finding is in a different category, also update that category
            if finding.category != cat:
                if finding.category not in categories:
                    categories[finding.category] = CategorySummary(
                        category=finding.category
                    )
                cross = categories[finding.category]
                cross.findings_by_severity[sev] = (
                    cross.findings_by_severity.get(sev, 0) + 1
                )

    total_tests = len(results)
    total_passed = sum(1 for r in results if r.passed)

    return ComplianceReport(
        agent_version=agent_version,
        model_version=model_version,
        timestamp=ts,
        categories=categories,
        total_tests=total_tests,
        total_passed=total_passed,
        total_findings=total_findings,
    )


def render_compliance_json(report: ComplianceReport) -> str:
    """Render the compliance report as a JSON string."""
    return json.dumps(report.to_dict(), indent=2)


def render_compliance_markdown(report: ComplianceReport) -> str:
    """Render the compliance report as Markdown."""
    lines: list[str] = []
    lines.append("# Safety Compliance Report\n")

    # Metadata
    lines.append("## Report Metadata\n")
    lines.append(f"- **Generated:** {report.timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}")
    if report.agent_version:
        lines.append(f"- **Agent Version:** {report.agent_version}")
    if report.model_version:
        lines.append(f"- **Model Version:** {report.model_version}")
    lines.append("")

    # Overall summary
    lines.append("## Summary\n")
    lines.append("| Metric | Value |")
    lines.append("|--------|-------|")
    lines.append(f"| Total safety tests | {report.total_tests} |")
    lines.append(f"| Passed | {report.total_passed} |")
    lines.append(f"| Failed | {report.total_failed} |")
    lines.append(f"| Overall resistance rate | {report.overall_resistance_rate:.1%} |")
    lines.append(f"| Total findings | {report.total_findings} |")
    status = "FAIL" if report.has_critical_findings else "PASS"
    lines.append(f"| Critical findings | {'Yes' if report.has_critical_findings else 'No'} |")
    lines.append("")

    # Per-category breakdown
    if report.categories:
        lines.append("## Category Breakdown\n")
        lines.append("| Category | OWASP | Tests | Passed | Failed | Resistance |")
        lines.append("|----------|-------|-------|--------|--------|------------|")
        for cat in sorted(report.categories, key=lambda c: c.value):
            s = report.categories[cat]
            lines.append(
                f"| {cat.value} | {s.owasp_id} | {s.total} | {s.passed} "
                f"| {s.failed} | {s.resistance_rate:.1%} |"
            )
        lines.append("")

    # Findings by severity
    severity_totals: Counter[Severity] = Counter()
    for s in report.categories.values():
        for sev, count in s.findings_by_severity.items():
            severity_totals[sev] += count

    if severity_totals:
        lines.append("## Findings by Severity\n")
        lines.append("| Severity | Count |")
        lines.append("|----------|-------|")
        for sev in [Severity.CRITICAL, Severity.HIGH, Severity.MEDIUM, Severity.LOW]:
            if severity_totals[sev] > 0:
                lines.append(f"| {sev.value.upper()} | {severity_totals[sev]} |")
        lines.append("")

    # Regulatory mapping
    lines.append("## Regulatory Mapping\n")
    lines.append("### OWASP LLM Top 10\n")
    owasp_seen: dict[str, list[str]] = {}
    for cat, s in report.categories.items():
        oid = s.owasp_id
        if oid != "N/A":
            owasp_seen.setdefault(oid, []).append(
                f"{cat.value} ({s.resistance_rate:.0%} resistance)"
            )
    if owasp_seen:
        for oid in sorted(owasp_seen):
            lines.append(f"- **{oid}:** {', '.join(owasp_seen[oid])}")
    else:
        lines.append("No OWASP-mapped categories tested.")
    lines.append("")

    lines.append("### EU AI Act\n")
    articles_seen: dict[str, list[str]] = {}
    for cat, s in report.categories.items():
        for article in s.eu_ai_act_articles:
            articles_seen.setdefault(article, []).append(cat.value)
    if articles_seen:
        for article in sorted(articles_seen):
            cats = ", ".join(articles_seen[article])
            lines.append(f"- **{article}:** {cats}")
    else:
        lines.append("No EU AI Act-mapped categories tested.")
    lines.append("")

    # Footer
    lines.append("---")
    lines.append(f"*Generated by [CheckAgent](https://github.com/checkagent/checkagent) — {report.timestamp.strftime('%Y-%m-%d %H:%M UTC')}*")

    return "\n".join(lines)


def render_compliance_html(report: ComplianceReport) -> str:
    """Render the compliance report as a self-contained HTML document."""
    md = render_compliance_markdown(report)
    # Minimal HTML wrapper — the Markdown content is rendered as preformatted
    # sections with basic styling. For full rendering, consumers can pipe
    # the Markdown through a proper renderer.
    ts = report.timestamp.strftime("%Y-%m-%d %H:%M UTC")
    status_color = "#dc3545" if report.has_critical_findings else "#28a745"
    status_text = "CRITICAL FINDINGS" if report.has_critical_findings else "PASSED"

    html_parts: list[str] = []
    html_parts.append("<!DOCTYPE html>")
    html_parts.append('<html lang="en">')
    html_parts.append("<head>")
    html_parts.append('<meta charset="UTF-8">')
    html_parts.append("<title>Safety Compliance Report</title>")
    html_parts.append("<style>")
    html_parts.append(_html_styles())
    html_parts.append("</style>")
    html_parts.append("</head>")
    html_parts.append("<body>")
    html_parts.append('<div class="container">')

    # Header
    html_parts.append("<h1>Safety Compliance Report</h1>")
    html_parts.append(f'<div class="status" style="background:{status_color}">{status_text}</div>')
    html_parts.append(f"<p>Generated: {ts}</p>")
    if report.agent_version:
        html_parts.append(f"<p>Agent: {report.agent_version}</p>")
    if report.model_version:
        html_parts.append(f"<p>Model: {report.model_version}</p>")

    # Summary table
    html_parts.append("<h2>Summary</h2>")
    html_parts.append("<table>")
    html_parts.append("<tr><th>Metric</th><th>Value</th></tr>")
    html_parts.append(f"<tr><td>Total safety tests</td><td>{report.total_tests}</td></tr>")
    html_parts.append(f"<tr><td>Passed</td><td>{report.total_passed}</td></tr>")
    html_parts.append(f"<tr><td>Failed</td><td>{report.total_failed}</td></tr>")
    html_parts.append(
        f"<tr><td>Overall resistance rate</td>"
        f"<td>{report.overall_resistance_rate:.1%}</td></tr>"
    )
    html_parts.append(f"<tr><td>Total findings</td><td>{report.total_findings}</td></tr>")
    html_parts.append("</table>")

    # Category breakdown
    if report.categories:
        html_parts.append("<h2>Category Breakdown</h2>")
        html_parts.append("<table>")
        html_parts.append(
            "<tr><th>Category</th><th>OWASP</th><th>Tests</th>"
            "<th>Passed</th><th>Failed</th><th>Resistance</th></tr>"
        )
        for cat in sorted(report.categories, key=lambda c: c.value):
            s = report.categories[cat]
            html_parts.append(
                f"<tr><td>{cat.value}</td><td>{s.owasp_id}</td>"
                f"<td>{s.total}</td><td>{s.passed}</td><td>{s.failed}</td>"
                f"<td>{s.resistance_rate:.1%}</td></tr>"
            )
        html_parts.append("</table>")

    # Regulatory sections
    html_parts.append("<h2>Regulatory Mapping</h2>")

    # OWASP
    html_parts.append("<h3>OWASP LLM Top 10</h3>")
    html_parts.append("<ul>")
    for cat, s in sorted(report.categories.items(), key=lambda x: x[0].value):
        oid = s.owasp_id
        if oid != "N/A":
            html_parts.append(
                f"<li><strong>{oid}:</strong> {cat.value} "
                f"({s.resistance_rate:.0%} resistance)</li>"
            )
    html_parts.append("</ul>")

    # EU AI Act
    html_parts.append("<h3>EU AI Act</h3>")
    html_parts.append("<ul>")
    articles_seen: dict[str, list[str]] = {}
    for cat, s in report.categories.items():
        for article in s.eu_ai_act_articles:
            articles_seen.setdefault(article, []).append(cat.value)
    for article in sorted(articles_seen):
        cats = ", ".join(articles_seen[article])
        html_parts.append(f"<li><strong>{article}:</strong> {cats}</li>")
    html_parts.append("</ul>")

    html_parts.append("</div>")
    html_parts.append("</body>")
    html_parts.append("</html>")

    return "\n".join(html_parts)


def _html_styles() -> str:
    """Return minimal CSS for the HTML compliance report."""
    return """
body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; margin: 0; padding: 20px; background: #f8f9fa; }
.container { max-width: 900px; margin: 0 auto; background: #fff; padding: 30px; border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }
h1 { margin-top: 0; }
h2 { border-bottom: 1px solid #dee2e6; padding-bottom: 8px; }
.status { display: inline-block; color: #fff; padding: 4px 12px; border-radius: 4px; font-weight: bold; font-size: 14px; }
table { width: 100%; border-collapse: collapse; margin: 12px 0 24px 0; }
th, td { text-align: left; padding: 8px 12px; border-bottom: 1px solid #dee2e6; }
th { background: #f1f3f5; font-weight: 600; }
ul { line-height: 1.8; }
""".strip()
