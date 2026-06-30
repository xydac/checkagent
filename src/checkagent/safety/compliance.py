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

from checkagent.safety.evaluator import SafetyResult
from checkagent.safety.taxonomy import (
    OWASP_MAPPING,
    SafetyCategory,
    Severity,
)

# EU AI Act article mapping for safety categories
EU_AI_ACT_MAPPING: dict[SafetyCategory, list[str]] = {
    SafetyCategory.PROMPT_INJECTION: ["Article 9 (Risk Management)", "Article 15 (Accuracy)"],
    SafetyCategory.JAILBREAK: ["Article 9 (Risk Management)", "Article 15 (Robustness)"],
    SafetyCategory.PII_LEAKAGE: ["Article 10 (Data Governance)", "Article 15 (Accuracy)"],
    SafetyCategory.TOOL_MISUSE: ["Article 9 (Risk Management)", "Article 14 (Human Oversight)"],
    SafetyCategory.GROUNDEDNESS: ["Article 15 (Accuracy)"],
    SafetyCategory.DATA_ENUMERATION: ["Article 10 (Data Governance)", "Article 15 (Accuracy)"],
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
        raw_findings: Per-probe finding details for HTML report rendering.
            Each dict has keys: probe_id, probe_description, category, severity,
            finding, probe_input, response, remediation (list[str]).
    """

    agent_version: str = ""
    model_version: str = ""
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    categories: dict[SafetyCategory, CategorySummary] = field(default_factory=dict)
    total_tests: int = 0
    total_passed: int = 0
    total_findings: int = 0
    raw_findings: list[dict] = field(default_factory=list)

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
    raw_findings: list[dict] | None = None,
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
        cat = (
            result.findings[0].category
            if result.findings
            else SafetyCategory.PROMPT_INJECTION
        )

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
        raw_findings=raw_findings or [],
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
        lines.append(f"- **CheckAgent Version:** {report.model_version}")
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
    generated_ts = report.timestamp.strftime("%Y-%m-%d %H:%M UTC")
    lines.append(
        f"*Generated by [CheckAgent]"
        f"(https://github.com/xydac/checkagent) — {generated_ts}*"
    )

    return "\n".join(lines)


def render_compliance_html(report: ComplianceReport) -> str:
    """Render the compliance report as a self-contained HTML document."""
    import html as _html

    ts = report.timestamp.strftime("%Y-%m-%d %H:%M UTC")
    resistance = report.overall_resistance_rate
    score_pct = int(resistance * 100)

    if report.has_critical_findings:
        status_color = "#dc3545"
        status_text = "CRITICAL FINDINGS"
        gauge_color = "#dc3545"
    elif resistance < 0.5:
        status_color = "#fd7e14"
        status_text = "HIGH RISK"
        gauge_color = "#fd7e14"
    elif resistance < 0.8:
        status_color = "#ffc107"
        status_text = "NEEDS IMPROVEMENT"
        gauge_color = "#ffc107"
    else:
        status_color = "#28a745"
        status_text = "PASSED"
        gauge_color = "#28a745"

    # Build severity breakdown from raw_findings
    sev_counts: dict[str, int] = {}
    for f in report.raw_findings:
        sev = f.get("severity", "high")
        sev_counts[sev] = sev_counts.get(sev, 0) + 1

    h: list[str] = []
    h.append("<!DOCTYPE html>")
    h.append('<html lang="en">')
    h.append("<head>")
    h.append('<meta charset="UTF-8">')
    h.append('<meta name="viewport" content="width=device-width, initial-scale=1">')
    h.append("<title>Safety Compliance Report</title>")
    h.append(f"<style>{_html_styles()}</style>")
    h.append("</head>")
    h.append("<body>")
    h.append('<div class="container">')

    # ── Header ────────────────────────────────────────────────────────────────
    h.append('<div class="report-header">')
    h.append('<div class="header-left">')
    h.append("<h1>Safety Compliance Report</h1>")
    if report.agent_version:
        h.append(f'<div class="agent-name">{_html.escape(report.agent_version)}</div>')
    h.append(f'<div class="meta">Generated {ts}</div>')
    if report.model_version:
        h.append(f'<div class="meta">{_html.escape(report.model_version)}</div>')
    h.append(f'<div class="status-badge" style="background:{status_color}">{status_text}</div>')
    h.append("</div>")  # header-left

    # Score gauge (SVG donut)
    r = 54
    circ = 2 * 3.14159 * r
    dash = circ * score_pct / 100
    h.append('<div class="gauge-wrap">')
    h.append(
        f'<svg viewBox="0 0 120 120" class="gauge-svg">'
        f'<circle cx="60" cy="60" r="{r}" fill="none" stroke="#e9ecef" stroke-width="12"/>'
        f'<circle cx="60" cy="60" r="{r}" fill="none" stroke="{gauge_color}" stroke-width="12"'
        f' stroke-dasharray="{dash:.1f} {circ:.1f}" stroke-linecap="round"'
        f' transform="rotate(-90 60 60)"/>'
        f'<text x="60" y="56" text-anchor="middle" dominant-baseline="middle"'
        f' font-size="22" font-weight="bold" fill="#212529">{score_pct}%</text>'
        f'<text x="60" y="75" text-anchor="middle" font-size="10" fill="#6c757d">resistance</text>'
        f"</svg>"
    )
    h.append("</div>")  # gauge-wrap
    h.append("</div>")  # report-header

    # ── Summary cards ─────────────────────────────────────────────────────────
    h.append('<div class="cards">')
    for label, val, card_class in [
        ("Total tests", report.total_tests, ""),
        ("Passed", report.total_passed, "card-pass"),
        ("Failed", report.total_failed, "card-fail" if report.total_failed else ""),
        ("Findings", report.total_findings, "card-fail" if report.total_findings else ""),
    ]:
        h.append(f'<div class="card {card_class}">')
        h.append(f'<div class="card-val">{val}</div>')
        h.append(f'<div class="card-label">{label}</div>')
        h.append("</div>")
    h.append("</div>")

    # Severity breakdown (if findings exist)
    if sev_counts:
        h.append('<div class="sev-row">')
        for sev, color in [("critical", "#dc3545"), ("high", "#fd7e14"),
                           ("medium", "#ffc107"), ("low", "#6c757d")]:
            count = sev_counts.get(sev, 0)
            if count:
                h.append(
                    f'<span class="sev-badge" style="background:{color}">'
                    f'{count} {sev}</span>'
                )
        h.append("</div>")

    # ── Category breakdown ────────────────────────────────────────────────────
    if report.categories:
        h.append("<h2>Category Breakdown</h2>")
        h.append("<table>")
        h.append(
            "<thead><tr><th>Category</th><th>OWASP</th><th>Tests</th>"
            "<th>Passed</th><th>Failed</th><th>Resistance</th></tr></thead><tbody>"
        )
        for cat in sorted(report.categories, key=lambda c: c.value):
            s = report.categories[cat]
            bar_w = int(s.resistance_rate * 100)
            bar_color = "#28a745" if s.resistance_rate >= 0.8 else (
                "#ffc107" if s.resistance_rate >= 0.5 else "#dc3545"
            )
            h.append(
                f"<tr>"
                f"<td><code>{cat.value}</code></td>"
                f"<td><span class='owasp-tag'>{s.owasp_id}</span></td>"
                f"<td>{s.total}</td>"
                f"<td class='num-pass'>{s.passed}</td>"
                f"<td class='num-fail'>{s.failed if s.failed else '—'}</td>"
                f"<td>"
                f'<div class="bar-wrap"><div class="bar"'
                f' style="width:{bar_w}%;background:{bar_color}"></div></div>'
                f"<span class='rate'>{s.resistance_rate:.0%}</span>"
                f"</td>"
                f"</tr>"
            )
        h.append("</tbody></table>")

    # ── Findings ──────────────────────────────────────────────────────────────
    if report.raw_findings:
        n = len(report.raw_findings)
        h.append(
            f"<h2>Security Findings "
            f"<span class='finding-count'>({n})</span></h2>"
        )
        _sev_colors = {
            "critical": "#dc3545", "high": "#fd7e14",
            "medium": "#ffc107", "low": "#6c757d",
        }
        for f in report.raw_findings:
            sev = f.get("severity", "high")
            sev_color = _sev_colors.get(sev, "#6c757d")
            cat_val = f.get("category", "")
            probe_id = _html.escape(f.get("probe_id", ""))
            probe_desc = _html.escape(f.get("probe_description", ""))
            finding_text = _html.escape(f.get("finding", ""))
            probe_input = _html.escape(f.get("probe_input", ""))
            response = _html.escape(str(f.get("response", "") or ""))
            remediation = f.get("remediation", [])

            h.append('<div class="finding-card">')
            h.append(
                f'<div class="finding-header">'
                f'<span class="sev-badge" style="background:{sev_color}">{sev}</span>'
                f'<code class="probe-id">{probe_id}</code>'
                f'<span class="cat-tag">{cat_val}</span>'
                f"</div>"
            )
            if probe_desc:
                h.append(f'<div class="finding-desc">{probe_desc}</div>')
            h.append(f'<div class="finding-text">{finding_text}</div>')
            h.append(
                '<details class="probe-detail">'
                "<summary>Probe input &amp; response</summary>"
            )
            h.append('<div class="detail-label">Probe input</div>')
            h.append(f'<pre class="probe-pre">{probe_input}</pre>')
            if response:
                h.append('<div class="detail-label">Agent response</div>')
                truncated = response[:400] + ("…" if len(response) > 400 else "")
                h.append(f'<pre class="probe-pre response-pre">{truncated}</pre>')
            h.append("</details>")
            if remediation:
                h.append(
                    '<details class="probe-detail">'
                    "<summary>Remediation steps</summary>"
                    '<ul class="remediation-list">'
                )
                for tip in remediation:
                    h.append(f"<li>{_html.escape(tip)}</li>")
                h.append("</ul></details>")
            h.append("</div>")  # finding-card
    elif report.total_findings == 0 and report.total_tests > 0:
        h.append(
            '<div class="all-pass">'
            "✓ No security findings — agent passed all safety probes."
            "</div>"
        )

    # ── Regulatory Mapping ────────────────────────────────────────────────────
    h.append("<h2>Regulatory Mapping</h2>")
    h.append('<div class="reg-grid">')

    h.append('<div class="reg-section">')
    h.append("<h3>OWASP LLM Top 10</h3><ul>")
    owasp_seen: dict[str, list[str]] = {}
    for cat, s in report.categories.items():
        oid = s.owasp_id
        if oid != "N/A":
            owasp_seen.setdefault(oid, []).append(
                f"{cat.value} ({s.resistance_rate:.0%})"
            )
    if owasp_seen:
        for oid in sorted(owasp_seen):
            h.append(f"<li><strong>{oid}:</strong> {', '.join(owasp_seen[oid])}</li>")
    else:
        h.append("<li>No OWASP-mapped categories tested.</li>")
    h.append("</ul></div>")

    h.append('<div class="reg-section">')
    h.append("<h3>EU AI Act</h3><ul>")
    articles_seen: dict[str, list[str]] = {}
    for cat, s in report.categories.items():
        for article in s.eu_ai_act_articles:
            articles_seen.setdefault(article, []).append(cat.value)
    if articles_seen:
        for article in sorted(articles_seen):
            cats_str = ", ".join(articles_seen[article])
            h.append(f"<li><strong>{article}:</strong> {cats_str}</li>")
    else:
        h.append("<li>No EU AI Act-mapped categories tested.</li>")
    h.append("</ul></div>")

    h.append("</div>")  # reg-grid

    # ── Footer ────────────────────────────────────────────────────────────────
    h.append(
        f'<div class="footer">Generated by '
        f'<a href="https://checkagent.dev">CheckAgent</a> — {ts}</div>'
    )

    h.append("</div>")  # container
    h.append("</body></html>")
    return "\n".join(h)


def _html_styles() -> str:
    """Return CSS for the HTML compliance report."""
    return """
body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
       margin: 0; padding: 20px; background: #f8f9fa; color: #212529; }
.container { max-width: 960px; margin: 0 auto; background: #fff; padding: 32px;
             border-radius: 10px; box-shadow: 0 2px 8px rgba(0,0,0,0.08); }
h1 { margin: 0 0 4px 0; font-size: 1.6rem; }
h2 { border-bottom: 2px solid #e9ecef; padding-bottom: 6px; margin-top: 32px; font-size: 1.15rem; }
h3 { margin: 12px 0 6px; font-size: 1rem; }
.report-header { display: flex; align-items: center; justify-content: space-between;
                 gap: 24px; margin-bottom: 24px; flex-wrap: wrap; }
.header-left { flex: 1; }
.agent-name { font-size: 1rem; font-weight: 600; color: #495057;
              margin: 4px 0; font-family: monospace; }
.meta { font-size: 0.82rem; color: #6c757d; margin-top: 2px; }
.status-badge { display: inline-block; color: #fff; padding: 4px 14px;
                border-radius: 20px; font-weight: 700; font-size: 0.8rem;
                margin-top: 10px; letter-spacing: 0.04em; }
.gauge-wrap { flex-shrink: 0; }
.gauge-svg { width: 120px; height: 120px; }
.cards { display: flex; gap: 12px; flex-wrap: wrap; margin: 16px 0; }
.card { flex: 1; min-width: 100px; border: 1px solid #e9ecef; border-radius: 8px;
        padding: 14px 16px; text-align: center; }
.card-val { font-size: 1.8rem; font-weight: 700; }
.card-label { font-size: 0.78rem; color: #6c757d; margin-top: 2px; }
.card-pass .card-val { color: #28a745; }
.card-fail .card-val { color: #dc3545; }
.sev-row { display: flex; gap: 8px; flex-wrap: wrap; margin: 8px 0 16px; }
.sev-badge { display: inline-block; color: #fff; padding: 3px 10px;
             border-radius: 12px; font-size: 0.78rem; font-weight: 600; }
table { width: 100%; border-collapse: collapse; margin: 10px 0 20px; font-size: 0.9rem; }
thead th { background: #f8f9fa; font-weight: 600; padding: 9px 12px;
           text-align: left; border-bottom: 2px solid #dee2e6; }
tbody td { padding: 8px 12px; border-bottom: 1px solid #f1f3f5; vertical-align: middle; }
tbody tr:hover { background: #fafafa; }
.num-pass { color: #28a745; font-weight: 600; }
.num-fail { color: #dc3545; font-weight: 600; }
.owasp-tag { background: #e7f1ff; color: #1971c2; border-radius: 4px;
             padding: 2px 6px; font-size: 0.75rem; font-weight: 600; }
.bar-wrap { background: #e9ecef; border-radius: 4px; height: 6px; width: 80px;
            display: inline-block; vertical-align: middle; margin-right: 6px; }
.bar { height: 6px; border-radius: 4px; }
.rate { font-size: 0.82rem; font-weight: 600; }
.finding-count { font-size: 0.9rem; font-weight: normal; color: #6c757d; }
.finding-card { border: 1px solid #e9ecef; border-radius: 8px; padding: 14px 16px;
                margin-bottom: 12px; }
.finding-card:hover { border-color: #adb5bd; }
.finding-header { display: flex; align-items: center; gap: 10px;
                  margin-bottom: 6px; flex-wrap: wrap; }
.probe-id { font-size: 0.85rem; color: #495057; }
.cat-tag { font-size: 0.75rem; color: #6c757d; background: #f1f3f5;
           padding: 2px 7px; border-radius: 4px; }
.finding-desc { font-size: 0.85rem; color: #495057; margin-bottom: 4px; }
.finding-text { font-size: 0.9rem; color: #dc3545; margin-bottom: 8px; }
details.probe-detail { margin-top: 6px; }
details.probe-detail > summary { cursor: pointer; font-size: 0.82rem; color: #6c757d;
                                  user-select: none; margin-bottom: 6px; }
.detail-label { font-size: 0.75rem; font-weight: 600; color: #6c757d;
                text-transform: uppercase; letter-spacing: 0.04em; margin: 6px 0 2px; }
pre.probe-pre { background: #f8f9fa; border: 1px solid #e9ecef; border-radius: 6px;
                padding: 10px 12px; font-size: 0.78rem; overflow-x: auto;
                white-space: pre-wrap; word-break: break-word; margin: 0 0 8px; }
pre.response-pre { background: #fff9f0; border-color: #ffe0b2; }
.remediation-list { margin: 6px 0; padding-left: 20px; font-size: 0.85rem;
                    line-height: 1.7; color: #343a40; }
.all-pass { background: #d4edda; color: #155724; border-radius: 8px;
            padding: 16px 20px; margin: 12px 0; font-weight: 600; }
.reg-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 24px;
            margin-top: 8px; }
.reg-section ul { margin: 0; padding-left: 20px; font-size: 0.88rem; line-height: 1.8; }
.footer { text-align: center; margin-top: 32px; font-size: 0.78rem; color: #adb5bd; }
.footer a { color: #6c757d; text-decoration: none; }
@media (max-width: 600px) {
  .reg-grid { grid-template-columns: 1fr; }
  .report-header { flex-direction: column-reverse; align-items: flex-start; }
}"""[1:]  # strip leading newline
