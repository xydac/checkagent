"""Tests for safety compliance report generation (F11.4)."""

from __future__ import annotations

import json
from datetime import datetime, timezone

from checkagent.safety.compliance import (
    EU_AI_ACT_MAPPING,
    CategorySummary,
    ComplianceReport,
    generate_compliance_report,
    render_compliance_html,
    render_compliance_json,
    render_compliance_markdown,
)
from checkagent.safety.evaluator import SafetyFinding, SafetyResult
from checkagent.safety.taxonomy import SafetyCategory, Severity

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

FIXED_TS = datetime(2026, 4, 5, 12, 0, 0, tzinfo=timezone.utc)


def _result(
    passed: bool,
    category: SafetyCategory = SafetyCategory.PROMPT_INJECTION,
    severity: Severity = Severity.HIGH,
    *,
    evaluator: str = "test_eval",
    n_findings: int | None = None,
) -> SafetyResult:
    """Create a SafetyResult with optional findings.

    Always includes at least one finding so the category is unambiguous.
    Use ``n_findings=0`` to explicitly create a result with no findings.
    """
    if n_findings is None:
        n_findings = 0 if passed else 1
    findings = [
        SafetyFinding(
            category=category,
            severity=severity,
            description=f"finding-{i}",
        )
        for i in range(n_findings)
    ]
    return SafetyResult(passed=passed, findings=findings, evaluator=evaluator)


# ---------------------------------------------------------------------------
# CategorySummary
# ---------------------------------------------------------------------------


class TestCategorySummary:
    def test_resistance_rate_all_passed(self):
        s = CategorySummary(
            category=SafetyCategory.PROMPT_INJECTION, total=10, passed=10
        )
        assert s.resistance_rate == 1.0

    def test_resistance_rate_none_passed(self):
        s = CategorySummary(
            category=SafetyCategory.PROMPT_INJECTION, total=5, passed=0, failed=5
        )
        assert s.resistance_rate == 0.0

    def test_resistance_rate_mixed(self):
        s = CategorySummary(
            category=SafetyCategory.PII_LEAKAGE, total=4, passed=3, failed=1
        )
        assert s.resistance_rate == 0.75

    def test_resistance_rate_empty(self):
        s = CategorySummary(category=SafetyCategory.JAILBREAK)
        assert s.resistance_rate == 0.0

    def test_owasp_id(self):
        s = CategorySummary(category=SafetyCategory.PII_LEAKAGE)
        assert s.owasp_id == "LLM06"

    def test_eu_ai_act_articles(self):
        s = CategorySummary(category=SafetyCategory.PROMPT_INJECTION)
        assert "Article 9 (Risk Management)" in s.eu_ai_act_articles

    def test_eu_ai_act_unknown_category(self):
        # OFF_TOPIC maps to Article 15
        s = CategorySummary(category=SafetyCategory.OFF_TOPIC)
        assert s.eu_ai_act_articles == ["Article 15 (Robustness)"]


# ---------------------------------------------------------------------------
# ComplianceReport properties
# ---------------------------------------------------------------------------


class TestComplianceReport:
    def test_total_failed(self):
        r = ComplianceReport(total_tests=10, total_passed=7)
        assert r.total_failed == 3

    def test_overall_resistance_rate(self):
        r = ComplianceReport(total_tests=8, total_passed=6)
        assert r.overall_resistance_rate == 0.75

    def test_overall_resistance_rate_empty(self):
        r = ComplianceReport()
        assert r.overall_resistance_rate == 0.0

    def test_has_critical_findings_true(self):
        r = ComplianceReport(
            categories={
                SafetyCategory.PROMPT_INJECTION: CategorySummary(
                    category=SafetyCategory.PROMPT_INJECTION,
                    findings_by_severity={Severity.CRITICAL: 1},
                )
            }
        )
        assert r.has_critical_findings is True

    def test_has_critical_findings_false(self):
        r = ComplianceReport(
            categories={
                SafetyCategory.PII_LEAKAGE: CategorySummary(
                    category=SafetyCategory.PII_LEAKAGE,
                    findings_by_severity={Severity.HIGH: 2},
                )
            }
        )
        assert r.has_critical_findings is False

    def test_to_dict_schema(self):
        r = ComplianceReport(
            agent_version="1.0.0",
            model_version="gpt-4",
            timestamp=FIXED_TS,
            total_tests=5,
            total_passed=3,
            total_findings=2,
        )
        d = r.to_dict()
        assert d["report_type"] == "checkagent_compliance"
        assert d["schema_version"] == "1.0"
        assert d["agent_version"] == "1.0.0"
        assert d["model_version"] == "gpt-4"
        assert d["summary"]["total_tests"] == 5
        assert d["summary"]["total_passed"] == 3
        assert d["summary"]["total_failed"] == 2
        assert d["summary"]["overall_resistance_rate"] == 0.6


# ---------------------------------------------------------------------------
# generate_compliance_report
# ---------------------------------------------------------------------------


class TestGenerateReport:
    def test_empty_results(self):
        report = generate_compliance_report([], timestamp=FIXED_TS)
        assert report.total_tests == 0
        assert report.total_passed == 0
        assert report.total_findings == 0
        assert report.categories == {}

    def test_all_passed(self):
        results = [
            _result(True, SafetyCategory.PROMPT_INJECTION),
            _result(True, SafetyCategory.PROMPT_INJECTION),
        ]
        report = generate_compliance_report(results, timestamp=FIXED_TS)
        assert report.total_tests == 2
        assert report.total_passed == 2
        assert report.overall_resistance_rate == 1.0

    def test_all_failed(self):
        results = [
            _result(False, SafetyCategory.PII_LEAKAGE, Severity.HIGH),
            _result(False, SafetyCategory.PII_LEAKAGE, Severity.CRITICAL),
        ]
        report = generate_compliance_report(results, timestamp=FIXED_TS)
        assert report.total_tests == 2
        assert report.total_passed == 0
        assert report.has_critical_findings is True
        cat = report.categories[SafetyCategory.PII_LEAKAGE]
        assert cat.failed == 2
        assert cat.findings_by_severity[Severity.HIGH] == 1
        assert cat.findings_by_severity[Severity.CRITICAL] == 1

    def test_mixed_categories(self):
        results = [
            _result(True, SafetyCategory.PROMPT_INJECTION, n_findings=1),
            _result(False, SafetyCategory.PII_LEAKAGE, Severity.MEDIUM),
            _result(True, SafetyCategory.JAILBREAK, n_findings=1),
            _result(False, SafetyCategory.TOOL_MISUSE, Severity.HIGH),
        ]
        report = generate_compliance_report(
            results,
            agent_version="2.0",
            model_version="claude-3",
            timestamp=FIXED_TS,
        )
        assert report.total_tests == 4
        assert report.total_passed == 2
        assert len(report.categories) == 4
        assert report.agent_version == "2.0"
        assert report.model_version == "claude-3"

    def test_multiple_findings_per_result(self):
        results = [
            _result(False, SafetyCategory.PROMPT_INJECTION, Severity.HIGH, n_findings=3),
        ]
        report = generate_compliance_report(results, timestamp=FIXED_TS)
        assert report.total_findings == 3
        cat = report.categories[SafetyCategory.PROMPT_INJECTION]
        assert cat.findings_by_severity[Severity.HIGH] == 3

    def test_version_tags_propagate(self):
        report = generate_compliance_report(
            [],
            agent_version="v1.2.3",
            model_version="gpt-4-turbo",
            timestamp=FIXED_TS,
        )
        assert report.agent_version == "v1.2.3"
        assert report.model_version == "gpt-4-turbo"
        assert report.timestamp == FIXED_TS

    def test_cross_category_findings(self):
        """A result with findings in a different category than the primary."""
        finding_pii = SafetyFinding(
            category=SafetyCategory.PII_LEAKAGE,
            severity=Severity.HIGH,
            description="leaked PII",
        )
        finding_injection = SafetyFinding(
            category=SafetyCategory.PROMPT_INJECTION,
            severity=Severity.CRITICAL,
            description="injection found",
        )
        result = SafetyResult(
            passed=False,
            findings=[finding_pii, finding_injection],
            evaluator="combined",
        )
        report = generate_compliance_report([result], timestamp=FIXED_TS)
        # Primary category is PII_LEAKAGE (first finding)
        assert SafetyCategory.PII_LEAKAGE in report.categories
        pii_cat = report.categories[SafetyCategory.PII_LEAKAGE]
        assert pii_cat.total == 1
        assert pii_cat.failed == 1
        # The injection finding should also appear in PROMPT_INJECTION severity counts
        assert SafetyCategory.PROMPT_INJECTION in report.categories
        inj_cat = report.categories[SafetyCategory.PROMPT_INJECTION]
        assert inj_cat.findings_by_severity.get(Severity.CRITICAL, 0) == 1


# ---------------------------------------------------------------------------
# JSON rendering
# ---------------------------------------------------------------------------


class TestRenderJson:
    def test_valid_json(self):
        report = generate_compliance_report(
            [_result(True), _result(False)],
            agent_version="1.0",
            timestamp=FIXED_TS,
        )
        text = render_compliance_json(report)
        data = json.loads(text)
        assert data["report_type"] == "checkagent_compliance"
        assert data["summary"]["total_tests"] == 2

    def test_empty_report_json(self):
        report = generate_compliance_report([], timestamp=FIXED_TS)
        data = json.loads(render_compliance_json(report))
        assert data["summary"]["total_tests"] == 0
        assert data["categories"] == {}


# ---------------------------------------------------------------------------
# Markdown rendering
# ---------------------------------------------------------------------------


class TestRenderMarkdown:
    def test_contains_title(self):
        report = generate_compliance_report([], timestamp=FIXED_TS)
        md = render_compliance_markdown(report)
        assert "# Safety Compliance Report" in md

    def test_contains_metadata(self):
        report = generate_compliance_report(
            [],
            agent_version="1.0.0",
            model_version="gpt-4",
            timestamp=FIXED_TS,
        )
        md = render_compliance_markdown(report)
        assert "Agent Version:** 1.0.0" in md
        assert "Model Version:** gpt-4" in md

    def test_contains_summary_table(self):
        report = generate_compliance_report(
            [_result(True), _result(False, severity=Severity.MEDIUM)],
            timestamp=FIXED_TS,
        )
        md = render_compliance_markdown(report)
        assert "Total safety tests" in md
        assert "| 2 |" in md
        assert "resistance rate" in md.lower()

    def test_contains_category_breakdown(self):
        results = [
            _result(True, SafetyCategory.PROMPT_INJECTION),
            _result(False, SafetyCategory.PII_LEAKAGE),
        ]
        report = generate_compliance_report(results, timestamp=FIXED_TS)
        md = render_compliance_markdown(report)
        assert "prompt_injection" in md
        assert "pii_leakage" in md
        assert "LLM01" in md
        assert "LLM06" in md

    def test_contains_regulatory_mapping(self):
        results = [_result(True, SafetyCategory.PROMPT_INJECTION)]
        report = generate_compliance_report(results, timestamp=FIXED_TS)
        md = render_compliance_markdown(report)
        assert "OWASP LLM Top 10" in md
        assert "EU AI Act" in md
        assert "Article 9" in md

    def test_severity_section(self):
        results = [
            _result(False, severity=Severity.CRITICAL),
            _result(False, severity=Severity.HIGH),
        ]
        report = generate_compliance_report(results, timestamp=FIXED_TS)
        md = render_compliance_markdown(report)
        assert "CRITICAL" in md
        assert "HIGH" in md

    def test_footer(self):
        report = generate_compliance_report([], timestamp=FIXED_TS)
        md = render_compliance_markdown(report)
        assert "CheckAgent" in md
        assert "2026-04-05" in md


# ---------------------------------------------------------------------------
# HTML rendering
# ---------------------------------------------------------------------------


class TestRenderHtml:
    def test_valid_html_structure(self):
        report = generate_compliance_report(
            [_result(True)], timestamp=FIXED_TS
        )
        html = render_compliance_html(report)
        assert "<!DOCTYPE html>" in html
        assert "<html" in html
        assert "</html>" in html
        assert "Safety Compliance Report" in html

    def test_status_passed(self):
        report = generate_compliance_report(
            [_result(True)], timestamp=FIXED_TS
        )
        html = render_compliance_html(report)
        assert "PASSED" in html
        assert "#28a745" in html  # green

    def test_status_critical(self):
        report = generate_compliance_report(
            [_result(False, severity=Severity.CRITICAL)],
            timestamp=FIXED_TS,
        )
        html = render_compliance_html(report)
        assert "CRITICAL FINDINGS" in html
        assert "#dc3545" in html  # red

    def test_contains_category_table(self):
        results = [
            _result(True, SafetyCategory.JAILBREAK, n_findings=1),
            _result(False, SafetyCategory.PII_LEAKAGE),
        ]
        report = generate_compliance_report(results, timestamp=FIXED_TS)
        html = render_compliance_html(report)
        assert "jailbreak" in html
        assert "pii_leakage" in html

    def test_contains_version_info(self):
        report = generate_compliance_report(
            [],
            agent_version="v2.0",
            model_version="claude-opus",
            timestamp=FIXED_TS,
        )
        html = render_compliance_html(report)
        assert "v2.0" in html
        assert "claude-opus" in html

    def test_contains_css(self):
        report = generate_compliance_report([], timestamp=FIXED_TS)
        html = render_compliance_html(report)
        assert "<style>" in html
        assert "font-family" in html


# ---------------------------------------------------------------------------
# EU AI Act mapping
# ---------------------------------------------------------------------------


class TestEuAiActMapping:
    def test_all_categories_mapped(self):
        """Every SafetyCategory should have an EU AI Act mapping."""
        for cat in SafetyCategory:
            assert cat in EU_AI_ACT_MAPPING, f"Missing EU AI Act mapping for {cat}"

    def test_prompt_injection_maps_to_article_9(self):
        articles = EU_AI_ACT_MAPPING[SafetyCategory.PROMPT_INJECTION]
        assert any("Article 9" in a for a in articles)

    def test_pii_maps_to_data_governance(self):
        articles = EU_AI_ACT_MAPPING[SafetyCategory.PII_LEAKAGE]
        assert any("Article 10" in a for a in articles)
