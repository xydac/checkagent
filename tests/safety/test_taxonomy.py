"""Tests for the safety taxonomy (F11.1)."""

from checkagent.safety.taxonomy import (
    OWASP_MAPPING,
    SEVERITY_ORDER,
    SafetyCategory,
    Severity,
    severity_meets_threshold,
)


class TestSafetyCategory:
    def test_all_categories_present(self):
        expected = {
            "prompt_injection",
            "jailbreak",
            "pii_leakage",
            "tool_misuse",
            "groundedness",
            "harmful_content",
            "off_topic",
        }
        assert {c.value for c in SafetyCategory} == expected

    def test_string_value_matches_enum_name(self):
        for cat in SafetyCategory:
            assert cat.value == cat.name.lower()

    def test_category_is_string_enum(self):
        assert isinstance(SafetyCategory.PROMPT_INJECTION, str)
        assert SafetyCategory.PROMPT_INJECTION == "prompt_injection"


class TestSeverity:
    def test_all_severities_present(self):
        assert {s.value for s in Severity} == {"low", "medium", "high", "critical"}

    def test_severity_ordering(self):
        assert SEVERITY_ORDER[Severity.LOW] < SEVERITY_ORDER[Severity.MEDIUM]
        assert SEVERITY_ORDER[Severity.MEDIUM] < SEVERITY_ORDER[Severity.HIGH]
        assert SEVERITY_ORDER[Severity.HIGH] < SEVERITY_ORDER[Severity.CRITICAL]


class TestSeverityThreshold:
    def test_critical_meets_all_thresholds(self):
        for threshold in Severity:
            assert severity_meets_threshold(Severity.CRITICAL, threshold)

    def test_low_only_meets_low(self):
        assert severity_meets_threshold(Severity.LOW, Severity.LOW)
        assert not severity_meets_threshold(Severity.LOW, Severity.MEDIUM)
        assert not severity_meets_threshold(Severity.LOW, Severity.HIGH)
        assert not severity_meets_threshold(Severity.LOW, Severity.CRITICAL)

    def test_medium_meets_medium_and_below(self):
        assert severity_meets_threshold(Severity.MEDIUM, Severity.LOW)
        assert severity_meets_threshold(Severity.MEDIUM, Severity.MEDIUM)
        assert not severity_meets_threshold(Severity.MEDIUM, Severity.HIGH)


class TestOWASPMapping:
    def test_all_categories_mapped(self):
        for cat in SafetyCategory:
            assert cat in OWASP_MAPPING

    def test_prompt_injection_maps_to_llm01(self):
        assert OWASP_MAPPING[SafetyCategory.PROMPT_INJECTION] == "LLM01"
        assert OWASP_MAPPING[SafetyCategory.JAILBREAK] == "LLM01"

    def test_pii_maps_to_llm06(self):
        assert OWASP_MAPPING[SafetyCategory.PII_LEAKAGE] == "LLM06"

    def test_tool_misuse_maps_to_llm08(self):
        assert OWASP_MAPPING[SafetyCategory.TOOL_MISUSE] == "LLM08"
