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
            "data_enumeration",
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

    def test_direct_comparison_lt(self):
        """F-023: Severity enum should support direct < comparison."""
        assert Severity.LOW < Severity.MEDIUM
        assert Severity.MEDIUM < Severity.HIGH
        assert Severity.HIGH < Severity.CRITICAL

    def test_direct_comparison_ge(self):
        """F-023: Severity enum should support direct >= comparison."""
        assert Severity.CRITICAL >= Severity.HIGH
        assert Severity.HIGH >= Severity.HIGH
        assert Severity.HIGH >= Severity.MEDIUM
        assert not (Severity.LOW >= Severity.MEDIUM)

    def test_direct_comparison_le(self):
        assert Severity.LOW <= Severity.LOW
        assert Severity.LOW <= Severity.CRITICAL
        assert not (Severity.CRITICAL <= Severity.HIGH)

    def test_severity_equality_with_string(self):
        """Severity should still compare equal to its string value."""
        assert Severity.LOW == "low"
        assert Severity.CRITICAL == "critical"

    def test_severity_hashable(self):
        """Severity values should be usable as dict keys and set members."""
        s = {Severity.LOW, Severity.HIGH}
        assert Severity.LOW in s
        assert len(s) == 2


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
