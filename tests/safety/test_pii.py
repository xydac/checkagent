"""Tests for the PIILeakageScanner (F11.2)."""

from checkagent.safety.pii import PIILeakageScanner
from checkagent.safety.taxonomy import SafetyCategory, Severity


class TestPIILeakageScanner:
    def setup_method(self):
        self.scanner = PIILeakageScanner()

    # -- Clean outputs --

    def test_clean_output_passes(self):
        result = self.scanner.evaluate("The project deadline is next Friday.")
        assert result.passed
        assert result.finding_count == 0

    def test_empty_string_passes(self):
        result = self.scanner.evaluate("")
        assert result.passed

    # -- Email detection --

    def test_detects_email(self):
        result = self.scanner.evaluate("Contact us at john.doe@example.com for help.")
        assert not result.passed
        assert any(f.probe == "email" for f in result.findings)

    def test_detects_multiple_emails(self):
        text = "Send to alice@test.com and bob@test.com"
        result = self.scanner.evaluate(text)
        assert result.finding_count == 2

    # -- Phone detection --

    def test_detects_us_phone(self):
        result = self.scanner.evaluate("Call me at 555-123-4567.")
        assert not result.passed
        assert any(f.probe == "phone_us" for f in result.findings)

    def test_detects_phone_with_parens(self):
        result = self.scanner.evaluate("Phone: (555) 123-4567")
        assert not result.passed

    def test_detects_phone_with_country_code(self):
        result = self.scanner.evaluate("Reach us at +1-555-123-4567")
        assert not result.passed

    # -- SSN detection --

    def test_detects_ssn(self):
        result = self.scanner.evaluate("SSN: 123-45-6789")
        assert not result.passed
        assert any(f.probe == "ssn" for f in result.findings)
        assert any(f.severity == Severity.CRITICAL for f in result.findings)

    def test_detects_ssn_no_dashes(self):
        result = self.scanner.evaluate("Social: 123456789")
        assert not result.passed

    # -- Credit card detection --

    def test_detects_credit_card(self):
        result = self.scanner.evaluate("Card: 4111-1111-1111-1111")
        assert not result.passed
        assert any(f.probe == "credit_card" for f in result.findings)
        assert any(f.severity == Severity.CRITICAL for f in result.findings)

    def test_detects_credit_card_spaces(self):
        result = self.scanner.evaluate("Card: 4111 1111 1111 1111")
        assert not result.passed

    # -- IPv4 detection --

    def test_detects_ipv4(self):
        result = self.scanner.evaluate("Server at 192.168.1.100")
        assert not result.passed
        assert any(f.probe == "ipv4" for f in result.findings)

    def test_ignores_invalid_ipv4(self):
        result = self.scanner.evaluate("Version 999.999.999.999 is not an IP")
        assert result.passed

    # -- Category and evaluator --

    def test_findings_category_is_pii(self):
        result = self.scanner.evaluate("Email: test@test.com")
        assert all(f.category == SafetyCategory.PII_LEAKAGE for f in result.findings)

    def test_evaluator_name(self):
        result = self.scanner.evaluate("clean")
        assert result.evaluator == "pii_leakage_scanner"

    # -- Deduplication --

    def test_duplicate_evidence_deduplicated(self):
        text = "Email test@test.com and again test@test.com"
        result = self.scanner.evaluate(text)
        evidences = [f.evidence for f in result.findings]
        assert len(evidences) == len(set(evidences))

    # -- Disabling patterns --

    def test_disable_email_pattern(self):
        scanner = PIILeakageScanner(disabled={"email"})
        result = scanner.evaluate("Contact john@test.com")
        assert result.passed

    def test_disable_does_not_affect_other_patterns(self):
        scanner = PIILeakageScanner(disabled={"email"})
        result = scanner.evaluate("SSN: 123-45-6789")
        assert not result.passed

    # -- Custom patterns --

    def test_custom_pattern(self):
        self.scanner.add_pattern(
            "passport",
            r"\b[A-Z]\d{8}\b",
            Severity.HIGH,
        )
        result = self.scanner.evaluate("Passport: A12345678")
        assert not result.passed
        assert result.findings[0].probe == "passport"

    # -- Mixed PII --

    def test_multiple_pii_types(self):
        text = "User john@test.com, SSN 123-45-6789, phone 555-123-4567"
        result = self.scanner.evaluate(text)
        assert not result.passed
        probes = {f.probe for f in result.findings}
        assert "email" in probes
        assert "ssn" in probes
        assert "phone_us" in probes
