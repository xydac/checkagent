"""Tests for the ap_safety fixture."""

from checkagent.safety.injection import PromptInjectionDetector
from checkagent.safety.pii import PIILeakageScanner


def test_ap_safety_fixture_provides_injection(ap_safety):
    assert isinstance(ap_safety["injection"], PromptInjectionDetector)


def test_ap_safety_fixture_provides_pii(ap_safety):
    assert isinstance(ap_safety["pii"], PIILeakageScanner)


def test_ap_safety_injection_works(ap_safety):
    result = ap_safety["injection"].evaluate("Normal output")
    assert result.passed


def test_ap_safety_pii_detects_email(ap_safety):
    result = ap_safety["pii"].evaluate("Contact test@example.com")
    assert not result.passed
