"""Tests for the ap_safety fixture."""

from checkagent.safety.injection import PromptInjectionDetector
from checkagent.safety.pii import PIILeakageScanner
from checkagent.safety.refusal import RefusalComplianceChecker
from checkagent.safety.system_prompt import SystemPromptLeakDetector
from checkagent.safety.tool_boundary import ToolCallBoundaryValidator


def test_ap_safety_fixture_provides_injection(ap_safety):
    assert isinstance(ap_safety["injection"], PromptInjectionDetector)


def test_ap_safety_fixture_provides_pii(ap_safety):
    assert isinstance(ap_safety["pii"], PIILeakageScanner)


def test_ap_safety_fixture_provides_system_prompt(ap_safety):
    assert isinstance(ap_safety["system_prompt"], SystemPromptLeakDetector)


def test_ap_safety_fixture_provides_tool_boundary(ap_safety):
    assert isinstance(ap_safety["tool_boundary"], ToolCallBoundaryValidator)


def test_ap_safety_fixture_provides_refusal(ap_safety):
    assert isinstance(ap_safety["refusal"], RefusalComplianceChecker)


def test_ap_safety_injection_works(ap_safety):
    result = ap_safety["injection"].evaluate("Normal output")
    assert result.passed


def test_ap_safety_pii_detects_email(ap_safety):
    result = ap_safety["pii"].evaluate("Contact test@example.com")
    assert not result.passed
