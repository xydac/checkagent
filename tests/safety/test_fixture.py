"""Tests for the ca_safety fixture."""

from checkagent.safety.injection import PromptInjectionDetector
from checkagent.safety.pii import PIILeakageScanner
from checkagent.safety.refusal import RefusalComplianceChecker
from checkagent.safety.system_prompt import SystemPromptLeakDetector
from checkagent.safety.tool_boundary import ToolCallBoundaryValidator


def test_ca_safety_fixture_provides_injection(ca_safety):
    assert isinstance(ca_safety["injection"], PromptInjectionDetector)


def test_ca_safety_fixture_provides_pii(ca_safety):
    assert isinstance(ca_safety["pii"], PIILeakageScanner)


def test_ca_safety_fixture_provides_system_prompt(ca_safety):
    assert isinstance(ca_safety["system_prompt"], SystemPromptLeakDetector)


def test_ca_safety_fixture_provides_tool_boundary(ca_safety):
    assert isinstance(ca_safety["tool_boundary"], ToolCallBoundaryValidator)


def test_ca_safety_fixture_provides_refusal(ca_safety):
    assert isinstance(ca_safety["refusal"], RefusalComplianceChecker)


def test_ca_safety_injection_works(ca_safety):
    result = ca_safety["injection"].evaluate("Normal output")
    assert result.passed


def test_ca_safety_pii_detects_email(ca_safety):
    result = ca_safety["pii"].evaluate("Contact test@example.com")
    assert not result.passed
