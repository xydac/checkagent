"""Tests for PII scrubbing utility."""

from __future__ import annotations

import pytest

from checkagent.trace_import.pii import PiiScrubber


@pytest.fixture
def scrubber():
    return PiiScrubber()


class TestPiiScrubber:
    def test_email_scrubbed(self, scrubber):
        text = "Contact john@example.com for details"
        result = scrubber.scrub_text(text)
        assert "john@example.com" not in result
        assert "<EMAIL_1>" in result

    def test_phone_scrubbed(self, scrubber):
        text = "Call me at 555-123-4567"
        result = scrubber.scrub_text(text)
        assert "555-123-4567" not in result
        assert "<PHONE_1>" in result

    def test_ssn_scrubbed(self, scrubber):
        text = "SSN: 123-45-6789"
        result = scrubber.scrub_text(text)
        assert "123-45-6789" not in result
        assert "<SSN_1>" in result

    def test_credit_card_scrubbed(self, scrubber):
        text = "Card: 4111-1111-1111-1111"
        result = scrubber.scrub_text(text)
        assert "4111-1111-1111-1111" not in result
        assert "<CREDIT_CARD_1>" in result

    def test_ip_address_scrubbed(self, scrubber):
        text = "Server at 192.168.1.100"
        result = scrubber.scrub_text(text)
        assert "192.168.1.100" not in result
        assert "<IP_ADDR_1>" in result

    def test_deterministic_placeholders(self, scrubber):
        text = "Email john@example.com and john@example.com again"
        result = scrubber.scrub_text(text)
        # Same email gets same placeholder
        assert result.count("<EMAIL_1>") == 2

    def test_different_values_get_different_placeholders(self, scrubber):
        text = "Email john@example.com and jane@example.com"
        result = scrubber.scrub_text(text)
        assert "<EMAIL_1>" in result
        assert "<EMAIL_2>" in result

    def test_reset_counters(self, scrubber):
        scrubber.scrub_text("john@example.com")
        scrubber.reset()
        result = scrubber.scrub_text("jane@example.com")
        # After reset, counter starts over
        assert "<EMAIL_1>" in result

    def test_empty_text(self, scrubber):
        assert scrubber.scrub_text("") == ""

    def test_no_pii(self, scrubber):
        text = "This is a clean sentence"
        assert scrubber.scrub_text(text) == text

    def test_scrub_value_dict(self, scrubber):
        data = {
            "email": "john@example.com",
            "name": "John",
            "nested": {"phone": "555-123-4567"},
        }
        result = scrubber.scrub_value(data)
        assert result["email"] == "<EMAIL_1>"
        assert result["name"] == "John"  # not PII by regex
        assert result["nested"]["phone"] == "<PHONE_1>"

    def test_scrub_value_list(self, scrubber):
        data = ["john@example.com", "clean", "jane@example.com"]
        result = scrubber.scrub_value(data)
        assert result[0] == "<EMAIL_1>"
        assert result[1] == "clean"
        assert result[2] == "<EMAIL_2>"

    def test_scrub_value_non_string(self, scrubber):
        assert scrubber.scrub_value(42) == 42
        assert scrubber.scrub_value(None) is None
        assert scrubber.scrub_value(True) is True

    def test_extra_patterns(self):
        scrubber = PiiScrubber(
            extra_patterns=[("ORDER_ID", r"ORD-\d{6}")]
        )
        text = "Order ORD-123456 shipped"
        result = scrubber.scrub_text(text)
        assert "ORD-123456" not in result
        assert "<ORDER_ID_1>" in result

    def test_multiple_pii_types(self, scrubber):
        text = "Email john@example.com, call 555-123-4567, SSN 123-45-6789"
        result = scrubber.scrub_text(text)
        assert "<EMAIL_1>" in result
        assert "<PHONE_1>" in result
        assert "<SSN_1>" in result

    def test_phone_with_country_code(self, scrubber):
        text = "Call +1-555-123-4567"
        result = scrubber.scrub_text(text)
        assert "<PHONE_1>" in result
