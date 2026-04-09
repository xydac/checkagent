"""Tests for PII extraction attack probes (F11.3)."""

from __future__ import annotations

from checkagent.safety.probes import Probe, ProbeSet, pii
from checkagent.safety.taxonomy import SafetyCategory


class TestPIIExtractionProbes:
    def test_probe_count(self):
        assert len(pii.extraction) >= 8

    def test_all_have_names(self):
        for probe in pii.extraction:
            assert probe.name, f"Probe missing name: {probe.input[:40]}"

    def test_all_have_pii_category(self):
        for probe in pii.extraction:
            assert probe.category == SafetyCategory.PII_LEAKAGE

    def test_all_have_extraction_tag(self):
        for probe in pii.extraction:
            assert "extraction" in probe.tags, f"{probe.name} missing 'extraction' tag"

    def test_all_have_nonempty_input(self):
        for probe in pii.extraction:
            assert len(probe.input) > 0

    def test_unique_names(self):
        names = [p.name for p in pii.extraction]
        assert len(names) == len(set(names)), "Duplicate probe names"

    def test_unique_inputs(self):
        inputs = [p.input for p in pii.extraction]
        assert len(inputs) == len(set(inputs)), "Duplicate probe inputs"

    def test_severity_distribution(self):
        severities = {p.severity for p in pii.extraction}
        assert len(severities) >= 2

    def test_parametrize_compatible(self):
        params = pii.extraction.all()
        assert isinstance(params, ProbeSet)
        assert all(isinstance(p, Probe) for p in params)

    def test_all_probes_equals_extraction(self):
        assert len(pii.all_probes) == len(pii.extraction)

    def test_no_name_collisions_with_injection(self):
        from checkagent.safety.probes import injection
        pii_names = {p.name for p in pii.all_probes}
        injection_names = {p.name for p in injection.all_probes}
        overlap = pii_names & injection_names
        assert not overlap, f"Name collisions: {overlap}"
