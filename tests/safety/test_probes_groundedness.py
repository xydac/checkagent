"""Tests for groundedness / overreliance attack probes (OWASP LLM09)."""

from __future__ import annotations

from checkagent.safety.probes import Probe, ProbeSet, groundedness
from checkagent.safety.taxonomy import SafetyCategory


class TestGroundednessProbes:
    def test_fabrication_probe_count(self):
        assert len(groundedness.fabrication) == 4

    def test_uncertainty_probe_count(self):
        assert len(groundedness.uncertainty) == 4

    def test_all_probes_total(self):
        assert len(groundedness.all_probes) == 8

    def test_all_have_groundedness_category(self):
        for probe in groundedness.all_probes:
            assert probe.category == SafetyCategory.GROUNDEDNESS

    def test_all_have_names(self):
        for probe in groundedness.all_probes:
            assert probe.name, f"Probe missing name: {probe.input[:40]}"

    def test_all_have_nonempty_input(self):
        for probe in groundedness.all_probes:
            assert len(probe.input) > 0

    def test_unique_names(self):
        names = [p.name for p in groundedness.all_probes]
        assert len(names) == len(set(names)), "Duplicate probe names"

    def test_severity_distribution(self):
        severities = {p.severity for p in groundedness.all_probes}
        assert len(severities) >= 2, "Need at least 2 severity levels"

    def test_fabrication_tag(self):
        for probe in groundedness.fabrication:
            assert "fabrication" in probe.tags, f"{probe.name} missing 'fabrication' tag"

    def test_uncertainty_tag(self):
        for probe in groundedness.uncertainty:
            assert "uncertainty" in probe.tags, f"{probe.name} missing 'uncertainty' tag"

    def test_parametrize_compatible(self):
        params = groundedness.all_probes.all()
        assert isinstance(params, ProbeSet)
        assert all(isinstance(p, Probe) for p in params)

    def test_no_name_collisions_across_modules(self):
        from checkagent.safety.probes import injection, jailbreak, pii, scope

        all_names = [
            {p.name for p in injection.all_probes},
            {p.name for p in jailbreak.all_probes},
            {p.name for p in pii.all_probes},
            {p.name for p in scope.all_probes},
            {p.name for p in groundedness.all_probes},
        ]
        combined = set()
        for name_set in all_names:
            overlap = combined & name_set
            assert not overlap, f"Name collisions: {overlap}"
            combined |= name_set

    def test_probeset_names(self):
        assert "fabrication" in groundedness.fabrication.name
        assert "uncertainty" in groundedness.uncertainty.name
        assert "groundedness" in groundedness.all_probes.name

    def test_filter_by_category(self):
        from checkagent.safety.probes import injection

        combined = groundedness.all_probes + injection.all_probes
        filtered = combined.filter(category=SafetyCategory.GROUNDEDNESS)
        assert len(filtered) == 8
