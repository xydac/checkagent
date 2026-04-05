"""Tests for scope boundary attack probes (F11.3)."""

from __future__ import annotations

from checkagent.safety.probes import Probe, scope
from checkagent.safety.taxonomy import SafetyCategory


class TestScopeBoundaryProbes:
    def test_probe_count(self):
        assert len(scope.boundary) >= 6

    def test_all_have_names(self):
        for probe in scope.boundary:
            assert probe.name, f"Probe missing name: {probe.input[:40]}"

    def test_all_have_off_topic_category(self):
        for probe in scope.boundary:
            assert probe.category == SafetyCategory.OFF_TOPIC

    def test_all_have_boundary_tag(self):
        for probe in scope.boundary:
            assert "boundary" in probe.tags, f"{probe.name} missing 'boundary' tag"

    def test_all_have_nonempty_input(self):
        for probe in scope.boundary:
            assert len(probe.input) > 0

    def test_unique_names(self):
        names = [p.name for p in scope.boundary]
        assert len(names) == len(set(names)), "Duplicate probe names"

    def test_severity_distribution(self):
        severities = {p.severity for p in scope.boundary}
        assert len(severities) >= 2

    def test_action_vs_topic_tags(self):
        """Should have both action and topic boundary probes."""
        all_tags = set()
        for probe in scope.boundary:
            all_tags.update(probe.tags)
        assert "action" in all_tags
        assert "topic" in all_tags

    def test_parametrize_compatible(self):
        params = scope.boundary.all()
        assert isinstance(params, list)
        assert all(isinstance(p, Probe) for p in params)

    def test_all_probes_equals_boundary(self):
        assert len(scope.all_probes) == len(scope.boundary)

    def test_no_name_collisions_across_modules(self):
        from checkagent.safety.probes import injection, jailbreak, pii
        all_names = [
            {p.name for p in injection.all_probes},
            {p.name for p in jailbreak.all_probes},
            {p.name for p in pii.all_probes},
            {p.name for p in scope.all_probes},
        ]
        combined = set()
        for name_set in all_names:
            overlap = combined & name_set
            assert not overlap, f"Name collisions: {overlap}"
            combined |= name_set


class TestTotalProbeCount:
    """F11.3 requires 50+ total probe templates."""

    def test_total_across_all_modules(self):
        from checkagent.safety.probes import injection, jailbreak, pii
        total = (
            len(injection.all_probes)
            + len(jailbreak.all_probes)
            + len(pii.all_probes)
            + len(scope.all_probes)
        )
        assert total >= 50, f"Only {total} probes, need 50+"
