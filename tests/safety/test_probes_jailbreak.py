"""Tests for jailbreak attack probes (F11.3)."""

from __future__ import annotations

from checkagent.safety.probes import Probe, ProbeSet, jailbreak
from checkagent.safety.taxonomy import SafetyCategory

# ---------------------------------------------------------------------------
# Encoding probes
# ---------------------------------------------------------------------------

class TestEncodingProbes:
    def test_encoding_probe_count(self):
        assert len(jailbreak.encoding) >= 6

    def test_all_have_names(self):
        for probe in jailbreak.encoding:
            assert probe.name, f"Probe missing name: {probe.input[:40]}"

    def test_all_have_jailbreak_category(self):
        for probe in jailbreak.encoding:
            assert probe.category == SafetyCategory.JAILBREAK

    def test_all_have_encoding_tag(self):
        for probe in jailbreak.encoding:
            assert "encoding" in probe.tags, f"{probe.name} missing 'encoding' tag"

    def test_all_have_nonempty_input(self):
        for probe in jailbreak.encoding:
            assert len(probe.input) > 0

    def test_unique_names(self):
        names = [p.name for p in jailbreak.encoding]
        assert len(names) == len(set(names)), "Duplicate probe names"

    def test_encoding_families(self):
        all_tags = set()
        for probe in jailbreak.encoding:
            all_tags.update(probe.tags)
        assert "base64" in all_tags
        assert "rot13" in all_tags
        assert "unicode" in all_tags

    def test_severity_distribution(self):
        severities = {p.severity for p in jailbreak.encoding}
        assert len(severities) >= 2

    def test_parametrize_compatible(self):
        params = jailbreak.encoding.all()
        assert isinstance(params, ProbeSet)
        assert all(isinstance(p, Probe) for p in params)


# ---------------------------------------------------------------------------
# Roleplay probes
# ---------------------------------------------------------------------------

class TestRoleplayProbes:
    def test_roleplay_probe_count(self):
        assert len(jailbreak.roleplay) >= 5

    def test_all_have_names(self):
        for probe in jailbreak.roleplay:
            assert probe.name, f"Probe missing name: {probe.input[:40]}"

    def test_all_have_jailbreak_category(self):
        for probe in jailbreak.roleplay:
            assert probe.category == SafetyCategory.JAILBREAK

    def test_all_have_roleplay_tag(self):
        for probe in jailbreak.roleplay:
            assert "roleplay" in probe.tags, f"{probe.name} missing 'roleplay' tag"

    def test_unique_names(self):
        names = [p.name for p in jailbreak.roleplay]
        assert len(names) == len(set(names)), "Duplicate probe names"


# ---------------------------------------------------------------------------
# Combined
# ---------------------------------------------------------------------------

class TestAllJailbreakProbes:
    def test_all_is_union(self):
        assert len(jailbreak.all_probes) == len(jailbreak.encoding) + len(jailbreak.roleplay)

    def test_composability(self):
        combined = jailbreak.encoding + jailbreak.roleplay
        assert len(combined) == len(jailbreak.all_probes)

    def test_filter_encoding_only(self):
        encoding_only = jailbreak.all_probes.filter(tags={"encoding"})
        assert len(encoding_only) == len(jailbreak.encoding)

    def test_filter_roleplay_only(self):
        roleplay_only = jailbreak.all_probes.filter(tags={"roleplay"})
        assert len(roleplay_only) == len(jailbreak.roleplay)

    def test_no_name_collisions_with_injection(self):
        from checkagent.safety.probes import injection
        jailbreak_names = {p.name for p in jailbreak.all_probes}
        injection_names = {p.name for p in injection.all_probes}
        overlap = jailbreak_names & injection_names
        assert not overlap, f"Name collisions: {overlap}"
