"""Tests for the attack probe library (F11.3)."""

from __future__ import annotations

import pytest

from checkagent.safety.probes import Probe, ProbeSet, injection
from checkagent.safety.taxonomy import SafetyCategory, Severity

# ---------------------------------------------------------------------------
# Probe dataclass
# ---------------------------------------------------------------------------

class TestProbe:
    def test_probe_is_frozen(self):
        p = Probe(input="test", category=SafetyCategory.PROMPT_INJECTION)
        with pytest.raises(AttributeError):
            p.input = "changed"  # type: ignore[misc]

    def test_probe_defaults(self):
        p = Probe(input="x", category=SafetyCategory.PROMPT_INJECTION)
        assert p.severity == Severity.HIGH
        assert p.name == ""
        assert p.description == ""
        assert p.tags == frozenset()

    def test_probe_str_uses_name(self):
        p = Probe(input="long text", category=SafetyCategory.PROMPT_INJECTION, name="short")
        assert str(p) == "short"

    def test_probe_str_falls_back_to_input(self):
        p = Probe(input="some input text", category=SafetyCategory.PROMPT_INJECTION)
        assert str(p) == "some input text"

    def test_probe_str_truncates_long_input(self):
        p = Probe(input="a" * 100, category=SafetyCategory.PROMPT_INJECTION)
        assert len(str(p)) == 60

    def test_probe_with_tags(self):
        p = Probe(
            input="x",
            category=SafetyCategory.PROMPT_INJECTION,
            tags=frozenset({"a", "b"}),
        )
        assert "a" in p.tags
        assert "b" in p.tags

    def test_probe_identity(self):
        """Same content → same object via frozen hash."""
        p1 = Probe(input="x", category=SafetyCategory.PROMPT_INJECTION, name="a")
        p2 = Probe(input="x", category=SafetyCategory.PROMPT_INJECTION, name="a")
        assert p1 == p2
        assert hash(p1) == hash(p2)


# ---------------------------------------------------------------------------
# ProbeSet
# ---------------------------------------------------------------------------

class TestProbeSet:
    def _make_probes(self, n: int = 5) -> list[Probe]:
        return [
            Probe(
                input=f"probe-{i}",
                category=SafetyCategory.PROMPT_INJECTION,
                name=f"p{i}",
                severity=Severity.HIGH if i % 2 == 0 else Severity.MEDIUM,
                tags=frozenset({"even"} if i % 2 == 0 else {"odd"}),
            )
            for i in range(n)
        ]

    def test_empty_probe_set(self):
        ps = ProbeSet()
        assert len(ps) == 0
        assert ps.all() == []

    def test_probe_set_len(self):
        ps = ProbeSet(self._make_probes(3))
        assert len(ps) == 3

    def test_probe_set_iter(self):
        probes = self._make_probes(3)
        ps = ProbeSet(probes)
        assert list(ps) == probes

    def test_probe_set_all_returns_copy(self):
        probes = self._make_probes(3)
        ps = ProbeSet(probes)
        result = ps.all()
        assert result == probes
        assert result is not ps._probes  # noqa: SLF001

    def test_probe_set_add(self):
        a = ProbeSet(self._make_probes(2), name="a")
        b = ProbeSet(self._make_probes(3), name="b")
        combined = a + b
        assert len(combined) == 5
        assert "a+b" in combined.name

    def test_probe_set_add_type_error(self):
        ps = ProbeSet()
        with pytest.raises(TypeError):
            ps + [1, 2, 3]  # type: ignore[operator]

    def test_probe_set_filter_by_tags(self):
        ps = ProbeSet(self._make_probes(6))
        even = ps.filter(tags={"even"})
        assert all("even" in p.tags for p in even)
        assert len(even) == 3

    def test_probe_set_filter_by_severity(self):
        ps = ProbeSet(self._make_probes(6))
        high = ps.filter(severity=Severity.HIGH)
        assert all(p.severity == Severity.HIGH for p in high)

    def test_probe_set_filter_by_category(self):
        probes = self._make_probes(3)
        extra = Probe(input="pii", category=SafetyCategory.PII_LEAKAGE, name="pii")
        ps = ProbeSet(probes + [extra])
        pii_only = ps.filter(category=SafetyCategory.PII_LEAKAGE)
        assert len(pii_only) == 1

    def test_probe_set_filter_combined(self):
        ps = ProbeSet(self._make_probes(6))
        result = ps.filter(tags={"even"}, severity=Severity.HIGH)
        assert len(result) > 0
        assert all(p.severity == Severity.HIGH and "even" in p.tags for p in result)

    def test_probe_set_filter_severity_case_insensitive(self):
        ps = ProbeSet(self._make_probes(6))
        upper = ps.filter(severity="HIGH")
        lower = ps.filter(severity="high")
        mixed = ps.filter(severity="High")
        enum_ = ps.filter(severity=Severity.HIGH)
        assert len(upper) == len(enum_)
        assert len(lower) == len(enum_)
        assert len(mixed) == len(enum_)

    def test_probe_set_filter_category_case_insensitive(self):
        probes = self._make_probes(3)
        extra = Probe(input="pii", category=SafetyCategory.PII_LEAKAGE, name="pii")
        ps = ProbeSet(probes + [extra])
        upper = ps.filter(category="PII_LEAKAGE")
        lower = ps.filter(category="pii_leakage")
        assert len(upper) == 1
        assert len(lower) == 1

    def test_probe_set_filter_invalid_severity_raises(self):
        ps = ProbeSet(self._make_probes(3))
        with pytest.raises(ValueError):
            ps.filter(severity="nonexistent")

    def test_probe_set_repr(self):
        ps = ProbeSet(self._make_probes(3), name="test")
        assert "test" in repr(ps)
        assert "3" in repr(ps)


# ---------------------------------------------------------------------------
# Injection probes — direct
# ---------------------------------------------------------------------------

class TestDirectInjectionProbes:
    def test_direct_probe_count(self):
        """F11.3 requires 50+ templates total; direct should have 20+."""
        assert len(injection.direct) >= 20

    def test_all_probes_have_names(self):
        for probe in injection.direct:
            assert probe.name, f"Probe missing name: {probe.input[:40]}"

    def test_all_probes_have_category(self):
        for probe in injection.direct:
            assert probe.category == SafetyCategory.PROMPT_INJECTION

    def test_all_probes_have_tags(self):
        for probe in injection.direct:
            assert len(probe.tags) > 0, f"Probe {probe.name} has no tags"

    def test_all_probes_have_nonempty_input(self):
        for probe in injection.direct:
            assert len(probe.input) > 0

    def test_unique_names(self):
        names = [p.name for p in injection.direct]
        assert len(names) == len(set(names)), "Duplicate probe names found"

    def test_unique_inputs(self):
        inputs = [p.input for p in injection.direct]
        assert len(inputs) == len(set(inputs)), "Duplicate probe inputs found"

    def test_tag_families_present(self):
        all_tags = set()
        for probe in injection.direct:
            all_tags.update(probe.tags)
        assert "ignore" in all_tags
        assert "extraction" in all_tags
        assert "persona" in all_tags
        assert "smuggling" in all_tags
        assert "authority" in all_tags

    def test_severity_distribution(self):
        """Should have a mix of severities, not all the same."""
        severities = {p.severity for p in injection.direct}
        assert len(severities) >= 2

    def test_parametrize_compatible(self):
        """Probes work with pytest.mark.parametrize."""
        params = injection.direct.all()
        assert isinstance(params, list)
        assert all(isinstance(p, Probe) for p in params)


# ---------------------------------------------------------------------------
# Injection probes — indirect
# ---------------------------------------------------------------------------

class TestIndirectInjectionProbes:
    def test_indirect_probe_count(self):
        assert len(injection.indirect) >= 8

    def test_all_indirect_have_indirect_tag(self):
        for probe in injection.indirect:
            assert "indirect" in probe.tags, f"{probe.name} missing 'indirect' tag"

    def test_unique_names(self):
        names = [p.name for p in injection.indirect]
        assert len(names) == len(set(names))


# ---------------------------------------------------------------------------
# Combined probes
# ---------------------------------------------------------------------------

class TestAllInjectionProbes:
    def test_all_probes_is_union(self):
        assert len(injection.all_probes) == len(injection.direct) + len(injection.indirect)

    def test_composability(self):
        combined = injection.direct + injection.indirect
        assert len(combined) == len(injection.all_probes)

    def test_filter_after_compose(self):
        combined = injection.direct + injection.indirect
        indirect_only = combined.filter(tags={"indirect"})
        assert len(indirect_only) == len(injection.indirect)
