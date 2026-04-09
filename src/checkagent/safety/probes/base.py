"""Base types for attack probes.

A :class:`Probe` is a single adversarial input with metadata.
A :class:`ProbeSet` is a composable collection of probes that
integrates with ``pytest.mark.parametrize``.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field

from checkagent.safety.taxonomy import SafetyCategory, Severity


@dataclass(frozen=True)
class Probe:
    """A single adversarial test input.

    Attributes:
        input: The adversarial text to send to the agent.
        category: The safety category this probe targets.
        severity: Expected severity if the agent fails this probe.
        name: Short human-readable label (used as pytest param ID).
        description: Longer explanation of what this probe tests.
        tags: Arbitrary tags for filtering (e.g. ``"encoding"``, ``"roleplay"``).
    """

    input: str
    category: SafetyCategory
    severity: Severity = Severity.HIGH
    name: str = ""
    description: str = ""
    tags: frozenset[str] = field(default_factory=frozenset)

    def __str__(self) -> str:
        return self.name or self.input[:60]


class ProbeSet:
    """A composable, iterable collection of :class:`Probe` instances.

    Supports:
    - Iteration: ``for probe in probe_set``
    - Concatenation: ``set_a + set_b``
    - Filtering: ``probe_set.filter(tags={"encoding"})``
    - pytest parametrize: ``@pytest.mark.parametrize("attack", probe_set.all())``
    """

    def __init__(self, probes: list[Probe] | None = None, *, name: str = "") -> None:
        self._probes: list[Probe] = list(probes) if probes else []
        self.name = name

    def all(self) -> ProbeSet:
        """Return a copy of this ProbeSet.

        The returned ``ProbeSet`` is iterable, so it works directly
        with ``pytest.mark.parametrize`` and supports ``+`` composition::

            combined = injection.direct.all() + jailbreak.all()
        """
        return ProbeSet(list(self._probes), name=self.name)

    def filter(
        self,
        *,
        tags: set[str] | None = None,
        category: SafetyCategory | str | None = None,
        severity: Severity | str | None = None,
    ) -> ProbeSet:
        """Return a new ProbeSet with only matching probes.

        ``category`` and ``severity`` accept enum values or case-insensitive
        strings (e.g. ``"CRITICAL"`` or ``"critical"``).
        """
        result = self._probes
        if tags:
            result = [p for p in result if tags & p.tags]
        if category is not None:
            if isinstance(category, str) and not isinstance(category, SafetyCategory):
                category = SafetyCategory(category.lower())
            result = [p for p in result if p.category == category]
        if severity is not None:
            if isinstance(severity, str) and not isinstance(severity, Severity):
                severity = Severity(severity.lower())
            result = [p for p in result if p.severity == severity]
        return ProbeSet(result, name=f"{self.name}[filtered]")

    def __add__(self, other: ProbeSet) -> ProbeSet:
        if not isinstance(other, ProbeSet):
            return NotImplemented
        return ProbeSet(
            self._probes + other._probes,
            name=f"{self.name}+{other.name}",
        )

    def __iter__(self) -> Iterator[Probe]:
        return iter(self._probes)

    def __len__(self) -> int:
        return len(self._probes)

    def __repr__(self) -> str:
        return f"ProbeSet(name={self.name!r}, count={len(self._probes)})"
