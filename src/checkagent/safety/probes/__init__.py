"""Attack probe library for adversarial safety testing.

Implements F11.3 from the PRD — pre-built attack suites that generate
adversarial test inputs for parametrized safety tests.

Usage::

    from checkagent.safety import probes

    @pytest.mark.parametrize("attack", probes.injection.direct.all())
    async def test_injection_resistance(my_agent, attack, ca_safety):
        result = await my_agent.run(attack.input)
        ca_safety.assert_no_injection(result)
"""

from checkagent.safety.probes import groundedness, injection, jailbreak, pii, scope
from checkagent.safety.probes.base import Probe, ProbeSet

__all__ = [
    "Probe",
    "ProbeSet",
    "groundedness",
    "injection",
    "jailbreak",
    "pii",
    "scope",
]
