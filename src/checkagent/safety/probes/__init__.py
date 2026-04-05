"""Attack probe library for adversarial safety testing.

Implements F11.3 from the PRD — pre-built attack suites that generate
adversarial test inputs for parametrized safety tests.

Usage::

    from checkagent.safety import probes

    @pytest.mark.parametrize("attack", probes.injection.direct.all())
    async def test_injection_resistance(my_agent, attack, ap_safety):
        result = await my_agent.run(attack.input)
        ap_safety.assert_no_injection(result)
"""

from checkagent.safety.probes.base import Probe, ProbeSet
from checkagent.safety.probes import injection

__all__ = [
    "Probe",
    "ProbeSet",
    "injection",
]
