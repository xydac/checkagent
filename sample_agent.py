"""Sample agent for CheckAgent CI and demos.

This module provides a simple agent callable used by:
- ``checkagent scan sample_agent:sample_agent`` in CI
- ``checkagent init`` as a scaffolding reference

The agent intentionally has minimal safety controls so the scan
produces findings — demonstrating that the tool works.
"""

from __future__ import annotations


async def sample_agent(prompt: str) -> str:
    """A simple echo agent that uppercases its input."""
    return f"AGENT: {prompt.upper()}"
