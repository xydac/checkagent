"""CheckAgent pytest plugin — registers markers, fixtures, and configuration."""

from __future__ import annotations

import pytest


def pytest_configure(config: pytest.Config) -> None:
    """Register custom markers."""
    config.addinivalue_line(
        "markers",
        "agent_test(layer): mark a test as an agent test with layer specification",
    )
    config.addinivalue_line(
        "markers",
        "safety(category, severity): mark a test as a safety test",
    )
    config.addinivalue_line(
        "markers",
        "cassette(path): specify a cassette file for record-replay testing",
    )
