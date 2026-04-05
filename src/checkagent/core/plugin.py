"""CheckAgent pytest plugin — registers markers, fixtures, and configuration."""

from __future__ import annotations

from typing import Sequence

import pytest

VALID_LAYERS = frozenset({"mock", "replay", "eval", "judge"})


def pytest_addoption(parser: pytest.Parser) -> None:
    """Add CheckAgent CLI options to pytest."""
    group = parser.getgroup("checkagent", "CheckAgent agent testing")
    group.addoption(
        "--agent-layer",
        action="store",
        default=None,
        help="Only run agent tests for the specified layer (mock, replay, eval, judge).",
    )


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


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    """Filter tests by --agent-layer if specified."""
    layer_filter = config.getoption("--agent-layer", default=None)
    if layer_filter is None:
        return

    layer_filter = layer_filter.lower()
    if layer_filter not in VALID_LAYERS:
        raise pytest.UsageError(
            f"Invalid --agent-layer '{layer_filter}'. "
            f"Valid layers: {', '.join(sorted(VALID_LAYERS))}"
        )

    selected: list[pytest.Item] = []
    deselected: list[pytest.Item] = []

    for item in items:
        marker = item.get_closest_marker("agent_test")
        if marker is None:
            # Non-agent tests always run
            selected.append(item)
        elif _marker_matches_layer(marker, layer_filter):
            selected.append(item)
        else:
            deselected.append(item)

    if deselected:
        config.hook.pytest_deselected(items=deselected)
    items[:] = selected


def _marker_matches_layer(marker: pytest.Mark, layer: str) -> bool:
    """Check if an agent_test marker matches the requested layer."""
    if marker.args:
        return marker.args[0].lower() == layer
    return marker.kwargs.get("layer", "").lower() == layer
