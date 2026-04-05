"""Tests for the checkagent pytest plugin."""


def test_plugin_loads(pytestconfig):
    """The checkagent plugin should be registered by pytest."""
    plugin = pytestconfig.pluginmanager.get_plugin("checkagent")
    assert plugin is not None


def test_agent_test_marker_registered(pytestconfig):
    """The agent_test marker should be registered."""
    markers = pytestconfig.getini("markers")
    marker_names = [m.split(":")[0].split("(")[0].strip() for m in markers]
    assert "agent_test" in marker_names


def test_safety_marker_registered(pytestconfig):
    """The safety marker should be registered."""
    markers = pytestconfig.getini("markers")
    marker_names = [m.split(":")[0].split("(")[0].strip() for m in markers]
    assert "safety" in marker_names


def test_cassette_marker_registered(pytestconfig):
    """The cassette marker should be registered."""
    markers = pytestconfig.getini("markers")
    marker_names = [m.split(":")[0].split("(")[0].strip() for m in markers]
    assert "cassette" in marker_names
