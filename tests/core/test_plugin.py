"""Tests for the checkagent pytest plugin."""

pytest_plugins = ["pytester"]

import pytest

from checkagent.core.plugin import VALID_LAYERS, _marker_matches_layer


def test_plugin_loads(pytestconfig):
    """The checkagent plugin should be registered by pytest."""
    plugin = pytestconfig.pluginmanager.get_plugin("checkagent")
    assert plugin is not None


def test_agent_test_marker_registered(pytestconfig):
    markers = pytestconfig.getini("markers")
    marker_names = [m.split(":")[0].split("(")[0].strip() for m in markers]
    assert "agent_test" in marker_names


def test_safety_marker_registered(pytestconfig):
    markers = pytestconfig.getini("markers")
    marker_names = [m.split(":")[0].split("(")[0].strip() for m in markers]
    assert "safety" in marker_names


def test_cassette_marker_registered(pytestconfig):
    markers = pytestconfig.getini("markers")
    marker_names = [m.split(":")[0].split("(")[0].strip() for m in markers]
    assert "cassette" in marker_names


def test_valid_layers():
    assert VALID_LAYERS == {"mock", "replay", "eval", "judge"}


class TestMarkerMatchesLayer:
    def test_positional_arg_match(self):
        mark = pytest.mark.agent_test("mock").mark
        assert _marker_matches_layer(mark, "mock") is True

    def test_positional_arg_no_match(self):
        mark = pytest.mark.agent_test("eval").mark
        assert _marker_matches_layer(mark, "mock") is False

    def test_keyword_arg_match(self):
        mark = pytest.mark.agent_test(layer="replay").mark
        assert _marker_matches_layer(mark, "replay") is True

    def test_keyword_arg_no_match(self):
        mark = pytest.mark.agent_test(layer="judge").mark
        assert _marker_matches_layer(mark, "mock") is False

    def test_case_insensitive(self):
        mark = pytest.mark.agent_test("MOCK").mark
        assert _marker_matches_layer(mark, "mock") is True


class TestLayerFiltering:
    """Integration tests using pytester to verify --agent-layer filtering."""

    def test_layer_filter_selects_matching(self, pytester):
        pytester.makepyfile("""
            import pytest

            @pytest.mark.agent_test("mock")
            def test_mock_agent():
                assert True

            @pytest.mark.agent_test("eval")
            def test_eval_agent():
                assert True

            def test_plain():
                assert True
        """)
        result = pytester.runpytest("--agent-layer=mock", "-v")
        result.assert_outcomes(passed=2)  # test_mock_agent + test_plain
        result.stdout.fnmatch_lines(["*test_mock_agent*PASSED*"])
        result.stdout.fnmatch_lines(["*test_plain*PASSED*"])

    def test_layer_filter_deselects_nonmatching(self, pytester):
        pytester.makepyfile("""
            import pytest

            @pytest.mark.agent_test("mock")
            def test_mock_agent():
                assert True

            @pytest.mark.agent_test("eval")
            def test_eval_agent():
                assert True
        """)
        result = pytester.runpytest("--agent-layer=mock", "-v")
        result.assert_outcomes(passed=1)
        result.stdout.fnmatch_lines(["*test_mock_agent*PASSED*"])

    def test_invalid_layer_errors(self, pytester):
        pytester.makepyfile("""
            def test_dummy():
                pass
        """)
        result = pytester.runpytest("--agent-layer=bogus")
        result.stderr.fnmatch_lines(["*Invalid --agent-layer*"])

    def test_no_filter_runs_all(self, pytester):
        pytester.makepyfile("""
            import pytest

            @pytest.mark.agent_test("mock")
            def test_mock():
                assert True

            @pytest.mark.agent_test("eval")
            def test_eval():
                assert True

            def test_plain():
                assert True
        """)
        result = pytester.runpytest("-v")
        result.assert_outcomes(passed=3)
