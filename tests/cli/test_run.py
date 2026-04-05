"""Tests for checkagent run CLI command."""

from __future__ import annotations

from checkagent.cli.run import build_pytest_args


class TestBuildPytestArgs:
    """Test argument construction for pytest.main()."""

    def test_adds_marker_filter_by_default(self) -> None:
        args = build_pytest_args((), layer=None)
        assert "-m" in args
        idx = args.index("-m")
        assert args[idx + 1] == "agent_test"

    def test_adds_verbose_by_default(self) -> None:
        args = build_pytest_args((), layer=None)
        assert "-v" in args

    def test_respects_user_quiet_flag(self) -> None:
        args = build_pytest_args(("-q",), layer=None)
        assert "-v" not in args
        assert "-q" in args

    def test_respects_user_verbose_flag(self) -> None:
        args = build_pytest_args(("-v",), layer=None)
        assert args.count("-v") == 1

    def test_layer_option_adds_agent_layer_flag(self) -> None:
        args = build_pytest_args((), layer="mock")
        assert "--agent-layer" in args
        idx = args.index("--agent-layer")
        assert args[idx + 1] == "mock"

    def test_passes_through_extra_pytest_args(self) -> None:
        args = build_pytest_args(("tests/core/", "-x", "--tb=short"), layer=None)
        assert "tests/core/" in args
        assert "-x" in args
        assert "--tb=short" in args

    def test_preserves_user_marker_expression(self) -> None:
        args = build_pytest_args(("-m", "slow"), layer=None)
        assert args.count("-m") == 1
        idx = args.index("-m")
        assert args[idx + 1] == "slow"

    def test_layer_and_custom_marker_both_work(self) -> None:
        args = build_pytest_args(("-m", "slow"), layer="eval")
        assert "--agent-layer" in args
        assert args.count("-m") == 1
