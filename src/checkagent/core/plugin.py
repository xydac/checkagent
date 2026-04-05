"""CheckAgent pytest plugin — registers markers, fixtures, and configuration."""

from __future__ import annotations

from pathlib import Path

import pytest

from checkagent.conversation.session import Conversation
from checkagent.core.config import CheckAgentConfig, load_config
from checkagent.mock.fault import FaultInjector
from checkagent.mock.llm import MockLLM
from checkagent.mock.tool import MockTool
from checkagent.streaming.collector import StreamCollector

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
    group.addoption(
        "--checkagent-config",
        action="store",
        default=None,
        help="Path to checkagent.yml config file (auto-discovered if not set).",
    )


def pytest_configure(config: pytest.Config) -> None:
    """Register custom markers and load CheckAgent configuration."""
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

    # Load configuration
    config_path_str = config.getoption("--checkagent-config", default=None)
    config_path = Path(config_path_str) if config_path_str else None
    ca_config = load_config(config_path)
    config.stash[_config_key] = ca_config


# Stash key for the CheckAgent config object
config_key = pytest.StashKey[CheckAgentConfig]()
_config_key = config_key  # internal alias


@pytest.fixture
def ap_mock_llm() -> MockLLM:
    """A fresh MockLLM instance for each test.

    Configure with rules in the test body::

        def test_agent(ap_mock_llm):
            ap_mock_llm.add_rule("weather", "It's sunny")
            result = await my_agent(ap_mock_llm).run("weather?")
            assert ap_mock_llm.call_count == 1
    """
    return MockLLM()


@pytest.fixture
def ap_mock_tool() -> MockTool:
    """A fresh MockTool instance for each test.

    Register tools and their responses in the test body::

        def test_agent(ap_mock_tool):
            ap_mock_tool.register("get_weather", response={"temp": 72})
            result = await my_agent(ap_mock_tool).run("weather in NYC?")
            ap_mock_tool.assert_tool_called("get_weather")
    """
    return MockTool()


@pytest.fixture
def ap_fault() -> FaultInjector:
    """A fresh FaultInjector instance for each test.

    Configure faults using the fluent API::

        def test_resilience(ap_fault):
            ap_fault.on_tool("search").timeout(5)
            ap_fault.on_llm().rate_limit(after_n=3)
            # ... run agent and assert graceful degradation
    """
    return FaultInjector()


@pytest.fixture
def ap_conversation() -> Conversation:
    """A conversation factory fixture for multi-turn testing.

    Returns a ``Conversation`` constructor. Pass your agent function
    to create a session::

        async def test_multi_turn(ap_conversation):
            conv = ap_conversation(my_agent_fn)
            r1 = await conv.say("Hello")
            r2 = await conv.say("What did I just say?")
            assert conv.total_turns == 2

    The agent function must accept an ``AgentInput`` and return
    an ``AgentRun``.
    """
    return Conversation


@pytest.fixture
def ap_stream_collector() -> StreamCollector:
    """A fresh StreamCollector instance for each test.

    Collects streaming events and provides assertion helpers::

        async def test_streaming(ap_mock_llm, ap_stream_collector):
            ap_mock_llm.stream_response("hello", ["Hi ", "there!"])
            await ap_stream_collector.collect_from(ap_mock_llm.stream("hello"))
            assert ap_stream_collector.aggregated_text == "Hi there!"
            assert ap_stream_collector.total_chunks == 2
    """
    return StreamCollector()


@pytest.fixture
def ap_config(request: pytest.FixtureRequest) -> CheckAgentConfig:
    """Access the loaded CheckAgent configuration."""
    return request.config.stash[_config_key]


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
