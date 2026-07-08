"""CheckAgent pytest plugin — registers markers, fixtures, and configuration."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from checkagent.conversation.session import Conversation
from checkagent.core.config import CheckAgentConfig, load_config
from checkagent.core.cost import CostTracker
from checkagent.core.tracer import (
    begin_probe_trace,
    end_probe_trace,
    install_patches,
    uninstall_patches,
)
from checkagent.judge.judge import RubricJudge
from checkagent.judge.types import Rubric
from checkagent.mock.fault import FaultInjector
from checkagent.mock.llm import MockLLM
from checkagent.mock.mcp import MockMCPServer
from checkagent.mock.tool import MockTool
from checkagent.replay.cassette import Cassette
from checkagent.replay.engine import MatchStrategy, ReplayEngine
from checkagent.replay.recorder import CassetteRecorder
from checkagent.safety.injection import PromptInjectionDetector
from checkagent.safety.pii import PIILeakageScanner
from checkagent.safety.refusal import RefusalComplianceChecker
from checkagent.safety.system_prompt import SystemPromptLeakDetector
from checkagent.safety.tool_boundary import ToolCallBoundaryValidator
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

    # Auto-configure pytest-asyncio to "auto" mode so async tests just work.
    # Only override if the user hasn't explicitly set asyncio_mode in their
    # own pytest config (pyproject.toml, pytest.ini, etc.).
    _auto_configure_asyncio(config)

    # Load configuration
    config_path_str = config.getoption("--checkagent-config", default=None)
    config_path = Path(config_path_str) if config_path_str else None
    ca_config = load_config(config_path)
    config.stash[_config_key] = ca_config


def _auto_configure_asyncio(config: pytest.Config) -> None:
    """Set asyncio_mode=auto if the user hasn't configured it themselves."""
    try:
        # Check if the user explicitly set asyncio_mode in their config file.
        # config.inicfg contains only values from the user's config file.
        inicfg = config.inicfg or {}
        if "asyncio_mode" not in inicfg:
            config._inicache["asyncio_mode"] = "auto"  # noqa: SLF001
    except Exception:  # noqa: BLE001
        # Don't crash if pytest-asyncio isn't installed or API changed
        pass


# Stash key for the CheckAgent config object
config_key = pytest.StashKey[CheckAgentConfig]()
_config_key = config_key  # internal alias


@pytest.fixture
def ca_mock_llm() -> MockLLM:
    """A fresh MockLLM instance for each test.

    Configure with the fluent API::

        async def test_agent(ca_mock_llm):
            ca_mock_llm.on_input(contains="weather").respond("It's sunny")
            response = await ca_mock_llm.complete("What's the weather?")
            assert response == "It's sunny"
            assert ca_mock_llm.call_count == 1
    """
    return MockLLM()


@pytest.fixture
def ca_mock_tool() -> MockTool:
    """A fresh MockTool instance for each test.

    Register tools with the fluent API::

        async def test_agent(ca_mock_tool):
            ca_mock_tool.on_call("get_weather").respond({"temp": 72})
            result = await ca_mock_tool.call("get_weather", {"city": "NYC"})
            ca_mock_tool.assert_tool_called("get_weather")
    """
    return MockTool()


@pytest.fixture
def ca_fault() -> FaultInjector:
    """A fresh FaultInjector instance for each test.

    Configure faults using the fluent API::

        def test_resilience(ca_fault):
            ca_fault.on_tool("search").timeout(5)
            ca_fault.on_llm().rate_limit(after_n=3)
            # ... run agent and assert graceful degradation
    """
    return FaultInjector()


@pytest.fixture
def ca_conversation() -> Conversation:
    """A conversation factory fixture for multi-turn testing.

    Returns a ``Conversation`` constructor. Pass your agent function
    to create a session::

        async def test_multi_turn(ca_conversation):
            conv = ca_conversation(my_agent_fn)
            r1 = await conv.say("Hello")
            r2 = await conv.say("What did I just say?")
            assert conv.total_turns == 2

    The agent function must accept an ``AgentInput`` and return
    an ``AgentRun``.
    """
    return Conversation


@pytest.fixture
def ca_stream_collector() -> StreamCollector:
    """A fresh StreamCollector instance for each test.

    Collects streaming events and provides assertion helpers::

        async def test_streaming(ca_mock_llm, ca_stream_collector):
            ca_mock_llm.on_input(contains="hello").stream(["Hi ", "there!"])
            await ca_stream_collector.collect_from(ca_mock_llm.stream("hello"))
            assert ca_stream_collector.aggregated_text == "Hi there!"
            assert ca_stream_collector.total_chunks == 2
    """
    return StreamCollector()


@pytest.fixture
def ca_mock_mcp_server() -> MockMCPServer:
    """A fresh MockMCPServer instance for each test.

    Simulates an MCP server for testing MCP-aware agents::

        async def test_mcp_agent(ca_mock_mcp_server):
            ca_mock_mcp_server.register_tool("search", response={"results": []})
            resp = await ca_mock_mcp_server.handle_message({
                "jsonrpc": "2.0", "id": 1, "method": "tools/call",
                "params": {"name": "search", "arguments": {"q": "test"}},
            })
            ca_mock_mcp_server.assert_tool_called("search")
    """
    return MockMCPServer()


@pytest.fixture
def ca_safety() -> dict[str, object]:
    """Safety evaluator instances for testing agent outputs.

    Provides all built-in safety evaluators::

        def test_agent_safety(ca_safety):
            result = ca_safety["injection"].evaluate(agent_output)
            assert result.passed

            result = ca_safety["pii"].evaluate(agent_output)
            assert result.passed

            result = ca_safety["tool_boundary"].evaluate_run(agent_run)
            assert result.passed

            result = ca_safety["refusal"].evaluate(agent_output)
            assert result.passed
    """
    return {
        "injection": PromptInjectionDetector(),
        "pii": PIILeakageScanner(),
        "system_prompt": SystemPromptLeakDetector(),
        "tool_boundary": ToolCallBoundaryValidator(),
        "refusal": RefusalComplianceChecker(),
    }


@pytest.fixture
def ca_judge() -> Callable[..., RubricJudge]:
    """Factory fixture for creating RubricJudge instances.

    Returns a factory that creates judges from a rubric and LLM callable::

        async def test_judge(ca_judge):
            rubric = Rubric(name="quality", criteria=[...])
            judge = ca_judge(rubric, my_llm_callable)
            score = await judge.evaluate(run)
            assert score.overall >= 0.7

    The factory accepts:
        rubric: Rubric — the evaluation rubric
        llm: async (system, user) -> str — LLM callable
        model_name: str — optional model identifier
    """

    def _factory(
        rubric: Rubric,
        llm: Callable[..., Any],
        model_name: str = "",
    ) -> RubricJudge:
        return RubricJudge(rubric=rubric, llm=llm, model_name=model_name)

    return _factory


@pytest.fixture
def ca_config(request: pytest.FixtureRequest) -> CheckAgentConfig:
    """Access the loaded CheckAgent configuration."""
    return request.config.stash[_config_key]


class TracerContext:
    """Collects LLM/tool call trace events during a test.

    Use :func:`begin` before calling the agent and :func:`end` to retrieve
    the captured events.  If you just need per-test auto-instrumentation
    without explicit begin/end, the fixture handles install/uninstall
    automatically — events from the last ``begin``/``end`` pair are in
    :attr:`events`.

    Events from :class:`~checkagent.mock.llm.MockLLM` have ``type="llm_call"``
    and include ``provider``, ``model``, ``prompt_preview``, ``response_preview``,
    and ``latency_ms``.  Events from :class:`~checkagent.mock.tool.MockTool` have
    ``type="tool_call"`` and include ``tool_name``, ``arguments``, ``result``,
    ``latency_ms``, and ``error`` (``None`` on success).

    Example::

        async def test_agent_calls(ca_tracer, ca_mock_llm, ca_mock_tool):
            ca_tracer.begin()
            await ca_mock_llm.complete("hello")
            await ca_mock_tool.call("search", {"query": "python"})
            ca_tracer.end()
            assert len(ca_tracer.llm_calls) == 1
            assert len(ca_tracer.tool_calls) == 1
            assert ca_tracer.tool_calls[0]["tool_name"] == "search"
    """

    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []
        self._active = False

    def begin(self) -> None:
        begin_probe_trace()
        self._active = True

    def end(self) -> list[dict[str, Any]]:
        self.events = end_probe_trace()
        self._active = False
        return self.events

    @property
    def llm_calls(self) -> list[dict[str, Any]]:
        return [e for e in self.events if e.get("type") == "llm_call"]

    @property
    def tool_calls(self) -> list[dict[str, Any]]:
        return [e for e in self.events if e.get("type") == "tool_call"]


@pytest.fixture
def ca_tracer() -> Any:
    """Auto-instrument OpenAI and Anthropic SDK calls for the duration of a test.

    Installs monkey-patches on SDK client methods at test start, removes them
    at test end.  Use ``ca_tracer.begin()`` / ``ca_tracer.end()`` around agent
    calls to capture LLM and tool call events::

        async def test_traces(ca_tracer, my_agent):
            ca_tracer.begin()
            result = await my_agent.run("hello")
            ca_tracer.end()
            assert len(ca_tracer.llm_calls) >= 1
    """
    install_patches()
    ctx = TracerContext()
    yield ctx
    if ctx._active:
        ctx.end()
    uninstall_patches()


class CassetteFixture:
    """Context object returned by the ``ca_cassette`` fixture.

    Attributes:
        mode: ``"record"`` when no cassette file exists yet; ``"replay"`` when
            an existing cassette was loaded.
        path: Resolved path to the cassette file.
        recorder: A :class:`~checkagent.replay.recorder.CassetteRecorder` in
            record mode, ``None`` in replay mode.
        engine: A :class:`~checkagent.replay.engine.ReplayEngine` in replay
            mode, ``None`` in record mode.
        cassette: The loaded :class:`~checkagent.replay.cassette.Cassette` in
            replay mode, ``None`` in record mode.
    """

    def __init__(
        self,
        mode: str,
        path: Path,
        recorder: CassetteRecorder | None = None,
        engine: ReplayEngine | None = None,
        cassette: Cassette | None = None,
    ) -> None:
        self.mode = mode
        self.path = path
        self.recorder = recorder
        self.engine = engine
        self.cassette = cassette

    def is_recording(self) -> bool:
        """Return True when in record mode."""
        return self.mode == "record"

    def is_replaying(self) -> bool:
        """Return True when in replay mode."""
        return self.mode == "replay"

    def replay_response(self, prompt: str) -> str:
        """Return the recorded response for *prompt* in replay mode.

        Matches by sequence position (same order as recorded).  Raises
        :class:`~checkagent.replay.engine.CassetteMismatchError` if the
        cassette is exhausted.

        Only valid in replay mode — raises ``RuntimeError`` when recording.
        """
        if self.mode == "record":
            raise RuntimeError(
                "replay_response() called in record mode. "
                "Use ca_cassette.recorder.record_response() instead."
            )
        from checkagent.replay.cassette import RecordedRequest

        req = RecordedRequest(
            kind="llm",
            method="chat.completions.create",
            body={"messages": [{"role": "user", "content": prompt}]},
        )
        interaction = self.engine.match(req)  # type: ignore[union-attr]
        body = interaction.response.body
        if isinstance(body, str):
            return body
        if isinstance(body, dict):
            try:
                return body["choices"][0]["message"]["content"]
            except (KeyError, IndexError, TypeError):
                return str(body)
        return str(body)

    async def arun(self, agent_fn: object, prompt: str) -> str:
        """Run *agent_fn(prompt)* in record mode or replay from cassette.

        This is the highest-level cassette API — it transparently handles
        both recording and replaying without any mode-switch logic in tests::

            @pytest.mark.agent_test(layer="replay")
            async def test_greeting(ca_cassette, my_agent):
                response = await ca_cassette.arun(my_agent, "hello")
                assert "hello" in response.lower()

        On the **first run** (no cassette yet): calls the real agent, saves the
        response.  On **subsequent runs**: returns the recorded response without
        calling the agent.
        """
        import inspect

        if self.mode == "record":
            if inspect.iscoroutinefunction(agent_fn):
                result = await agent_fn(prompt)  # type: ignore[operator]
            else:
                result = agent_fn(prompt)  # type: ignore[operator]
            response = str(result)
            self.recorder.record_response(prompt, response)  # type: ignore[union-attr]
            return response
        else:
            return self.replay_response(prompt)


@pytest.fixture
def ca_cassette(request: pytest.FixtureRequest) -> Any:
    """Record-and-replay cassette fixture.

    The fixture automatically selects the operating mode:

    * **Record mode** — when no cassette file exists at the target path.
      A fresh :class:`~checkagent.replay.recorder.CassetteRecorder` is
      created.  After the test completes the cassette is finalized and
      saved to disk.
    * **Replay mode** — when a cassette file already exists.  The cassette
      is loaded and a :class:`~checkagent.replay.engine.ReplayEngine` is
      created in ``SEQUENCE`` strategy (matches by call order).

    The cassette path is resolved in priority order:

    1. ``@pytest.mark.cassette(path="cassettes/my.json")`` on the test.
    2. ``cassettes/<module_path>/<test_name>.json`` relative to the
       directory that contains the test file.

    The simplest usage via :meth:`CassetteFixture.arun` — no mode-switch
    logic needed in your test::

        @pytest.mark.agent_test(layer="replay")
        async def test_greeting(ca_cassette, my_agent):
            response = await ca_cassette.arun(my_agent, "hello")
            assert "Hello" in response

    For finer control, use :meth:`CassetteFixture.is_recording` to branch::

        @pytest.mark.agent_test(layer="replay")
        async def test_greeting(ca_cassette, my_agent):
            if ca_cassette.is_recording():
                result = await my_agent.run("hello")
                ca_cassette.recorder.record_response("hello", result)
            else:
                result = ca_cassette.replay_response("hello")
            assert "Hello" in result
    """
    # --- Resolve cassette path ---
    marker = request.node.get_closest_marker("cassette")
    if marker and marker.args:
        cassette_path = Path(marker.args[0])
    elif marker and "path" in marker.kwargs:
        cassette_path = Path(marker.kwargs["path"])
    else:
        # Default: cassettes/<relative_module>/<test_name>.json
        test_file = Path(request.node.fspath)
        module_rel = test_file.stem  # e.g. "test_agent"
        test_name = request.node.name.replace("::", "_").replace("[", "_").replace("]", "")
        cassette_path = test_file.parent / "cassettes" / module_rel / f"{test_name}.json"

    # --- Choose mode ---
    if cassette_path.exists():
        # REPLAY mode
        cassette = Cassette.load(cassette_path)
        engine = ReplayEngine(cassette, strategy=MatchStrategy.SEQUENCE)
        ctx = CassetteFixture(mode="replay", path=cassette_path, engine=engine, cassette=cassette)
        yield ctx
    else:
        # RECORD mode — save cassette after the test body completes
        test_id = request.node.nodeid
        recorder = CassetteRecorder(test_id=test_id)
        ctx = CassetteFixture(mode="record", path=cassette_path, recorder=recorder)
        yield ctx
        # Post-test: finalize and save
        cassette = recorder.finalize()
        cassette.save(cassette_path)


@pytest.fixture(scope="session")
def ca_cost_tracker(pytestconfig: pytest.Config) -> Any:
    """Session-scoped CostTracker with automatic budget enforcement.

    Provides a :class:`~checkagent.core.cost.CostTracker` pre-configured with
    the budget from ``checkagent.yml``.  After the session completes the
    tracker calls ``check_suite_budget()`` — if the total cost exceeds the
    configured budget the session teardown raises ``BudgetExceededError``.

    Example::

        async def test_my_agent_cost(ca_cost_tracker, my_agent):
            result = await my_agent.run("hello")
            ca_cost_tracker.record(result)
            assert ca_cost_tracker.total_cost < 0.01
    """
    cfg: CheckAgentConfig = pytestconfig.stash.get(_config_key, CheckAgentConfig())
    budget = getattr(cfg, "budget", None)
    tracker = CostTracker(budget=budget)
    yield tracker
    tracker.check_suite_budget()


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
