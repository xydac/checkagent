"""Tests for the CrewAI adapter.

crewai is an optional dependency — all tests mock the framework
objects to test the adapter's conversion logic.
"""

from __future__ import annotations

import sys
import types
from unittest.mock import AsyncMock, MagicMock

import pytest

from checkagent.core.types import AgentInput, StreamEventType

# ---------------------------------------------------------------------------
# Fake crewai module so the import guard passes
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _fake_crewai_module():
    """Inject a fake crewai module for the import check."""
    fake = types.ModuleType("crewai")
    sys.modules["crewai"] = fake
    yield
    sys.modules.pop("crewai", None)


# ---------------------------------------------------------------------------
# Helpers to build mock CrewOutput objects
# ---------------------------------------------------------------------------


def _make_task_output(
    raw: str = "task result",
    description: str = "do something",
    agent: str = "researcher",
    tool_calls: list | None = None,
) -> MagicMock:
    """Create a mock TaskOutput."""
    task = MagicMock()
    task.raw = raw
    task.description = description
    task.agent = agent
    task.tool_calls = tool_calls or []
    return task


def _make_crew_output(
    raw: str = "final answer",
    tasks_output: list | None = None,
    token_usage: dict | None = None,
) -> MagicMock:
    """Create a mock CrewOutput."""
    result = MagicMock()
    result.raw = raw
    result.tasks_output = tasks_output or []
    result.token_usage = token_usage
    return result


# ---------------------------------------------------------------------------
# Tests: basic run
# ---------------------------------------------------------------------------


class TestCrewAIAdapterRun:
    async def test_basic_run_async(self):
        from checkagent.adapters.crewai import CrewAIAdapter

        output = _make_crew_output(
            raw="final answer",
            tasks_output=[_make_task_output()],
        )
        crew = MagicMock()
        crew.kickoff_async = AsyncMock(return_value=output)

        adapter = CrewAIAdapter(crew)
        result = await adapter.run("hello")

        assert result.succeeded
        assert result.final_output == "final answer"
        crew.kickoff_async.assert_awaited_once()

    async def test_string_to_agent_input(self):
        from checkagent.adapters.crewai import CrewAIAdapter

        output = _make_crew_output()
        crew = MagicMock()
        crew.kickoff_async = AsyncMock(return_value=output)

        adapter = CrewAIAdapter(crew)
        result = await adapter.run("test query")

        assert result.input.query == "test query"

    async def test_agent_input_forwarded(self):
        from checkagent.adapters.crewai import CrewAIAdapter

        output = _make_crew_output()
        crew = MagicMock()
        crew.kickoff_async = AsyncMock(return_value=output)

        adapter = CrewAIAdapter(crew)
        inp = AgentInput(query="hello", context={"key": "val"})
        await adapter.run(inp)

        call_kwargs = crew.kickoff_async.call_args
        inputs = call_kwargs.kwargs.get("inputs") or call_kwargs[1].get("inputs")
        assert inputs["query"] == "hello"
        assert inputs["key"] == "val"

    async def test_steps_from_tasks_output(self):
        from checkagent.adapters.crewai import CrewAIAdapter

        tasks = [
            _make_task_output(raw="step 1", description="task 1", agent="agent1"),
            _make_task_output(raw="step 2", description="task 2", agent="agent2"),
        ]
        output = _make_crew_output(tasks_output=tasks)
        crew = MagicMock()
        crew.kickoff_async = AsyncMock(return_value=output)

        adapter = CrewAIAdapter(crew)
        result = await adapter.run("query")

        assert len(result.steps) == 2
        assert result.steps[0].output_text == "step 1"
        assert result.steps[0].metadata["agent"] == "agent1"
        assert result.steps[1].output_text == "step 2"

    async def test_no_tasks_output_fallback(self):
        from checkagent.adapters.crewai import CrewAIAdapter

        output = _make_crew_output(raw="just raw", tasks_output=[])
        crew = MagicMock()
        crew.kickoff_async = AsyncMock(return_value=output)

        adapter = CrewAIAdapter(crew)
        result = await adapter.run("query")

        assert len(result.steps) == 1
        assert result.steps[0].output_text == "just raw"

    async def test_tool_calls_extracted(self):
        from checkagent.adapters.crewai import CrewAIAdapter

        task = _make_task_output(
            tool_calls=[{"name": "search", "arguments": {"q": "test"}}]
        )
        output = _make_crew_output(tasks_output=[task])
        crew = MagicMock()
        crew.kickoff_async = AsyncMock(return_value=output)

        adapter = CrewAIAdapter(crew)
        result = await adapter.run("query")

        assert len(result.steps[-1].tool_calls) == 1
        assert result.steps[-1].tool_calls[0].name == "search"

    async def test_token_usage_dict(self):
        from checkagent.adapters.crewai import CrewAIAdapter

        output = _make_crew_output(
            token_usage={"prompt_tokens": 100, "completion_tokens": 50}
        )
        crew = MagicMock()
        crew.kickoff_async = AsyncMock(return_value=output)

        adapter = CrewAIAdapter(crew)
        result = await adapter.run("query")

        assert result.total_prompt_tokens == 100
        assert result.total_completion_tokens == 50

    async def test_error_handling(self):
        from checkagent.adapters.crewai import CrewAIAdapter

        crew = MagicMock()
        crew.kickoff_async = AsyncMock(side_effect=RuntimeError("crew crashed"))

        adapter = CrewAIAdapter(crew)
        result = await adapter.run("oops")

        assert not result.succeeded
        assert "RuntimeError: crew crashed" in result.error
        assert result.duration_ms >= 0

    async def test_duration_recorded(self):
        from checkagent.adapters.crewai import CrewAIAdapter

        output = _make_crew_output()
        crew = MagicMock()
        crew.kickoff_async = AsyncMock(return_value=output)

        adapter = CrewAIAdapter(crew)
        result = await adapter.run("query")

        assert result.duration_ms >= 0

    async def test_sync_kickoff_fallback(self):
        from checkagent.adapters.crewai import CrewAIAdapter

        output = _make_crew_output(raw="sync result")
        crew = MagicMock(spec=["kickoff"])  # no kickoff_async
        crew.kickoff = MagicMock(return_value=output)

        adapter = CrewAIAdapter(crew)
        result = await adapter.run("query")

        assert result.succeeded
        assert result.final_output == "sync result"


# ---------------------------------------------------------------------------
# Tests: streaming (synthesized)
# ---------------------------------------------------------------------------


class TestCrewAIAdapterStream:
    async def test_stream_synthesized(self):
        from checkagent.adapters.crewai import CrewAIAdapter

        tasks = [_make_task_output(raw="step output")]
        output = _make_crew_output(tasks_output=tasks)
        crew = MagicMock()
        crew.kickoff_async = AsyncMock(return_value=output)

        adapter = CrewAIAdapter(crew)
        events = []
        async for event in adapter.run_stream("query"):
            events.append(event)

        types = [e.event_type for e in events]
        assert types[0] == StreamEventType.RUN_START
        assert StreamEventType.TEXT_DELTA in types
        assert types[-1] == StreamEventType.RUN_END

    async def test_stream_error(self):
        from checkagent.adapters.crewai import CrewAIAdapter

        crew = MagicMock()
        crew.kickoff_async = AsyncMock(side_effect=RuntimeError("boom"))

        adapter = CrewAIAdapter(crew)
        events = []
        async for event in adapter.run_stream("query"):
            events.append(event)

        types = [e.event_type for e in events]
        assert StreamEventType.ERROR in types
        assert types[-1] == StreamEventType.RUN_END


# ---------------------------------------------------------------------------
# Tests: import guard
# ---------------------------------------------------------------------------


class TestCrewAIImportGuard:
    async def test_import_error_without_crewai(self):
        from checkagent.adapters.crewai import _ensure_crewai

        saved = sys.modules.get("crewai")
        sys.modules["crewai"] = None  # type: ignore[assignment]
        try:
            with pytest.raises(ImportError, match="crewai"):
                _ensure_crewai()
        finally:
            if saved is not None:
                sys.modules["crewai"] = saved
            else:
                sys.modules.pop("crewai", None)
