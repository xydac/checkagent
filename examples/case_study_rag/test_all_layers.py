"""Case study: RAG agent tested with replay-based regression detection.

This file demonstrates how replay testing catches retrieval regressions
that mock-only tests miss:
  Layer 1 (MOCK)   — Verify tool call structure, deterministic
  Layer 2 (REPLAY) — Record baseline, detect regressions via cassette diff
  Layer 3 (EVAL)   — Metric evaluation of retrieval quality
"""

from __future__ import annotations

import pytest
from agent import rag_agent

from checkagent.eval.metrics import task_completion, tool_correctness, trajectory_match
from checkagent.mock import MockLLM, MockTool
from checkagent.replay import CassetteRecorder, ReplayEngine

# ---------------------------------------------------------------------------
# Knowledge base: baseline documents
# ---------------------------------------------------------------------------

BASELINE_DOCS = [
    {
        "id": "doc-1",
        "title": "Python asyncio",
        "content": "asyncio is a library for writing concurrent code using "
        "the async/await syntax. It is used as a foundation for "
        "multiple Python asynchronous frameworks.",
    },
    {
        "id": "doc-2",
        "title": "Python threading",
        "content": "The threading module provides a higher-level threading "
        "API on top of the lower-level _thread module. Threads share "
        "the process memory space.",
    },
    {
        "id": "doc-3",
        "title": "Python multiprocessing",
        "content": "The multiprocessing module allows spawning processes. "
        "Each process has its own memory space, avoiding the GIL.",
    },
]

RANKED_DOCS = [BASELINE_DOCS[0], BASELINE_DOCS[1]]  # Top 2 after ranking

QUERY = "How does Python handle concurrency?"


def _setup_baseline_mocks():
    """Configure mocks for the baseline RAG scenario."""
    llm = MockLLM()
    llm.on_input(contains="answer:").respond(
        "Python handles concurrency through three main approaches: "
        "asyncio for async I/O, threading for concurrent threads "
        "sharing memory, and multiprocessing for parallel processes "
        "with separate memory spaces."
    )

    tools = MockTool()
    # Wrap lists in [list] because MockTool treats list responses as sequences
    tools.on_call("retrieve_docs").respond([BASELINE_DOCS])
    tools.on_call("rank_docs").respond([RANKED_DOCS])
    return llm, tools


# ===========================================================================
# Layer 1: MOCK — Structural tests (free, milliseconds)
# ===========================================================================


@pytest.mark.agent_test(layer="mock")
async def test_mock_retrieves_and_synthesizes():
    """Verify the agent calls retrieve, rank, then synthesize."""
    llm, tools = _setup_baseline_mocks()
    run = await rag_agent(QUERY, llm=llm, tools=tools)

    tools.assert_tool_called("retrieve_docs")
    tools.assert_tool_called("rank_docs")
    assert run.final_output is not None
    assert len(run.steps) == 3  # retrieve, rank, synthesize


@pytest.mark.agent_test(layer="mock")
async def test_mock_retrieve_passes_query():
    """Verify retrieve_docs receives the original query."""
    llm, tools = _setup_baseline_mocks()
    await rag_agent(QUERY, llm=llm, tools=tools)

    record = tools.assert_tool_called("retrieve_docs")
    assert record.arguments["query"] == QUERY
    assert record.arguments["top_k"] == 3


@pytest.mark.agent_test(layer="mock")
async def test_mock_output_references_sources():
    """Verify the synthesized answer mentions key concepts."""
    llm, tools = _setup_baseline_mocks()
    run = await rag_agent(QUERY, llm=llm, tools=tools)

    output = str(run.final_output).lower()
    assert "asyncio" in output
    assert "threading" in output


# ===========================================================================
# Layer 2: REPLAY — Regression detection via cassette comparison
# ===========================================================================


def _record_cassette(llm, tools, run):
    """Record a cassette from a completed agent run."""
    recorder = CassetteRecorder(test_id="rag_case_study::baseline")

    for call in llm.calls:
        recorder.record_llm_call(
            method="complete",
            request_body={"input": call.input_text},
            response_body={"output": call.response_text},
            model="mock",
            prompt_tokens=call.prompt_tokens,
            completion_tokens=call.completion_tokens,
        )

    for step in run.steps:
        for tc in step.tool_calls:
            recorder.record_tool_call(
                tool_name=tc.name,
                arguments=tc.arguments,
                result=tc.result,
            )

    return recorder.finalize()


@pytest.mark.agent_test(layer="mock")
async def test_replay_baseline_round_trip():
    """Record baseline, verify all interactions captured."""
    llm, tools = _setup_baseline_mocks()
    run = await rag_agent(QUERY, llm=llm, tools=tools)

    cassette = _record_cassette(llm, tools, run)
    # 1 LLM call + 2 tool calls = 3 interactions
    assert len(cassette.interactions) == 3

    engine = ReplayEngine(cassette)
    assert engine.remaining == 3
    assert not engine.all_used


@pytest.mark.agent_test(layer="mock")
async def test_replay_detects_content_drift():
    """Replay detects when retrieved document content changes.

    This simulates an embedding index update where the same query
    returns documents with different content — a retrieval regression.
    """
    # Record baseline
    llm, tools = _setup_baseline_mocks()
    run = await rag_agent(QUERY, llm=llm, tools=tools)
    baseline_cassette = _record_cassette(llm, tools, run)

    # Simulate content drift: doc-1 content changed after index rebuild
    drifted_docs = [
        {
            "id": "doc-1",
            "title": "Python asyncio",
            "content": "asyncio has been deprecated in favor of trio.",  # WRONG
        },
        BASELINE_DOCS[1],
        BASELINE_DOCS[2],
    ]

    llm2 = MockLLM()
    llm2.on_input(contains="answer:").respond(
        "Python concurrency now uses trio instead of asyncio."
    )
    tools2 = MockTool()
    tools2.on_call("retrieve_docs").respond([drifted_docs])
    tools2.on_call("rank_docs").respond([[drifted_docs[0], drifted_docs[1]]])

    run2 = await rag_agent(QUERY, llm=llm2, tools=tools2)
    drifted_cassette = _record_cassette(llm2, tools2, run2)

    # interactions: [0]=LLM, [1]=retrieve_docs, [2]=rank_docs
    # The retrieve_docs response differs — content drift detected
    assert baseline_cassette.interactions[1].response.body != \
        drifted_cassette.interactions[1].response.body, \
        "Content drift should produce different cassette responses"


@pytest.mark.agent_test(layer="mock")
async def test_replay_detects_missing_document():
    """Replay detects when a document disappears from retrieval.

    Simulates a document being removed from the index — fewer
    results returned than baseline.
    """
    # Record baseline
    llm, tools = _setup_baseline_mocks()
    run = await rag_agent(QUERY, llm=llm, tools=tools)
    baseline_cassette = _record_cassette(llm, tools, run)

    # Simulate missing document: only 2 docs returned instead of 3
    partial_docs = BASELINE_DOCS[:2]

    llm2 = MockLLM()
    llm2.on_input(contains="answer:").respond(
        "Python uses asyncio and threading for concurrency."
    )
    tools2 = MockTool()
    tools2.on_call("retrieve_docs").respond([partial_docs])
    tools2.on_call("rank_docs").respond([partial_docs])

    run2 = await rag_agent(QUERY, llm=llm2, tools=tools2)
    partial_cassette = _record_cassette(llm2, tools2, run2)

    # interactions: [0]=LLM, [1]=retrieve_docs, [2]=rank_docs
    baseline_retrieve = baseline_cassette.interactions[1]
    partial_retrieve = partial_cassette.interactions[1]
    assert len(baseline_retrieve.response.body) == 3
    assert len(partial_retrieve.response.body) == 2
    assert baseline_retrieve.response.body != partial_retrieve.response.body


@pytest.mark.agent_test(layer="mock")
async def test_replay_detects_reordering():
    """Replay detects when document ranking changes.

    Simulates a reranker model update that changes the order of
    results — potentially changing the synthesized answer.
    """
    # Record baseline
    llm, tools = _setup_baseline_mocks()
    run = await rag_agent(QUERY, llm=llm, tools=tools)
    baseline_cassette = _record_cassette(llm, tools, run)

    # Simulate reordering: rank_docs returns docs in different order
    reordered_ranked = [BASELINE_DOCS[1], BASELINE_DOCS[0]]  # Swapped

    llm2 = MockLLM()
    llm2.on_input(contains="answer:").respond(
        "Python handles concurrency primarily through threading "
        "and also asyncio for async operations."
    )
    tools2 = MockTool()
    tools2.on_call("retrieve_docs").respond([BASELINE_DOCS])
    tools2.on_call("rank_docs").respond([reordered_ranked])

    run2 = await rag_agent(QUERY, llm=llm2, tools=tools2)
    reordered_cassette = _record_cassette(llm2, tools2, run2)

    # interactions: [0]=LLM, [1]=retrieve_docs, [2]=rank_docs
    baseline_rank = baseline_cassette.interactions[2]
    reordered_rank = reordered_cassette.interactions[2]
    assert baseline_rank.response.body != reordered_rank.response.body


# ===========================================================================
# Layer 3: EVAL — Retrieval quality metrics
# ===========================================================================


@pytest.mark.agent_test(layer="eval")
async def test_eval_tool_correctness():
    """Verify correct tool usage: retrieve then rank."""
    llm, tools = _setup_baseline_mocks()
    run = await rag_agent(QUERY, llm=llm, tools=tools)

    score = tool_correctness(
        run,
        expected_tools=["retrieve_docs", "rank_docs"],
        threshold=1.0,
    )
    assert score.passed
    assert score.metadata["precision"] == 1.0
    assert score.metadata["recall"] == 1.0


@pytest.mark.agent_test(layer="eval")
async def test_eval_trajectory():
    """Verify retrieve → rank → synthesize order."""
    llm, tools = _setup_baseline_mocks()
    run = await rag_agent(QUERY, llm=llm, tools=tools)

    score = trajectory_match(
        run,
        expected_trajectory=["retrieve_docs", "rank_docs"],
        mode="ordered",
    )
    assert score.passed


@pytest.mark.agent_test(layer="eval")
async def test_eval_task_completion():
    """Verify the answer addresses the question with source material."""
    llm, tools = _setup_baseline_mocks()
    run = await rag_agent(QUERY, llm=llm, tools=tools)

    score = task_completion(
        run,
        expected_output_contains=["asyncio", "threading"],
        check_no_error=True,
        threshold=0.6,
    )
    assert score.passed
    assert score.value >= 0.6
