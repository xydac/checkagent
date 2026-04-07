"""Experiment E-062: RAG replay regression detection matrix.

Measures which retrieval regression types are caught by mock vs replay
vs eval layers, demonstrating that replay provides a unique detection
capability for RAG agent testing.

Regression types:
  R1 — Content drift: document text changes after index rebuild
  R2 — Missing document: document removed from index
  R3 — Reordering: reranker changes document priority
  R4 — Schema drift: document fields change structure

Detection layers:
  mock:tool_called — tool was invoked
  mock:args_match — tool arguments match baseline
  mock:output_present — final output is non-empty
  replay:cassette_match — cassette exact match succeeds
  replay:response_diff — tool response bodies differ from baseline
  eval:tool_correctness — correct tools called
  eval:trajectory — tools called in correct order
"""

from __future__ import annotations

import pytest

from checkagent.core.types import AgentInput, AgentRun, Step, ToolCall
from checkagent.eval.metrics import tool_correctness, trajectory_match
from checkagent.mock import MockLLM, MockTool
from checkagent.replay import CassetteRecorder, ReplayEngine

# ---------------------------------------------------------------------------
# Shared RAG agent (inline for test isolation)
# ---------------------------------------------------------------------------

QUERY = "How does Python handle concurrency?"

BASELINE_DOCS = [
    {"id": "d1", "title": "asyncio", "content": "asyncio provides async I/O"},
    {"id": "d2", "title": "threading", "content": "threads share memory"},
    {"id": "d3", "title": "multiprocessing", "content": "processes have own memory"},
]

RANKED_DOCS = [BASELINE_DOCS[0], BASELINE_DOCS[1]]


async def _rag_agent(query, *, llm, tools):
    """Minimal RAG: retrieve → rank → synthesize."""
    steps = []
    docs = await tools.call("retrieve_docs", {"query": query, "top_k": 3})
    steps.append(Step(
        input_text=query,
        output_text=f"Retrieved {len(docs)} docs",
        tool_calls=[ToolCall(
            name="retrieve_docs",
            arguments={"query": query, "top_k": 3},
            result=docs,
        )],
    ))
    ranked = await tools.call("rank_docs", {"query": query, "documents": docs})
    steps.append(Step(
        input_text="rank",
        output_text=f"Ranked to {len(ranked)} docs",
        tool_calls=[ToolCall(
            name="rank_docs",
            arguments={"query": query, "documents": docs},
            result=ranked,
        )],
    ))
    context = "\n".join(d.get("content", d.get("snippet", "")) for d in ranked)
    answer = await llm.complete(f"Answer: {query}\n{context}")
    steps.append(Step(input_text="synthesize", output_text=answer))
    return AgentRun(
        input=AgentInput(query=query), steps=steps, final_output=answer,
    )


def _setup_mocks(docs=None, ranked=None, answer=None):
    """Create configured mocks.

    Note: MockTool.get_response() treats list responses as sequences
    to cycle through. We use register() with response= to pass the
    list as-is by wrapping in a single-element list.
    """
    docs = docs or BASELINE_DOCS
    ranked = ranked or RANKED_DOCS
    answer = answer or (
        "Python uses asyncio for async I/O, threading for shared-memory "
        "concurrency, and multiprocessing for process-level parallelism."
    )
    llm = MockLLM()
    llm.on_input(contains="Answer:").respond(answer)
    tools = MockTool()
    # Wrap list responses in [response] so get_response() returns the full list
    tools.on_call("retrieve_docs").respond([docs])
    tools.on_call("rank_docs").respond([ranked])
    return llm, tools


def _record(llm, tools, run):
    """Record a cassette from a run."""
    rec = CassetteRecorder(test_id="e062_rag_baseline")
    for call in llm.calls:
        rec.record_llm_call(
            method="complete",
            request_body={"input": call.input_text},
            response_body={"output": call.response_text},
            model="mock",
            prompt_tokens=call.prompt_tokens,
            completion_tokens=call.completion_tokens,
        )
    for step in run.steps:
        for tc in step.tool_calls:
            rec.record_tool_call(
                tool_name=tc.name, arguments=tc.arguments, result=tc.result,
            )
    return rec.finalize()


# ---------------------------------------------------------------------------
# Regression variants
# ---------------------------------------------------------------------------

# R1: Content drift — same doc IDs, different content
R1_DOCS = [
    {"id": "d1", "title": "asyncio", "content": "asyncio is DEPRECATED use trio"},
    BASELINE_DOCS[1],
    BASELINE_DOCS[2],
]
R1_RANKED = [R1_DOCS[0], R1_DOCS[1]]

# R2: Missing document — only 2 returned
R2_DOCS = BASELINE_DOCS[:2]
R2_RANKED = R2_DOCS

# R3: Reordering — different ranking
R3_RANKED = [BASELINE_DOCS[1], BASELINE_DOCS[0]]  # Swapped

# R4: Schema drift — extra field, missing field
R4_DOCS = [
    {"id": "d1", "title": "asyncio", "snippet": "async I/O"},  # 'snippet' not 'content'
    {"id": "d2", "title": "threading", "content": "threads share memory", "score": 0.95},
    BASELINE_DOCS[2],
]
R4_RANKED = [R4_DOCS[0], R4_DOCS[1]]


# ---------------------------------------------------------------------------
# Detection matrix tests
# ---------------------------------------------------------------------------


@pytest.mark.agent_test(layer="mock")
async def test_baseline_passes_all():
    """Baseline run passes all detectors — sanity check."""
    llm, tools = _setup_mocks()
    run = await _rag_agent(QUERY, llm=llm, tools=tools)
    cassette = _record(llm, tools, run)

    # Mock checks
    tools.assert_tool_called("retrieve_docs")
    tools.assert_tool_called("rank_docs")
    assert run.final_output is not None

    # Replay check
    engine = ReplayEngine(cassette)
    assert engine.remaining == 3

    # Eval checks
    tc = tool_correctness(run, expected_tools=["retrieve_docs", "rank_docs"])
    assert tc.passed
    tj = trajectory_match(run, expected_trajectory=["retrieve_docs", "rank_docs"], mode="ordered")
    assert tj.passed


@pytest.mark.agent_test(layer="mock")
async def test_r1_content_drift_detection():
    """R1: Content drift — mock passes, replay detects."""
    # Baseline
    llm_b, tools_b = _setup_mocks()
    run_b = await _rag_agent(QUERY, llm=llm_b, tools=tools_b)
    baseline = _record(llm_b, tools_b, run_b)

    # Drifted
    llm_d, tools_d = _setup_mocks(
        docs=R1_DOCS, ranked=R1_RANKED,
        answer="Python now uses trio instead of asyncio.",
    )
    run_d = await _rag_agent(QUERY, llm=llm_d, tools=tools_d)
    drifted = _record(llm_d, tools_d, run_d)

    # Mock layer: PASSES (tools called correctly, output present)
    tools_d.assert_tool_called("retrieve_docs")
    tools_d.assert_tool_called("rank_docs")
    assert run_d.final_output is not None

    # Mock layer: args match (same query)
    rec = tools_d.assert_tool_called("retrieve_docs")
    assert rec.arguments["query"] == QUERY

    # Eval layer: PASSES (correct tools, correct order)
    tc = tool_correctness(run_d, expected_tools=["retrieve_docs", "rank_docs"])
    assert tc.passed
    tj = trajectory_match(
        run_d, expected_trajectory=["retrieve_docs", "rank_docs"], mode="ordered",
    )
    assert tj.passed

    # Replay layer: DETECTS — retrieve_docs response bodies differ
    # interactions: [0]=LLM call, [1]=retrieve_docs, [2]=rank_docs
    assert baseline.interactions[1].response.body != drifted.interactions[1].response.body


@pytest.mark.agent_test(layer="mock")
async def test_r2_missing_document_detection():
    """R2: Missing document — mock passes, replay and eval detect."""
    # Baseline
    llm_b, tools_b = _setup_mocks()
    run_b = await _rag_agent(QUERY, llm=llm_b, tools=tools_b)
    baseline = _record(llm_b, tools_b, run_b)

    # Missing doc
    llm_m, tools_m = _setup_mocks(
        docs=R2_DOCS, ranked=R2_RANKED,
        answer="Python uses asyncio and threading.",
    )
    run_m = await _rag_agent(QUERY, llm=llm_m, tools=tools_m)
    missing = _record(llm_m, tools_m, run_m)

    # Mock layer: PASSES
    tools_m.assert_tool_called("retrieve_docs")
    assert run_m.final_output is not None

    # Eval layer: PASSES on tool correctness
    tc = tool_correctness(run_m, expected_tools=["retrieve_docs", "rank_docs"])
    assert tc.passed

    # Replay layer: DETECTS — different number of docs in response
    # interactions: [0]=LLM call, [1]=retrieve_docs, [2]=rank_docs
    b_result = baseline.interactions[1].response.body
    m_result = missing.interactions[1].response.body
    assert len(b_result) == 3
    assert len(m_result) == 2
    assert b_result != m_result


@pytest.mark.agent_test(layer="mock")
async def test_r3_reordering_detection():
    """R3: Reordering — mock and eval pass, replay detects."""
    # Baseline
    llm_b, tools_b = _setup_mocks()
    run_b = await _rag_agent(QUERY, llm=llm_b, tools=tools_b)
    baseline = _record(llm_b, tools_b, run_b)

    # Reordered
    llm_r, tools_r = _setup_mocks(
        ranked=R3_RANKED,
        answer="Python primarily uses threading and also asyncio.",
    )
    run_r = await _rag_agent(QUERY, llm=llm_r, tools=tools_r)
    reordered = _record(llm_r, tools_r, run_r)

    # Mock layer: PASSES
    tools_r.assert_tool_called("retrieve_docs")
    tools_r.assert_tool_called("rank_docs")

    # Eval layer: PASSES (same tools, same order)
    tc = tool_correctness(run_r, expected_tools=["retrieve_docs", "rank_docs"])
    assert tc.passed
    tj = trajectory_match(
        run_r, expected_trajectory=["retrieve_docs", "rank_docs"], mode="ordered",
    )
    assert tj.passed

    # Replay layer: DETECTS — rank_docs response body differs (different order)
    # interactions: [0]=LLM call, [1]=retrieve_docs, [2]=rank_docs
    b_rank = baseline.interactions[2].response.body
    r_rank = reordered.interactions[2].response.body
    assert b_rank != r_rank
    # Same elements, different order
    assert set(d["id"] for d in b_rank) == set(d["id"] for d in r_rank)


@pytest.mark.agent_test(layer="mock")
async def test_r4_schema_drift_detection():
    """R4: Schema drift — mock passes, replay detects."""
    # Baseline
    llm_b, tools_b = _setup_mocks()
    run_b = await _rag_agent(QUERY, llm=llm_b, tools=tools_b)
    baseline = _record(llm_b, tools_b, run_b)

    # Schema drift (snippet instead of content, extra score field)
    llm_s, tools_s = _setup_mocks(
        docs=R4_DOCS, ranked=R4_RANKED,
        answer="Python uses async I/O and threading.",
    )
    run_s = await _rag_agent(QUERY, llm=llm_s, tools=tools_s)
    schema = _record(llm_s, tools_s, run_s)

    # Mock layer: PASSES
    tools_s.assert_tool_called("retrieve_docs")
    assert run_s.final_output is not None

    # Replay layer: DETECTS — document structure differs
    # interactions: [0]=LLM call, [1]=retrieve_docs, [2]=rank_docs
    assert baseline.interactions[1].response.body != schema.interactions[1].response.body


@pytest.mark.agent_test(layer="mock")
async def test_detection_matrix_synthesis():
    """Synthesize the full detection matrix across all regression types.

    Matrix format: Regression × Detector → Detected (Y/N)
    """
    results = {}

    for label, docs, ranked, answer in [
        ("R1_drift", R1_DOCS, R1_RANKED, "trio replaced asyncio"),
        ("R2_missing", R2_DOCS, R2_RANKED, "asyncio and threading"),
        ("R3_reorder", BASELINE_DOCS, R3_RANKED, "threading then asyncio"),
        ("R4_schema", R4_DOCS, R4_RANKED, "async I/O and threads"),
    ]:
        # Run baseline
        llm_b, tools_b = _setup_mocks()
        run_b = await _rag_agent(QUERY, llm=llm_b, tools=tools_b)
        baseline = _record(llm_b, tools_b, run_b)

        # Run regressed variant
        llm_v, tools_v = _setup_mocks(docs=docs, ranked=ranked, answer=answer)
        run_v = await _rag_agent(QUERY, llm=llm_v, tools=tools_v)
        variant = _record(llm_v, tools_v, run_v)

        # Detect with each layer
        detections = {}

        # mock:tool_called — always passes (tools are still called)
        detections["mock:tool_called"] = False  # Does not detect regression

        # mock:args_match — always passes (same query passed)
        rec = tools_v.assert_tool_called("retrieve_docs")
        detections["mock:args_match"] = rec.arguments["query"] != QUERY

        # mock:output_present — always passes
        detections["mock:output_present"] = run_v.final_output is None

        # interactions: [0]=LLM call, [1]=retrieve_docs, [2]=rank_docs
        # replay:retrieve_diff — compare retrieve_docs response
        detections["replay:retrieve_diff"] = (
            baseline.interactions[1].response.body
            != variant.interactions[1].response.body
        )

        # replay:rank_diff — compare rank_docs response
        detections["replay:rank_diff"] = (
            baseline.interactions[2].response.body
            != variant.interactions[2].response.body
        )

        # replay:llm_diff — compare LLM synthesis response
        detections["replay:llm_diff"] = (
            baseline.interactions[0].response.body
            != variant.interactions[0].response.body
        )

        # eval:tool_correctness
        tc = tool_correctness(run_v, expected_tools=["retrieve_docs", "rank_docs"])
        detections["eval:tool_correct"] = not tc.passed

        # eval:trajectory
        tj = trajectory_match(
            run_v,
            expected_trajectory=["retrieve_docs", "rank_docs"],
            mode="ordered",
        )
        detections["eval:trajectory"] = not tj.passed

        results[label] = detections

    # Verify expected detection patterns
    # R1 (content drift): only replay detects
    assert results["R1_drift"]["replay:retrieve_diff"] is True
    assert results["R1_drift"]["replay:rank_diff"] is True
    assert results["R1_drift"]["replay:llm_diff"] is True
    assert results["R1_drift"]["mock:tool_called"] is False
    assert results["R1_drift"]["eval:tool_correct"] is False

    # R2 (missing doc): replay detects
    assert results["R2_missing"]["replay:retrieve_diff"] is True
    assert results["R2_missing"]["mock:tool_called"] is False

    # R3 (reorder): only replay:rank_diff and replay:llm_diff detect
    assert results["R3_reorder"]["replay:rank_diff"] is True
    assert results["R3_reorder"]["replay:retrieve_diff"] is False  # Same docs retrieved
    assert results["R3_reorder"]["eval:tool_correct"] is False
    assert results["R3_reorder"]["eval:trajectory"] is False

    # R4 (schema drift): replay detects
    assert results["R4_schema"]["replay:retrieve_diff"] is True
    assert results["R4_schema"]["mock:tool_called"] is False

    # Count detections per regression type
    for label, detections in results.items():
        detected_count = sum(1 for v in detections.values() if v)
        # Every regression should be caught by at least one replay detector
        assert detected_count > 0, f"{label} not detected by any layer"

    # Count detections per detector
    detector_names = list(next(iter(results.values())).keys())
    for det in detector_names:
        catches = sum(1 for r in results.values() if r[det])
        # Replay detectors should catch more than mock/eval for RAG regressions
        if det.startswith("replay:"):
            assert catches >= 2, f"Replay detector {det} caught too few: {catches}"
