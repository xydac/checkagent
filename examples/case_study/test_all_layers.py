"""Case study: Customer support agent tested at all four layers.

This file demonstrates CheckAgent's testing pyramid on a single agent:
  Layer 1 (MOCK)   — Deterministic, free, milliseconds
  Layer 2 (REPLAY) — Recorded interactions, cheap, seconds
  Layer 3 (EVAL)   — Metric evaluation against expectations
  Layer 4 (JUDGE)  — LLM-as-judge with statistical assertions
"""

from __future__ import annotations

import pytest
from agent import support_agent

from checkagent.eval.metrics import task_completion, tool_correctness, trajectory_match
from checkagent.judge import (
    Criterion,
    Rubric,
    RubricJudge,
    ScaleType,
    compute_verdict,
)
from checkagent.mock import MockLLM, MockTool
from checkagent.replay import CassetteRecorder, ReplayEngine

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

REFUND_QUERY = "I want a refund for order ORD-456"


def _setup_mocks():
    """Configure mock LLM and tools for a refund scenario."""
    llm = MockLLM()
    llm.on_input(contains="Summarize for customer").respond(
        "Your order ORD-456 is eligible for a full refund. "
        "The refund of $49.99 will be processed within 3-5 business days."
    )
    llm.on_input(contains="refund").respond(
        "Customer is requesting a refund. Let me look up the order."
    )

    tools = MockTool()
    tools.on_call("lookup_order").respond({
        "order_id": "ORD-456",
        "status": "delivered",
        "total": 49.99,
        "items": ["Wireless Mouse"],
    })
    tools.on_call("check_refund_policy").respond({
        "eligible": True,
        "reason": "Within 30-day return window",
        "refund_amount": 49.99,
    })
    return llm, tools


# ===========================================================================
# Layer 1: MOCK — Deterministic unit tests (free, milliseconds)
# ===========================================================================


@pytest.mark.agent_test(layer="mock")
async def test_mock_refund_calls_correct_tools():
    """Verify the agent calls the right tools in the right order."""
    llm, tools = _setup_mocks()
    await support_agent(REFUND_QUERY, llm=llm, tools=tools)

    # Verify both tools were called
    tools.assert_tool_called("lookup_order")
    tools.assert_tool_called("check_refund_policy")

    # Verify tool call arguments
    record = tools.assert_tool_called("lookup_order")
    assert record.arguments == {"order_id": "ORD-456"}


@pytest.mark.agent_test(layer="mock")
async def test_mock_refund_output_content():
    """Verify the final output contains key information."""
    llm, tools = _setup_mocks()
    run = await support_agent(REFUND_QUERY, llm=llm, tools=tools)

    output = str(run.final_output)
    assert "refund" in output.lower()
    assert "ORD-456" in output
    assert "49.99" in output
    assert "3-5 business days" in output


@pytest.mark.agent_test(layer="mock")
async def test_mock_no_refund_skips_policy_check():
    """Non-refund queries should NOT call check_refund_policy."""
    llm = MockLLM()
    llm.on_input(contains="status").respond("Let me check your order status.")
    llm.on_input(contains="Summarize").respond("Order ORD-789 is in transit.")

    tools = MockTool()
    tools.on_call("lookup_order").respond({
        "order_id": "ORD-789",
        "status": "in_transit",
        "total": 29.99,
    })

    await support_agent(
        "What's the status of ORD-789?", llm=llm, tools=tools
    )

    tools.assert_tool_called("lookup_order")
    assert not tools.was_called("check_refund_policy")


# ===========================================================================
# Layer 2: REPLAY — Record-and-replay regression testing
# ===========================================================================


@pytest.mark.agent_test(layer="mock")
async def test_replay_round_trip():
    """Record a session, then replay it and verify identical output."""
    llm, tools = _setup_mocks()

    # Record phase: run the agent and capture interactions
    recorder = CassetteRecorder(test_id="case_study::test_refund")
    run_original = await support_agent(REFUND_QUERY, llm=llm, tools=tools)

    # Record what the LLM returned
    for call in llm.calls:
        recorder.record_llm_call(
            method="complete",
            request_body={"input": call.input_text},
            response_body={"output": call.response_text},
            model="mock",
            prompt_tokens=call.prompt_tokens,
            completion_tokens=call.completion_tokens,
        )

    # Record tool calls from each step
    for step in run_original.steps:
        for tc in step.tool_calls:
            recorder.record_tool_call(
                tool_name=tc.name,
                arguments=tc.arguments,
                result=tc.result,
            )

    cassette = recorder.finalize()
    assert len(cassette.interactions) == 4  # 2 LLM + 2 tool calls

    # Replay phase: verify the engine can match all interactions
    engine = ReplayEngine(cassette)
    assert not engine.all_used
    assert engine.remaining == 4


# ===========================================================================
# Layer 3: EVAL — Metric evaluation with quantitative scores
# ===========================================================================


@pytest.mark.agent_test(layer="eval")
async def test_eval_task_completion():
    """Score the agent's task completion on the refund scenario."""
    llm, tools = _setup_mocks()
    run = await support_agent(REFUND_QUERY, llm=llm, tools=tools)

    score = task_completion(
        run,
        expected_output_contains=["refund", "ORD-456"],
        check_no_error=True,
        threshold=0.6,
    )
    assert score.passed, f"Task completion failed: {score.reason}"
    assert score.value >= 0.6


@pytest.mark.agent_test(layer="eval")
async def test_eval_tool_correctness():
    """Measure precision and recall of tool usage."""
    llm, tools = _setup_mocks()
    run = await support_agent(REFUND_QUERY, llm=llm, tools=tools)

    score = tool_correctness(
        run,
        expected_tools=["lookup_order", "check_refund_policy"],
        threshold=1.0,
    )
    assert score.passed, f"Tool correctness failed: {score.reason}"
    assert score.metadata["precision"] == 1.0
    assert score.metadata["recall"] == 1.0


@pytest.mark.agent_test(layer="eval")
async def test_eval_trajectory():
    """Verify tools are called in the expected order."""
    llm, tools = _setup_mocks()
    run = await support_agent(REFUND_QUERY, llm=llm, tools=tools)

    score = trajectory_match(
        run,
        expected_trajectory=["lookup_order", "check_refund_policy"],
        mode="ordered",
    )
    assert score.passed, f"Trajectory mismatch: {score.reason}"


# ===========================================================================
# Layer 4: JUDGE — LLM-as-judge with statistical assertions
# ===========================================================================

SUPPORT_RUBRIC = Rubric(
    name="support_quality",
    description="Evaluate customer support response quality",
    criteria=[
        Criterion(
            name="helpfulness",
            description="Does the response address the customer's request?",
            scale_type=ScaleType.BINARY,
            scale=["fail", "pass"],
            weight=2.0,
        ),
        Criterion(
            name="completeness",
            description="Does the response include order ID, amount, and timeline?",
            scale_type=ScaleType.NUMERIC,
            scale=[1, 2, 3, 4, 5],
        ),
    ],
)


@pytest.mark.agent_test(layer="judge")
async def test_judge_support_quality():
    """Use an LLM judge (mocked) to evaluate response quality."""
    llm, tools = _setup_mocks()
    run = await support_agent(REFUND_QUERY, llm=llm, tools=tools)

    # Mock the judge LLM to return structured scores
    async def mock_judge_llm(system: str, user: str) -> str:
        import json
        return json.dumps({
            "scores": [
                {
                    "criterion": "helpfulness",
                    "value": "pass",
                    "reasoning": "Addresses refund request directly",
                },
                {
                    "criterion": "completeness",
                    "value": 5,
                    "reasoning": "Includes order ID, amount, and timeline",
                },
            ],
            "overall_reasoning": "Response is helpful and complete",
        })

    judge = RubricJudge(
        rubric=SUPPORT_RUBRIC,
        llm=mock_judge_llm,
        model_name="mock-judge",
    )

    verdict = await compute_verdict(
        judge,
        run,
        num_trials=3,
        threshold=0.7,
        min_pass_rate=0.6,
    )
    assert verdict.passed, f"Judge verdict: {verdict.verdict.value}"
    assert verdict.pass_rate == 1.0
