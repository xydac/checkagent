"""Tests for statistical verdict computation."""

import json

import pytest

from checkagent.core.types import AgentInput, AgentRun
from checkagent.judge.judge import RubricJudge
from checkagent.judge.types import Criterion, Rubric, Verdict
from checkagent.judge.verdict import compute_verdict


def _make_run() -> AgentRun:
    return AgentRun(
        input=AgentInput(query="test"),
        final_output="result",
    )


def _make_judge(score_value: float) -> RubricJudge:
    """Create a judge that always returns a fixed score."""
    rubric = Rubric(
        name="fixed",
        criteria=[Criterion(name="q", description="d")],
    )

    async def mock_llm(system: str, user: str) -> str:
        # Return score that normalizes to the desired value
        # Scale is 1-5, so value = 1 + score_value * 4
        raw = 1 + score_value * 4
        return json.dumps({
            "scores": [{"criterion": "q", "value": raw, "reasoning": ""}],
            "overall_reasoning": "",
        })

    return RubricJudge(rubric=rubric, llm=mock_llm, model_name="mock")


class TestComputeVerdict:
    @pytest.mark.asyncio
    async def test_clear_pass(self):
        judge = _make_judge(0.9)
        v = await compute_verdict(judge, _make_run(), num_trials=3, threshold=0.7)
        assert v.verdict == Verdict.PASS
        assert v.pass_rate == 1.0
        assert v.num_trials == 3
        assert v.passed is True

    @pytest.mark.asyncio
    async def test_clear_fail(self):
        judge = _make_judge(0.1)
        v = await compute_verdict(judge, _make_run(), num_trials=3, threshold=0.7)
        assert v.verdict == Verdict.FAIL
        assert v.pass_rate == 0.0
        assert v.passed is False

    @pytest.mark.asyncio
    async def test_single_trial(self):
        judge = _make_judge(0.9)
        v = await compute_verdict(judge, _make_run(), num_trials=1, threshold=0.7)
        assert v.verdict == Verdict.PASS
        assert v.num_trials == 1

    @pytest.mark.asyncio
    async def test_invalid_num_trials(self):
        judge = _make_judge(0.5)
        with pytest.raises(ValueError, match="num_trials must be at least 1"):
            await compute_verdict(judge, _make_run(), num_trials=0)

    @pytest.mark.asyncio
    async def test_reasoning_includes_stats(self):
        judge = _make_judge(0.9)
        v = await compute_verdict(judge, _make_run(), num_trials=3, threshold=0.7)
        assert "3/3" in v.reasoning
        assert "100.0%" in v.reasoning

    @pytest.mark.asyncio
    async def test_inconclusive_near_boundary(self):
        """When pass rate is near the boundary, verdict should be inconclusive."""
        # Create a judge that returns scores right at the threshold
        rubric = Rubric(
            name="borderline",
            criteria=[Criterion(name="q", description="d")],
        )
        call_count = 0

        async def varying_llm(system: str, user: str) -> str:
            nonlocal call_count
            call_count += 1
            # Alternate: pass, fail, pass -> 2/3 = 0.667 pass rate
            raw = 5 if call_count % 2 == 1 else 1
            return json.dumps({
                "scores": [{"criterion": "q", "value": raw, "reasoning": ""}],
                "overall_reasoning": "",
            })

        judge = RubricJudge(rubric=rubric, llm=varying_llm)
        v = await compute_verdict(
            judge, _make_run(),
            num_trials=3, threshold=0.7,
            min_pass_rate=0.6, inconclusive_band=0.1,
        )
        # 2/3 = 0.667, min_pass_rate=0.6, band=0.1
        # Range [0.5, 0.7] is inconclusive
        assert v.verdict == Verdict.INCONCLUSIVE
        assert v.confidence == 0.0

    @pytest.mark.asyncio
    async def test_five_trials_all_pass(self):
        judge = _make_judge(0.95)
        v = await compute_verdict(judge, _make_run(), num_trials=5, threshold=0.5)
        assert v.verdict == Verdict.PASS
        assert v.pass_rate == 1.0
        assert v.num_trials == 5

    @pytest.mark.asyncio
    async def test_custom_threshold(self):
        judge = _make_judge(0.5)
        # With threshold=0.3, score of 0.5 should pass
        v = await compute_verdict(judge, _make_run(), num_trials=3, threshold=0.3)
        assert v.verdict == Verdict.PASS

    @pytest.mark.asyncio
    async def test_custom_min_pass_rate(self):
        judge = _make_judge(0.9)
        v = await compute_verdict(
            judge, _make_run(),
            num_trials=3, threshold=0.7,
            min_pass_rate=0.9, inconclusive_band=0.05,
        )
        # 3/3 = 1.0, min_pass_rate=0.9, band=0.05 -> 1.0 >= 0.95 -> PASS
        assert v.verdict == Verdict.PASS
