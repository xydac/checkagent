"""Tests for multi-judge consensus evaluation (F4.4)."""

import json

import pytest

from checkagent.core.types import AgentInput, AgentRun
from checkagent.judge.consensus import multi_judge_evaluate
from checkagent.judge.judge import RubricJudge
from checkagent.judge.types import (
    ConsensusVerdict,
    Criterion,
    Rubric,
    Verdict,
)


def _make_run() -> AgentRun:
    return AgentRun(
        input=AgentInput(query="test"),
        final_output="result",
    )


def _make_judge(score_value: float, name: str = "judge") -> RubricJudge:
    """Create a judge that always returns a fixed normalized score."""
    rubric = Rubric(
        name=f"rubric_{name}",
        criteria=[Criterion(name="q", description="d")],
    )

    async def mock_llm(system: str, user: str) -> str:
        raw = 1 + score_value * 4  # scale 1-5 → normalized score_value
        return json.dumps({
            "scores": [{"criterion": "q", "value": raw, "reasoning": "ok"}],
            "overall_reasoning": "summary",
        })

    return RubricJudge(rubric=rubric, llm=mock_llm, model_name=f"mock-{name}")


class TestConsensusVerdict:
    def test_consensus_verdict_properties(self):
        cv = ConsensusVerdict(
            verdict=Verdict.PASS,
            judge_verdicts={},
            agreement_rate=1.0,
        )
        assert cv.passed is True
        assert cv.num_judges == 0

    def test_disagreement_flag(self):
        cv = ConsensusVerdict(
            verdict=Verdict.PASS,
            has_disagreement=True,
            agreement_rate=0.67,
        )
        assert cv.has_disagreement is True


class TestMultiJudgeEvaluate:
    @pytest.mark.asyncio
    async def test_requires_at_least_two_judges(self):
        judge = _make_judge(0.9, "solo")
        with pytest.raises(ValueError, match="at least 2 judges"):
            await multi_judge_evaluate([judge], _make_run())

    @pytest.mark.asyncio
    async def test_unanimous_pass(self):
        judges = [_make_judge(0.9, f"j{i}") for i in range(3)]
        result = await multi_judge_evaluate(
            judges, _make_run(), num_trials=3, threshold=0.7
        )
        assert result.verdict == Verdict.PASS
        assert result.passed is True
        assert result.agreement_rate == 1.0
        assert result.has_disagreement is False
        assert result.num_judges == 3
        assert len(result.judge_verdicts) == 3

    @pytest.mark.asyncio
    async def test_unanimous_fail(self):
        judges = [_make_judge(0.1, f"j{i}") for i in range(3)]
        result = await multi_judge_evaluate(
            judges, _make_run(), num_trials=3, threshold=0.7
        )
        assert result.verdict == Verdict.FAIL
        assert result.passed is False
        assert result.agreement_rate == 1.0
        assert result.has_disagreement is False

    @pytest.mark.asyncio
    async def test_majority_pass_with_disagreement(self):
        # 2 passing judges, 1 failing → majority PASS
        judges = [
            _make_judge(0.9, "pass1"),
            _make_judge(0.9, "pass2"),
            _make_judge(0.1, "fail1"),
        ]
        result = await multi_judge_evaluate(
            judges, _make_run(), num_trials=3, threshold=0.7
        )
        assert result.verdict == Verdict.PASS
        assert result.has_disagreement is True
        assert result.agreement_rate == pytest.approx(2 / 3, abs=0.01)

    @pytest.mark.asyncio
    async def test_majority_fail_with_disagreement(self):
        # 1 passing, 2 failing → majority FAIL
        judges = [
            _make_judge(0.9, "pass1"),
            _make_judge(0.1, "fail1"),
            _make_judge(0.1, "fail2"),
        ]
        result = await multi_judge_evaluate(
            judges, _make_run(), num_trials=3, threshold=0.7
        )
        assert result.verdict == Verdict.FAIL
        assert result.has_disagreement is True

    @pytest.mark.asyncio
    async def test_two_judges_agree(self):
        judges = [
            _make_judge(0.9, "j1"),
            _make_judge(0.9, "j2"),
        ]
        result = await multi_judge_evaluate(
            judges, _make_run(), num_trials=3, threshold=0.7
        )
        assert result.verdict == Verdict.PASS
        assert result.num_judges == 2

    @pytest.mark.asyncio
    async def test_two_judges_disagree(self):
        judges = [
            _make_judge(0.9, "pass"),
            _make_judge(0.1, "fail"),
        ]
        result = await multi_judge_evaluate(
            judges, _make_run(), num_trials=3, threshold=0.7
        )
        # With 1 PASS and 1 FAIL, most_common picks one
        assert result.verdict in (Verdict.PASS, Verdict.FAIL)
        assert result.has_disagreement is True
        assert result.agreement_rate == 0.5

    @pytest.mark.asyncio
    async def test_sequential_mode(self):
        judges = [
            _make_judge(0.9, "j1"),
            _make_judge(0.9, "j2"),
        ]
        result = await multi_judge_evaluate(
            judges, _make_run(), num_trials=2, threshold=0.7, concurrent=False
        )
        assert result.verdict == Verdict.PASS

    @pytest.mark.asyncio
    async def test_per_judge_verdicts_keyed_by_name_and_model(self):
        judges = [
            _make_judge(0.9, "alpha"),
            _make_judge(0.1, "beta"),
        ]
        result = await multi_judge_evaluate(
            judges, _make_run(), num_trials=3, threshold=0.7
        )
        # Keys include model_name for uniqueness
        assert "rubric_judge:rubric_alpha:mock-alpha" in result.judge_verdicts
        assert "rubric_judge:rubric_beta:mock-beta" in result.judge_verdicts

    @pytest.mark.asyncio
    async def test_reasoning_includes_verdict_summary(self):
        judges = [
            _make_judge(0.9, "j1"),
            _make_judge(0.9, "j2"),
        ]
        result = await multi_judge_evaluate(
            judges, _make_run(), num_trials=3, threshold=0.7
        )
        assert "Consensus: pass" in result.reasoning
        assert "2 judges" in result.reasoning

    @pytest.mark.asyncio
    async def test_custom_threshold_and_trials(self):
        judges = [
            _make_judge(0.6, "j1"),
            _make_judge(0.6, "j2"),
        ]
        # With threshold=0.5, score 0.6 should pass
        result = await multi_judge_evaluate(
            judges, _make_run(), num_trials=3, threshold=0.5
        )
        assert result.verdict == Verdict.PASS

    @pytest.mark.asyncio
    async def test_shared_rubric_different_models_no_collision(self):
        """F-052: judges sharing a rubric but with different models get unique keys."""
        shared_rubric = Rubric(
            name="quality",
            criteria=[Criterion(name="q", description="d")],
        )

        async def make_llm(score_val: float):
            async def mock_llm(system: str, user: str) -> str:
                raw = 1 + score_val * 4
                return json.dumps({
                    "scores": [{"criterion": "q", "value": raw, "reasoning": "ok"}],
                    "overall_reasoning": "summary",
                })
            return mock_llm

        judge_gpt = RubricJudge(
            rubric=shared_rubric,
            llm=await make_llm(0.9),
            model_name="gpt-4",
        )
        judge_claude = RubricJudge(
            rubric=shared_rubric,
            llm=await make_llm(0.8),
            model_name="claude-3",
        )
        judge_gemini = RubricJudge(
            rubric=shared_rubric,
            llm=await make_llm(0.7),
            model_name="gemini-pro",
        )

        result = await multi_judge_evaluate(
            [judge_gpt, judge_claude, judge_gemini],
            _make_run(),
            num_trials=3,
            threshold=0.7,
        )

        # All three judges should have distinct entries
        assert len(result.judge_verdicts) == 3
        assert "rubric_judge:quality:gpt-4" in result.judge_verdicts
        assert "rubric_judge:quality:claude-3" in result.judge_verdicts
        assert "rubric_judge:quality:gemini-pro" in result.judge_verdicts

    @pytest.mark.asyncio
    async def test_identical_judges_get_indexed_keys(self):
        """Edge case: same name AND same model_name still get unique keys."""
        judges = [
            _make_judge(0.9, "same"),
            _make_judge(0.8, "same"),
        ]
        # Both have name="rubric_judge:rubric_same" and model_name="mock-same"
        result = await multi_judge_evaluate(
            judges, _make_run(), num_trials=3, threshold=0.7
        )
        # Should have 2 unique entries, not 1
        assert len(result.judge_verdicts) == 2
