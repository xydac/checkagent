"""Tests for the ca_judge pytest fixture."""

import json

import pytest

from checkagent.core.types import AgentInput, AgentRun
from checkagent.judge.judge import RubricJudge
from checkagent.judge.types import Criterion, Rubric


def _make_rubric() -> Rubric:
    return Rubric(
        name="test",
        criteria=[Criterion(name="quality", description="Is it good?")],
    )


async def _mock_llm(system: str, user: str) -> str:
    return json.dumps({
        "scores": [{"criterion": "quality", "value": 4, "reasoning": "good"}],
        "overall_reasoning": "solid",
    })


class TestApJudgeFixture:
    def test_creates_rubric_judge(self, ca_judge):
        judge = ca_judge(_make_rubric(), _mock_llm)
        assert isinstance(judge, RubricJudge)

    def test_with_model_name(self, ca_judge):
        judge = ca_judge(_make_rubric(), _mock_llm, model_name="gpt-4o")
        assert judge.model_name == "gpt-4o"

    @pytest.mark.asyncio
    async def test_evaluate_with_fixture(self, ca_judge):
        judge = ca_judge(_make_rubric(), _mock_llm)
        run = AgentRun(input=AgentInput(query="test"), final_output="result")
        score = await judge.evaluate(run)
        assert score.overall > 0.5
