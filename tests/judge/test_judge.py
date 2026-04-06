"""Tests for judge evaluation logic."""

import json

import pytest

from checkagent.core.types import AgentInput, AgentRun, Step, ToolCall
from checkagent.judge.judge import (
    JudgeParseError,
    RubricJudge,
    _build_system_prompt,
    _build_user_prompt,
    _normalize_score,
    _parse_judge_response,
)
from checkagent.judge.types import Criterion, Rubric, ScaleType


def _make_run(query: str = "hello", output: str = "world") -> AgentRun:
    """Helper to create a minimal agent run."""
    return AgentRun(
        input=AgentInput(query=query),
        steps=[
            Step(
                step_index=0,
                input_text=query,
                output_text=output,
                tool_calls=[
                    ToolCall(name="search", arguments={"q": "test"}, result="found")
                ],
            )
        ],
        final_output=output,
    )


def _make_rubric() -> Rubric:
    """Helper to create a standard test rubric."""
    return Rubric(
        name="test_quality",
        description="Evaluate response quality",
        criteria=[
            Criterion(name="accuracy", description="Factually correct?"),
            Criterion(
                name="tone",
                description="Professional tone?",
                scale_type=ScaleType.BINARY,
                scale=["pass", "fail"],
            ),
        ],
    )


class TestNormalizeScore:
    def test_numeric_mid_range(self):
        c = Criterion(name="q", description="d", scale=[1, 2, 3, 4, 5])
        assert _normalize_score(c, 3) == 0.5

    def test_numeric_min(self):
        c = Criterion(name="q", description="d", scale=[1, 2, 3, 4, 5])
        assert _normalize_score(c, 1) == 0.0

    def test_numeric_max(self):
        c = Criterion(name="q", description="d", scale=[1, 2, 3, 4, 5])
        assert _normalize_score(c, 5) == 1.0

    def test_numeric_clamped(self):
        c = Criterion(name="q", description="d", scale=[1, 2, 3, 4, 5])
        assert _normalize_score(c, 10) == 1.0
        assert _normalize_score(c, -5) == 0.0

    def test_binary_pass(self):
        c = Criterion(
            name="q", description="d",
            scale_type=ScaleType.BINARY, scale=["pass", "fail"],
        )
        assert _normalize_score(c, "pass") == 1.0

    def test_binary_fail(self):
        c = Criterion(
            name="q", description="d",
            scale_type=ScaleType.BINARY, scale=["pass", "fail"],
        )
        assert _normalize_score(c, "fail") == 0.0

    def test_binary_bool(self):
        c = Criterion(
            name="q", description="d",
            scale_type=ScaleType.BINARY, scale=["pass", "fail"],
        )
        assert _normalize_score(c, True) == 1.0
        assert _normalize_score(c, False) == 0.0

    def test_categorical(self):
        c = Criterion(
            name="q", description="d",
            scale_type=ScaleType.CATEGORICAL,
            scale=["bad", "ok", "good"],
        )
        assert _normalize_score(c, "bad") == 0.0
        assert _normalize_score(c, "ok") == 0.5
        assert _normalize_score(c, "good") == 1.0

    def test_categorical_unknown(self):
        c = Criterion(
            name="q", description="d",
            scale_type=ScaleType.CATEGORICAL,
            scale=["bad", "ok", "good"],
        )
        assert _normalize_score(c, "unknown") == 0.0

    def test_single_value_numeric(self):
        c = Criterion(name="q", description="d", scale=[5])
        assert _normalize_score(c, 5) == 1.0


class TestParseJudgeResponse:
    def test_valid_json(self):
        rubric = _make_rubric()
        response = json.dumps({
            "scores": [
                {"criterion": "accuracy", "value": 4, "reasoning": "Good"},
                {"criterion": "tone", "value": "pass", "reasoning": "Professional"},
            ],
            "overall_reasoning": "Well done",
        })
        scores, reasoning = _parse_judge_response(response, rubric)
        assert len(scores) == 2
        assert scores[0].criterion_name == "accuracy"
        assert scores[0].normalized == 0.75  # 4 on 1-5 scale
        assert scores[1].normalized == 1.0  # pass
        assert reasoning == "Well done"

    def test_json_with_markdown_fences(self):
        rubric = _make_rubric()
        response = "```json\n" + json.dumps({
            "scores": [
                {"criterion": "accuracy", "value": 3, "reasoning": "Ok"},
            ],
            "overall_reasoning": "Adequate",
        }) + "\n```"
        scores, reasoning = _parse_judge_response(response, rubric)
        assert len(scores) == 1
        assert reasoning == "Adequate"

    def test_unknown_criterion_skipped_with_warning(self):
        rubric = _make_rubric()
        response = json.dumps({
            "scores": [
                {"criterion": "nonexistent", "value": 5, "reasoning": "?"},
            ],
            "overall_reasoning": "",
        })
        with pytest.warns(UserWarning, match="unknown criterion names"):
            scores, _ = _parse_judge_response(response, rubric)
        assert len(scores) == 0

    def test_unknown_criterion_warning_includes_details(self):
        rubric = _make_rubric()
        response = json.dumps({
            "scores": [
                {"criterion": "misspelled_accuracy", "value": 5, "reasoning": "?"},
                {"criterion": "accuracy", "value": 4, "reasoning": "ok"},
            ],
            "overall_reasoning": "",
        })
        with pytest.warns(UserWarning, match="misspelled_accuracy") as rec:
            scores, _ = _parse_judge_response(response, rubric)
        assert len(scores) == 1  # only "accuracy" matched
        assert scores[0].criterion_name == "accuracy"
        # Warning mentions expected criteria
        assert "accuracy" in str(rec[0].message)
        assert "tone" in str(rec[0].message)

    def test_all_criteria_matched_no_warning(self):
        rubric = _make_rubric()
        response = json.dumps({
            "scores": [
                {"criterion": "accuracy", "value": 4, "reasoning": "good"},
                {"criterion": "tone", "value": "pass", "reasoning": "polite"},
            ],
            "overall_reasoning": "nice",
        })
        # No warning when all criteria match
        import warnings as _w
        with _w.catch_warnings():
            _w.simplefilter("error")
            scores, _ = _parse_judge_response(response, rubric)
        assert len(scores) == 2

    def test_invalid_json_raises_judge_parse_error(self):
        rubric = _make_rubric()
        with pytest.raises(JudgeParseError, match="non-JSON response") as exc_info:
            _parse_judge_response("not json at all", rubric)
        assert exc_info.value.raw_response == "not json at all"
        assert "accuracy" in str(exc_info.value)  # includes expected criteria

    def test_judge_parse_error_includes_raw_response(self):
        rubric = _make_rubric()
        verbose_response = "I think the answer is great! " * 50
        with pytest.raises(JudgeParseError) as exc_info:
            _parse_judge_response(verbose_response, rubric)
        # Raw response preserved in full
        assert exc_info.value.raw_response == verbose_response
        # Error message truncates to first 500 chars
        assert "500 chars" in str(exc_info.value)


class TestBuildPrompts:
    def test_system_prompt_includes_criteria(self):
        rubric = _make_rubric()
        prompt = _build_system_prompt(rubric)
        assert "accuracy" in prompt
        assert "tone" in prompt
        assert "test_quality" in prompt

    def test_user_prompt_includes_run_details(self):
        run = _make_run(query="What is 2+2?", output="4")
        prompt = _build_user_prompt(run)
        assert "What is 2+2?" in prompt
        assert "4" in prompt
        assert "search" in prompt


class TestRubricJudge:
    @pytest.mark.asyncio
    async def test_evaluate_with_mock_llm(self):
        rubric = _make_rubric()

        async def mock_llm(system: str, user: str) -> str:
            return json.dumps({
                "scores": [
                    {"criterion": "accuracy", "value": 4, "reasoning": "Good"},
                    {"criterion": "tone", "value": "pass", "reasoning": "Ok"},
                ],
                "overall_reasoning": "Solid",
            })

        judge = RubricJudge(rubric=rubric, llm=mock_llm, model_name="mock")
        run = _make_run()
        score = await judge.evaluate(run)

        assert score.rubric_name == "test_quality"
        assert score.judge_model == "mock"
        assert len(score.criterion_scores) == 2
        # accuracy: 0.75 (weight 1), tone: 1.0 (weight 1) -> avg 0.875
        assert abs(score.overall - 0.875) < 0.01
        assert score.reasoning == "Solid"

    @pytest.mark.asyncio
    async def test_evaluate_with_weighted_criteria(self):
        rubric = Rubric(
            name="weighted",
            criteria=[
                Criterion(name="a", description="d", weight=3.0),
                Criterion(name="b", description="d", weight=1.0),
            ],
        )

        async def mock_llm(system: str, user: str) -> str:
            return json.dumps({
                "scores": [
                    {"criterion": "a", "value": 5, "reasoning": ""},
                    {"criterion": "b", "value": 1, "reasoning": ""},
                ],
                "overall_reasoning": "",
            })

        judge = RubricJudge(rubric=rubric, llm=mock_llm)
        score = await judge.evaluate(_make_run())

        # a: 1.0 * 3.0 = 3.0, b: 0.0 * 1.0 = 0.0, total = 3.0/4.0 = 0.75
        assert abs(score.overall - 0.75) < 0.01

    @pytest.mark.asyncio
    async def test_evaluate_empty_response(self):
        rubric = _make_rubric()

        async def mock_llm(system: str, user: str) -> str:
            return json.dumps({
                "scores": [],
                "overall_reasoning": "No scores",
            })

        judge = RubricJudge(rubric=rubric, llm=mock_llm)
        score = await judge.evaluate(_make_run())
        assert score.overall == 0.0

    @pytest.mark.asyncio
    async def test_judge_name(self):
        rubric = _make_rubric()

        async def mock_llm(system: str, user: str) -> str:
            return '{"scores": [], "overall_reasoning": ""}'

        judge = RubricJudge(rubric=rubric, llm=mock_llm)
        assert judge.name == "rubric_judge:test_quality"
        assert "rubric_judge" in repr(judge)
