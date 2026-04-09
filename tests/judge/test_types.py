"""Tests for judge layer types."""

import pytest

from checkagent.judge.types import (
    Criterion,
    CriterionScore,
    JudgeScore,
    JudgeVerdict,
    Rubric,
    ScaleType,
    Verdict,
)


class TestCriterion:
    def test_numeric_defaults(self):
        c = Criterion(name="quality", description="How good?")
        assert c.scale_type == ScaleType.NUMERIC
        assert c.scale == [1, 2, 3, 4, 5]
        assert c.weight == 1.0
        assert c.min_value == 1.0
        assert c.max_value == 5.0

    def test_binary_criterion(self):
        c = Criterion(
            name="correct",
            description="Is it correct?",
            scale_type=ScaleType.BINARY,
            scale=["pass", "fail"],
        )
        assert c.min_value == 0.0
        assert c.max_value == 1.0

    def test_categorical_criterion(self):
        c = Criterion(
            name="compliance",
            description="Policy compliance",
            scale_type=ScaleType.CATEGORICAL,
            scale=["non_compliant", "unclear", "compliant"],
        )
        assert c.min_value == 0.0
        assert c.max_value == 2.0

    def test_custom_weight(self):
        c = Criterion(name="x", description="y", weight=2.5)
        assert c.weight == 2.5

    def test_weight_must_be_positive(self):
        with pytest.raises(ValueError):
            Criterion(name="x", description="y", weight=0.0)

    def test_binary_default_scale(self):
        """BINARY scale_type defaults to ['pass', 'fail'] not [1,2,3,4,5] (F-060)."""
        c = Criterion(name="ok", description="is it ok?", scale_type=ScaleType.BINARY)
        assert c.scale == ["pass", "fail"]

    def test_binary_explicit_scale_preserved(self):
        """Explicit scale on BINARY is not overridden."""
        c = Criterion(
            name="ok", description="x",
            scale_type=ScaleType.BINARY, scale=["yes", "no"],
        )
        assert c.scale == ["yes", "no"]


class TestRubric:
    def test_basic_rubric(self):
        r = Rubric(
            name="test_rubric",
            criteria=[
                Criterion(name="a", description="first"),
                Criterion(name="b", description="second"),
            ],
        )
        assert r.name == "test_rubric"
        assert len(r.criteria) == 2

    def test_get_criterion(self):
        r = Rubric(
            name="r",
            criteria=[Criterion(name="empathy", description="shows empathy")],
        )
        assert r.get_criterion("empathy") is not None
        assert r.get_criterion("nonexistent") is None

    def test_requires_at_least_one_criterion(self):
        with pytest.raises(ValueError):
            Rubric(name="empty", criteria=[])


class TestCriterionScore:
    def test_basic_score(self):
        cs = CriterionScore(
            criterion_name="quality",
            raw_value=4,
            normalized=0.75,
            reasoning="Good quality",
        )
        assert cs.criterion_name == "quality"
        assert cs.normalized == 0.75

    def test_normalized_bounds(self):
        with pytest.raises(ValueError):
            CriterionScore(criterion_name="x", raw_value=1, normalized=1.5)
        with pytest.raises(ValueError):
            CriterionScore(criterion_name="x", raw_value=1, normalized=-0.1)


class TestJudgeScore:
    def test_score_for_criterion(self):
        js = JudgeScore(
            rubric_name="test",
            criterion_scores=[
                CriterionScore(criterion_name="a", raw_value=3, normalized=0.5),
                CriterionScore(criterion_name="b", raw_value=5, normalized=1.0),
            ],
            overall=0.75,
        )
        assert js.score_for("a") is not None
        assert js.score_for("a").normalized == 0.5
        assert js.score_for("missing") is None

    def test_overall_bounds(self):
        with pytest.raises(ValueError):
            JudgeScore(rubric_name="r", overall=1.5)

    def test_passed_above_threshold(self):
        """JudgeScore.passed returns True when overall >= 0.5 (F-058)."""
        assert JudgeScore(rubric_name="r", overall=0.8).passed is True
        assert JudgeScore(rubric_name="r", overall=0.5).passed is True

    def test_passed_below_threshold(self):
        """JudgeScore.passed returns False when overall < 0.5 (F-058)."""
        assert JudgeScore(rubric_name="r", overall=0.3).passed is False
        assert JudgeScore(rubric_name="r", overall=0.0).passed is False


class TestVerdict:
    def test_enum_values(self):
        assert Verdict.PASS == "pass"
        assert Verdict.FAIL == "fail"
        assert Verdict.INCONCLUSIVE == "inconclusive"


class TestJudgeVerdict:
    def test_pass_verdict(self):
        jv = JudgeVerdict(
            verdict=Verdict.PASS,
            pass_rate=0.9,
            threshold=0.7,
        )
        assert jv.passed is True
        assert jv.num_trials == 0

    def test_fail_verdict(self):
        jv = JudgeVerdict(
            verdict=Verdict.FAIL,
            pass_rate=0.1,
            threshold=0.7,
        )
        assert jv.passed is False

    def test_inconclusive_verdict(self):
        jv = JudgeVerdict(
            verdict=Verdict.INCONCLUSIVE,
            pass_rate=0.5,
            threshold=0.7,
        )
        assert jv.passed is False

    def test_num_trials_from_list(self):
        trials = [
            JudgeScore(rubric_name="r", overall=0.8),
            JudgeScore(rubric_name="r", overall=0.6),
        ]
        jv = JudgeVerdict(
            verdict=Verdict.PASS,
            trials=trials,
            pass_rate=0.5,
            threshold=0.7,
        )
        assert jv.num_trials == 2
