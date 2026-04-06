"""LLM-as-Judge evaluation layer (F4.x).

Rubric-based evaluation with statistical verdicts.
"""

from checkagent.judge.judge import Judge, RubricJudge
from checkagent.judge.types import (
    Criterion,
    CriterionScore,
    JudgeScore,
    JudgeVerdict,
    Rubric,
    ScaleType,
    Verdict,
)
from checkagent.judge.verdict import compute_verdict

__all__ = [
    "Criterion",
    "CriterionScore",
    "Judge",
    "JudgeScore",
    "JudgeVerdict",
    "Rubric",
    "RubricJudge",
    "ScaleType",
    "Verdict",
    "compute_verdict",
]
