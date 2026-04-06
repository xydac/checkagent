"""LLM-as-Judge evaluation layer (F4.x).

Rubric-based evaluation with statistical verdicts.
"""

from checkagent.judge.consensus import multi_judge_evaluate
from checkagent.judge.judge import Judge, RubricJudge
from checkagent.judge.types import (
    ConsensusVerdict,
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
    "ConsensusVerdict",
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
    "multi_judge_evaluate",
]
