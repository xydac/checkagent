"""Judge layer types — rubrics, criteria, verdicts.

Defines the data model for LLM-as-judge evaluations (F4.1, F4.2).
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class ScaleType(str, Enum):
    """Types of scoring scales for rubric criteria."""

    NUMERIC = "numeric"
    BINARY = "binary"
    CATEGORICAL = "categorical"


class Criterion(BaseModel):
    """A single evaluation criterion within a rubric.

    Examples::

        Criterion(name="empathy", description="Acknowledged frustration?",
                  scale_type=ScaleType.NUMERIC, scale=[1, 2, 3, 4, 5])

        Criterion(name="accuracy", description="Factually correct?",
                  scale_type=ScaleType.BINARY, scale=["pass", "fail"])
    """

    name: str
    description: str
    scale_type: ScaleType = ScaleType.NUMERIC
    scale: list[Any] = Field(default_factory=lambda: [1, 2, 3, 4, 5])
    weight: float = Field(default=1.0, gt=0.0)

    @property
    def max_value(self) -> float:
        """Maximum numeric score for this criterion."""
        if self.scale_type == ScaleType.NUMERIC:
            return float(max(self.scale))
        if self.scale_type == ScaleType.BINARY:
            return 1.0
        # Categorical: last item is "best"
        return float(len(self.scale) - 1)

    @property
    def min_value(self) -> float:
        """Minimum numeric score for this criterion."""
        if self.scale_type == ScaleType.NUMERIC:
            return float(min(self.scale))
        return 0.0


class Rubric(BaseModel):
    """A complete evaluation rubric with named criteria.

    Rubrics are defined in YAML config or code and describe how a
    judge should evaluate an agent run.
    """

    name: str
    description: str = ""
    criteria: list[Criterion] = Field(min_length=1)
    metadata: dict[str, Any] = Field(default_factory=dict)

    def get_criterion(self, name: str) -> Criterion | None:
        """Look up a criterion by name."""
        for c in self.criteria:
            if c.name == name:
                return c
        return None


class CriterionScore(BaseModel):
    """Score for a single criterion from one judge trial."""

    criterion_name: str
    raw_value: Any
    normalized: float = Field(ge=0.0, le=1.0)
    reasoning: str = ""


class JudgeScore(BaseModel):
    """Complete judge output for one evaluation trial."""

    rubric_name: str
    criterion_scores: list[CriterionScore] = Field(default_factory=list)
    overall: float = Field(ge=0.0, le=1.0)
    reasoning: str = ""
    judge_model: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)

    def score_for(self, criterion_name: str) -> CriterionScore | None:
        """Look up a criterion score by name."""
        for cs in self.criterion_scores:
            if cs.criterion_name == criterion_name:
                return cs
        return None


class Verdict(str, Enum):
    """Three-valued verdict for statistical assertions (F4.2)."""

    PASS = "pass"
    FAIL = "fail"
    INCONCLUSIVE = "inconclusive"


class JudgeVerdict(BaseModel):
    """Statistical verdict computed from multiple judge trials.

    Aggregates K trials into a three-valued verdict using a
    configurable threshold and minimum pass rate.
    """

    verdict: Verdict
    trials: list[JudgeScore] = Field(default_factory=list)
    pass_rate: float = Field(ge=0.0, le=1.0)
    threshold: float = Field(ge=0.0, le=1.0)
    min_pass_rate: float = Field(default=0.5, ge=0.0, le=1.0)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    reasoning: str = ""

    @property
    def num_trials(self) -> int:
        return len(self.trials)

    @property
    def passed(self) -> bool:
        return self.verdict == Verdict.PASS


class ConsensusVerdict(BaseModel):
    """Aggregated verdict from multiple judges (F4.4).

    Runs multiple judges on the same agent run and computes a
    majority-vote consensus. Disagreements are flagged when judges
    produce different verdicts.
    """

    verdict: Verdict
    judge_verdicts: dict[str, JudgeVerdict] = Field(default_factory=dict)
    agreement_rate: float = Field(ge=0.0, le=1.0)
    has_disagreement: bool = False
    reasoning: str = ""

    @property
    def num_judges(self) -> int:
        return len(self.judge_verdicts)

    @property
    def passed(self) -> bool:
        return self.verdict == Verdict.PASS
