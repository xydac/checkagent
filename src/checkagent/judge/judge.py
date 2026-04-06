"""Judge protocol and rubric-based evaluation (F4.1, F4.3).

Judges evaluate agent runs against rubrics by calling an LLM to
score each criterion. The judge model is always separate from the
agent model being tested.
"""

from __future__ import annotations

import json
import warnings
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any, Protocol, runtime_checkable

from checkagent.core.types import AgentRun
from checkagent.judge.types import (
    Criterion,
    CriterionScore,
    JudgeScore,
    Rubric,
    ScaleType,
)


class JudgeParseError(Exception):
    """Raised when the judge LLM returns unparseable output.

    Includes the raw response so users can diagnose the issue.
    """

    def __init__(self, message: str, raw_response: str) -> None:
        self.raw_response = raw_response
        super().__init__(message)


@runtime_checkable
class LLMCallable(Protocol):
    """Protocol for an async function that calls an LLM.

    Takes a system prompt and user prompt, returns the raw text response.
    This keeps the judge decoupled from any specific LLM SDK.
    """

    async def __call__(self, system: str, user: str) -> str: ...


class Judge(ABC):
    """Base class for judges that evaluate agent runs.

    Subclass to implement custom judging strategies. The simplest
    approach is to use ``RubricJudge`` with a rubric and LLM callable.
    """

    name: str = "unnamed_judge"

    @abstractmethod
    async def evaluate(self, run: AgentRun, **kwargs: Any) -> JudgeScore:
        """Evaluate an agent run and return a structured score."""
        ...

    def __repr__(self) -> str:
        return f"{type(self).__name__}(name={self.name!r})"


def _build_system_prompt(rubric: Rubric) -> str:
    """Build the system prompt for rubric-based evaluation."""
    criteria_text = []
    for c in rubric.criteria:
        scale_desc = f"Scale: {c.scale}"
        if c.scale_type == ScaleType.BINARY:
            scale_desc = f"Binary: {c.scale}"
        elif c.scale_type == ScaleType.CATEGORICAL:
            scale_desc = f"Categories: {c.scale} (first=worst, last=best)"
        criteria_text.append(
            f"- {c.name}: {c.description} ({scale_desc}, weight={c.weight})"
        )

    return f"""You are an evaluation judge. Score the agent's output against this rubric.

Rubric: {rubric.name}
{rubric.description}

Criteria:
{chr(10).join(criteria_text)}

Respond with ONLY a JSON object (no markdown fences) with this structure:
{{
  "scores": [
    {{"criterion": "<name>", "value": <score>, "reasoning": "<brief explanation>"}}
  ],
  "overall_reasoning": "<summary>"
}}

Use the exact scale values defined above. Be specific in your reasoning."""


def _build_user_prompt(run: AgentRun) -> str:
    """Build the user prompt with the agent run details."""
    steps_text = []
    for i, step in enumerate(run.steps):
        parts = [f"Step {i + 1}:"]
        if step.input_text:
            parts.append(f"  Input: {step.input_text}")
        if step.output_text:
            parts.append(f"  Output: {step.output_text}")
        for tc in step.tool_calls:
            result_str = f" -> {tc.result}" if tc.result is not None else ""
            parts.append(f"  Tool: {tc.name}({tc.arguments}){result_str}")
        steps_text.append("\n".join(parts))

    return f"""Agent Input: {run.input.query}

Agent Steps:
{chr(10).join(steps_text) if steps_text else "(no steps recorded)"}

Final Output: {run.final_output}"""


def _normalize_score(criterion: Criterion, raw_value: Any) -> float:
    """Normalize a raw criterion score to [0, 1]."""
    if criterion.scale_type == ScaleType.BINARY:
        # Accept string or bool
        if isinstance(raw_value, bool):
            return 1.0 if raw_value else 0.0
        val = str(raw_value).lower().strip()
        passing = {"pass", "true", "yes", "1"}
        return 1.0 if val in passing else 0.0

    if criterion.scale_type == ScaleType.NUMERIC:
        min_val = criterion.min_value
        max_val = criterion.max_value
        if max_val == min_val:
            return 1.0
        return max(0.0, min(1.0, (float(raw_value) - min_val) / (max_val - min_val)))

    if criterion.scale_type == ScaleType.CATEGORICAL:
        try:
            # Normalize by position in the scale list
            str_scale = [str(s).lower().strip() for s in criterion.scale]
            idx = str_scale.index(str(raw_value).lower().strip())
            if len(criterion.scale) <= 1:
                return 1.0
            return idx / (len(criterion.scale) - 1)
        except ValueError:
            return 0.0

    return 0.0


def _parse_judge_response(
    response: str, rubric: Rubric
) -> tuple[list[CriterionScore], str]:
    """Parse the JSON response from the judge LLM.

    Returns (criterion_scores, overall_reasoning).
    Raises ValueError if the response cannot be parsed.
    """
    # Strip markdown code fences if present
    text = response.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        # Remove first and last fence lines
        lines = [ln for ln in lines[1:] if not ln.strip().startswith("```")]
        text = "\n".join(lines)

    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        expected_names = [c.name for c in rubric.criteria]
        raise JudgeParseError(
            f"Judge LLM returned non-JSON response. "
            f"Expected JSON with keys 'scores' and 'overall_reasoning'. "
            f"Expected criterion names: {expected_names}. "
            f"Raw response (first 500 chars): {response[:500]!r}",
            raw_response=response,
        ) from exc

    scores_data = data.get("scores", [])
    overall_reasoning = data.get("overall_reasoning", "")

    criterion_scores: list[CriterionScore] = []
    expected_names = {c.name for c in rubric.criteria}
    matched_names: set[str] = set()
    unmatched_names: list[str] = []

    for item in scores_data:
        crit_name = item.get("criterion", "")
        criterion = rubric.get_criterion(crit_name)
        if criterion is None:
            unmatched_names.append(crit_name)
            continue
        matched_names.add(crit_name)
        raw = item.get("value")
        normalized = _normalize_score(criterion, raw)
        criterion_scores.append(
            CriterionScore(
                criterion_name=crit_name,
                raw_value=raw,
                normalized=normalized,
                reasoning=item.get("reasoning", ""),
            )
        )

    if unmatched_names:
        missing = sorted(expected_names - matched_names)
        warnings.warn(
            f"Judge LLM returned unknown criterion names: {unmatched_names}. "
            f"Expected: {sorted(expected_names)}. "
            f"Unmatched rubric criteria: {missing}. "
            f"These scores will be ignored, which may produce overall=0.0.",
            stacklevel=2,
        )

    return criterion_scores, overall_reasoning


class RubricJudge(Judge):
    """Evaluates agent runs against a rubric using an LLM.

    The LLM callable is a simple async function that takes
    (system, user) prompts and returns a string. This keeps
    the judge decoupled from any specific SDK.

    Example::

        async def call_llm(system: str, user: str) -> str:
            response = await client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            )
            return response.choices[0].message.content

        judge = RubricJudge(rubric=my_rubric, llm=call_llm, model_name="gpt-4o")
        score = await judge.evaluate(run)
    """

    def __init__(
        self,
        rubric: Rubric,
        llm: Callable[..., Any],
        model_name: str = "",
    ) -> None:
        self.rubric = rubric
        self.llm = llm
        self.model_name = model_name
        self.name = f"rubric_judge:{rubric.name}"

    async def evaluate(self, run: AgentRun, **kwargs: Any) -> JudgeScore:
        """Evaluate an agent run against the rubric."""
        system_prompt = _build_system_prompt(self.rubric)
        user_prompt = _build_user_prompt(run)

        response = await self.llm(system_prompt, user_prompt)

        criterion_scores, overall_reasoning = _parse_judge_response(
            response, self.rubric
        )

        # Compute weighted overall score
        total_weight = 0.0
        weighted_sum = 0.0
        for cs in criterion_scores:
            criterion = self.rubric.get_criterion(cs.criterion_name)
            weight = criterion.weight if criterion else 1.0
            weighted_sum += cs.normalized * weight
            total_weight += weight

        overall = weighted_sum / total_weight if total_weight > 0 else 0.0

        return JudgeScore(
            rubric_name=self.rubric.name,
            criterion_scores=criterion_scores,
            overall=overall,
            reasoning=overall_reasoning,
            judge_model=self.model_name,
            metadata=kwargs,
        )
