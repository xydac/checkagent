"""Statistical verdict computation for multi-trial evaluation (F4.2).

Runs K trials of a judge evaluation and computes a three-valued
verdict: PASS, FAIL, or INCONCLUSIVE based on configurable thresholds.
"""

from __future__ import annotations

from typing import Any

from checkagent.core.types import AgentRun
from checkagent.judge.judge import Judge
from checkagent.judge.types import JudgeVerdict, Verdict


async def compute_verdict(
    judge: Judge,
    run: AgentRun,
    *,
    num_trials: int = 3,
    threshold: float = 0.7,
    min_pass_rate: float = 0.5,
    inconclusive_band: float = 0.15,
    **kwargs: Any,
) -> JudgeVerdict:
    """Run multiple judge trials and compute a statistical verdict.

    Args:
        judge: The judge to run evaluations with.
        run: The agent run to evaluate.
        num_trials: Number of evaluation trials (K). Default 3.
        threshold: Score threshold for a trial to count as passing.
        min_pass_rate: Minimum fraction of passing trials for PASS verdict.
        inconclusive_band: Band around min_pass_rate where verdict is
            INCONCLUSIVE. If pass_rate is within this band of the
            threshold, the result is inconclusive.
        **kwargs: Extra arguments passed to judge.evaluate().

    Returns:
        JudgeVerdict with aggregated results.
    """
    if num_trials < 1:
        raise ValueError("num_trials must be at least 1")

    trials = []
    for _ in range(num_trials):
        score = await judge.evaluate(run, **kwargs)
        trials.append(score)

    passing = sum(1 for t in trials if t.overall >= threshold)
    pass_rate = passing / num_trials

    # Determine verdict
    if pass_rate >= min_pass_rate + inconclusive_band:
        verdict = Verdict.PASS
    elif pass_rate <= min_pass_rate - inconclusive_band:
        verdict = Verdict.FAIL
    else:
        verdict = Verdict.INCONCLUSIVE

    # Simple confidence: distance from the inconclusive band edge
    if verdict == Verdict.INCONCLUSIVE:
        confidence = 0.0
    else:
        edge = min_pass_rate + (
            inconclusive_band if verdict == Verdict.PASS else -inconclusive_band
        )
        confidence = min(1.0, abs(pass_rate - edge) / (1.0 - min_pass_rate))

    avg_score = sum(t.overall for t in trials) / num_trials
    reasoning = (
        f"{passing}/{num_trials} trials passed (threshold={threshold}). "
        f"Pass rate: {pass_rate:.1%}. Average score: {avg_score:.3f}."
    )

    return JudgeVerdict(
        verdict=verdict,
        trials=trials,
        pass_rate=pass_rate,
        threshold=threshold,
        min_pass_rate=min_pass_rate,
        confidence=confidence,
        reasoning=reasoning,
    )
