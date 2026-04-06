"""Multi-judge consensus evaluation (F4.4).

Runs multiple judges on the same agent run and aggregates their
verdicts via majority vote. Disagreements are flagged when judges
produce different verdicts.
"""

from __future__ import annotations

import asyncio
from collections import Counter
from typing import Any

from checkagent.core.types import AgentRun
from checkagent.judge.judge import Judge
from checkagent.judge.types import ConsensusVerdict, Verdict
from checkagent.judge.verdict import compute_verdict


async def multi_judge_evaluate(
    judges: list[Judge],
    run: AgentRun,
    *,
    num_trials: int = 3,
    threshold: float = 0.7,
    min_pass_rate: float = 0.5,
    inconclusive_band: float = 0.15,
    concurrent: bool = True,
    **kwargs: Any,
) -> ConsensusVerdict:
    """Run multiple judges and aggregate verdicts via majority vote.

    Each judge runs ``num_trials`` trials (via ``compute_verdict``),
    producing a per-judge verdict. The consensus verdict is the
    majority among non-INCONCLUSIVE verdicts.

    Args:
        judges: List of judges to evaluate with (minimum 2).
        run: The agent run to evaluate.
        num_trials: Trials per judge.
        threshold: Score threshold per trial.
        min_pass_rate: Minimum pass rate for a PASS verdict per judge.
        inconclusive_band: Band around threshold for INCONCLUSIVE.
        concurrent: Run judges concurrently (default True).
        **kwargs: Passed through to each judge.evaluate().

    Returns:
        ConsensusVerdict with per-judge results and majority vote.

    Raises:
        ValueError: If fewer than 2 judges are provided.
    """
    if len(judges) < 2:
        raise ValueError("multi_judge_evaluate requires at least 2 judges")

    verdict_kwargs = dict(
        num_trials=num_trials,
        threshold=threshold,
        min_pass_rate=min_pass_rate,
        inconclusive_band=inconclusive_band,
        **kwargs,
    )

    if concurrent:
        coros = [
            compute_verdict(judge, run, **verdict_kwargs) for judge in judges
        ]
        results = await asyncio.gather(*coros)
    else:
        results = []
        for judge in judges:
            result = await compute_verdict(judge, run, **verdict_kwargs)
            results.append(result)

    # Collect per-judge verdicts keyed by unique identifier.
    # Use model_name when available to distinguish judges sharing a rubric.
    judge_verdicts: dict[str, Any] = {}
    for judge, result in zip(judges, results, strict=True):
        model = getattr(judge, "model_name", "")
        key = f"{judge.name}:{model}" if model else judge.name
        # If key still collides (e.g. same model_name), append index
        if key in judge_verdicts:
            idx = 2
            while f"{key}:{idx}" in judge_verdicts:
                idx += 1
            key = f"{key}:{idx}"
        judge_verdicts[key] = result

    # Majority vote among non-INCONCLUSIVE verdicts
    decisive_verdicts = [
        r.verdict for r in results if r.verdict != Verdict.INCONCLUSIVE
    ]

    if not decisive_verdicts:
        # All judges inconclusive
        consensus = Verdict.INCONCLUSIVE
    else:
        counts = Counter(decisive_verdicts)
        consensus = counts.most_common(1)[0][0]

    # Check for disagreement: any judge disagrees with the consensus
    unique_verdicts = set(r.verdict for r in results)
    has_disagreement = len(unique_verdicts) > 1

    # Agreement rate: fraction of judges matching consensus
    matching = sum(1 for r in results if r.verdict == consensus)
    agreement_rate = matching / len(results)

    # Build reasoning
    verdict_summary = ", ".join(
        f"{name}: {v.verdict.value}" for name, v in judge_verdicts.items()
    )
    reasoning = (
        f"Consensus: {consensus.value} from {len(judges)} judges. "
        f"Verdicts: [{verdict_summary}]. "
        f"Agreement: {agreement_rate:.0%}."
    )
    if has_disagreement:
        reasoning += " DISAGREEMENT detected among judges."

    return ConsensusVerdict(
        verdict=consensus,
        judge_verdicts=judge_verdicts,
        agreement_rate=agreement_rate,
        has_disagreement=has_disagreement,
        reasoning=reasoning,
    )
