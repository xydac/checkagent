# Judge Layer

The judge layer uses an LLM to evaluate subjective qualities like helpfulness, accuracy, and safety. It includes statistical assertions so you can make confident pass/fail decisions despite LLM non-determinism.

## Basic Usage

```python
from checkagent import RubricJudge, Criterion

@pytest.mark.agent_test(layer="judge")
async def test_response_quality(my_agent, judge_llm):
    judge = RubricJudge(
        llm=judge_llm,
        criteria=[
            Criterion(name="helpfulness", description="Was the response helpful and actionable?"),
            Criterion(name="accuracy", description="Was the factual information correct?"),
        ],
    )

    result = await my_agent.run("What's the capital of France?")
    score = await judge.evaluate(result)
    assert score.passed  # overall >= 0.5
```

## Criteria

A `Criterion` defines what the judge evaluates:

```python
from checkagent import Criterion

# Likert scale (default) — judge rates 1-5
helpfulness = Criterion(
    name="helpfulness",
    description="Was the response helpful and actionable?",
)

# Binary scale — judge says pass or fail
safety = Criterion(
    name="safety",
    description="Did the response avoid harmful content?",
    scale_type="binary",
)
```

## Statistical Assertions

A single LLM judge call is noisy. Run multiple trials and use statistical assertions for confidence:

```python
score = await judge.evaluate(result, trials=5)
# score.passed uses statistical aggregation across trials
assert score.passed
```

The verdict is `PASS`, `FAIL`, or `INCONCLUSIVE` based on agreement across trials.

## Multi-Judge Consensus

Use multiple judges for higher confidence:

```python
from checkagent.judge import multi_judge_evaluate

score = await multi_judge_evaluate(
    judges=[judge_a, judge_b],
    run=result,
)
print(f"Agreement: {score.agreement_rate:.0%}")
assert score.passed
```

## JudgeScore

The `JudgeScore` object returned by evaluation includes:

- `score.overall` — aggregate score (0.0 to 1.0)
- `score.passed` — boolean (`overall >= 0.5`)
- `score.criteria_scores` — per-criterion scores
- `score.verdict` — `PASS`, `FAIL`, or `INCONCLUSIVE`

```python
score = await judge.evaluate(result)
for name, criterion_score in score.criteria_scores.items():
    print(f"{name}: {criterion_score.value}")
```
