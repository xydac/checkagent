# Eval Layer

The eval layer measures agent quality using metrics and golden datasets. It answers "is my agent good enough?" with numbers.

## Metrics

CheckAgent provides built-in evaluation metrics:

### Task Completion

Did the agent produce the expected output?

```python
from checkagent.eval.metrics import task_completion

score = task_completion(result, expected="The capital of France is Paris.")
assert score.value >= 0.8
```

### Tool Correctness

Did the agent call the right tools with the right arguments?

```python
from checkagent.eval.metrics import tool_correctness

expected_tools = [
    {"name": "search", "arguments": {"query": "capital of France"}},
]
score = tool_correctness(result, expected_tools)
assert score.value == 1.0
```

### Step Efficiency

Did the agent solve the task in a reasonable number of steps?

```python
from checkagent.eval.metrics import step_efficiency

score = step_efficiency(result, max_steps=5)
assert score.value >= 0.6  # Penalizes excess steps
```

## Golden Datasets

Define test cases in a JSON file and run your agent against all of them:

```json
{
  "name": "booking-tests",
  "version": 1,
  "cases": [
    {
      "input": "Book a flight to Tokyo",
      "expected_output": "Flight booked",
      "expected_tools": ["search_flights", "book_flight"],
      "metadata": {"category": "booking"}
    }
  ]
}
```

Load and use in tests:

```python
from checkagent import load_dataset

@pytest.mark.agent_test(layer="eval")
async def test_against_golden_dataset(my_agent):
    dataset = load_dataset("tests/golden/booking_cases.json")
    for case in dataset.cases:
        result = await my_agent.run(case.input)
        score = task_completion(result, case.expected_output)
        assert score.value >= 0.8, f"Failed: {case.input}"
```

Validate your dataset schema:

```bash
checkagent dataset validate tests/golden/booking_cases.json
```

## Aggregate Scoring

Compute aggregate metrics across a test suite:

```python
from checkagent.eval.aggregate import TestRunSummary

summary = TestRunSummary()
summary.add_result("test_1", scores=[score1, score2])
summary.add_result("test_2", scores=[score3])

print(f"Pass rate: {summary.pass_rate:.0%}")
print(f"Total tests: {summary.total}")
```

## Custom Evaluators

Write your own evaluator by implementing the evaluator protocol:

```python
from checkagent.eval.evaluator import BaseEvaluator
from checkagent.core.types import AgentRun, Score

class MyEvaluator(BaseEvaluator):
    name = "my_metric"

    def evaluate(self, run: AgentRun, **kwargs) -> Score:
        # Your evaluation logic
        value = 1.0 if run.final_output else 0.0
        return Score(name=self.name, value=value, threshold=0.5)
```

Register custom evaluators as plugins via entry points:

```toml
[project.entry-points."checkagent.evaluators"]
my_metric = "my_package:MyEvaluator"
```
