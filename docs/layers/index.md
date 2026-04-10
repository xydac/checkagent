# Testing Layers

CheckAgent organizes tests into four layers, each with different cost, speed, and confidence tradeoffs.

## Layer 1: Mock

**Cost:** Free | **Speed:** Milliseconds | **When:** Every commit

Replace LLMs and tools with deterministic mocks. Test your agent's logic without making any API calls.

```python
@pytest.mark.agent_test(layer="mock")
async def test_agent_uses_correct_tool(ca_mock_llm, ca_mock_tool):
    ca_mock_llm.on_input(contains="schedule").respond("I'll create an event.")
    ca_mock_tool.on_call("create_event").respond({"id": "evt-1"})

    result = await my_agent("Schedule a meeting", llm=ca_mock_llm, tools=ca_mock_tool)
    assert_tool_called(result, "create_event")
```

[Mock Layer Guide →](mock.md)

## Layer 2: Replay

**Cost:** Free (after recording) | **Speed:** Seconds | **When:** Every PR

Record real API responses as JSON cassettes, then replay them for deterministic regression tests.

```python
@pytest.mark.agent_test(layer="replay")
@pytest.mark.cassette("tests/cassettes/booking_flow.json")
async def test_booking_regression(my_agent):
    result = await my_agent.run("Book a flight to Tokyo")
    assert result.succeeded
    assert result.tool_was_called("search_flights")
```

[Replay Layer Guide →](replay.md)

## Layer 3: Eval

**Cost:** Moderate | **Speed:** Seconds | **When:** On merge

Evaluate agent quality against golden datasets using metrics like task completion, tool correctness, and step efficiency.

```python
from checkagent import load_dataset, task_completion

@pytest.mark.agent_test(layer="eval")
async def test_agent_quality(my_agent):
    dataset = load_dataset("tests/golden/booking_cases.json")
    for case in dataset.cases:
        result = await my_agent.run(case.input)
        score = task_completion(result, case.expected_output)
        assert score.value >= 0.8, f"Failed on: {case.input}"
```

[Eval Layer Guide →](eval.md)

## Layer 4: Judge

**Cost:** Expensive (LLM calls) | **Speed:** Minutes | **When:** Nightly

Use an LLM as a judge to evaluate subjective quality — helpfulness, accuracy, safety — with statistical assertions.

```python
from checkagent import RubricJudge, Criterion

@pytest.mark.agent_test(layer="judge")
async def test_response_quality(my_agent, judge_llm):
    judge = RubricJudge(
        llm=judge_llm,
        criteria=[
            Criterion(name="helpfulness", description="Was the response helpful?"),
            Criterion(name="accuracy", description="Was the information correct?"),
        ],
    )
    result = await my_agent.run("What's the capital of France?")
    score = await judge.evaluate(result)
    assert score.passed
```

[Judge Layer Guide →](judge.md)

## Choosing a Layer

| Question | Layer |
|----------|-------|
| Does my agent call the right tools? | Mock |
| Did a code change break existing behavior? | Replay |
| Is my agent good enough on a benchmark? | Eval |
| Is the output actually helpful/correct/safe? | Judge |

Start with mock tests. They're free, fast, and catch most logic bugs. Add higher layers as your agent matures.
