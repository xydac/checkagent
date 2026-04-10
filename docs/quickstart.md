# Quickstart

Get CheckAgent running with your agent in under 5 minutes.

## Install

```bash
pip install checkagent
```

For framework-specific adapters, install the relevant extra:

```bash
pip install checkagent[langchain]    # LangChain / LangGraph
pip install checkagent[openai-agents]  # OpenAI Agents SDK
pip install checkagent[pydantic-ai]  # PydanticAI
pip install checkagent[anthropic]    # Anthropic SDK
pip install checkagent[crewai]       # CrewAI
pip install checkagent[all]          # Everything
```

## Scaffold a Test Project

```bash
checkagent init my-agent-tests
cd my-agent-tests
pytest tests/ -v
```

This generates a working test project with a sample agent and two passing tests. No API keys needed.

## Write Your First Test

Every CheckAgent test follows the same pattern: set up mocks, run your agent, assert on the result.

```python
import pytest
from checkagent import AgentRun, Step, ToolCall, assert_tool_called

# Your agent — any async callable
async def my_agent(query: str, *, llm, tools):
    response = await llm.complete(query)
    result = await tools.call("search", {"query": query})
    return AgentRun(
        input=query,
        steps=[Step(output_text=response, tool_calls=[
            ToolCall(name="search", arguments={"query": query}, result=result),
        ])],
        final_output=result,
    )

@pytest.mark.agent_test(layer="mock")
async def test_my_agent(ca_mock_llm, ca_mock_tool):
    # Set up deterministic responses
    ca_mock_llm.on_input(contains="weather").respond("Let me search for that.")
    ca_mock_tool.on_call("search").respond({"temp": 72, "unit": "F"})

    # Run your agent
    result = await my_agent("What's the weather?", llm=ca_mock_llm, tools=ca_mock_tool)

    # Assert on the result
    assert result.succeeded
    assert_tool_called(result, "search", query="What's the weather?")
    assert result.final_output["temp"] == 72
```

Run it:

```bash
pytest tests/ -v
```

!!! note "No async boilerplate needed"
    CheckAgent automatically configures `pytest-asyncio` with `asyncio_mode = "auto"`. Write `async def test_*` and it just works.

## Key Concepts

### Agent Runs

An `AgentRun` is the complete trace of one agent execution: input, steps, tool calls, and final output. Every test produces one.

```python
from checkagent import AgentRun, Step, ToolCall, AgentInput

run = AgentRun(
    input="What's 2+2?",           # String or AgentInput
    steps=[
        Step(
            output_text="The answer is 4.",
            tool_calls=[
                ToolCall(name="calculator", arguments={"expr": "2+2"}, result="4")
            ],
        )
    ],
    final_output="4",
)
```

### Testing Layers

Mark tests by layer to control when they run:

```python
@pytest.mark.agent_test(layer="mock")    # Free, milliseconds — every commit
@pytest.mark.agent_test(layer="replay")  # Recorded responses — every PR
@pytest.mark.agent_test(layer="eval")    # Real metrics — on merge
@pytest.mark.agent_test(layer="judge")   # LLM grading — nightly
```

Filter by layer:

```bash
checkagent run --layer mock    # Only mock tests
checkagent run --layer eval    # Only eval tests
```

### Fixtures

CheckAgent provides fixtures with the `ap_` prefix:

| Fixture | Purpose |
|---------|---------|
| `ca_mock_llm` | Mock LLM with pattern-based responses |
| `ca_mock_tool` | Mock tool with schema validation and call recording |
| `ca_fault` | Fault injection (timeouts, rate limits, errors) |
| `ca_conversation` | Multi-turn conversation session |
| `ca_stream_collector` | Streaming event collector |
| `ca_safety` | Safety assertion helpers |

## What's Next?

- [Mock Layer Guide](layers/mock.md) — pattern matching, streaming, structured output
- [Safety Testing](guides/safety.md) — prompt injection, PII leakage, tool boundary testing
- [Fault Injection](guides/faults.md) — test how your agent handles failures
- [CLI Reference](cli.md) — all commands and options
