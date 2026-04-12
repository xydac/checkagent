# Test Your LangChain Agent in 5 Minutes

This guide walks you through adding CheckAgent tests to an existing LangChain agent. By the end you will have deterministic unit tests that run in milliseconds (no API keys, no network calls) and a safety scan against the OWASP LLM Top 10.

---

## Prerequisites

- Python 3.10 or later
- A LangChain agent you want to test
- `langchain` and `langchain-openai` already installed

---

## 1. Install CheckAgent

```bash
pip install checkagent
```

Verify the install:

```bash
checkagent --version
```

---

## 2. Your Agent

If you do not have an agent yet, here is a minimal ReAct agent to work from. It has two tools — a web search stub and a calculator — and uses `ChatOpenAI` as its LLM.

```python
# my_agent.py
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_react_agent
from langchain.tools import tool
from langchain import hub


@tool
def web_search(query: str) -> str:
    """Search the web for current information."""
    # In production this would call a real search API.
    return f"Search results for: {query}"


@tool
def calculator(expression: str) -> str:
    """Evaluate a safe arithmetic expression and return the result."""
    try:
        result = eval(expression, {"__builtins__": {}})  # noqa: S307
        return str(result)
    except Exception as exc:
        return f"Error: {exc}"


def build_agent() -> AgentExecutor:
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    prompt = hub.pull("hwchase17/react")
    agent = create_react_agent(llm, [web_search, calculator], prompt)
    return AgentExecutor(agent=agent, tools=[web_search, calculator], verbose=False)
```

---

## 3. Add a `checkagent.yml`

Create `checkagent.yml` at your project root. This tells CheckAgent where your tests live and sets some defaults.

```yaml
# checkagent.yml
version: 1

layers:
  mock:
    enabled: true

agent:
  module: my_agent
  factory: build_agent
```

---

## 4. Write Your First Mock Test

Create `tests/test_agent.py`. The mock layer intercepts every LLM call and tool call — nothing leaves your machine.

```python
# tests/test_agent.py
import pytest
from checkagent.adapters import LangChainAdapter
from checkagent.core.types import AgentRun

from my_agent import build_agent


@pytest.fixture
def agent():
    """Build a fresh agent instance for each test."""
    return LangChainAdapter(build_agent())


# ---------------------------------------------------------------------------
# Basic response test
# ---------------------------------------------------------------------------

@pytest.mark.agent_test(layer="mock")
async def test_agent_answers_question(agent, ca_mock_llm):
    """Agent returns a direct answer without using any tools."""
    ca_mock_llm.on_input(contains="capital of France").respond("Paris")

    run: AgentRun = await agent.run("What is the capital of France?")

    assert "Paris" in run.final_output
    assert ca_mock_llm.call_count == 1


# ---------------------------------------------------------------------------
# Tool call test
# ---------------------------------------------------------------------------

@pytest.mark.agent_test(layer="mock")
async def test_agent_uses_calculator(agent, ca_mock_llm, ca_mock_tool):
    """Agent calls the calculator tool then summarises the result."""
    # First LLM turn: decide to use the calculator
    ca_mock_llm.on_input(contains="square root").respond([
        "I need to calculate this. Let me use the calculator.",
        "The answer is 12.",  # Second turn: final summary
    ])

    # Mock the calculator tool
    ca_mock_tool.on_call("calculator").respond("12")

    run: AgentRun = await agent.run("What is the square root of 144?")

    # Assert the tool was called with the right expression
    ca_mock_tool.assert_tool_called("calculator", with_args={"expression": "144**0.5"})

    # Assert the final output references the result
    assert "12" in run.final_output


# ---------------------------------------------------------------------------
# Tool NOT called test
# ---------------------------------------------------------------------------

@pytest.mark.agent_test(layer="mock")
async def test_agent_does_not_search_for_simple_facts(agent, ca_mock_llm, ca_mock_tool):
    """Agent should answer a simple arithmetic question without a web search."""
    ca_mock_llm.on_input(contains="2 + 2").respond("2 + 2 is 4.")

    run: AgentRun = await agent.run("What is 2 + 2?")

    ca_mock_tool.assert_tool_not_called("web_search")
    assert "4" in run.final_output


# ---------------------------------------------------------------------------
# Multi-step trace inspection
# ---------------------------------------------------------------------------

@pytest.mark.agent_test(layer="mock")
async def test_agent_search_then_summarise(agent, ca_mock_llm, ca_mock_tool):
    """Agent searches the web and then writes a summary."""
    ca_mock_llm.on_input(contains="latest Python").respond([
        "I should search for this.",
        "Based on the search results, Python 3.13 was released in October 2024.",
    ])
    ca_mock_tool.on_call("web_search").respond(
        "Python 3.13 released October 2024 with free-threaded mode."
    )

    run: AgentRun = await agent.run("What is the latest Python release?")

    # Inspect the execution trace
    tool_calls = [step.tool_call for step in run.steps if step.tool_call]
    assert any(tc.tool_name == "web_search" for tc in tool_calls)
    assert "3.13" in run.final_output
```

### What is happening here

| Concept | What it does |
|---------|-------------|
| `LangChainAdapter` | Wraps your `AgentExecutor` to conform to the CheckAgent `AgentAdapter` protocol |
| `ca_mock_llm` | Intercepts every `ChatOpenAI` call and returns your scripted responses |
| `ca_mock_tool` | Intercepts every tool invocation and returns your scripted results |
| `AgentRun` | The full execution trace — final output, steps, tool calls, token counts |
| `@pytest.mark.agent_test(layer="mock")` | Marks the test for the mock layer; CheckAgent sets `asyncio_mode = "auto"` automatically |

---

## 5. Run the Tests

```bash
checkagent run --layer mock
```

Or use pytest directly:

```bash
pytest tests/ -v
```

Expected output:

```
tests/test_agent.py::test_agent_answers_question PASSED              [  0.012s]
tests/test_agent.py::test_agent_uses_calculator PASSED               [  0.008s]
tests/test_agent.py::test_agent_does_not_search_for_simple_facts PASSED [  0.007s]
tests/test_agent.py::test_agent_search_then_summarise PASSED         [  0.009s]

4 passed in 0.04s
```

No API keys. No network calls. Under 50 ms total.

---

## 6. Run a Safety Scan

`checkagent scan` runs 68 attack probes against your agent, covering the [OWASP LLM Top 10](../owasp-mapping.md). Point it at the factory function in your module:

```bash
checkagent scan my_agent:build_agent
```

The scan covers:

| Category | What it tests |
|----------|--------------|
| **Prompt injection** (LLM01) | Can an attacker override your system prompt? |
| **Insecure output** (LLM02) | Does the agent echo back unsanitised content? |
| **Sensitive information** (LLM06) | Does the agent leak PII or secrets? |
| **Insecure plugin design** (LLM07) | Can probes trick the agent into calling tools it should not? |
| **Overreliance** (LLM09) | Does the agent refuse clearly harmful requests? |

Example output:

```
Scanning my_agent:build_agent — 68 probes
──────────────────────────────────────────────────────
[PASS] injection/direct-override            0.31s
[PASS] injection/indirect-tool-output       0.28s
[FAIL] jailbreak/roleplay-override          0.19s  ← agent followed injected role
[PASS] pii/extraction-ssn                   0.24s
...

Results: 65 passed, 3 failed
Run `checkagent scan my_agent:build_agent -g tests/test_safety.py` to generate regression tests.
```

Failed probes become actionable items. Use `--generate-tests` to pin them as regression tests so they cannot silently regress:

```bash
checkagent scan my_agent:build_agent -g tests/test_safety.py
pytest tests/test_safety.py -v
```

---

## What's Next

- **Replay layer** — Record a real agent session once and replay it deterministically on every PR. See [Replay Layer](../layers/replay.md).
- **Eval layer** — Run your agent against golden datasets and measure task completion rates. See [Eval Layer](../layers/eval.md).
- **Safety guide** — Write programmatic safety assertions using `ca_safety` and the `ProbeSet` API. See [Safety Testing](safety.md).
