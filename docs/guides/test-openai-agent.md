# Test Your OpenAI Agent in 5 Minutes

This guide walks you through adding CheckAgent tests to an existing OpenAI Agents SDK project. By the end you'll have a passing mock test suite that runs in milliseconds with no API keys required.

## Prerequisites

- Python 3.10+
- [`openai-agents`](https://github.com/openai/openai-agents-python) installed
- An existing agent (or follow along with the example below)

## Install

```bash
pip install checkagent
```

No other configuration is needed to get started. The pytest plugin activates automatically once CheckAgent is installed.

## Your Agent

Here's a realistic customer support agent with two tools. If you already have an agent, skip ahead — the testing patterns apply regardless of what your tools do.

```python
# support_agent.py
from agents import Agent, Runner, function_tool


@function_tool
def get_order_status(order_id: str) -> dict:
    """Look up the current status of an order."""
    # In production this calls your orders database
    return {"order_id": order_id, "status": "shipped", "eta": "2 days"}


@function_tool
def check_inventory(sku: str) -> dict:
    """Check whether a product SKU is in stock."""
    # In production this calls your inventory service
    return {"sku": sku, "in_stock": True, "quantity": 42}


support_agent = Agent(
    name="SupportAgent",
    instructions=(
        "You are a helpful customer support agent. "
        "Use get_order_status to look up orders and check_inventory "
        "to answer stock questions. Be concise."
    ),
    tools=[get_order_status, check_inventory],
)


async def run_support_agent(user_input: str) -> str:
    result = await Runner.run(support_agent, user_input)
    return result.final_output
```

## Write Your First Mock Test

Create `tests/test_support_agent.py`:

```python
# tests/test_support_agent.py
import pytest
from checkagent.adapters import OpenAIAgentAdapter
from support_agent import support_agent, run_support_agent


@pytest.fixture
def agent():
    return OpenAIAgentAdapter(support_agent, runner=run_support_agent)


@pytest.mark.agent_test(layer="mock")
async def test_order_status_query(agent, ca_mock_llm, ca_mock_tool):
    # Define what the LLM will decide to do
    ca_mock_llm.on_input(contains="order").respond([
        "I'll look that up for you.",          # Step 1: agent decides to call tool
        "Your order ORD-123 has shipped and will arrive in 2 days.",  # Step 2: final answer
    ])

    # Define what the tool returns when called
    ca_mock_tool.on_call("get_order_status").respond({
        "order_id": "ORD-123",
        "status": "shipped",
        "eta": "2 days",
    })

    agent_run = await agent.run(
        "What's the status of my order ORD-123?",
        mock_llm=ca_mock_llm,
        mock_tools=ca_mock_tool,
    )

    # Assert on the final answer
    assert "shipped" in agent_run.final_output.lower()

    # Assert the agent called the right tool with the right argument
    ca_mock_tool.assert_tool_called("get_order_status", with_args={"order_id": "ORD-123"})

    # Assert on trace shape — two steps: tool call + final response
    assert len(agent_run.steps) == 2


@pytest.mark.agent_test(layer="mock")
async def test_inventory_check(agent, ca_mock_llm, ca_mock_tool):
    ca_mock_llm.on_input(contains="stock").respond([
        "Let me check the inventory.",
        "Yes, SKU-789 is in stock with 42 units available.",
    ])

    ca_mock_tool.on_call("check_inventory").respond({
        "sku": "SKU-789",
        "in_stock": True,
        "quantity": 42,
    })

    agent_run = await agent.run(
        "Is SKU-789 in stock?",
        mock_llm=ca_mock_llm,
        mock_tools=ca_mock_tool,
    )

    assert "in stock" in agent_run.final_output.lower()
    ca_mock_tool.assert_tool_called("check_inventory", with_args={"sku": "SKU-789"})
    # Inventory tool should never be called more than once for a simple query
    ca_mock_tool.assert_tool_called("check_inventory", times=1)


@pytest.mark.agent_test(layer="mock")
async def test_agent_does_not_call_wrong_tool(agent, ca_mock_llm, ca_mock_tool):
    """An order-status question should never trigger check_inventory."""
    ca_mock_llm.on_input(contains="order").respond([
        "Checking that order now.",
        "Order ORD-456 is still processing.",
    ])
    ca_mock_tool.on_call("get_order_status").respond({
        "order_id": "ORD-456",
        "status": "processing",
        "eta": "unknown",
    })

    agent_run = await agent.run(
        "Where is order ORD-456?",
        mock_llm=ca_mock_llm,
        mock_tools=ca_mock_tool,
    )

    ca_mock_tool.assert_tool_called("get_order_status")
    ca_mock_tool.assert_tool_not_called("check_inventory")
```

A few things to notice:

- No `asyncio_mode` setting needed — the plugin enables it automatically.
- `ca_mock_llm` and `ca_mock_tool` are pytest fixtures provided by CheckAgent; just declare them as test parameters.
- `agent_run` is an `AgentRun` object. Its `.steps` list gives you the full execution trace; each `Step` may contain a `ToolCall`.
- Mock tests make zero network calls, so they run in milliseconds and never fail due to rate limits or model changes.

## Run It

```bash
checkagent run tests/test_support_agent.py
```

Expected output:

```
collected 3 items

tests/test_support_agent.py::test_order_status_query    PASSED   [  8ms]
tests/test_support_agent.py::test_inventory_check       PASSED   [  6ms]
tests/test_support_agent.py::test_agent_does_not_call_wrong_tool  PASSED   [  5ms]

====== 3 passed in 0.04s (layer=mock) ======
```

To run only mock-layer tests across your whole project:

```bash
checkagent run --layer mock
```

## Add a `checkagent.yml`

Drop this at your project root to pin configuration:

```yaml
# checkagent.yml
project: support-agent

layers:
  mock:
    timeout_ms: 100   # Fail any mock test that takes longer than 100 ms

cassettes:
  dir: tests/cassettes
```

## Safety Scan

Run the OWASP LLM Top 10 probe suite against your agent with one command:

```bash
checkagent scan --agent support_agent:run_support_agent
```

CheckAgent fires 68 attack probes covering prompt injection, jailbreaks, PII extraction, and scope boundary violations, then prints a pass/fail report mapped to OWASP categories. No API key setup is needed for the probe runner itself; your agent's LLM key is used only if you point the scan at a live agent.

See [OWASP Mapping](../owasp-mapping.md) for the full category-to-probe breakdown.

## Handoffs and Multi-Agent Systems

If your support agent delegates to a specialized sub-agent — for example, handing a billing question off to a `BillingAgent` — CheckAgent tracks the full handoff chain automatically via `MultiAgentTrace`. You can assert that the right sub-agent was invoked, inspect the payload passed at each handoff, and use `assign_blame` to pinpoint which agent caused a failure when the system returns a wrong answer.

See [Multi-Agent Testing](multiagent.md) for the complete guide.

## What's Next

- [Mock Layer reference](../layers/mock.md) — full `ca_mock_llm` and `ca_mock_tool` API
- [Replay Layer](../layers/replay.md) — record real agent runs once, replay them on every PR
- [Safety Testing](safety.md) — write targeted injection and PII tests in pytest
- [Fault Injection](faults.md) — test how your agent handles tool timeouts and partial failures
- [CI Integration](../github-action.md) — run CheckAgent in GitHub Actions with a quality gate
