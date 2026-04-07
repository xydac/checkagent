# CheckAgent

**The open-source testing framework for AI agents.**

*pytest-native · async-first · CI/CD-first · safety-aware*

[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org)

---

CheckAgent is a pytest plugin for testing AI agent workflows. It provides layered testing — from free, millisecond unit tests to LLM-judged evaluations with statistical rigor — so you can ship agents with the same confidence you ship traditional software.

## Why CheckAgent

- **pytest-native** — tests are `.py` files, assertions are `assert`, markers and fixtures are standard pytest
- **Async-first** — most agent frameworks are async; CheckAgent is too
- **Framework-agnostic** — works with LangChain, OpenAI Agents SDK, CrewAI, PydanticAI, Anthropic, or any Python callable
- **Cost-aware** — every test run tracks token usage and estimated cost, with budget limits
- **Safety built-in** — prompt injection, PII leakage, and tool misuse testing ships as core

## The Testing Pyramid

```
┌─────────────────────────────────────────────┐
│  Layer 4: JUDGE                             │
│  LLM-as-judge · statistical assertions      │
│  ⏱ Minutes  💰 $$$  📍 Nightly              │
├─────────────────────────────────────────────┤
│  Layer 3: EVAL                              │
│  Agent metrics · golden datasets            │
│  ⏱ Seconds  💰 $$   📍 On merge             │
├─────────────────────────────────────────────┤
│  Layer 2: REPLAY                            │
│  Record-and-replay · regression testing     │
│  ⏱ Seconds  💰 $    📍 On every PR          │
├─────────────────────────────────────────────┤
│  Layer 1: MOCK                              │
│  Deterministic unit tests · zero LLM cost   │
│  ⏱ Milliseconds  💰 Free  📍 On every commit│
└─────────────────────────────────────────────┘
```

## Quick Start

```bash
pip install checkagent
mkdir my-agent-tests && cd my-agent-tests
checkagent init
pytest tests/ -v
```

Or try the zero-config demo (no API keys needed):

```bash
pip install checkagent
checkagent demo
```

## Example Test

```python
import pytest
from checkagent import AgentInput, AgentRun, Step, ToolCall, assert_tool_called

# Your agent — any async function that calls LLMs and tools
async def booking_agent(query, *, llm, tools):
    plan = await llm.complete(query)
    event = await tools.call("create_event", {"title": "Meeting"})
    return AgentRun(
        input=AgentInput(query=query),
        steps=[Step(output_text=plan, tool_calls=[
            ToolCall(name="create_event", arguments={"title": "Meeting"}, result=event),
        ])],
        final_output=event,
    )

# Test with zero LLM cost, deterministic, milliseconds
@pytest.mark.agent_test(layer="mock")
async def test_booking(ap_mock_llm, ap_mock_tool):
    ap_mock_llm.on_input(contains="book").respond("Booking your meeting now.")
    ap_mock_tool.on_call("create_event").respond(
        {"confirmed": True, "event_id": "evt-123"}
    )

    result = await booking_agent(
        "Book a meeting", llm=ap_mock_llm, tools=ap_mock_tool
    )

    assert_tool_called(result, "create_event", title="Meeting")
    assert result.final_output["confirmed"] is True
```

## Features

| Feature | Status |
|---------|--------|
| Mock LLM & tool providers | ✅ Implemented |
| Streaming mock support | ✅ Implemented |
| Fault injection (timeouts, rate limits) | ✅ Implemented |
| Structured output assertions | ✅ Implemented |
| Multi-turn conversation testing | ✅ Implemented |
| Record-and-replay cassettes | ✅ Implemented |
| Evaluation metrics (task completion, tool correctness) | ✅ Implemented |
| Safety testing (prompt injection, PII leakage) | ✅ Implemented |
| LLM-as-judge with statistical assertions | ✅ Implemented |
| CI/CD quality gates (GitHub Action) | ✅ Implemented |
| Production trace import | ✅ Implemented |
| Cost tracking & budget limits | ✅ Implemented |
| Multi-agent trace & credit assignment | ✅ Implemented |

## Roadmap

All core milestones are implemented. See [ROADMAP.md](ROADMAP.md) for what's next:

- More framework adapters (AutoGen, DSPy, Marvin)
- Production trace import from OpenTelemetry/Langfuse
- VS Code extension for inline test results
- Community plugin ecosystem

## Contributing

We're building this in public and contributions are welcome from day one. See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

Apache-2.0. See [LICENSE](LICENSE).
