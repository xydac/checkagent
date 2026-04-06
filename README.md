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
checkagent init
cd checkagent-tests
checkagent run
```

Or try the zero-config demo (no API keys needed):

```bash
pip install checkagent
checkagent demo
```

## Example Test

```python
import pytest
from checkagent import assert_tool_called, assert_output_schema
from pydantic import BaseModel

class BookingResult(BaseModel):
    confirmed: bool
    event_id: str

@pytest.mark.agent_test(layer="mock")
async def test_booking_agent(booking_agent, ap_mock_llm, ap_mock_tool):
    # Mock the LLM response for the booking request
    ap_mock_llm.on_input(contains="book a meeting").respond(
        "I'll book a meeting for April 10 at 2pm. Checking the calendar now."
    )
    # Mock the tool that the agent calls
    ap_mock_tool.on_call("check_calendar").respond({"available": True})
    ap_mock_tool.on_call("create_event").respond(
        {"confirmed": True, "event_id": "evt-123"}
    )

    result = await booking_agent.run("Book a meeting for April 10 at 2pm")

    assert_tool_called(result, "check_calendar", date="2026-04-10")
    assert_tool_called(result, "create_event")
    assert_output_schema(result, BookingResult)
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

See [ROADMAP.md](ROADMAP.md) for the full plan.

**Phase 1 — Ship Something People Love (Weeks 1-12)**
- Milestone 0: Foundation + Demo
- Milestone 1: Mock Layer + Fault Injection + Structured Output

**Phase 2 — Make It Useful for Real Agents (Weeks 9-24)**
- Milestone 2: Evaluation Metrics
- Milestone 3: Safety Testing + CI/CD

**Phase 3 — Make It Complete (Weeks 25-52)**
- Milestones 4-7: Replay, Judge, Adapters, Multi-Agent

## Contributing

We're building this in public and contributions are welcome from day one. See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

Apache-2.0. See [LICENSE](LICENSE).
