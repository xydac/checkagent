# CheckAgent

**The open-source testing framework for AI agents.**

*pytest-native · async-first · CI/CD-first · safety-aware*

---

CheckAgent is a pytest plugin for testing AI agent workflows. It provides layered testing — from free, millisecond unit tests to LLM-judged evaluations with statistical rigor — so you can ship agents with the same confidence you ship traditional software.

## Why CheckAgent?

- **pytest-native** — tests are `.py` files, assertions are `assert`, markers and fixtures are standard pytest
- **Async-first** — most agent frameworks are async; CheckAgent is too
- **Framework-agnostic** — works with LangChain, OpenAI Agents SDK, CrewAI, PydanticAI, Anthropic, or any Python callable
- **Cost-aware** — every test run tracks token usage and estimated cost, with budget limits
- **Safety built-in** — prompt injection, PII leakage, and tool misuse testing ships as core
- **Zero config** — auto-discovers tests, auto-configures asyncio, works out of the box

## The Testing Pyramid

```
                  ╱‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾╲
                 │   JUDGE  · $$$     │          Minutes · Nightly
                 │   LLM-as-judge     │
                ╱‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾╲
               │   EVAL  · $$          │         Seconds · On merge
               │   Metrics & datasets  │
              ╱‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾╲
             │   REPLAY  · $              │      Seconds · On PR
             │   Record & replay          │
            ╱‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾╲
           │   MOCK  · Free                  │   Milliseconds · Every commit
           │   Deterministic unit tests      │
            ╲_______________________________╱
```

Each layer builds on the one below it. Start with mocks (free, fast, deterministic), then add replay tests for regression, eval metrics for quality, and judge assertions for subjective quality.

## Get Started in 60 Seconds

```bash
pip install checkagent
checkagent demo
```

That's it. No API keys, no configuration. The demo runs 8 tests across mock, eval, and safety layers and shows you what CheckAgent can do.

Ready to test your own agent? See the [Quickstart](quickstart.md).
