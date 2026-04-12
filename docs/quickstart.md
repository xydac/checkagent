# Quickstart

## Step 1: Install

```bash
pip install checkagent
```

## Step 2: Run the demo

```bash
checkagent demo
```

The demo runs 8 tests across mock, eval, and safety layers in under 30 seconds — no API keys, no configuration needed.

## What you saw

CheckAgent tests are organized into four layers, each with a different cost and speed profile:

- **MOCK** — deterministic unit tests with mocked LLMs and tools. Free, milliseconds. See [Mock Layer](layers/mock.md).
- **REPLAY** — record real agent responses once, replay them deterministically on every PR. Cheap, seconds.
- **EVAL** — run your agent against golden datasets and measure quality metrics. Moderate cost, seconds.
- **JUDGE** — LLM-as-judge evaluations with statistical assertions for subjective quality. Expensive, minutes.

The demo exercised the first three. You can run each layer independently in CI so you're not paying for LLM calls on every commit.

## Next steps

- [Mock Layer](layers/mock.md) — write your first unit test
- [Safety Testing](guides/safety.md) — scan for prompt injection and PII leakage
- [Fault Injection](guides/faults.md) — test how your agent handles failures
- [Configuration](configuration.md) — checkagent.yml reference
- [CLI Reference](cli.md) — all commands
