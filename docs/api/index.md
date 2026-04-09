# API Reference

Auto-generated reference for all public CheckAgent APIs.

## Modules

| Module | Description |
|--------|-------------|
| [Core Types](types.md) | `AgentRun`, `Step`, `ToolCall`, `AgentInput`, `Score`, `StreamEvent` |
| [Mock Layer](mock.md) | `MockLLM`, `MockTool`, `FaultInjector`, `MockMCPServer` |
| [Eval & Assertions](eval.md) | Metrics, assertions, datasets, cost tracking |
| [Safety](safety.md) | Evaluators, attack probes, taxonomy |
| [Adapters](adapters.md) | `AgentAdapter` protocol and framework adapters |
| [Replay](replay.md) | `Cassette`, `ReplayEngine`, cassette management |
| [Judge](judge.md) | `RubricJudge`, `Criterion`, `JudgeScore`, statistical verdicts |

## Top-Level Imports

Most commonly used classes and functions are available directly from `checkagent`:

```python
from checkagent import (
    AgentRun, Step, ToolCall, AgentInput, Score,
    MockLLM, MockTool, FaultInjector,
    GenericAdapter, wrap,
    assert_tool_called, assert_output_schema, assert_output_matches,
    Cassette, ReplayEngine,
    RubricJudge, Criterion, JudgeScore,
    GoldenDataset, load_dataset,
    CostTracker, CostReport,
    Conversation, Turn,
    StreamCollector, StreamEvent,
)
```
