# CheckAgent

Open-source, pytest-native testing framework for AI agents.

## What This Project Does

CheckAgent is a pytest plugin for testing AI agent workflows across four layers:

1. **MOCK** — Deterministic unit tests with mocked LLMs and tools (free, milliseconds)
2. **REPLAY** — Record-and-replay regression testing (cheap, seconds)
3. **EVAL** — Metric evaluation against golden datasets (moderate, seconds)
4. **JUDGE** — LLM-as-judge with statistical assertions (expensive, minutes)

## Source Layout

```
src/checkagent/
├── core/          # Plugin, adapter protocol, types, tracing, cost, streaming
├── mock/          # MockLLM, MockTool, MCP mock, fault injection, fixtures
├── replay/        # Record/replay, versioned cassettes, stream cassettes
├── eval/          # Metrics: task completion, tool correctness, trajectory, etc.
├── safety/        # Attack probes, safety evaluators, compliance reports
├── conversation/  # Multi-turn session management
├── judge/         # LLM-as-judge, rubrics, statistical verdicts
├── ci/            # GitHub Action, GitLab CI, quality gates, reporters
├── adapters/      # LangChain, OpenAI Agents SDK, CrewAI, PydanticAI, Anthropic, generic
├── datasets/      # Golden dataset loader, schema, generator
└── cli/           # init, run, demo, record, report, cost, migrate, import-trace
```

## Code Conventions

### Async-First

Most agent frameworks are async-native. All agent-facing APIs are `async def`. The plugin sets `asyncio_mode = "auto"` so any `async def test_*` just works.

Sync agents are supported via the `@wrap` decorator which auto-detects sync callables and runs them in a thread pool executor.

```python
# Async test — the default
@pytest.mark.agent_test(layer="mock")
async def test_my_agent(my_agent, ap_mock_llm):
    result = await my_agent.run("hello")
    assert result.final_output is not None
```

### Fixture Naming

All fixtures use the `ap_` prefix to avoid conflicts with other pytest plugins:

- `ap_mock_llm` — mock LLM provider
- `ap_mock_tool` — mock tool executor
- `ap_fault` — fault injection
- `ap_conversation` — multi-turn conversation session
- `ap_stream_collector` — streaming event collector
- `ap_safety` — safety assertion helpers

### Adapters

Adapters wrap agent frameworks to conform to the `AgentAdapter` protocol. Rules:
- Keep each adapter under 200 lines
- No deep framework integrations — thin wrappers only
- The `GenericAdapter` handles any Python callable as a fallback
- Adapters go in `src/checkagent/adapters/`

### Types

Core data types live in `src/checkagent/core/types.py`:
- `AgentRun` — complete execution trace
- `Step` — single agent step
- `ToolCall` — tool invocation + result
- `AgentInput` — input to the agent
- `StreamEvent` — streaming chunk event
- `Score` — evaluation score

### Cassettes

- Format: **JSON** (not YAML) — smaller diffs, fewer git merge conflicts
- Filenames: content-addressed (`cassettes/{test_id}/{short_hash}.json`)
- Include `_meta` block with schema version, timestamps, content hash
- API keys and secrets are redacted at recording time
- Versioned format with migration support (`checkagent migrate-cassettes`)

### Configuration

All config in `checkagent.yml` at project root, validated with Pydantic. See docs for full schema reference.

## Testing the Framework

```bash
pytest tests/                  # Run all framework tests
pytest tests/mock/             # Run mock layer tests only
pytest tests/ -x --tb=short   # Stop on first failure
```

- CI runs on Linux, macOS, and Windows
- Layer 1 (mock) tests must execute in < 100ms each
- Layer 2 (replay) tests must execute in < 1s each
- Zero flaky tests in Layers 1 and 2 — they are deterministic by design

## CLI

```bash
checkagent init                        # Scaffold new test project
checkagent run                         # Run tests (thin pytest wrapper)
checkagent demo                        # Zero-config demo, no API keys needed
checkagent record <agent> <input>      # Record session as cassette
checkagent report <results>            # Generate HTML report
checkagent cost <results>              # Cost breakdown
checkagent migrate-cassettes [dir]     # Upgrade cassette format
checkagent import-trace --source ...   # Import production traces
checkagent dataset validate <file>     # Validate golden dataset
```

## Dependencies

**Required:**
- `pytest` >= 7.0
- `pytest-asyncio` >= 0.23
- `pluggy` >= 1.0
- `pydantic` >= 2.0
- `click`
- `rich`

**Optional:**
- `opentelemetry-api` — trace emission
- `dirty-equals` — structured output fuzzy matching
- `deepdiff` — detailed failure diagnostics
- `spacy` — NER-based PII detection (for trace import)

## Plugin System

Community extensions are separate PyPI packages that auto-register via entry points:

```toml
# pyproject.toml for a community plugin
[project.entry-points."checkagent.adapters"]
my_framework = "checkagent_myframework:MyAdapter"
```

Entry point groups: `checkagent.adapters`, `checkagent.evaluators`, `checkagent.safety`, `checkagent.judges`

## Contributing

See CONTRIBUTING.md for guidelines. Key points:
- All PRs need tests
- Run `pytest tests/` locally before pushing
- Adapters belong in `src/checkagent/adapters/` (core) or as separate packages (community)
- Safety probes must be non-destructive
