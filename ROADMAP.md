# Roadmap

CheckAgent is being built in public. Here's where we're headed.

## Phase 1: Ship Something People Love

### Milestone 0: Foundation + Demo (In Progress)
- [x] Core types (`AgentRun`, `Step`, `ToolCall`, `AgentInput`, `StreamEvent`, `Score`)
- [x] `AgentAdapter` protocol (async-first)
- [x] `GenericAdapter` for any Python callable
- [x] pytest plugin with `@pytest.mark.agent_test` marker
- [x] `checkagent init` CLI
- [x] `checkagent demo` — zero-config, no API keys, < 30 seconds
- [x] `checkagent run` — thin pytest wrapper
- [x] CI on Linux, macOS, Windows
- [x] PyPI v0.0.1-alpha (version set, publish workflow ready)

### Milestone 1: Mock Layer + Fault Injection
- [x] `MockLLM` with pattern-based responses and streaming mode
- [x] `MockTool` with schema validation and call recording
- [x] Fault injection (`ap_fault`): timeouts, rate limits, malformed responses
- [x] Structured output assertions (Pydantic, JSON Schema)
- [x] Multi-turn conversation fixture (`ap_conversation`)
- [x] Streaming support: `StreamCollector`, `ap_stream_collector`, stream event assertions
- [x] MCP-aware mock server
- [ ] PyPI v0.1.0 — **public alpha**

## Phase 2: Make It Useful for Real Agents

### Milestone 2: Evaluation Metrics
- [x] Task completion, tool correctness, step efficiency metrics
- [x] Golden dataset loader with schema validation
- [x] Aggregate scoring and regression detection
- [x] Token tracking and cost reporting
- [x] Custom evaluator plugin interface
- [ ] PyPI v0.2.0

### Milestone 3: Safety Testing + CI/CD
- [x] Safety marker with OWASP LLM Top 10 taxonomy
- [x] Built-in evaluators: prompt injection, PII leakage, system prompt leak, tool boundary, refusal compliance
- [x] Attack probe library — 68 probe templates (injection, jailbreak, PII, scope)
- [x] GitHub Action with quality gates and PR comments
- [x] Compliance report generation
- [x] JUnit XML output for CI dashboards
- [ ] PyPI v0.3.0

## Phase 3: Make It Complete

### Milestone 4: Record-and-Replay
- [x] Versioned JSON cassettes with stream support
- [x] Replay engine with configurable matching
- [x] Cassette migration tooling
- [ ] PyPI v0.4.0

### Milestone 5: LLM-as-Judge
- [ ] Rubric-based evaluation
- [ ] Statistical assertions (PASS/FAIL/INCONCLUSIVE)
- [ ] Multi-judge consensus
- [ ] PyPI v0.5.0

### Milestone 6: Framework Adapters + Production Loop
- [ ] LangChain, OpenAI Agents SDK, CrewAI, PydanticAI, Anthropic adapters
- [ ] Production trace import (Langfuse, Phoenix, OpenTelemetry)
- [ ] PyPI v1.0.0

### Milestone 7: Multi-Agent + Ecosystem
- [ ] Multi-agent trace capture and handoff testing
- [ ] Credit assignment heuristics
- [ ] PyPI v1.1.0

---

Want to help? Check out [CONTRIBUTING.md](CONTRIBUTING.md) and look for issues labeled `good first issue`.
