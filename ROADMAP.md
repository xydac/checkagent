# Roadmap

CheckAgent is being built in public. Here's where we're headed.

## Phase 1: Ship Something People Love

### Milestone 0: Foundation + Demo (Complete)
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
- [x] Rubric-based evaluation
- [x] Statistical assertions (PASS/FAIL/INCONCLUSIVE)
- [x] Multi-judge consensus
- [ ] PyPI v0.5.0

### Milestone 6: Framework Adapters + Production Loop
- [x] LangChain adapter
- [x] OpenAI Agents SDK adapter
- [x] CrewAI, PydanticAI, Anthropic adapters
- [x] Production trace import (JSON/JSONL, OpenTelemetry; Langfuse/Phoenix API planned)
- [ ] PyPI v1.0.0

### Milestone 7: Multi-Agent + Ecosystem
- [x] Multi-agent trace capture and handoff testing
- [x] Credit assignment heuristics
- [ ] PyPI v1.1.0

## Phase 4: Launch Readiness

### Milestone 8: Onboarding Polish (In Progress)
- [ ] `checkagent init` generates tests that pass on a clean install
- [ ] Strict Pydantic types prevent silent field drops
- [ ] All dependencies declared correctly
- [ ] `pip install checkagent && checkagent init && pytest` works end-to-end

### Milestone 9: Documentation Site
- [ ] MkDocs Material with GitHub Pages
- [ ] Quickstart guide
- [ ] Layer guides (Mock, Replay, Eval, Judge)
- [ ] Safety testing guide
- [ ] CLI + config reference
- [ ] API reference

### Milestone 10: One-Command Safety Scan
- [ ] `checkagent scan` — point at any agent, get a safety report
- [ ] Interactive TUI with drill-down results
- [ ] Generate pytest tests from scan findings
- [ ] HTML/PDF compliance report export

### Milestone 11: Framework Validation
- [ ] Test adapters against real framework agents (not just mocks)
- [ ] Verified support for LangChain, OpenAI Agents SDK, PydanticAI

### Milestone 12: CI/CD Polish
- [ ] Quality gates auto-enforce from config
- [ ] PR comment generation with eval metrics
- [ ] `checkagent ci-init` for easy CI setup

### Milestone 13: v0.1.0 Launch
- [ ] PyPI v0.1.0
- [ ] All README examples verified working
- [ ] Docs site live
- [ ] 3+ framework adapters validated with real agents

### Future
- [ ] Local dashboard for test history and trends
- [ ] Auto-instrumentation (one import, zero config)
- [ ] Production trace import from more providers

---

Want to help? Check out [CONTRIBUTING.md](CONTRIBUTING.md) and look for issues labeled `good first issue`.
