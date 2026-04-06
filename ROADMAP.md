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

### Milestone 8: Documentation Site (In Progress)
- [ ] MkDocs Material setup with GitHub Pages deployment
- [ ] Landing page (index.md) — what, why, install, 30-sec demo
- [ ] Quickstart guide — pip install → first test → green in 5 min
- [ ] Testing pyramid concept guide
- [ ] Layer guides: Mock, Replay, Eval, Judge
- [ ] Feature guides: Safety, Fault Injection, Multi-Turn, Streaming, CI/CD, Cost Tracking
- [ ] CLI reference
- [ ] Configuration reference (checkagent.yml)
- [ ] API reference (auto-generated with mkdocstrings)

### Milestone 9: Local Dashboard
- [ ] Test run history storage (`.checkagent/` directory with JSON results)
- [ ] `checkagent dashboard` — local web UI showing test history, score trends, cost trends
- [ ] Safety probe detection rate visualization
- [ ] Publishable results export (charts/screenshots for paper and launch)

### Milestone 10: PyPI + Launch
- [ ] PyPI v0.1.0 publish
- [ ] All README examples verified working
- [ ] checkagent demo runs clean end-to-end
- [ ] QA critical findings < 5 open

---

Want to help? Check out [CONTRIBUTING.md](CONTRIBUTING.md) and look for issues labeled `good first issue`.
