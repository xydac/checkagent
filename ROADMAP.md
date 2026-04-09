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

## Phase 4: Launch Readiness — Feature Arcs

Each arc is 3-5 cycles of connected work delivering a complete, coherent capability.

---

### Arc A: "First 5 Minutes" — Onboarding Fix (Priority 1)
**Why:** `checkagent init` is DX 1/1 (generates tests that fail immediately). Silent field drops on core types (F-027) mean beginners hit invisible bugs. This is THE adoption blocker — nothing else matters if the first 5 minutes are broken.

**

| Stage | Cycle | Deliverable |
|-------|-------|-------------|
| 1 | 084 | Fix `checkagent init` — generated tests must pass. Fix silent field drops (extra='forbid' on remaining types). Fix `assert_json_schema` missing dep (F-008). |
| 2 | 085 | Fix DX < 3 paper cuts: case-sensitive filters (F-023 severity), `was_called_with` substring confusion (F-009), `RunSummary` name collision (F-029). |
| 3 | 086 | End-to-end onboarding test: `pip install checkagent && checkagent init && pytest` must work in a clean venv. Add `checkagent[all]` extra (F-088). |
| 4 | 087 | Verify with QA agent. Update DX scores. Log experiment on onboarding success rate. |

**Delivers:** A new user can go from zero to green tests in under 2 minutes.

---

### Arc B: Documentation Site (Priority 2)
**Why:** No docs = no launch. 1668 tests mean nothing without a way for users to discover features. Every competitor has docs. We have a README.

**

| Stage | Cycle | Deliverable |
|-------|-------|-------------|
| 1 | 088 | MkDocs Material setup, GitHub Pages deploy, landing page (what/why/install/demo). |
| 2 | 089 | Quickstart guide + testing pyramid concept page + mock layer guide. |
| 3 | 090 (meta-review) | — |
| 4 | 091 | Remaining layer guides (replay, eval, judge) + safety guide. |
| 5 | 092 | CLI reference, config reference, API reference (mkdocstrings). |

**Delivers:** A complete documentation site at checkagent.dev (or GitHub Pages).

---

### Arc C: One-Command Safety Scan (Priority 3)

**

| Stage | Cycle | Deliverable |
|-------|-------|-------------|
| 1 | 093 (planning) | — |
| 2 | 094 | `checkagent scan` CLI — accepts a Python callable or HTTP endpoint, runs all safety probes, prints Rich-formatted report. |
| 3 | 095 (innovation) | Interactive TUI with drill-down into individual probe results. Press [t] to generate a pytest test file from scan results. |
| 4 | 096 | HTML/PDF compliance report export (OWASP mapping, EU AI Act references). |
| 5 | 097 | Docs page + README section + example with real agent scan. |

**Delivers:** `checkagent scan my_agent.py` — one command, full safety report, tweetable screenshot.

---

### Arc D: Real-Agent Validation (Priority 4)
**Why:** We claim "works with LangChain, OpenAI, CrewAI, PydanticAI, Anthropic" but the QA agent found real bugs in PydanticAI adapter (F-085, F-087). We cannot launch claiming framework support without testing real framework agents.

| Stage | Cycle | Deliverable |
|-------|-------|-------------|
| 1 | 098 | Test with real PydanticAI agent. Fix F-085 (empty input_text) and F-087 (deprecated tokens). |
| 2 | 099 | Test with real LangChain agent. Fix any adapter bugs found. |
| 3 | 100 (meta-review) | — |
| 4 | 101 | Test with OpenAI Agents SDK agent. Fix any adapter bugs found. |
| 5 | 102 | Update README with verified "works with" claims. Paper Section 5.1 data. |

**Delivers:** Verified, tested adapter support for top 3 frameworks.

---

### Arc E: CI/CD Integration Polish (Priority 5)
**Why:** QA found CI quality gates are disconnected from pytest (F-033, F-034). The GitHub Action exists but the pytest integration is manual. Enterprise users need this to work automatically.

| Stage | Cycle | Deliverable |
|-------|-------|-------------|
| 1 | 103 | pytest_sessionfinish hook that auto-evaluates quality gates from config. |
| 2 | 104 | `generate_pr_comment` integration with eval metrics and regressions. |
| 3 | 105 (innovation) | Auto-generated CI config (`checkagent ci-init`) for GitHub Actions, GitLab CI. |

**Delivers:** Quality gates that actually enforce in CI without manual wiring.

---

### Milestone 9: Local Dashboard (Future)
- [ ] Test run history storage (`.checkagent/` directory with JSON results)
- [ ] `checkagent dashboard` — local web UI showing test history, score trends, cost trends
- [ ] Safety probe detection rate visualization
- [ ] Publishable results export (charts/screenshots for paper and launch)

### Milestone 10: PyPI v0.1.0 + Launch (After Arcs A-D)
- [ ] PyPI v0.1.0 publish
- [ ] All README examples verified working
- [ ] checkagent demo runs clean end-to-end
- [ ] QA critical findings < 5 open
- [ ] Docs site live
- [ ] 3+ framework adapters validated with real agents

---

Want to help? Check out [CONTRIBUTING.md](CONTRIBUTING.md) and look for issues labeled `good first issue`.
