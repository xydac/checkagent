# CheckAgent vs DeepEval

CheckAgent and DeepEval are both open-source frameworks for testing AI systems, and they overlap in the "Eval" and "Judge" layers of the testing pyramid. The meaningful differences are in philosophy and focus: DeepEval is built around LLM output quality metrics — especially for RAG pipelines — while CheckAgent is built around the full agent development lifecycle, from deterministic unit tests through safety probes and CI integration. If you are evaluating which tool to adopt, or wondering whether to use both, this page lays out an honest comparison.

---

## Quick Comparison

| Dimension | CheckAgent | DeepEval |
|-----------|-----------|---------|
| **Test style** | pytest-native (`assert`, fixtures, markers) | Own test runner (`deepeval test run`); pytest integration available |
| **Setup time (no API key)** | ~0 min for layers 1–2 (mock/replay) | Most metrics require an LLM API key; minimal useful coverage without one |
| **Setup time (full)** | ~5 min | ~25 min (API key + optional Confident AI account) |
| **RAG evaluation** | Basic groundedness metric | Deep: faithfulness, contextual precision/recall, answer relevancy, and more |
| **Agent tool-call tracing** | Full: per-step traces, tool boundary validation | Limited |
| **Safety probes** | 68 probes, OWASP LLM Top 10 mapped | Not a primary focus |
| **Cost tracking** | Built-in; per-test token usage and budget assertions | Not built-in |
| **Self-hosted** | Fully self-hosted; zero telemetry | Open-source core; Confident AI is a hosted platform |
| **Dashboard** | None (SARIF + CI annotations) | Confident AI hosted dashboard |
| **License** | Apache-2.0 | MIT (core); Confident AI platform has separate terms |

---

## Where DeepEval Excels

- **RAG pipeline evaluation.** DeepEval was designed with retrieval-augmented generation in mind. Its metrics for faithfulness, contextual precision, contextual recall, and answer relevancy are well-designed, actively maintained, and provide fine-grained signal that is hard to replicate from scratch.

- **Breadth of pre-built LLM metrics.** DeepEval ships G-Eval, hallucination detection, summarization quality, and other metrics out of the box. For teams that primarily need output quality scoring and want to reach for proven implementations rather than build their own, it is a strong starting point.

- **Hosted results dashboard.** Confident AI gives teams a visual interface for tracking metric trends across runs. If your team is not comfortable reading CI logs or SARIF files, or if you need to share results with non-engineers, the hosted dashboard is a genuine advantage.

- **Documentation and community.** DeepEval has thorough documentation, worked examples for many frameworks, and an active community. The learning curve is reasonable and there is a lot of written guidance available.

---

## Where CheckAgent Excels

- **Deterministic, free-tier testing.** The Mock layer runs in milliseconds with no API calls, no keys, and no cost. You can write hundreds of agent unit tests that run on every commit without touching a rate limit or spending a dollar. DeepEval does not have an equivalent tier.

- **Full agent lifecycle coverage.** CheckAgent covers the testing pyramid from deterministic unit tests (Mock) through record-and-replay regression (Replay), metric evaluation (Eval), and LLM-judged assertions (Judge) — with explicit cost and frequency guidance for each layer. The framework is designed around the question: "which tests should I run, when, and at what cost?"

- **Safety testing as a first-class feature.** 68 attack probes covering OWASP LLM Top 10 categories — prompt injection, PII leakage, jailbreak, and tool scope violations — ship as part of the core framework, not as an add-on. Probes produce SARIF output that integrates directly with GitHub Advanced Security.

- **Agent tool-call tracing and boundary validation.** CheckAgent captures per-step execution traces including which tools were called, with what arguments, and what they returned. Tool boundary tests (`ToolCallBoundaryValidator`) let you assert that your agent never calls tools outside its permitted scope — a concern that output quality metrics do not cover.

---

## When to Use DeepEval

- You are building or auditing a RAG pipeline and need detailed retrieval quality metrics (faithfulness, contextual precision/recall, answer relevancy). DeepEval's RAG metrics are mature and purpose-built.

- Your team values a hosted visual dashboard for tracking metric trends over time, and you are comfortable with the Confident AI platform's terms and data handling.

- You primarily need output quality scoring for conversational or Q&A systems, and agent tool orchestration, safety probes, and CI-native workflows are not a priority.

---

## When to Use CheckAgent

- You are building an agent (tool-using, multi-step, or autonomous) and want a full testing strategy: unit tests on every commit, regression tests on every PR, quality metrics on merge, and LLM-judged assertions nightly.

- Security and compliance matter: you need a self-hosted solution with zero telemetry, auditable Apache-2.0 source code, and OWASP-mapped safety probes that produce SARIF output for your security toolchain.

- You want to test agent behavior — tool call sequences, cost per run, boundary violations — rather than (or in addition to) output text quality.

---

## Can I Use Both?

Yes. CheckAgent and DeepEval address different parts of the problem. A team building a RAG-backed agent could reasonably use CheckAgent for Mock-layer unit tests, Replay regression tests, agent tracing, and safety probes — and use DeepEval's faithfulness and contextual recall metrics in the Eval layer to score retrieval quality. Both frameworks use Python and pytest, so they can coexist in the same test suite without conflict.

If you find a useful pattern for combining them, consider opening a discussion on the [CheckAgent GitHub](https://github.com/xydac/checkagent/discussions).
