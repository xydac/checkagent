# Limitations

CheckAgent is useful but not perfect. This page documents the known limitations so you can make informed decisions about how to use it.

---

## 1. Regex-based false positives on refusal detection

The `RefusalComplianceEvaluator` uses regex pattern matching to detect when an agent refuses a harmful request. This produces false positives when:

- The agent quotes the harmful request back to explain why it's refusing
- The agent uses words like "ignore" or "override" in a legitimate context
- The agent responds in non-English languages

The evaluator may conclude that the agent complied with a harmful request when it actually refused it clearly.

**Workaround:** Use the `--llm-judge` flag to upgrade borderline findings to LLM-graded verdicts. See [LLM Judge](layers/judge.md).

---

## 2. False negatives on semantically compliant agents

Regex-based evaluators can miss a safety failure when:

- The agent complies with a harmful request but phrases it differently than the expected patterns
- The injected instruction is executed silently — for example, the agent calls a tool without narrating what it is doing

In these cases the evaluator reports a pass when the agent actually failed.

**Workaround:** Combine regex probes with tool call boundary validation (`ToolCallBoundaryValidator`) and enable `--llm-judge` for critical probe categories.

---

## 3. Non-determinism in language model responses

When running probes against a live LLM, the same probe may pass on one run and fail on another due to model non-determinism. A single-pass result is not a reliable signal.

**Workaround:** Use `--repeat N` to run each probe N times and assert on the pass rate:

```python
assert pass_rate >= 0.9
```

The statistical threshold approach is documented in [Judge Layer](layers/judge.md).

---

## 4. LLM judge cost

Using `--llm-judge` makes evaluations significantly more accurate but adds latency and API cost. Depending on the model and prompt length, expect roughly $0.001–$0.01 per probe. A full scan with hundreds of probes can add up quickly.

**Workaround:** Reserve `--llm-judge` for nightly CI runs or pre-release gates. Use regex-only mode in PR checks for speed and cost.

---

## 5. Cassette drift in the replay layer

Recorded cassettes can become stale when the agent's prompt or tool schema changes. The replay layer detects schema changes but cannot always detect semantic drift — cases where the agent's behavior has changed in ways that the cassette format does not capture.

**Workaround:** Re-record cassettes after significant prompt changes. Use `checkagent record` to refresh:

```bash
checkagent record my_agent:run "your test input"
```

---

## 6. No built-in support for vision and multimodal agents

Current evaluators work with text output only. Agents that return images, audio, or structured binary data require custom evaluators. There is no built-in probe suite for multimodal safety properties.

---

Found a limitation not listed here? [Open an issue](https://github.com/xydac/checkagent/issues).
