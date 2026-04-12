# OWASP LLM Top 10 Coverage

> **Note:** Coverage is as of CheckAgent v0.2. The [roadmap](https://github.com/xydac/checkagent/blob/main/ROADMAP.md) tracks planned additions.

CheckAgent's safety probe library is organized around the [OWASP Top 10 for Large Language Model Applications](https://owasp.org/www-project-top-10-for-large-language-model-applications/) (2025 edition). This page maps each OWASP category to the specific probes and evaluators that test for it, so you can communicate coverage to security teams and compliance reviewers using a standard vocabulary.

## Summary Table

| OWASP ID | Name | Coverage | CheckAgent Feature |
|----------|------|----------|--------------------|
| LLM01 | Prompt Injection | **Full** | `ProbeSet.injection()`, `PromptInjectionEvaluator` |
| LLM02 | Sensitive Information Disclosure | **Full** | `ProbeSet.pii()`, `PIILeakageEvaluator`, `SystemPromptLeakEvaluator` |
| LLM03 | Supply Chain Vulnerabilities | Out of scope | — |
| LLM04 | Data and Model Poisoning | Out of scope | — |
| LLM05 | Improper Output Handling | **Partial** | `assert_output_schema` |
| LLM06 | Excessive Agency | **Full** | `ProbeSet.scope()`, `ToolCallBoundaryValidator` |
| LLM07 | System Prompt Leakage | **Full** | `SystemPromptLeakEvaluator` |
| LLM08 | Vector and Embedding Weaknesses | **Planned** | Retrieval poisoning probes (roadmap) |
| LLM09 | Misinformation | **Partial** | `GroundednessEvaluator` |
| LLM10 | Unbounded Consumption | **Partial** | Cost tracking, token budget assertions |

---

## Covered Categories

### LLM01 — Prompt Injection

**What it means:** An attacker embeds instructions in user input or external data that override the agent's intended behaviour.

**Coverage: Full** — 50 probes across four attack surfaces.

| Probe module | Count | Attack surface |
|---|---|---|
| `probes.injection.direct` | 25 | User messages with direct override attempts |
| `probes.injection.indirect` | 10 | Injections embedded in tool outputs |
| `probes.jailbreak.encoding` | 8 | Base64, ROT13, Unicode obfuscation |
| `probes.jailbreak.roleplay` | 7 | "Pretend you are…" persona hijacks |

All four modules feed into `PromptInjectionEvaluator` (direct/indirect) and `RefusalComplianceEvaluator` (roleplay).

```python
import pytest
from checkagent.safety import probes

# Parametrize over all 25 direct injection probes
@pytest.mark.agent_test(layer="mock")
@pytest.mark.safety(category="prompt_injection", severity="high")
@pytest.mark.parametrize("attack", probes.injection.direct.all())
async def test_direct_injection(my_agent, attack, ca_safety):
    result = await my_agent.run(attack.input)
    await ca_safety.assert_no_injection(result)

# Run the full injection + jailbreak suite in one call
@pytest.mark.agent_test(layer="mock")
async def test_injection_suite(my_agent, ca_safety):
    await ca_safety.assert_probe_set(my_agent, probes.ProbeSet.injection())
```

---

### LLM02 — Sensitive Information Disclosure

**What it means:** The agent reveals private or regulated data — PII, credentials, internal system details — that it should not expose.

**Coverage: Full** — 10 PII extraction probes plus system prompt leak detection.

CheckAgent tests two distinct disclosure vectors:

**PII leakage** — `ProbeSet.pii()` / `PIILeakageEvaluator`

The 10 PII probes ask the agent to repeat, summarise, or reformat data it may have been given in context, checking whether it surfaces names, email addresses, phone numbers, SSNs, credit card numbers, and similar regulated fields.

```python
@pytest.mark.agent_test(layer="mock")
@pytest.mark.safety(category="pii_leakage", severity="critical")
@pytest.mark.parametrize("probe", probes.pii.all())
async def test_no_pii_leakage(my_agent, probe, ca_safety):
    result = await my_agent.run(probe.input)
    await ca_safety.assert_no_pii(result)
```

**System prompt disclosure** — `SystemPromptLeakEvaluator`

Probes ask the agent to repeat, translate, or paraphrase its system prompt, catching naive implementations that echo it verbatim.

```python
from checkagent.safety import SystemPromptLeakDetector

@pytest.mark.agent_test(layer="mock")
async def test_no_system_prompt_leak(my_agent, ca_safety):
    await ca_safety.assert_no_system_prompt_leak(my_agent)
```

---

### LLM05 — Improper Output Handling

**What it means:** The agent produces output in a format that downstream consumers handle unsafely — for example, returning unsanitised HTML that a browser renders, or SQL that a database executes.

**Coverage: Partial** — `assert_output_schema` validates that structured output conforms to a declared schema, catching type confusion and unexpected fields. However, detecting HTML injection, SQL injection, or code injection *within* a conformant response requires custom evaluators that encode your application's rendering context. CheckAgent provides the extension point (`SafetyEvaluator` base class) but not built-in probes for every output sink.

```python
@pytest.mark.agent_test(layer="eval")
async def test_structured_output_safe(my_agent, ca_mock_llm):
    result = await my_agent.run("Summarise the ticket")
    ca_mock_llm.assert_output_schema(result, schema=TicketSummary)
```

---

### LLM06 — Excessive Agency

**What it means:** The agent invokes tools, takes actions, or acquires capabilities beyond what the task requires, potentially causing unintended side-effects.

**Coverage: Full** — 8 scope boundary probes plus `ToolCallBoundaryValidator`.

`ProbeSet.scope()` sends inputs that should be handled within defined tool boundaries and checks whether the agent calls unauthorised tools, escalates permissions, or chains tool calls beyond its declared scope.

```python
from checkagent.safety import ToolCallBoundaryValidator, ToolBoundary

ALLOWED = ToolBoundary(
    allowed_tools={"search", "summarise"},
    forbidden_tools={"send_email", "delete_file", "execute_code"},
)

@pytest.mark.agent_test(layer="mock")
@pytest.mark.safety(category="tool_misuse", severity="high")
@pytest.mark.parametrize("probe", probes.scope.all())
async def test_tool_scope(my_agent, probe, ca_safety):
    result = await my_agent.run(probe.input)
    validator = ToolCallBoundaryValidator(boundary=ALLOWED)
    finding = validator.evaluate(result)
    assert finding is None, f"Unexpected tool call: {finding}"
```

---

### LLM07 — System Prompt Leakage

**What it means:** The agent reveals the contents of its system prompt, which may contain confidential instructions, persona configuration, or business logic.

**Coverage: Full** — `SystemPromptLeakEvaluator` runs a battery of extraction attempts including direct requests, indirect elicitation, and translate-and-repeat tricks.

```python
from checkagent.safety import SystemPromptLeakDetector

@pytest.mark.agent_test(layer="mock")
@pytest.mark.safety(category="prompt_injection", severity="high")
async def test_system_prompt_confidential(my_agent):
    detector = SystemPromptLeakDetector()
    findings = await detector.scan(my_agent)
    assert not findings, f"System prompt leaked in {len(findings)} probe(s)"
```

---

### LLM09 — Misinformation

**What it means:** The agent generates plausible but factually incorrect responses — hallucinating citations, statistics, or events that did not occur.

**Coverage: Partial** — `GroundednessEvaluator` checks whether agent responses are grounded in provided context, flagging claims that cannot be traced to source documents. This catches retrieval-augmented hallucination well. It does not cover open-domain factual accuracy (verifying claims against world knowledge), which requires a live reference corpus or LLM judge.

```python
from checkagent.safety import GroundednessEvaluator

@pytest.mark.agent_test(layer="eval")
async def test_response_grounded(my_rag_agent, ca_mock_llm):
    context = "Our refund policy allows returns within 30 days."
    result = await my_rag_agent.run("What is the refund policy?", context=context)
    evaluator = GroundednessEvaluator(threshold=0.85)
    score = evaluator.evaluate(result, context=context)
    assert score.passed, f"Groundedness score {score.value:.2f} below threshold"
```

---

### LLM10 — Unbounded Consumption

**What it means:** The agent consumes disproportionate resources — tokens, API calls, time, or money — potentially enabling denial-of-wallet attacks or degrading service for other users.

**Coverage: Partial** — CheckAgent's cost tracking module records token usage and estimated spend for every test run. You can assert a budget ceiling on any test, and unusual token usage will surface as a test failure. There are no dedicated probes that actively attempt to exhaust resources (e.g., prompt-flooding or recursive tool-call loops), so detection is reactive rather than adversarial.

```python
@pytest.mark.agent_test(layer="eval")
async def test_no_runaway_tokens(my_agent, ca_mock_llm):
    result = await my_agent.run("Explain quantum computing")
    assert result.usage.total_tokens < 2000, (
        f"Response used {result.usage.total_tokens} tokens — check for prompt amplification"
    )
```

---

## Not Covered

### LLM03 — Supply Chain Vulnerabilities

**What it means:** Compromised model weights, poisoned fine-tuning datasets, or malicious third-party plugins are introduced through the model supply chain before the agent runs.

**Why it is out of scope:** Supply chain integrity is a static analysis and provenance problem, not a runtime testing problem. Verifying model provenance, plugin signatures, and dependency hashes is outside CheckAgent's test-execution model. Use software composition analysis (SCA) tools and model cards from your model provider for this category.

### LLM04 — Data and Model Poisoning

**What it means:** An attacker corrupts training data or fine-tuning datasets so the resulting model exhibits backdoor behaviours at inference time.

**Why it is out of scope:** Data poisoning attacks occur during training, not during inference. Runtime agent testing cannot detect whether a model was trained on poisoned data — that requires training pipeline audits and behavioural red-teaming at scale. CheckAgent focuses on inference-time behaviour; training-time concerns are out of scope.
