# CheckAgent vs. Braintrust

CheckAgent and Braintrust solve related but distinct problems. CheckAgent is a developer-centric, pytest-native testing framework that runs entirely in your repo and CI pipeline — no external services required. Braintrust is a SaaS platform built around collaborative experiment tracking, human review workflows, and A/B comparisons across model versions. Depending on your team and workflow, they can complement each other rather than compete.

---

## Quick Comparison

| | CheckAgent | Braintrust |
|---|---|---|
| **Setup** | `pip install checkagent`, add tests to repo | SDK install + account, data flows to cloud |
| **License / Hosting** | Apache-2.0, fully self-hosted | Proprietary SaaS |
| **Test style** | pytest tests in your repo (`assert`, markers, fixtures) | SDK calls; results visible in web dashboard |
| **Experiment tracking** | Not a primary feature | Strong: compare runs, track history, A/B experiments |
| **Team collaboration** | Via standard code review (PRs, diffs) | Built-in: annotation, scoring, human review in UI |
| **Data ownership** | Stays in your repo and infrastructure | Flows to Braintrust cloud (data residency implications) |
| **CI integration** | First-class: SARIF output, GitHub Actions, fails PRs | Possible via SDK, but evaluation results live in dashboard |
| **Safety testing** | OWASP LLM Top 10 probes, data enumeration detection | Not a focus |
| **Cost** | Free (open source); no per-eval fees | Usage-based SaaS pricing |

---

## Where Braintrust Excels

- **Experiment history and A/B comparisons.** Braintrust makes it easy to compare eval results across model versions, prompts, or configurations over time. The dashboard lets you visually diff runs side by side without writing custom reporting code.

- **Team collaboration and human review.** Non-engineer team members — product managers, domain experts, annotators — can browse traces, add scores, and annotate outputs through the UI. This is a genuine strength for teams that need structured human feedback loops.

- **Breadth of SDK support.** Braintrust provides well-maintained Python and TypeScript SDKs that integrate with popular LLM frameworks, and the logging API is straightforward to add to existing applications.

- **Prompt and model management.** Braintrust includes tooling for versioning prompts and tracking which prompt version produced which results, which can streamline iterative prompt engineering.

---

## Where CheckAgent Excels

- **Developer-native workflow.** Tests are pytest tests. They live in your repo, run with `pytest`, block PRs via standard CI, and are reviewed like code. There is no external service to sign up for, no dashboard to check, no SDK call to add.

- **Deterministic layers without API keys.** The Mock and Replay layers (Layers 1 and 2) require no LLM API keys and run in milliseconds. You can write and run hundreds of agent regression tests for free, locally, in CI, with zero flakiness.

- **Safety testing built in.** CheckAgent includes probes mapped to the [OWASP LLM Top 10](owasp-mapping.md) — prompt injection, data enumeration, sensitive data leakage — that run as ordinary pytest tests. Failing a safety probe fails the CI job.

- **Full data sovereignty.** Nothing leaves your infrastructure. Cassettes, traces, and results are files in your repo. This matters for regulated industries, air-gapped environments, or teams with strict data residency requirements.

---

## When to Use Braintrust

- Your team includes non-engineers who need to review, annotate, or score model outputs through a UI.
- You are running systematic A/B experiments across multiple model providers or prompt variants and want a structured way to track result history over weeks or months.
- You want a managed platform and are comfortable with data flowing to a third-party cloud.
- Rapid experiment iteration is more important than CI gate integration.

---

## When to Use CheckAgent

- You want agent tests to live in your repo and fail PRs, the same way unit tests do.
- You need deterministic, zero-cost regression testing (no LLM API calls, no fees, no flakiness).
- Data must stay on-premises or within your own cloud account — no third-party services.
- You need safety probes (OWASP LLM Top 10) as part of your standard CI pipeline.
- Your team works primarily in the terminal and in code review, and a separate dashboard adds friction rather than value.

---

## Can You Use Both?

Yes — they address different parts of the development lifecycle and can work together.

A practical split:

- **Braintrust** for experiment tracking during active development: comparing prompt versions, reviewing output quality across model upgrades, getting human feedback from non-engineers.
- **CheckAgent** for CI gates: deterministic mock and replay tests on every PR, safety probes before deploy, SARIF reports in GitHub Security, and regression coverage without API costs.

The two tools do not conflict. Braintrust instrumentation can be added to the same agents that CheckAgent tests. If you already use Braintrust for experiments and want stricter PR gates or safety coverage, CheckAgent adds that layer without replacing what Braintrust does well.
