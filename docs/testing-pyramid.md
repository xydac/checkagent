# The Agent Testing Pyramid

Traditional software testing has the unit/integration/e2e pyramid. Agent testing needs its own hierarchy — one that accounts for non-determinism, LLM costs, tool side effects, and multi-step reasoning chains. CheckAgent's four-layer pyramid is that hierarchy.

This page explains the conceptual model. By the end you should understand what each layer tests, when to run it, and why the ratio matters.

---

## Why testing AI agents is different

Testing a pure function is straightforward: given input X, assert output Y. Agents break this model in three ways.

**Non-determinism.** A language model samples from a probability distribution. The same prompt does not guarantee the same output. You cannot assert `result == "Paris"` the way you would for a deterministic function. You need statistical framing ("this output passes quality criteria ≥ 80% of the time") or you need to remove the non-determinism from specific test runs entirely.

**Tool use and side effects.** Agents invoke tools — search APIs, databases, code interpreters, external services. A test that fires real tool calls is an integration test with all the costs that implies: latency, flakiness, money, potential mutations to production data. You need a seam where you can swap real tools for controlled fakes and still exercise the agent's routing and decision logic.

**Multi-step reasoning.** An agent's output is the product of many decisions: which tool to call, how to interpret its result, whether to call another tool or return a final answer, how to format the response. A single end-to-end correctness check tells you whether the agent succeeded but not where it went wrong. You need finer-grained assertions at the level of individual steps, tool selections, and trajectory shapes.

The four-layer pyramid addresses all three. Each layer sacrifices some realism for some combination of speed, cost control, and determinism. Used together they give you comprehensive coverage at sustainable cost.

---

## The pyramid

```
                        ╱‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾╲
                       │   JUDGE                     │
                       │   LLM-as-judge · rubrics     │
                       │   ~1% of tests · $$$          │
                       │   Minutes · Pre-release        │
                      ╱‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾╲
                     │   EVAL                            │
                     │   Metrics · golden datasets        │
                     │   ~4% of tests · $$                 │
                     │   Seconds–minutes · On merge         │
                    ╱‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾╲
                   │   REPLAY                                │
                   │   Cassette record & replay               │
                   │   ~15% of tests · $                       │
                   │   Seconds · Every PR                       │
                  ╱‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾╲
                 │   MOCK                                        │
                 │   MockLLM · MockTool · deterministic           │
                 │   ~80% of tests · Free                          │
                 │   Milliseconds · Every commit                    │
                  ╲______________________________________________╱
```

The base is the widest: the most tests, the cheapest, the fastest. The apex is the narrowest: the fewest tests, the most expensive, the slowest. Layers build on each other — a solid mock foundation makes replay tests more stable, which makes eval results more meaningful.

---

## Layer-by-layer breakdown

### Layer 1: Mock

**What it tests:** Agent logic in isolation. Routing decisions ("given this user intent, does the agent select the right tool?"), error handling ("when a tool returns an error, does the agent retry or surface it cleanly?"), output formatting, guardrail enforcement.

The LLM and all external tools are replaced with deterministic fakes. `ca_mock_llm` lets you pre-program exactly what the model returns. `ca_mock_tool` captures tool calls and lets you assert on arguments.

**When to run:** Every commit. Pre-push hooks. CI on every push. These tests are free and finish in under 100ms each — there is no reason not to run them constantly.

**Cost:** $0. No API calls leave your machine.

**Speed:** Milliseconds per test.

```python
@pytest.mark.agent_test(layer="mock")
async def test_routes_to_search_tool(my_agent, ca_mock_llm, ca_mock_tool):
    ca_mock_llm.add_response("I'll search for that.", tool_calls=["search"])
    result = await my_agent.run("What is the capital of France?")
    ca_mock_tool.assert_called("search", times=1)
```

---

### Layer 2: Replay

**What it tests:** Regressions. You record a real agent session once — the actual LLM responses, actual tool outputs, actual execution trajectory — and store it as a JSON cassette. On every subsequent PR, CheckAgent replays that cassette deterministically. If a prompt change, tool schema change, or logic refactor breaks the expected trajectory, the test fails.

Cassettes are content-addressed JSON files. They live in version control alongside your tests. Reviewing a cassette diff in a PR tells you exactly how agent behavior changed.

**When to run:** Every PR. Before merging feature branches. After any change to prompts, tool schemas, or agent logic.

**Cost:** Cheap. No new LLM calls are made during replay — only the original recording incurs model cost.

**Speed:** Seconds per test.

```python
@pytest.mark.agent_test(layer="replay")
async def test_summarization_regression(my_agent, ca_cassette):
    async with ca_cassette("summarize_article_v1"):
        result = await my_agent.run("Summarize this article: ...")
        assert result.final_output is not None
```

---

### Layer 3: Eval

**What it tests:** Quality against a golden dataset. You define a set of inputs with expected outputs or quality criteria, run the agent against them, and measure metrics: task completion rate, tool call correctness, trajectory adherence, output similarity.

Unlike mock and replay, eval uses real LLM calls (or a cached subset). The results tell you whether your agent actually solves the problems you care about, not just whether it follows a recorded script.

**When to run:** On merge to main. On scheduled nightly runs. Before and after significant prompt changes to measure impact.

**Cost:** Moderate. Each test makes real LLM calls; cost scales with dataset size and agent complexity.

**Speed:** Seconds to minutes depending on dataset size.

```python
@pytest.mark.agent_test(layer="eval")
async def test_task_completion_rate(my_agent, ca_dataset):
    results = await ca_dataset.evaluate(my_agent, "golden/qa_pairs.json")
    assert results.task_completion_rate >= 0.85
```

---

### Layer 4: Judge

**What it tests:** Subjective quality at statistical scale. A judge layer test runs the agent many times (e.g., 50 runs) and uses a second LLM to evaluate each response against a rubric. The assertion is statistical: "at least 80% of responses must pass the rubric."

This is the right tool for measuring qualities that cannot be captured with exact string matching or simple metrics: helpfulness, tone, reasoning quality, instruction following on open-ended tasks.

**When to run:** Before major releases. After significant model or prompt upgrades. On a weekly or monthly cadence as a quality baseline.

**Cost:** Expensive. Each run makes at least two LLM calls (agent + judge), multiplied by the sample size.

**Speed:** Minutes.

```python
@pytest.mark.agent_test(layer="judge")
async def test_response_quality(my_agent, ca_judge):
    verdict = await ca_judge.evaluate(
        my_agent,
        prompt="Explain recursion to a 10-year-old.",
        rubric="rubrics/clarity.yml",
        n=50,
    )
    assert verdict.pass_rate >= 0.80
```

---

## The rule of thumb: 80-15-4-1

A healthy agent test suite follows this distribution:

| Layer | Share | Rationale |
|-------|-------|-----------|
| Mock | 80% | Free and fast — maximize coverage here |
| Replay | 15% | Regression net for real-world trajectories |
| Eval | 4% | Quality measurement; run on merges not commits |
| Judge | 1% | Statistical quality gates; reserve for releases |

This is not an arbitrary ratio. It reflects cost and feedback loop constraints.

If you invert the pyramid — heavy on judge and eval, light on mock — you pay for LLM calls on every commit, your CI pipeline takes minutes instead of seconds, and your feedback loop degrades to the point where developers stop running tests locally. You end up with expensive tests that run rarely and a codebase that drifts.

The 80% mock base gives you a fast, free feedback loop that catches logic errors before they reach code review. The 15% replay layer catches behavioral regressions that mock tests cannot see. The 4% eval layer validates quality on real inputs periodically. The 1% judge layer provides statistical quality assurance before high-stakes releases.

!!! tip "Start at the base"
    If you're adding CheckAgent to an existing project, start by writing mock tests for your most critical agent behaviors. Add replay tests for your most important user flows. Eval and judge layers can come later — they are most valuable once you have a stable behavioral baseline to measure against.

---

## Running each layer

### By layer marker

```bash
# Mock only — runs in seconds, free
pytest -m "agent_test and layer_mock"

# Mock + replay
pytest -m "agent_test and (layer_mock or layer_replay)"

# All layers
pytest -m agent_test
```

### Via the CLI

```bash
# Run only mock tests
checkagent run --layer mock

# Run mock and replay
checkagent run --layer mock --layer replay

# Run all layers with cost reporting
checkagent run --all-layers --cost
```

### Cost guardrails

Set per-run budget limits in `checkagent.yml` so eval and judge runs cannot exceed your budget:

```yaml
layers:
  eval:
    max_cost_usd: 5.00
  judge:
    max_cost_usd: 20.00
    runs_per_test: 50
```

---

## The pyramid in CI

Map layers to CI trigger events so you pay for expensive layers only when they add value.

```yaml
# .github/workflows/test.yml

on:
  push:
    branches: ["**"]       # Every commit
  pull_request:            # Every PR
  merge_group:             # Merge queue
  schedule:
    - cron: "0 2 * * *"    # Nightly at 02:00 UTC

jobs:
  mock:
    # Runs on every commit — fast, free
    if: always()
    steps:
      - uses: checkagent/checkagent-action@v1
        with:
          layer: mock

  replay:
    # Runs on every PR — catches behavioral regressions
    if: github.event_name == 'pull_request' || github.event_name == 'merge_group'
    steps:
      - uses: checkagent/checkagent-action@v1
        with:
          layer: replay

  eval:
    # Runs on merge to main — validates quality metrics
    if: github.event_name == 'merge_group'
    steps:
      - uses: checkagent/checkagent-action@v1
        with:
          layer: eval
          max-cost: "5.00"

  judge:
    # Runs nightly — statistical quality baseline
    if: github.event_name == 'schedule'
    steps:
      - uses: checkagent/checkagent-action@v1
        with:
          layer: judge
          max-cost: "20.00"
```

This structure ensures every developer gets sub-second mock feedback locally and on push, regression protection on every PR, quality validation on every merge, and statistical quality assurance nightly — with costs proportional to the value each layer provides.

!!! note "Quality gates"
    Eval and judge layers can be configured to block merges if quality metrics drop below a threshold. See [GitHub Action](github-action.md) for quality gate configuration.

---

## Getting started

The fastest way to see the pyramid in action:

```bash
pip install checkagent
checkagent demo
```

The demo runs tests across mock, eval, and safety layers in under 30 seconds with no API keys.

Ready to test your own agent? See the [Quickstart](quickstart.md).
