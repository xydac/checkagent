# Configuration

CheckAgent is configured via `checkagent.yml` at your project root. All settings have sensible defaults — you only need a config file if you want to customize behavior.

## Minimal Configuration

```yaml
# checkagent.yml
project:
  name: my-agent-tests
```

## Full Reference

```yaml
project:
  name: my-agent-tests
  description: Tests for my AI agent

# Test layer defaults
layers:
  mock:
    timeout_ms: 100          # Max time per mock test
  replay:
    cassette_dir: tests/cassettes
    strategy: exact           # exact | subset | sequence
    block_unmatched: true     # Raise on unmatched requests
  eval:
    dataset_dir: tests/golden
    threshold: 0.8            # Default pass threshold for metrics
  judge:
    trials: 3                 # Number of judge trials per evaluation
    model: gpt-4              # Default judge model

# Cost tracking
cost:
  budget_limit: 10.00         # Max spend per test run (USD)
  warn_at: 5.00               # Warn when spend exceeds this

# Safety
safety:
  enabled: true
  probe_categories:
    - injection
    - pii
    - jailbreak

# CI quality gates
ci:
  gates:
    - metric: task_completion
      min: 0.8
    - metric: safety_score
      min: 1.0
```

## Configuration Discovery

CheckAgent looks for configuration in this order:

1. `checkagent.yml` in the current directory
2. `checkagent.toml` in the current directory
3. `[tool.checkagent]` section in `pyproject.toml`

If no configuration is found, sensible defaults are used.

## Environment Variables

| Variable | Description |
|----------|-------------|
| `CHECKAGENT_CONFIG` | Path to config file (overrides auto-discovery) |
| `CHECKAGENT_LAYER` | Default layer filter for `checkagent run` |
