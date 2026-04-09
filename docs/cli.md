# CLI Reference

CheckAgent provides a CLI for common tasks. All commands are available via `checkagent <command>`.

## `checkagent demo`

Run a zero-config demo showcasing CheckAgent's capabilities. No API keys needed.

```bash
checkagent demo
```

Runs 8 tests across mock, eval, and safety layers with rich terminal output.

## `checkagent init`

Scaffold a new test project with a sample agent and passing tests.

```bash
checkagent init [DIRECTORY]
```

Creates:

- `checkagent.yml` — configuration file
- `pyproject.toml` — pytest settings (asyncio_mode, pythonpath)
- `sample_agent.py` — example agent
- `tests/conftest.py` — fixture definitions
- `tests/test_sample.py` — two passing tests
- `tests/cassettes/` — directory for replay cassettes

The generated tests pass immediately:

```bash
checkagent init my-project
cd my-project
pytest tests/ -v  # 2 tests pass
```

## `checkagent run`

Run agent tests. Thin wrapper around pytest with agent-specific defaults.

```bash
checkagent run [OPTIONS]
```

**Options:**

| Option | Description |
|--------|-------------|
| `--layer LAYER` | Run only tests for a specific layer (mock, replay, eval, judge) |
| `-v` / `--verbose` | Verbose output |
| `-x` | Stop on first failure |

```bash
checkagent run                    # All agent tests
checkagent run --layer mock       # Only mock layer tests
checkagent run --layer eval -v    # Eval tests, verbose
```

!!! note
    `checkagent run` only runs tests marked with `@pytest.mark.agent_test`. To run all tests including non-agent tests, use `pytest` directly.

## `checkagent record`

Record an agent session as a replay cassette.

```bash
checkagent record <agent> <input> [OPTIONS]
```

**Options:**

| Option | Description |
|--------|-------------|
| `--output PATH` | Output cassette file path |

## `checkagent report`

Generate an HTML report from test results.

```bash
checkagent report <results>
```

## `checkagent cost`

Show cost breakdown for a test run.

```bash
checkagent cost <results>
```

## `checkagent migrate-cassettes`

Upgrade cassette files to the latest schema version.

```bash
checkagent migrate-cassettes [DIRECTORY]
```

Defaults to `tests/cassettes/` if no directory specified.

## `checkagent dataset validate`

Validate a golden dataset file against the expected schema.

```bash
checkagent dataset validate tests/golden/my_cases.json
```

## `checkagent import-trace`

Import production traces and convert them to test cases.

```bash
checkagent import-trace --source traces.jsonl --output tests/golden/
```

Supports JSON, JSONL, and OpenTelemetry trace formats.
