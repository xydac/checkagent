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

## `checkagent scan`

Scan an agent for safety vulnerabilities. Runs 68 attack probes across prompt injection, jailbreak, PII leakage, and scope violation categories.

```bash
checkagent scan <TARGET>
```

TARGET is a Python callable in `module:function` format.

```bash
checkagent scan my_agent:run
checkagent scan my_app.agents.booking:handle_request
```

**Options:**

| Option | Description |
|--------|-------------|
| `-c`, `--category` | Run only probes from a category: `injection`, `jailbreak`, `pii`, `scope` |
| `-t`, `--timeout FLOAT` | Timeout in seconds per probe (default: 10.0) |
| `-v`, `--verbose` | Show all probes, not just failures |
| `-g`, `--generate-tests FILE` | Generate a pytest file from findings |

**Examples:**

```bash
checkagent scan my_agent:run                              # Full scan
checkagent scan my_agent:run --category injection         # Injection probes only
checkagent scan my_agent:run -g test_safety.py            # Generate regression tests
checkagent scan my_agent:run --timeout 5 --verbose        # Custom timeout, verbose
```

The `--generate-tests` flag creates a pytest file with one test per finding, so you can track safety regressions in CI:

```bash
checkagent scan my_agent:run -g test_safety.py
pytest test_safety.py -v
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
