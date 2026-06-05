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

Scan an agent for safety vulnerabilities. Runs 101 attack probes across six categories: prompt injection, jailbreak, PII leakage, scope violation, data enumeration, and groundedness.

Scan a Python callable:

```bash
checkagent scan my_agent:run
checkagent scan my_app.agents.booking:handle_request
```

Or scan any HTTP endpoint — works with agents in any language or framework:

```bash
checkagent scan --url http://localhost:8000/chat
checkagent scan --url http://localhost:8000/api --input-field query
checkagent scan --url http://localhost:8000/api -H 'Authorization: Bearer tok'
```

**Options:**

| Option | Description |
|--------|-------------|
| `-u`, `--url URL` | Scan an HTTP endpoint instead of a Python callable |
| `--input-field TEXT` | JSON field name for the probe input in HTTP requests (default: `message`) |
| `--output-field TEXT` | JSON field name to extract from HTTP responses (auto-detected if not set) |
| `-H`, `--header TEXT` | HTTP header as `Name: Value` (repeatable) |
| `-c`, `--category` | Run only probes from a category: `injection`, `jailbreak`, `pii`, `scope`, `data_enumeration`, `groundedness` |
| `-t`, `--timeout FLOAT` | Timeout in seconds per probe (default: 10.0) |
| `-v`, `--verbose` | Show all probes, not just failures |
| `-g`, `--generate-tests FILE` | Generate a pytest file from findings |
| `--json` | Output results as JSON to stdout |
| `--badge FILE` | Generate a shields.io-style SVG badge |
| `--sarif FILE` | Write scan results as SARIF 2.1.0 to FILE (for GitHub Code Scanning integration) |
| `--comment-file FILE` | Write a Markdown PR comment summary to FILE (suitable for GitHub PR comments) |
| `--report FILE` | Write a full HTML compliance report to FILE (e.g. `--report safety.html`) |
| `-r`, `--repeat N` | Run each probe N times and aggregate results; reports a stability score (default: 1) |
| `--llm-judge MODEL` | Use an LLM to judge each probe response. Accepts `gpt-4o-mini`, `claude-haiku-4-5-20251001`, or `claude-code` (uses your local Claude Code CLI — no API key required). |
| `--agent-description TEXT` | Describe what your agent does and what it should refuse. Used by `--llm-judge`. |
| `--prompt-file FILE` | Path to a system prompt file. Runs static prompt analysis alongside the dynamic scan. |
| `--diff` | Compare results against the previous scan from history and display new/fixed findings. |

**Examples:**

```bash
checkagent scan my_agent:run                              # Full scan (101 probes)
checkagent scan --url http://localhost:8000/chat           # Scan HTTP endpoint
checkagent scan my_agent:run --category injection         # Injection probes only
checkagent scan my_agent:run --category data_enumeration  # Data enumeration probes only
checkagent scan my_agent:run -g test_safety.py            # Generate regression tests
checkagent scan my_agent:run --timeout 5 --verbose        # Custom timeout, verbose
checkagent scan my_agent:run --json                       # JSON output
checkagent scan my_agent:run --sarif scan.sarif           # SARIF output for GitHub Code Scanning
checkagent scan my_agent:run --badge badge.svg            # Generate SVG badge
checkagent scan my_agent:run --comment-file comment.md   # PR comment Markdown
checkagent scan my_agent:run --repeat 3                   # Run each probe 3 times for stability score
checkagent scan my_agent:run \
    --llm-judge gpt-4o-mini \
    --agent-description "Customer support bot. Must refuse instruction overrides."
checkagent scan my_agent:run --llm-judge claude-code      # No API key needed — uses local Claude
checkagent scan my_agent:run --prompt-file system_prompt.txt
checkagent scan my_agent:run --report safety.html         # Full HTML compliance report
checkagent scan my_agent:run --diff                        # Compare against last scan
```

The `--generate-tests` flag creates a pytest file with one test per finding, so you can track safety regressions in CI:

```bash
checkagent scan my_agent:run -g test_safety.py
pytest test_safety.py -v
```

The `--sarif` flag writes results in SARIF 2.1.0 format, which GitHub Code Scanning can ingest directly to surface findings as pull request annotations:

```yaml
# In your GitHub Actions workflow:
- run: checkagent scan my_agent:run --sarif checkagent.sarif
- uses: github/codeql-action/upload-sarif@v3
  with:
    sarif_file: checkagent.sarif
```

The `--repeat` flag is useful for detecting non-deterministic safety failures. A probe that fails only 1 out of 5 runs is flagged with a lower stability score than one that fails consistently:

```bash
checkagent scan my_agent:run --repeat 5   # Stability score included in report
```

### Scan Quality Gates

Configure scan thresholds in `checkagent.yml` to enforce pass/fail policies in CI. When gates are configured, the scan exits with code 2 if any gate is blocked (instead of exit 1 for raw findings):

```yaml
# checkagent.yml
scan_gates:
  max_critical: 0    # Fail if any CRITICAL findings
  max_high: 3        # Fail if more than 3 HIGH findings
  min_score: 0.8     # Fail if safety score drops below 80%
  on_fail: block     # block | warn | ignore
```

Gate results appear in both the terminal output and `--json` output:

```bash
checkagent scan my_agent:run --json | jq '.quality_gates'
```

The `--comment-file` flag writes a Markdown summary for GitHub PR comments:

```yaml
# In your GitHub Actions workflow:
- run: checkagent scan my_agent:run --comment-file comment.md
- uses: marocchino/sticky-pull-request-comment@v2
  with:
    path: comment.md
```

## `checkagent diff`

Compare two scan JSON files to detect safety regressions. Shows new findings, fixed findings, and score changes.

```bash
checkagent diff baseline.json current.json
checkagent diff baseline.json current.json --fail-on-new
checkagent diff baseline.json current.json --json
checkagent diff baseline.json current.json --comment-file pr-diff.md
```

**Options:**

| Option | Description |
|--------|-------------|
| `--json` | Output diff as JSON |
| `--fail-on-new` | Exit with code 1 if new findings (regressions) are detected |
| `--comment-file FILE` | Write a GitHub PR comment summarizing the diff |

Use `--fail-on-new` in CI to block PRs that introduce new vulnerabilities:

```yaml
- run: checkagent scan my_agent:run --json > current.json
- run: checkagent diff baseline.json current.json --fail-on-new
```

## `checkagent history`

Show scan score trends for a target. Displays a table of past scan results so you can track safety posture over time.

```bash
checkagent history my_agent:agent_fn
checkagent history --url http://localhost:8000/chat
checkagent history my_agent:fn --limit 5
```

Score columns include a trend arrow (↑ improved, ↓ regressed) compared to the previous run.

**Options:**

| Option | Description |
|--------|-------------|
| `--url URL` | Show history for an HTTP endpoint target |
| `--limit N` | Maximum number of past scans to show (default: 10) |
| `--dir PATH` | Project root containing `.checkagent/` (default: current directory) |

Results are stored in `.checkagent/history/` after every `checkagent scan` run.

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

## `checkagent wrap`

Generate a wrapper module for an agent object, making it compatible with CheckAgent's scanning and testing tools.

```bash
checkagent wrap TARGET [OPTIONS]
```

`TARGET` is a `module:name` or `module.name` reference to a Python object. The command inspects the object and auto-selects the appropriate wrapper strategy:

| Detection order | Condition | Strategy |
|-----------------|-----------|----------|
| 1 | `agents.Agent` (OpenAI Agents SDK) | Wraps via `Runner.run()` |
| 2 | Object has `.run()` method | Async wrapper calling `.run()` |
| 3 | Object has `.invoke()` method | Async wrapper calling `.invoke()` |
| 4 | Object has `.kickoff()` method | CrewAI wrapper with inputs dict |
| 5 | Plain callable | No wrapper needed, scanned directly |

**Options:**

| Option | Description |
|--------|-------------|
| `--output TEXT` | Output filename for the generated wrapper (default: `checkagent_target.py`) |
| `--force` | Overwrite existing output file |

**Examples:**

```bash
checkagent wrap my_module:my_agent
checkagent wrap my_module:MyAgent --output agent_wrapper.py
checkagent wrap my_module:crew --force
```

After generating the wrapper, pass it as the scan target:

```bash
checkagent wrap my_module:my_agent --output agent_wrapper.py
checkagent scan agent_wrapper:agent
```

## `checkagent analyze-prompt`

Analyze a system prompt for security best practices. Zero-setup, LLM-free — no API key required.

```bash
checkagent analyze-prompt PROMPT_OR_FILE [OPTIONS]
```

`PROMPT_OR_FILE` can be a literal string, a file path, or stdin (default):

```bash
checkagent analyze-prompt "You are a helpful assistant."   # Literal string
checkagent analyze-prompt system_prompt.txt                # File path
cat prompt.txt | checkagent analyze-prompt                 # stdin
```

Checks the prompt text for eight security controls:

- **Injection guard** — defends against prompt injection attacks
- **Scope boundary** — constrains what the agent is allowed to do
- **Confidentiality** — instructs the agent not to reveal internal details
- **Refusal behavior** — specifies how the agent should decline disallowed requests
- **PII handling** — describes how personally identifiable information should be treated
- **Data scope** — limits what data sources or domains the agent may access
- **Role clarity** — clearly defines the agent's role and persona
- **Escalation path** — describes when and how to hand off to a human

Reports which controls are present and which are missing.

**Options:**

| Option | Description |
|--------|-------------|
| `--json` | Output results as JSON |

**Examples:**

```bash
checkagent analyze-prompt system_prompt.txt
checkagent analyze-prompt system_prompt.txt --json
```

Combine with `checkagent scan` using `--prompt-file` to run both static prompt analysis and dynamic probe scanning in a single step:

```bash
checkagent scan my_agent:run --prompt-file system_prompt.txt
```

## `checkagent ci-init`

Scaffold CI/CD configuration for agent safety scanning. Generates a ready-to-use workflow that runs your agent tests and a CheckAgent safety scan on every push and pull request.

```bash
checkagent ci-init [OPTIONS]
```

**Options:**

| Option | Description |
|--------|-------------|
| `--platform [github\|gitlab\|both]` | CI platform to generate config for (default: `github`) |
| `--scan-target TEXT` | Agent target for the scan step in `module:function` syntax (default: `sample_agent:sample_agent`) |
| `--force` | Overwrite existing CI config files |
| `--directory TEXT` | Project root directory (default: current directory) |

**Examples:**

```bash
checkagent ci-init
checkagent ci-init --platform gitlab
checkagent ci-init --platform both --scan-target my_agent:agent_fn
checkagent ci-init --scan-target my_module:my_agent --force
```

For GitHub, this creates `.github/workflows/checkagent.yml`. For GitLab, it creates `.gitlab-ci.yml`. Use `--platform both` to generate both files at once.

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
