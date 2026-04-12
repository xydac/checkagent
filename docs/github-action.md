# GitHub Action

## Overview

The `xydac/checkagent` composite action runs a CheckAgent safety scan against your agent and uploads the results to GitHub Code Scanning as a SARIF file. Once uploaded, probe failures appear as code scanning alerts in the **Security** tab and as inline annotations on pull requests — giving reviewers a clear view of new safety regressions without leaving the PR review interface.

---

## Basic Usage

The minimal configuration requires only a `target` — the Python importable pointing to your agent's entry function.

```yaml
name: CheckAgent Safety Scan

on:
  push:
    branches: [main]
  pull_request:

jobs:
  safety-scan:
    runs-on: ubuntu-latest
    permissions:
      contents: read
      security-events: write

    steps:
      - uses: actions/checkout@v4

      - name: Run CheckAgent scan
        uses: xydac/checkagent@v0.2
        with:
          target: my_agent:run
```

The action installs Python 3.11, installs `checkagent`, installs your repo's dependencies from `requirements.txt`, runs `checkagent scan`, and uploads the SARIF results to Code Scanning — even if the scan finds failures.

---

## With LLM Judge

Adding `llm-judge: true` passes each borderline finding through an LLM evaluator to reduce false positives. This makes the scan more accurate but adds approximately $0.01–$0.10 per run depending on your agent's complexity and the number of borderline findings.

```yaml
jobs:
  safety-scan:
    runs-on: ubuntu-latest
    permissions:
      contents: read
      security-events: write

    steps:
      - uses: actions/checkout@v4

      - name: Run CheckAgent scan
        uses: xydac/checkagent@v0.2
        with:
          target: my_agent:run
          llm-judge: "true"
        env:
          OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
```

You can use `ANTHROPIC_API_KEY` instead of `OPENAI_API_KEY` — the action passes whichever key is present through to the scan process.

---

## Full Workflow Example

The workflow below shows a complete production setup: fast scan on every PR (no LLM judge, free), proper permissions, SARIF upload with fork protection, and a badge generation step.

```yaml
name: CheckAgent Safety Scan

on:
  push:
    branches: [main]
  pull_request:

jobs:
  safety-scan:
    runs-on: ubuntu-latest
    permissions:
      contents: read
      security-events: write

    steps:
      - uses: actions/checkout@v4

      # Run the scan — LLM judge disabled for speed and cost
      - name: Run CheckAgent scan
        id: scan
        uses: xydac/checkagent@v0.2
        with:
          target: my_agent:run
          llm-judge: "false"
          sarif-file: checkagent-results.sarif
          requirements: requirements.txt

      # Upload SARIF — always runs, even if scan found failures
      # Fork PR protection: only upload when we have write access
      - name: Upload SARIF to Code Scanning
        uses: github/codeql-action/upload-sarif@v3
        if: >
          always() &&
          (github.event_name != 'pull_request' ||
           github.event.pull_request.head.repo.full_name == github.repository)
        with:
          sarif_file: checkagent-results.sarif
          category: checkagent-scan

      # Generate a status badge and commit it to the repo
      - name: Generate badge
        if: github.ref == 'refs/heads/main'
        run: checkagent report --badge checkagent-results.sarif

      # Fail the job explicitly if critical probes fired
      # (the scan exits non-zero on critical findings by default,
      #  but this step makes the failure reason visible in the log)
      - name: Assert no critical findings
        run: |
          CRITICAL=$(python - <<'EOF'
          import json, sys
          with open("checkagent-results.sarif") as f:
              sarif = json.load(f)
          runs = sarif.get("runs", [])
          results = runs[0].get("results", []) if runs else []
          criticals = [r for r in results if r.get("level") == "error"]
          print(len(criticals))
          EOF
          )
          if [ "$CRITICAL" -gt "0" ]; then
            echo "::error::CheckAgent found $CRITICAL critical safety finding(s). See the Security tab for details."
            exit 1
          fi
```

---

## Inputs Reference

| Input | Required | Default | Description |
|-------|----------|---------|-------------|
| `target` | Yes | — | Agent target: `module:function` (e.g. `my_agent:run`) or `--url http://localhost:8080` for HTTP agents |
| `llm-judge` | No | `"false"` | Set to `"true"` to use LLM-as-judge for borderline findings. Requires `OPENAI_API_KEY` or `ANTHROPIC_API_KEY` in the environment. |
| `sarif-file` | No | `"results.sarif"` | Path where the SARIF output file is written. Pass the same path to `upload-sarif` if you upload separately. |
| `requirements` | No | `"requirements.txt"` | Path to the requirements file for the agent under test. Falls back to `pyproject.toml` if the file is not found. |

---

## SARIF and Code Scanning

Once SARIF is uploaded, findings surface in two places:

**Security tab** — Navigate to **Security** > **Code scanning** in your repository. Each probe failure becomes an alert with:
- The probe name and category (e.g. `direct_injection / LLM01 Prompt Injection`)
- Severity (`note`, `warning`, or `error` mapped from CheckAgent's `low`/`medium`/`high`/`critical`)
- A description of what the probe tested and why the finding was raised

**Pull request annotations** — For any probe failure introduced by a PR (i.e. not present on the base branch), GitHub adds an inline annotation to the PR diff. Reviewers see the finding without navigating away from the PR.

Findings are dismissed or resolved the same way as any other code scanning alert — through the Security UI or by fixing the underlying behaviour so the probe passes.

---

## Permissions

The `security-events: write` permission is required for the SARIF upload step. Without it, `github/codeql-action/upload-sarif` will fail with a 403 error.

**Fork pull requests** do not have access to `security-events: write` — GitHub restricts this permission for workflows triggered by forks to prevent malicious PRs from polluting the repository's security findings. Add a conditional to the upload step to skip it for fork PRs:

```yaml
- name: Upload SARIF to Code Scanning
  uses: github/codeql-action/upload-sarif@v3
  if: >
    always() &&
    (github.event_name != 'pull_request' ||
     github.event.pull_request.head.repo.full_name == github.repository)
  with:
    sarif_file: results.sarif
    category: checkagent-scan
```

The scan itself still runs for fork PRs — only the SARIF upload is skipped. Probe results are still visible in the workflow logs.
