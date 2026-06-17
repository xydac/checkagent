"""checkagent ci-init — scaffold CI/CD configuration for agent safety scanning."""

from __future__ import annotations

from pathlib import Path

import click
from rich.console import Console

console = Console()

# ---------------------------------------------------------------------------
# GitHub Actions template
# ---------------------------------------------------------------------------

_GITHUB_WORKFLOW = """\
name: CheckAgent Safety Scan

on:
  push:
    branches: [main, master]
  pull_request:
    branches: [main, master]

jobs:
  # ─── Job 1: Run scan on every push and PR ───────────────────────────────────
  scan:
    name: Agent Safety Scan
    runs-on: ubuntu-latest
    permissions:
      contents: read

    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.11"
          cache: "pip"

      - name: Install dependencies
        run: |
          pip install checkagent
          # pip install -r requirements.txt

      - name: Run agent tests
        run: pytest tests/ -q --tb=short

      - name: Run safety scan
        run: |
          # --repeat 3: run each probe 3 times for stability scoring on LLM-backed agents.
          # --exit-zero: always exit 0 so the artifact uploads even when findings exist.
          # Quality gates in the diff step (--fail-on-new, --min-score) control CI pass/fail.
          checkagent scan {scan_target} --repeat 3 --json --exit-zero > scan.json
          # For HTTP endpoints: checkagent scan --url http://localhost:8000/chat --repeat 3 --exit-zero
          # For LLM eval:       checkagent scan {scan_target} --repeat 3 --llm-judge gpt-4o-mini --exit-zero
        env:
          OPENAI_API_KEY: ${{{{ secrets.OPENAI_API_KEY }}}}

      - name: Upload scan result
        uses: actions/upload-artifact@v4
        with:
          name: scan-result-${{{{ github.sha }}}}
          path: scan.json
          retention-days: 30

  # ─── Job 2: Diff against main-branch baseline and post PR comment ───────────
  pr-diff:
    name: PR Safety Diff
    runs-on: ubuntu-latest
    needs: scan
    if: github.event_name == 'pull_request'
    permissions:
      pull-requests: write
      actions: read

    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.11"
          cache: "pip"

      - name: Install checkagent
        run: pip install checkagent

      - name: Download current scan
        uses: actions/download-artifact@v4
        with:
          name: scan-result-${{{{ github.sha }}}}

      - name: Download baseline from main branch
        id: baseline
        run: |
          # Find the latest successful main-branch run that produced a scan artifact.
          RUN_ID=$(gh api "repos/${{{{ github.repository }}}}/actions/runs?branch=main&status=success&per_page=20" \\
            --jq '.workflow_runs | map(select(.name == "CheckAgent Safety Scan")) | first | .id // empty')
          if [ -z "$RUN_ID" ]; then
            echo "No completed main-branch scan found — skipping diff on first PR."
            echo "found=false" >> "$GITHUB_OUTPUT"
            exit 0
          fi
          ARTIFACT_ID=$(gh api "repos/${{{{ github.repository }}}}/actions/runs/$RUN_ID/artifacts" \\
            --jq ".artifacts[] | select(.name | startswith(\\"scan-result-\\")) | .id // empty" | head -1)
          if [ -z "$ARTIFACT_ID" ]; then
            echo "Baseline artifact not found — skipping diff."
            echo "found=false" >> "$GITHUB_OUTPUT"
            exit 0
          fi
          gh api "repos/${{{{ github.repository }}}}/actions/artifacts/$ARTIFACT_ID/zip" > baseline.zip
          unzip -p baseline.zip > baseline.json && echo "found=true" >> "$GITHUB_OUTPUT"
        env:
          GH_TOKEN: ${{{{ secrets.GITHUB_TOKEN }}}}

      - name: Diff and enforce quality gates
        if: steps.baseline.outputs.found == 'true'
        run: |
          checkagent diff baseline.json scan.json \\
            --comment-file pr-comment.md \\
            --fail-on-new \\
            --min-score 0.8
          # Remove --fail-on-new to comment without blocking CI.
          # Remove --min-score to skip score-threshold enforcement.
          # Add --min-stability 0.9 if both scans used --repeat N.

      - name: Post PR comment
        if: steps.baseline.outputs.found == 'true' && always()
        uses: actions/github-script@v7
        with:
          script: |
            const fs = require('fs')
            if (!fs.existsSync('pr-comment.md')) process.exit(0)
            const body = fs.readFileSync('pr-comment.md', 'utf8')
            const {{ data: comments }} = await github.rest.issues.listComments({{
              issue_number: context.issue.number,
              owner: context.repo.owner,
              repo: context.repo.repo,
            }})
            const existing = comments.find(c => c.user.type === 'Bot' && c.body.includes('CheckAgent'))
            if (existing) {{
              await github.rest.issues.updateComment({{
                comment_id: existing.id,
                owner: context.repo.owner,
                repo: context.repo.repo,
                body,
              }})
            }} else {{
              await github.rest.issues.createComment({{
                issue_number: context.issue.number,
                owner: context.repo.owner,
                repo: context.repo.repo,
                body,
              }})
            }}
"""

# ---------------------------------------------------------------------------
# GitLab CI template
# ---------------------------------------------------------------------------

_GITLAB_CI = """\
# CheckAgent Safety Scan — GitLab CI
# Add OPENAI_API_KEY to your project's CI/CD variables if using --llm-judge

stages:
  - test
  - safety

agent-tests:
  stage: test
  image: python:3.11-slim
  script:
    - pip install checkagent
    # Install your agent's dependencies, e.g.:
    # - pip install -r requirements.txt
    - pytest tests/ -q --tb=short

safety-scan:
  stage: safety
  image: python:3.11-slim
  script:
    - pip install checkagent
    # Install your agent's dependencies, e.g.:
    # - pip install -r requirements.txt
    # --repeat 3: run each probe 3 times to catch flaky LLM-backed agents
    # --exit-zero: always exit 0; quality gates below control CI pass/fail
    # --diff: show new/fixed findings vs. previous scan inline
    - checkagent scan {scan_target} --repeat 3 --json --exit-zero --diff > scan.json
    # For HTTP endpoints: checkagent scan --url http://localhost:8000/chat --repeat 3 --exit-zero
    # For LLM judge:      checkagent scan {scan_target} --repeat 3 --llm-judge gpt-4o-mini --exit-zero
    # Quality gates (uncomment to enforce):
    # - checkagent diff baseline.json scan.json --min-score 0.8 --min-stability 0.9 --fail-on-new
  variables:
    OPENAI_API_KEY: $OPENAI_API_KEY
"""


def _write_file(path: Path, content: str, force: bool, root: Path) -> bool:
    """Write a file and return True if created/updated, False if skipped."""
    if path.exists() and not force:
        console.print(
            f"  [dim]skip[/dim] {path.relative_to(root)} (exists, use --force to overwrite)"
        )
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return True


@click.command("ci-init")
@click.option(
    "--platform",
    type=click.Choice(["github", "gitlab", "both"], case_sensitive=False),
    default="github",
    show_default=True,
    help="CI platform to generate config for.",
)
@click.option(
    "--scan-target",
    default="sample_agent:sample_agent",
    show_default=True,
    help="Agent target for the scan step (module:function syntax).",
)
@click.option(
    "--force",
    is_flag=True,
    help="Overwrite existing CI config files.",
)
@click.option(
    "--directory",
    default=".",
    help="Project root directory (default: current directory).",
)
def ci_init_cmd(
    platform: str,
    scan_target: str,
    force: bool,
    directory: str,
) -> None:
    """Scaffold CI/CD configuration for agent safety scanning.

    Generates a ready-to-use workflow that runs your agent tests and a
    CheckAgent safety scan on every push and pull request.

    \\b
    Examples:
      checkagent ci-init
      checkagent ci-init --platform gitlab
      checkagent ci-init --platform both --scan-target my_agent:agent_fn
      checkagent ci-init --scan-target my_module:my_agent --force
    """
    root = Path(directory).resolve()

    console.print(f"\n[bold]Initializing CheckAgent CI config in[/bold] {root}\n")

    created: list[str] = []
    skipped: list[str] = []

    platform_lower = platform.lower()

    if platform_lower in ("github", "both"):
        gh_path = root / ".github" / "workflows" / "checkagent.yml"
        content = _GITHUB_WORKFLOW.format(scan_target=scan_target)
        if _write_file(gh_path, content, force, root):
            rel = gh_path.relative_to(root).as_posix()
            created.append(rel)
            console.print(f"  [green]✓[/green] {rel}")
        else:
            skipped.append(gh_path.relative_to(root).as_posix())

    if platform_lower in ("gitlab", "both"):
        gl_path = root / ".gitlab-ci.yml"
        content = _GITLAB_CI.format(scan_target=scan_target)
        if _write_file(gl_path, content, force, root):
            rel = gl_path.relative_to(root).as_posix()
            created.append(rel)
            console.print(f"  [green]✓[/green] {rel}")
        else:
            skipped.append(gl_path.relative_to(root).as_posix())

    if not created and skipped:
        console.print(
            "\n[yellow]No files created.[/yellow] Use --force to overwrite existing files."
        )
        return

    console.print("\n[bold]Next steps:[/bold]")
    if platform_lower in ("github", "both"):
        console.print("  1. Commit and push the workflow file to your repository")
        console.print("  2. Push to main to run the first scan (creates the baseline)")
        console.print("  3. Open a pull request — CheckAgent will diff against the baseline")
        console.print("     and post a safety summary comment on the PR automatically")
        console.print("  4. [optional] Add OPENAI_API_KEY to repository secrets for LLM judge")
    if platform_lower in ("gitlab", "both"):
        console.print("  1. Commit .gitlab-ci.yml to your repository")
        console.print("  2. Add OPENAI_API_KEY to CI/CD → Variables in GitLab settings")
        console.print("     (only needed if using --llm-judge in the scan step)")

    console.print()
    console.print(
        "[dim]Tip: edit the scan step to match your agent's module path, "
        "or use --url for HTTP endpoints.[/dim]"
    )
    console.print()
