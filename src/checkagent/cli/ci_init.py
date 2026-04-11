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
  agent-safety:
    name: Agent Safety Scan
    runs-on: ubuntu-latest

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
          # Install your agent's dependencies, e.g.:
          # pip install -r requirements.txt

      - name: Run agent tests
        run: pytest tests/ -q --tb=short

      - name: Run safety scan
        run: |
          checkagent scan {scan_target}
          # For HTTP endpoints: checkagent scan --url http://localhost:8000/chat
          # For LLM judge:      checkagent scan {scan_target} --llm-judge gpt-4o-mini
        env:
          OPENAI_API_KEY: ${{{{ secrets.OPENAI_API_KEY }}}}
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
    - checkagent scan {scan_target}
    # For HTTP endpoints: checkagent scan --url http://localhost:8000/chat
    # For LLM judge:      checkagent scan {scan_target} --llm-judge gpt-4o-mini
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
        console.print("  2. Add OPENAI_API_KEY to your GitHub repository secrets")
        console.print("     (only needed if using --llm-judge in the scan step)")
        console.print("  3. Open a pull request to trigger the workflow")
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
