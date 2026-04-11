"""CLI command: checkagent analyze-prompt

Zero-setup, LLM-free static analysis of a system prompt.  Checks the
text for eight security best practices that map directly to the probe
categories used by ``checkagent scan``.
"""

from __future__ import annotations

import sys

import click
from rich import box
from rich.console import Console
from rich.table import Table

from checkagent.safety.prompt_analyzer import PromptAnalysisResult, PromptAnalyzer

_console = Console()


def _score_bar(score: float, width: int = 20) -> str:
    """Return a unicode progress bar string for *score* (0.0–1.0)."""
    filled = round(score * width)
    empty = width - filled
    return "█" * filled + "░" * empty


def _severity_color(severity: str) -> str:
    return {"high": "red", "medium": "yellow", "low": "cyan"}.get(severity, "white")


def _render_result(result: PromptAnalysisResult, prompt_preview: str) -> None:
    pct = int(result.score * 100)
    bar = _score_bar(result.score)

    # Headline
    _console.print()
    _console.print("[bold]System Prompt Analysis[/bold]", style="white")
    _console.print("─" * 46)
    if prompt_preview:
        preview = prompt_preview[:72] + ("…" if len(prompt_preview) > 72 else "")
        _console.print(f"[dim]Prompt:[/dim] {preview}")
        _console.print()

    # Score bar
    if pct >= 75:
        score_style = "green"
    elif pct >= 50:
        score_style = "yellow"
    else:
        score_style = "red"

    counts = f"{result.passed_count}/{result.total_count} ({pct}%)"
    _console.print(
        f"Score: [{score_style}]{counts}[/{score_style}]  "
        f"[{score_style}]{bar}[/{score_style}]"
    )
    _console.print()

    # Per-check table
    table = Table(box=box.SIMPLE, show_header=True, header_style="bold dim")
    table.add_column("Check", style="white", min_width=22)
    table.add_column("Status", min_width=8)
    table.add_column("Severity", min_width=8)
    table.add_column("Note", style="dim")

    for cr in result.check_results:
        if cr.passed:
            status = "[green]✓ PRESENT[/green]"
            if cr.evidence and len(cr.evidence) > 48:
                note = f'Found: "{cr.evidence[:48]}…"'
            elif cr.evidence:
                note = f'Found: "{cr.evidence}"'
            else:
                note = ""
        else:
            status = "[red]✗ MISSING[/red]"
            note = ""
        sev_color = _severity_color(cr.check.severity)
        severity = f"[{sev_color}]{cr.check.severity.upper()}[/{sev_color}]"
        table.add_row(cr.check.name, status, severity, note)

    _console.print(table)

    # Recommendations
    recs = result.recommendations
    if recs:
        _console.print("[bold]Recommendations[/bold]")
        for i, rec in enumerate(recs, 1):
            _console.print(f"  [dim]{i}.[/dim] {rec}")
        _console.print()
    else:
        _console.print(
            "[green]All checks passed — your system prompt covers all eight "
            "security controls.[/green]"
        )
        _console.print()

    # Footer
    _console.print(
        "[dim]This is a static guidelines check, not a security guarantee. "
        "Run [bold]checkagent scan[/bold] for dynamic probe testing.[/dim]"
    )
    _console.print()


@click.command("analyze-prompt")
@click.argument("prompt_source", metavar="PROMPT_OR_FILE", default="-")
@click.option("--json", "output_json", is_flag=True, default=False, help="Output results as JSON.")
def analyze_prompt_cmd(prompt_source: str, output_json: bool) -> None:
    """Analyze a system prompt for security best practices.

    PROMPT_OR_FILE can be:

    \b
      - A literal string: checkagent analyze-prompt "You are a helpful assistant."
      - A file path:      checkagent analyze-prompt system_prompt.txt
      - stdin (default):  cat prompt.txt | checkagent analyze-prompt

    Checks the prompt text for eight security controls (injection guard,
    scope boundary, confidentiality, refusal behavior, PII handling,
    data scope, role clarity, escalation path) and reports which are
    present or missing.

    This is a zero-setup, LLM-free check — no API key required.
    """
    # Resolve prompt text
    prompt_text: str
    if prompt_source == "-":
        if sys.stdin.isatty():
            raise click.UsageError(
                "No prompt provided. Pass a string, a file path, or pipe via stdin.\n"
                "  checkagent analyze-prompt \"You are a helpful assistant.\"\n"
                "  checkagent analyze-prompt system_prompt.txt\n"
                "  cat prompt.txt | checkagent analyze-prompt"
            )
        prompt_text = sys.stdin.read()
    else:
        # Try as file path first, fall back to treating as literal string
        import pathlib
        p = pathlib.Path(prompt_source)
        try:
            is_file = p.exists() and p.is_file()
        except OSError:
            # Path too long or invalid as a filesystem path — treat as literal
            is_file = False
        prompt_text = p.read_text(encoding="utf-8") if is_file else prompt_source

    prompt_text = prompt_text.strip()
    if not prompt_text:
        raise click.UsageError("Prompt is empty.")

    analyzer = PromptAnalyzer()
    result = analyzer.analyze(prompt_text)

    if output_json:
        import json

        data = {
            "score": round(result.score, 4),
            "passed_count": result.passed_count,
            "total_count": result.total_count,
            "checks": [
                {
                    "id": cr.check.id,
                    "name": cr.check.name,
                    "passed": cr.passed,
                    "severity": cr.check.severity,
                    "evidence": cr.evidence,
                    "recommendation": cr.check.recommendation if not cr.passed else None,
                }
                for cr in result.check_results
            ],
        }
        click.echo(json.dumps(data, indent=2))
    else:
        _render_result(result, prompt_text)

    # Exit with non-zero if any HIGH checks are missing (useful for CI)
    if result.missing_high:
        sys.exit(1)
