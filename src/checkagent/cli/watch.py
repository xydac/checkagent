"""CLI command: checkagent watch

Live-updating safety analysis that re-runs analyze-prompt whenever a
system prompt file is saved.  Ideal for iterating on a system prompt and
seeing the score change in real time.

Usage:
    checkagent watch system_prompt.txt
    checkagent watch system_prompt.txt --llm gpt-4o-mini
    checkagent watch system_prompt.txt --interval 0.5
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path

import click
from rich.console import Console
from rich.live import Live
from rich.panel import Panel

from checkagent.safety.prompt_analyzer import PromptAnalyzer

_console = Console()


def _score_bar(score: float, width: int = 20) -> str:
    filled = round(score * width)
    return "█" * filled + "░" * (width - filled)


def _render_panel(
    path: Path,
    prompt_text: str,
    result,
    llm_verified: dict | None = None,
    llm_model: str | None = None,
    last_modified: float | None = None,
    elapsed: float | None = None,
) -> Panel:
    """Build a Rich Panel with the current analysis result."""
    llm_verified = llm_verified or {}
    llm_pass_ids = {cid for cid, (passed, _) in llm_verified.items() if passed}

    total = result.total_count
    passed_count = result.passed_count + len(llm_pass_ids)
    score = passed_count / total if total > 0 else 0.0
    pct = int(score * 100)
    bar = _score_bar(score)

    score_style = "green" if pct >= 75 else "yellow" if pct >= 50 else "red"

    lines: list[str] = []

    # Header
    preview = prompt_text[:80] + ("…" if len(prompt_text) > 80 else "")
    lines.append(f"[dim]File:[/dim] {path}")
    lines.append(f"[dim]Prompt:[/dim] {preview}")
    lines.append("")

    # Score
    lines.append(
        f"Score: [{score_style}]{passed_count}/{total} ({pct}%)[/{score_style}]  "
        f"[{score_style}]{bar}[/{score_style}]"
    )
    if llm_model and llm_pass_ids:
        lines.append(
            f"[dim]Pattern: {result.passed_count}/{total} "
            f"· LLM ({llm_model}): +{len(llm_pass_ids)} semantic[/dim]"
        )
    lines.append("")

    # Per-check rows
    for cr in result.check_results:
        llm_result = llm_verified.get(cr.check.id)
        llm_passed = llm_result[0] if llm_result else False
        llm_evidence = llm_result[1] if llm_result else ""

        if cr.passed:
            icon = "[green]✓[/green]"
            status = "[green]PRESENT[/green]"
            note = f'  [dim]Found: "{cr.evidence[:50]}"[/dim]' if cr.evidence else ""
        elif llm_passed:
            icon = "[yellow]~[/yellow]"
            status = "[yellow]SEMANTIC[/yellow]"
            note = f'  [dim]LLM: "{llm_evidence[:50]}"[/dim]' if llm_evidence else ""
        else:
            icon = "[red]✗[/red]"
            status = "[red]MISSING[/red] "
            note = ""

        sev_colors = {"high": "red", "medium": "yellow", "low": "cyan"}
        sev_color = sev_colors.get(cr.check.severity, "white")
        sev = f"[{sev_color}]{cr.check.severity.upper()}[/{sev_color}]"
        name = cr.check.name.ljust(24)
        lines.append(f"  {icon} {name} {status}  {sev}{note}")

    # Missing recommendations
    still_missing = [
        cr for cr in result.check_results
        if not cr.passed and cr.check.id not in llm_pass_ids
    ]
    recs = [cr.check.recommendation for cr in still_missing if cr.check.recommendation]
    if recs:
        lines.append("")
        lines.append("[bold yellow]To fix:[/bold yellow]")
        for i, rec in enumerate(recs[:3], 1):
            lines.append(f"  [dim]{i}.[/dim] {rec}")
        if len(recs) > 3:
            lines.append(f"  [dim]… and {len(recs) - 3} more[/dim]")

    # Footer
    lines.append("")
    ts = time.strftime("%H:%M:%S")
    extra = f" via {llm_model}" if llm_model else ""
    elapsed_str = f" in {elapsed:.1f}s" if elapsed else ""
    lines.append(f"[dim]Last checked{elapsed_str}{extra} at {ts}. Ctrl+C to stop.[/dim]")

    body = "\n".join(lines)
    title = "[bold]checkagent watch[/bold]"
    return Panel(body, title=title, border_style=score_style)


@click.command("watch")
@click.argument("prompt_file", type=click.Path(exists=True, dir_okay=False))
@click.option(
    "--llm",
    "llm_model",
    default=None,
    metavar="MODEL",
    help="Use an LLM for semantic verification (e.g. gpt-4o-mini).",
)
@click.option(
    "--interval",
    default=1.0,
    show_default=True,
    metavar="SECONDS",
    help="How often to poll for file changes.",
)
def watch_cmd(prompt_file: str, llm_model: str | None, interval: float) -> None:
    """Watch a system prompt file and re-analyze on every save.

    Displays a live score that updates instantly whenever you save the file.
    Perfect for iterating on a system prompt until all security checks pass.

    \b
    Examples:
      checkagent watch system_prompt.txt
      checkagent watch prompt.txt --llm gpt-4o-mini
      checkagent watch prompt.txt --interval 0.5
    """
    if llm_model:
        from checkagent.core.llm_call import detect_provider
        detect_provider(llm_model, param_hint="--llm")

    path = Path(prompt_file)
    analyzer = PromptAnalyzer()

    last_mtime: float = 0.0
    last_content: str = ""

    async def _llm_verify(prompt_text: str, result):
        from checkagent.cli.analyze_prompt import _llm_verify_failing_checks
        from checkagent.core.llm_call import check_api_key

        missing_key = check_api_key(llm_model)
        if missing_key:
            return {}
        failing = [cr.check for cr in result.check_results if not cr.passed]
        if not failing:
            return {}
        return await _llm_verify_failing_checks(prompt_text, failing, llm_model)

    def _analyze(prompt_text: str):
        result = analyzer.analyze(prompt_text)
        llm_verified: dict = {}
        if llm_model:
            t0 = time.time()
            llm_verified = asyncio.run(_llm_verify(prompt_text, result))
            elapsed = time.time() - t0
        else:
            elapsed = None
        return result, llm_verified, elapsed

    _console.print(
        f"\n[bold]Watching[/bold] [cyan]{path}[/cyan] for changes… "
        "(save the file to see your score update)\n"
    )

    # Initial render with placeholder
    placeholder = Panel(
        f"[dim]Reading {path}…[/dim]",
        title="[bold]checkagent watch[/bold]",
        border_style="dim",
    )

    with Live(placeholder, console=_console, refresh_per_second=4, screen=False) as live:
        while True:
            try:
                mtime = path.stat().st_mtime
            except OSError:
                live.update(Panel(
                    f"[red]File not found: {path}[/red]",
                    title="[bold]checkagent watch[/bold]",
                    border_style="red",
                ))
                time.sleep(interval)
                continue

            if mtime != last_mtime:
                last_mtime = mtime
                try:
                    content = path.read_text(encoding="utf-8").strip()
                except OSError:
                    time.sleep(interval)
                    continue

                if content == last_content and last_content:
                    time.sleep(interval)
                    continue

                last_content = content
                if not content:
                    live.update(Panel(
                        "[dim]File is empty.[/dim]",
                        title="[bold]checkagent watch[/bold]",
                        border_style="dim",
                    ))
                    time.sleep(interval)
                    continue

                # Transient "analyzing…" state
                live.update(Panel(
                    "[dim]Analyzing…[/dim]",
                    title="[bold]checkagent watch[/bold]",
                    border_style="dim",
                ))

                try:
                    result, llm_verified, elapsed = _analyze(content)
                except Exception as exc:
                    live.update(Panel(
                        f"[red]Error: {exc}[/red]",
                        title="[bold]checkagent watch[/bold]",
                        border_style="red",
                    ))
                    time.sleep(interval)
                    continue

                panel = _render_panel(
                    path,
                    content,
                    result,
                    llm_verified=llm_verified,
                    llm_model=llm_model,
                    last_modified=mtime,
                    elapsed=elapsed,
                )
                live.update(panel)

            time.sleep(interval)
