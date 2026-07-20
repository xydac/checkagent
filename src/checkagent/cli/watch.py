"""CLI command: checkagent watch

Two modes depending on the argument:

  File mode  — watch a system prompt text file, re-run analyze-prompt on save.
  Agent mode — watch a Python module containing an agent (module:fn), re-run
               `checkagent scan` on every file change.

Usage:
    checkagent watch system_prompt.txt
    checkagent watch system_prompt.txt --llm gpt-4o-mini
    checkagent watch system_prompt.txt --interval 0.5
    checkagent watch my_module:my_agent
    checkagent watch my_module:my_agent --interval 1.0
"""

from __future__ import annotations

import asyncio
import importlib
import json
import subprocess
import sys
import time
from pathlib import Path

import click
from rich.console import Console
from rich.live import Live
from rich.panel import Panel

from checkagent.safety.prompt_analyzer import PromptAnalyzer

_console = Console()


# ---------------------------------------------------------------------------
# Prompt-file watch helpers
# ---------------------------------------------------------------------------

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
    """Build a Rich Panel with the current prompt analysis result."""
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


def _watch_prompt_file(path: Path, llm_model: str | None, interval: float) -> None:
    """Original watch mode: analyze a system prompt file on every save."""
    if llm_model:
        from checkagent.core.llm_call import detect_provider
        detect_provider(llm_model, param_hint="--llm")

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


# ---------------------------------------------------------------------------
# Agent scan watch helpers
# ---------------------------------------------------------------------------

def _resolve_module_file(target: str) -> Path | None:
    """Given 'module:fn', return the source file path of the module, or None."""
    if ":" not in target:
        return None
    module_part = target.split(":")[0].replace(".", "/")
    # Try to import and get __file__
    try:
        spec = importlib.util.find_spec(target.split(":")[0])  # type: ignore[attr-defined]
        if spec and spec.origin:
            p = Path(spec.origin)
            if p.suffix == ".py":
                return p
    except (ModuleNotFoundError, ValueError):
        pass
    # Fallback: look for module_part.py relative to cwd
    for candidate in [
        Path.cwd() / f"{module_part}.py",
        Path.cwd() / "src" / f"{module_part}.py",
    ]:
        if candidate.exists():
            return candidate
    return None


def _category_counts(scan_data: dict) -> dict[str, int]:
    """Count findings per category from scan JSON output."""
    counts: dict[str, int] = {}
    for f in scan_data.get("findings", []):
        cat = f.get("category", "unknown")
        counts[cat] = counts.get(cat, 0) + 1
    return counts


def _render_category_delta(
    prev_counts: dict[str, int], curr_counts: dict[str, int]
) -> list[str]:
    """Render per-category finding change between two scans. Returns Rich-markup lines."""
    all_cats = sorted(set(list(prev_counts) + list(curr_counts)))
    rows: list[str] = []
    for cat in all_cats:
        prev_n = prev_counts.get(cat, 0)
        curr_n = curr_counts.get(cat, 0)
        d = curr_n - prev_n
        if d < 0:
            indicator = f"[green]↓{abs(d)} fixed[/green]"
        elif d > 0:
            indicator = f"[red]↑{d} new[/red]"
        else:
            indicator = "[dim]= unchanged[/dim]"
        rows.append(f"  [dim]{cat:<22}[/dim] {prev_n} → {curr_n}  {indicator}")
    return rows


def _render_scan_panel(
    target: str,
    source_file: Path | None,
    scan_data: dict | None,
    elapsed: float | None,
    error: str | None,
    prev_counts: dict[str, int] | None = None,
) -> Panel:
    """Build a Rich Panel showing agent scan results."""
    lines: list[str] = []
    lines.append(f"[dim]Target:[/dim] {target}")
    if source_file:
        lines.append(f"[dim]File:[/dim]   {source_file}")
    lines.append("")

    if error:
        lines.append(f"[red]Error: {error}[/red]")
        ts = time.strftime("%H:%M:%S")
        lines.append("")
        lines.append(f"[dim]Last checked at {ts}. Ctrl+C to stop.[/dim]")
        return Panel("\n".join(lines), title="[bold]checkagent watch[/bold]", border_style="red")

    if scan_data is None:
        lines.append("[dim]Waiting for first scan…[/dim]")
        return Panel("\n".join(lines), title="[bold]checkagent watch[/bold]", border_style="dim")

    summary = scan_data.get("summary", {})
    score = summary.get("score", 0.0)
    passed = summary.get("passed", 0)
    total = summary.get("total", 0)

    pct = int(score * 100)
    bar = _score_bar(score)
    score_style = "green" if pct >= 75 else "yellow" if pct >= 50 else "red"

    lines.append(
        f"Score: [{score_style}]{passed}/{total} ({pct}%)[/{score_style}]  "
        f"[{score_style}]{bar}[/{score_style}]"
    )
    lines.append("")

    # Finding summary by severity
    findings = scan_data.get("findings", [])
    if findings:
        critical = [f for f in findings if f.get("severity") == "critical"]
        high = [f for f in findings if f.get("severity") == "high"]
        medium = [f for f in findings if f.get("severity") == "medium"]

        if critical:
            lines.append(f"  [red]● {len(critical)} critical[/red]")
        if high:
            lines.append(f"  [red]○ {len(high)} high[/red]")
        if medium:
            lines.append(f"  [yellow]○ {len(medium)} medium[/yellow]")

        lines.append("")
        lines.append("[bold yellow]Top findings:[/bold yellow]")
        for f in findings[:4]:
            probe = f.get("probe_id", f.get("probe", "?"))
            cat = f.get("category", "")
            sev = f.get("severity", "")
            sev_colors = {"critical": "red", "high": "red", "medium": "yellow", "low": "cyan"}
            sc = sev_colors.get(sev, "white")
            lines.append(f"  [dim]·[/dim] [{sc}]{probe}[/{sc}] [dim]({cat})[/dim]")
        if len(findings) > 4:
            lines.append(f"  [dim]… and {len(findings) - 4} more[/dim]")
    else:
        lines.append("  [green]No findings — all probes passed.[/green]")

    # Per-category delta vs previous scan (only shown on rescan)
    if prev_counts is not None:
        curr_counts = _category_counts(scan_data)
        if curr_counts != prev_counts or prev_counts:
            delta_rows = _render_category_delta(prev_counts, curr_counts)
            if delta_rows:
                lines.append("")
                lines.append("[bold]Change from last scan:[/bold]")
                lines.extend(delta_rows)

    lines.append("")
    ts = time.strftime("%H:%M:%S")
    elapsed_str = f" in {elapsed:.1f}s" if elapsed else ""
    lines.append(
        f"[dim]Last scanned{elapsed_str} at {ts}. "
        "Edit the agent file to trigger a rescan. Ctrl+C to stop.[/dim]"
    )

    body = "\n".join(lines)
    return Panel(body, title="[bold]checkagent watch[/bold]", border_style=score_style)


def _run_scan(target: str) -> tuple[dict | None, str | None]:
    """Run `checkagent scan target --json` in a subprocess, return (data, error)."""
    try:
        result = subprocess.run(
            [sys.executable, "-m", "checkagent", "scan", target, "--json", "--exit-zero"],
            capture_output=True,
            text=True,
            timeout=120,
        )
        stdout = result.stdout.strip()
        if not stdout:
            stderr = result.stderr.strip()
            if stderr:
                return None, f"scan produced no output. stderr: {stderr[:200]}"
            return None, "scan produced no output"
        data = json.loads(stdout)
        return data, None
    except subprocess.TimeoutExpired:
        return None, "scan timed out (120s)"
    except json.JSONDecodeError as exc:
        return None, f"JSON parse error: {exc}"
    except Exception as exc:
        return None, str(exc)


def _watch_agent(target: str, interval: float) -> None:
    """Agent mode: re-run checkagent scan whenever the agent's source file changes."""
    source_file = _resolve_module_file(target)

    if source_file:
        _console.print(
            f"\n[bold]Watching[/bold] [cyan]{source_file}[/cyan] "
            f"for changes to [cyan]{target}[/cyan]…\n"
        )
    else:
        _console.print(
            f"\n[bold]Watching[/bold] agent [cyan]{target}[/cyan] "
            f"(polling every {interval}s)…\n"
        )

    placeholder = Panel(
        "[dim]Running initial scan…[/dim]",
        title="[bold]checkagent watch[/bold]",
        border_style="dim",
    )

    last_mtime: float = 0.0
    last_data: dict | None = None

    with Live(placeholder, console=_console, refresh_per_second=4, screen=False) as live:
        # Force initial scan
        force_scan = True

        while True:
            changed = force_scan
            force_scan = False

            if source_file:
                try:
                    mtime = source_file.stat().st_mtime
                    if mtime != last_mtime:
                        last_mtime = mtime
                        changed = True
                except OSError:
                    pass

            if changed:
                live.update(Panel(
                    "[dim]Scanning…[/dim]",
                    title="[bold]checkagent watch[/bold]",
                    border_style="dim",
                ))
                # Snapshot category counts from the previous scan before overwriting
                counts_before = _category_counts(last_data) if last_data is not None else None
                t0 = time.time()
                data, error = _run_scan(target)
                elapsed = time.time() - t0
                last_data = data
                # Pass prev_counts only on rescan (not on first scan)
                panel = _render_scan_panel(
                    target, source_file, data, elapsed, error, prev_counts=counts_before
                )
                live.update(panel)
            elif last_data is not None:
                # Re-render to keep footer timestamp fresh (optional — skip for performance)
                pass

            time.sleep(interval)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _is_module_target(arg: str) -> bool:
    """Return True if arg looks like module:fn (not a file path)."""
    if not arg:
        return False
    # Has colon but not a Windows drive letter (C:\...)
    if ":" not in arg:
        return False
    # Windows drive letter: single letter followed by :\ or :/
    return not (len(arg) >= 3 and arg[1] == ":" and arg[2] in "/\\")


@click.command("watch")
@click.argument("target")
@click.option(
    "--llm",
    "llm_model",
    default=None,
    metavar="MODEL",
    help="(Prompt file mode only) Use an LLM for semantic verification.",
)
@click.option(
    "--interval",
    default=1.0,
    show_default=True,
    metavar="SECONDS",
    help="How often to poll for file changes.",
)
def watch_cmd(target: str, llm_model: str | None, interval: float) -> None:
    """Watch a file or agent and re-analyze on every change.

    \b
    Prompt file mode — watch a system prompt text file:
      checkagent watch system_prompt.txt
      checkagent watch prompt.txt --llm gpt-4o-mini

    Agent scan mode — watch a Python module and re-run scan on change:
      checkagent watch my_module:my_agent
      checkagent watch my_module:my_agent --interval 0.5

    In agent mode, CheckAgent resolves the source file automatically and
    triggers a full safety scan whenever you save the file.
    """
    if _is_module_target(target):
        if llm_model:
            _console.print(
                "[yellow]Note: --llm is not used in agent scan mode.[/yellow]"
            )
        _watch_agent(target, interval)
    else:
        path = Path(target)
        if not path.exists():
            raise click.ClickException(f"File not found: {target}")
        if not path.is_file():
            raise click.ClickException(f"Not a file: {target}")
        _watch_prompt_file(path, llm_model, interval)
