"""checkagent history — list past scan results for a target.

Shows a table of previous scan runs with scores and finding counts,
so you can track safety posture over time.
"""

from __future__ import annotations

from pathlib import Path

import click
from rich.console import Console
from rich.table import Table

console = Console()

# Braille sparkline blocks (empty → full, 8 levels)
_SPARK_CHARS = " ▁▂▃▄▅▆▇█"


def _sparkline(scores: list[float]) -> str:
    """Return a single-line ASCII sparkline for a list of 0-1 scores."""
    if not scores:
        return ""
    chars = []
    for s in scores:
        idx = min(int(s * (len(_SPARK_CHARS) - 1)), len(_SPARK_CHARS) - 1)
        chars.append(_SPARK_CHARS[idx])
    return "".join(chars)


def _trend_summary(records: list) -> str:
    """Return a human-readable trend sentence given history records (newest first)."""
    if len(records) < 2:
        return ""
    scores = [r.get("summary", {}).get("score", 0.0) for r in records]
    oldest = scores[-1]
    newest = scores[0]
    delta = newest - oldest
    n = len(records)

    if abs(delta) <= 0.005:
        return f"[dim]Score stable across {n} scans ({int(round(newest * 100))}%).[/dim]"
    direction = "improved" if delta > 0 else "regressed"
    color = "green" if delta > 0 else "red"
    arrow = "↑" if delta > 0 else "↓"
    old_pct = int(round(oldest * 100))
    new_pct = int(round(newest * 100))
    change = abs(int(round(delta * 100)))
    return (
        f"[{color}]{arrow} Score {direction} {old_pct}% → {new_pct}% "
        f"(+{change}% over {n} scans)[/{color}]"
        if delta > 0
        else f"[{color}]{arrow} Score {direction} {old_pct}% → {new_pct}% "
        f"(-{change}% over {n} scans)[/{color}]"
    )


@click.command("history")
@click.argument("target", required=False, default=None)
@click.option(
    "--url",
    "url_target",
    default=None,
    metavar="URL",
    help="Show history for an HTTP endpoint target (alternative to positional TARGET).",
)
@click.option(
    "--limit",
    type=int,
    default=10,
    show_default=True,
    help="Maximum number of past scans to display.",
)
@click.option(
    "--dir",
    "base_dir",
    type=click.Path(file_okay=False),
    default=None,
    help="Project root directory containing .checkagent/. Defaults to current directory.",
)
def history_cmd(
    target: str | None, url_target: str | None, limit: int, base_dir: str | None
) -> None:
    """Show scan history for a target.

    TARGET is the agent target you previously scanned (module:function or URL).
    For HTTP endpoints, you can use --url instead of the positional argument.

    \b
    Examples:
        checkagent history my_agent:agent_fn
        checkagent history http://localhost:8000/chat
        checkagent history --url http://localhost:8000/chat
        checkagent history sample_agent:sample_agent --limit 5
    """
    from checkagent.cli.history import list_history

    resolved = url_target or target
    if not resolved:
        raise click.UsageError(
            "Provide a TARGET (module:function or URL) or use --url http://..."
        )

    bdir = Path(base_dir) if base_dir else None
    records = list_history(resolved, limit=limit, base_dir=bdir)

    if not records:
        console.print(
            f"[yellow]No scan history found for[/yellow] [cyan]{resolved}[/cyan]"
        )
        console.print(
            "[dim]Run [bold]checkagent scan[/bold] first to record a result.[/dim]"
        )
        return

    table = Table(title=f"Scan history — {resolved}", show_lines=False)
    table.add_column("Date", style="dim")
    table.add_column("Time", style="dim")
    table.add_column("Score", justify="right")
    table.add_column("Passed", justify="right", style="green")
    table.add_column("Failed", justify="right", style="red")
    table.add_column("Total", justify="right")
    table.add_column("Time (s)", justify="right", style="dim")

    prev_score: float | None = None
    for record in records:
        s = record.get("summary", {})
        score = s.get("score", 0.0)
        pct = f"{int(round(score * 100))}%"

        # Trend indicator vs. the previous row (records are newest-first)
        if prev_score is not None:
            if score > prev_score + 0.005:
                pct = f"[red]{pct} ↓[/red]"  # older record was higher = going down
            elif score < prev_score - 0.005:
                pct = f"[green]{pct} ↑[/green]"  # older record was lower = going up
        prev_score = score

        table.add_row(
            record.get("date", ""),
            record.get("time", ""),
            pct,
            str(s.get("passed", 0)),
            str(s.get("failed", 0)),
            str(s.get("total", 0)),
            str(s.get("elapsed_seconds", "")),
        )

    console.print()
    console.print(table)

    # Trend summary and sparkline
    scores = [r.get("summary", {}).get("score", 0.0) for r in records]
    if len(scores) >= 2:
        spark = _sparkline(list(reversed(scores)))  # oldest → newest for sparkline
        trend = _trend_summary(records)
        console.print(f"\n  Trend: {spark}  {trend}")

    console.print(
        f"\n[dim]{len(records)} scan(s) shown. "
        f"Run [bold]checkagent scan {resolved}[/bold] to add a new result.[/dim]"
    )
