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


@click.command("history")
@click.argument("target")
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
def history_cmd(target: str, limit: int, base_dir: str | None) -> None:
    """Show scan history for a target.

    TARGET is the agent target you previously scanned (module:function or URL).

    \b
    Examples:
        checkagent history my_agent:agent_fn
        checkagent history --url http://localhost:8000/chat
        checkagent history sample_agent:sample_agent --limit 5
    """
    from checkagent.cli.history import list_history

    bdir = Path(base_dir) if base_dir else None
    records = list_history(target, limit=limit, base_dir=bdir)

    if not records:
        console.print(
            f"[yellow]No scan history found for[/yellow] [cyan]{target}[/cyan]"
        )
        console.print(
            "[dim]Run [bold]checkagent scan[/bold] first to record a result.[/dim]"
        )
        return

    table = Table(title=f"Scan history — {target}", show_lines=False)
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

        # Trend indicator vs. the previous row
        if prev_score is not None:
            if score > prev_score + 0.005:
                pct = f"[green]{pct} ↑[/green]"
            elif score < prev_score - 0.005:
                pct = f"[red]{pct} ↓[/red]"
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
    console.print(
        f"\n[dim]{len(records)} scan(s) shown. "
        "Run [bold]checkagent scan[/bold] to add a new result.[/dim]"
    )
