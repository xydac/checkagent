"""checkagent dashboard — overview of all agents scanned in this project.

Shows the latest safety score for every target that has scan history,
with trend direction vs. the previous scan.
"""

from __future__ import annotations

import json
from pathlib import Path

import click
from rich.console import Console
from rich.table import Table

console = Console()

_HISTORY_DIR = ".checkagent/history"
def _score_style(score: float) -> str:
    pct = int(round(score * 100))
    if pct >= 80:
        return f"[green]{pct}%[/green]"
    if pct >= 60:
        return f"[yellow]{pct}%[/yellow]"
    return f"[red]{pct}%[/red]"


def _load_agent_summary(tdir: Path) -> dict | None:
    """Load latest + second-most-recent record for one target directory."""
    latest_path = tdir / "latest.json"
    if not latest_path.exists():
        return None
    try:
        latest = json.loads(latest_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None

    # Collect all timestamped records for sparkline
    try:
        files = sorted(
            (f for f in tdir.iterdir() if f.name != "latest.json" and f.suffix == ".json"),
            key=lambda p: p.stem,
        )
    except OSError:
        files = []

    scores: list[float] = []
    for f in files:
        try:
            r = json.loads(f.read_text(encoding="utf-8"))
            scores.append(r.get("summary", {}).get("score", 0.0))
        except (json.JSONDecodeError, OSError):
            continue

    prev_score: float | None = scores[-2] if len(scores) >= 2 else None
    current_score = latest.get("summary", {}).get("score", 0.0)

    trend = ""
    if prev_score is not None:
        delta = current_score - prev_score
        if delta > 0.005:
            trend = "[green]↑[/green]"
        elif delta < -0.005:
            trend = "[red]↓[/red]"
        else:
            trend = "[dim]→[/dim]"

    return {
        "target": latest.get("target", tdir.name),
        "score": current_score,
        "failed": latest.get("summary", {}).get("failed", 0),
        "total": latest.get("summary", {}).get("total", 0),
        "date": latest.get("date", ""),
        "scans": len(files),
        "trend": trend,
    }


@click.command("dashboard")
@click.option(
    "--dir",
    "base_dir",
    type=click.Path(file_okay=False),
    default=None,
    help="Project root directory containing .checkagent/. Defaults to current directory.",
)
@click.option(
    "--top",
    type=int,
    default=20,
    show_default=True,
    help="Show only the N lowest-scoring agents (most attention needed first).",
)
@click.option(
    "--json",
    "json_output",
    is_flag=True,
    default=False,
    help="Output results as JSON.",
)
def dashboard_cmd(base_dir: str | None, top: int, json_output: bool) -> None:
    """Show a safety overview for all agents scanned in this project.

    Reads scan history from .checkagent/history/ and displays the latest
    score, trend, and finding counts for every target.

    \b
    Examples:
        checkagent dashboard
        checkagent dashboard --json
        checkagent dashboard --dir /path/to/project
    """
    import json as json_mod

    base = Path(base_dir) if base_dir else Path.cwd()
    history_root = base / _HISTORY_DIR

    if not history_root.exists():
        if json_output:
            print(json_mod.dumps({"agents": [], "total": 0}))
        else:
            console.print("[yellow]No scan history found.[/yellow]")
            console.print(
                "[dim]Run [bold]checkagent scan <target>[/bold] to record your first result.[/dim]"
            )
        return

    agents = []
    try:
        for tdir in sorted(history_root.iterdir()):
            if not tdir.is_dir():
                continue
            summary = _load_agent_summary(tdir)
            if summary:
                agents.append(summary)
    except OSError as exc:
        console.print(f"[red]Error reading history: {exc}[/red]")
        return

    if not agents:
        if json_output:
            print(json_mod.dumps({"agents": [], "total": 0}))
        else:
            console.print("[yellow]No scan history found.[/yellow]")
            console.print(
                "[dim]Run [bold]checkagent scan <target>[/bold] to record your first result.[/dim]"
            )
        return

    # Sort by score ascending so lowest-scoring agents appear first (most attention needed)
    agents.sort(key=lambda a: a["score"])
    total_agents = len(agents)
    displayed = agents[:top]

    if json_output:
        out = {
            "agents": [
                {
                    "target": a["target"],
                    "score": round(a["score"], 4),
                    "failed": a["failed"],
                    "total": a["total"],
                    "date": a["date"],
                    "scans": a["scans"],
                }
                for a in displayed
            ],
            "total": total_agents,
            "showing": len(displayed),
        }
        print(json_mod.dumps(out, indent=2))
        return

    title = f"Safety Dashboard — {total_agents} agent(s)"
    if total_agents > top:
        title += f" (showing bottom {top} by score)"
    table = Table(title=title, show_lines=False, expand=False)
    table.add_column("Agent", no_wrap=True)
    table.add_column("Score", justify="right")
    table.add_column("Trend", justify="center")
    table.add_column("Fail/Total", justify="right")
    table.add_column("Scans", justify="right")
    table.add_column("Last Scan", style="dim")

    for a in displayed:
        table.add_row(
            a["target"],
            _score_style(a["score"]),
            a["trend"],
            f"{a['failed']}/{a['total']}",
            str(a["scans"]),
            a["date"],
        )

    console.print()
    console.print(table)

    avg = sum(a["score"] for a in agents) / total_agents
    avg_pct = int(round(avg * 100))
    console.print(
        f"\n[dim]Average score across {total_agents} agent(s): "
        f"{avg_pct}%.[/dim]"
    )
    console.print(
        "[dim]Run [bold]checkagent history <target>[/bold] for per-agent scan history.[/dim]"
    )
