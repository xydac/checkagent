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


def _render_category_trends(records: list) -> None:
    """Show per-category finding counts across scans (newest-first records)."""
    # Collect all categories across all scans
    all_cats: dict[str, list[int]] = {}
    for record in records:
        findings = record.get("findings", [])
        cats_in_scan: dict[str, int] = {}
        for f in findings:
            cat = f.get("category", "unknown")
            cats_in_scan[cat] = cats_in_scan.get(cat, 0) + 1
        for cat in cats_in_scan:
            if cat not in all_cats:
                all_cats[cat] = []
        for cat in all_cats:
            all_cats[cat].append(cats_in_scan.get(cat, 0))

    if not all_cats:
        console.print("\n[dim]No findings recorded — no category trend data.[/dim]")
        return

    console.print()
    console.print("[bold]Category Trends[/bold] (newest → oldest)")
    console.print("─" * 50)

    cat_table = Table(show_header=True, header_style="bold dim", show_lines=False)
    cat_table.add_column("Category", style="white", min_width=22)
    cat_table.add_column("Trend (newest→oldest)", min_width=18)
    cat_table.add_column("Now", justify="right", min_width=5)
    cat_table.add_column("Change", justify="right", min_width=8)

    for cat in sorted(all_cats):
        counts = all_cats[cat]
        newest = counts[0]
        oldest = counts[-1] if len(counts) > 1 else newest

        # Sparkline: each count → bar height (inverted: fewer findings = better)
        if counts:
            max_count = max(counts) or 1
            spark_chars = []
            for c in reversed(counts):  # oldest → newest for left-to-right reading
                idx = int((c / max_count) * (len(_SPARK_CHARS) - 1))
                spark_chars.append(_SPARK_CHARS[idx])
            spark = "".join(spark_chars)
        else:
            spark = ""

        # Change indicator (fewer findings = better = green)
        if len(counts) < 2:
            change_str = "[dim]—[/dim]"
        else:
            delta = newest - oldest
            if delta < 0:
                change_str = f"[green]↓ {abs(delta)} improved[/green]"
            elif delta > 0:
                change_str = f"[red]↑ {delta} worse[/red]"
            else:
                change_str = "[dim]= stable[/dim]"

        cat_table.add_row(cat, spark, str(newest), change_str)

    console.print(cat_table)
    console.print(
        "[dim]Each bar = one scan. Taller bar = more findings in that category "
        "(lower is better).[/dim]"
    )
    console.print()


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
    "--categories",
    "show_categories",
    is_flag=True,
    default=False,
    help="Show per-category finding trends across scans.",
)
@click.option(
    "--dir",
    "base_dir",
    type=click.Path(file_okay=False),
    default=None,
    help="Project root directory containing .checkagent/. Defaults to current directory.",
)
def history_cmd(
    target: str | None,
    url_target: str | None,
    limit: int,
    show_categories: bool,
    base_dir: str | None,
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
        checkagent history my_agent:fn --categories
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

    if show_categories:
        _render_category_trends(records)
