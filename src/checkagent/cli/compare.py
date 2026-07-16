"""checkagent compare — side-by-side safety comparison of two agents.

Compare the safety scan results of two agents, highlighting strengths,
weaknesses, and category-level differences.

Usage::

    checkagent compare agent_a:fn agent_b:fn
    checkagent compare agent_a:fn agent_b:fn --json

Uses scan history from ``.checkagent/history/``.  Run ``checkagent scan``
on both targets first.
"""

from __future__ import annotations

import json as json_mod
from collections import defaultdict
from pathlib import Path

import click
from rich.console import Console
from rich.table import Table

from checkagent.cli.history import _target_id

console = Console()


def _load_latest(target: str, base_dir: Path) -> dict | None:
    history_dir = base_dir / ".checkagent" / "history" / _target_id(target)
    latest = history_dir / "latest.json"
    if not latest.exists():
        return None
    try:
        return json_mod.loads(latest.read_text(encoding="utf-8"))
    except (json_mod.JSONDecodeError, OSError):
        return None


def _findings_by_category(record: dict) -> dict[str, int]:
    cats: dict[str, int] = defaultdict(int)
    for f in record.get("findings", []):
        cat = f.get("category", "unknown")
        cats[cat] += 1
    return dict(cats)


def build_comparison(record_a: dict, record_b: dict) -> dict:
    """Build a structured comparison between two scan results."""
    sum_a = record_a.get("summary", {})
    sum_b = record_b.get("summary", {})

    score_a = sum_a.get("score", 0.0)
    score_b = sum_b.get("score", 0.0)

    cats_a = _findings_by_category(record_a)
    cats_b = _findings_by_category(record_b)
    all_cats = sorted(set(cats_a) | set(cats_b))

    categories = []
    for cat in all_cats:
        a_count = cats_a.get(cat, 0)
        b_count = cats_b.get(cat, 0)
        categories.append({
            "category": cat,
            "agent_a_findings": a_count,
            "agent_b_findings": b_count,
            "delta": b_count - a_count,
        })

    # Unique failing probe IDs: probes that failed on one but not the other
    def _probe_key(f: dict) -> str:
        return f.get("probe_id") or f.get("probe") or f.get("description") or ""

    a_probes = {_probe_key(f) for f in record_a.get("findings", []) if _probe_key(f)}
    b_probes = {_probe_key(f) for f in record_b.get("findings", []) if _probe_key(f)}
    only_a = sorted(a_probes - b_probes)
    only_b = sorted(b_probes - a_probes)

    return {
        "agent_a": {
            "target": record_a.get("target", "?"),
            "score": score_a,
            "passed": sum_a.get("passed", 0),
            "failed": sum_a.get("failed", 0),
            "total": sum_a.get("total", 0),
            "date": record_a.get("date", "?"),
        },
        "agent_b": {
            "target": record_b.get("target", "?"),
            "score": score_b,
            "passed": sum_b.get("passed", 0),
            "failed": sum_b.get("failed", 0),
            "total": sum_b.get("total", 0),
            "date": record_b.get("date", "?"),
        },
        "score_delta": round(score_b - score_a, 4),
        "categories": categories,
        "only_agent_a": only_a,
        "only_agent_b": only_b,
        "winner": (
            "agent_a" if score_a > score_b
            else "agent_b" if score_b > score_a
            else "tie"
        ),
    }


def _display_comparison(comparison: dict) -> None:
    a = comparison["agent_a"]
    b = comparison["agent_b"]

    console.print()

    # Summary table
    summary = Table(title="Agent Safety Comparison", show_header=True)
    summary.add_column("Metric", style="bold")
    summary.add_column(a["target"], justify="center")
    summary.add_column(b["target"], justify="center")

    score_a_str = f"{a['score']:.0%}"
    score_b_str = f"{b['score']:.0%}"
    if a["score"] > b["score"]:
        score_a_str = f"[green]{score_a_str}[/green]"
        score_b_str = f"[red]{score_b_str}[/red]"
    elif b["score"] > a["score"]:
        score_a_str = f"[red]{score_a_str}[/red]"
        score_b_str = f"[green]{score_b_str}[/green]"

    summary.add_row("Safety Score", score_a_str, score_b_str)
    summary.add_row("Passed", str(a["passed"]), str(b["passed"]))
    summary.add_row("Failed", str(a["failed"]), str(b["failed"]))
    summary.add_row("Total Probes", str(a["total"]), str(b["total"]))
    summary.add_row("Scan Date", a["date"], b["date"])
    console.print(summary)

    # Category breakdown
    cats = comparison["categories"]
    if cats:
        console.print()
        cat_table = Table(title="Findings by Category", show_header=True)
        cat_table.add_column("Category", style="bold")
        cat_table.add_column(a["target"], justify="center")
        cat_table.add_column(b["target"], justify="center")
        cat_table.add_column("Delta", justify="center")

        for c in cats:
            delta = c["delta"]
            if delta > 0:
                delta_str = f"[red]+{delta}[/red]"
            elif delta < 0:
                delta_str = f"[green]{delta}[/green]"
            else:
                delta_str = "="
            cat_table.add_row(
                c["category"],
                str(c["agent_a_findings"]),
                str(c["agent_b_findings"]),
                delta_str,
            )
        console.print(cat_table)

    # Unique findings
    only_a = comparison["only_agent_a"]
    only_b = comparison["only_agent_b"]
    if only_a:
        console.print(
            f"\n[yellow]Only {a['target']}[/yellow] fails ({len(only_a)}):"
        )
        for desc in only_a[:10]:
            console.print(f"  • {desc}")
        if len(only_a) > 10:
            console.print(f"  … and {len(only_a) - 10} more")

    if only_b:
        console.print(
            f"\n[yellow]Only {b['target']}[/yellow] fails ({len(only_b)}):"
        )
        for desc in only_b[:10]:
            console.print(f"  • {desc}")
        if len(only_b) > 10:
            console.print(f"  … and {len(only_b) - 10} more")

    # Winner
    winner = comparison["winner"]
    if winner == "tie":
        console.print("\n[bold]Result:[/bold] Tie — both agents scored equally.")
    else:
        w = a if winner == "agent_a" else b
        console.print(
            f"\n[bold green]Winner:[/bold green] {w['target']}"
            f" ({w['score']:.0%} vs"
            f" {(b if winner == 'agent_a' else a)['score']:.0%})"
        )
    console.print()


@click.command("compare")
@click.argument("target_a")
@click.argument("target_b")
@click.option(
    "--json", "json_output",
    is_flag=True,
    help="Output comparison as JSON.",
)
@click.option(
    "--base-dir",
    type=click.Path(exists=True, file_okay=False),
    default=None,
    help="Base directory for .checkagent/history/ (default: cwd).",
)
def compare_cmd(
    target_a: str,
    target_b: str,
    json_output: bool,
    base_dir: str | None,
) -> None:
    """Compare safety scan results of two agents side-by-side.

    Uses the latest scan history for each target.  Run ``checkagent scan``
    on both targets first.

    Examples::

        checkagent compare agent_a:fn agent_b:fn
        checkagent compare --url-a http://a/chat --url-b http://b/chat --json
    """
    base = Path(base_dir) if base_dir else Path.cwd()

    record_a = _load_latest(target_a, base)
    record_b = _load_latest(target_b, base)

    if record_a is None:
        console.print(
            f"[red]No scan history for '{target_a}'.[/red]"
            " Run: checkagent scan " + target_a
        )
        raise SystemExit(1)
    if record_b is None:
        console.print(
            f"[red]No scan history for '{target_b}'.[/red]"
            " Run: checkagent scan " + target_b
        )
        raise SystemExit(1)

    comparison = build_comparison(record_a, record_b)

    if json_output:
        click.echo(json_mod.dumps(comparison, indent=2))
    else:
        _display_comparison(comparison)
