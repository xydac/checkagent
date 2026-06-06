"""checkagent diff — compare two scan results and show safety regressions.

Compare scan JSON files side-by-side to detect new vulnerabilities,
fixed findings, and score changes. Designed for CI workflows:
scan main, scan PR branch, diff the results.

Usage::

    checkagent diff baseline.json current.json
    checkagent diff baseline.json current.json --json
    checkagent diff baseline.json current.json --fail-on-new
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import click
from rich.console import Console
from rich.table import Table

console = Console()


def _load_scan(path: Path) -> dict[str, Any]:
    """Load and validate a scan JSON file."""
    text = path.read_text(encoding="utf-8")
    data = json.loads(text)
    if "summary" not in data or "findings" not in data:
        raise click.ClickException(
            f"{path} does not look like a checkagent scan JSON file "
            "(missing 'summary' or 'findings' keys)"
        )
    return data


def _finding_key(f: dict[str, Any]) -> str:
    """Stable identity for a finding — probe_id + category."""
    return f"{f.get('probe_id', '')}::{f.get('category', '')}"


def compute_diff(
    baseline: dict[str, Any],
    current: dict[str, Any],
) -> dict[str, Any]:
    """Compute the structured diff between two scan results.

    Returns a dict with new_findings, fixed_findings, unchanged_findings,
    and summary statistics.
    """
    base_findings = {_finding_key(f): f for f in baseline.get("findings", [])}
    curr_findings = {_finding_key(f): f for f in current.get("findings", [])}

    base_keys = set(base_findings.keys())
    curr_keys = set(curr_findings.keys())

    new_keys = curr_keys - base_keys
    fixed_keys = base_keys - curr_keys
    common_keys = base_keys & curr_keys

    new_findings = [curr_findings[k] for k in sorted(new_keys)]
    fixed_findings = [base_findings[k] for k in sorted(fixed_keys)]
    unchanged_findings = [curr_findings[k] for k in sorted(common_keys)]

    base_summary = baseline.get("summary", {})
    curr_summary = current.get("summary", {})

    base_score = base_summary.get("score", 0.0)
    curr_score = curr_summary.get("score", 0.0)
    score_delta = curr_score - base_score

    base_stability = baseline.get("stability")
    curr_stability = current.get("stability")
    stability: dict[str, Any] | None = None
    if base_stability is not None and curr_stability is not None:
        base_stab_score = base_stability.get("stability_score", 1.0)
        curr_stab_score = curr_stability.get("stability_score", 1.0)
        stab_delta = curr_stab_score - base_stab_score
        stability = {
            "baseline": round(base_stab_score, 4),
            "current": round(curr_stab_score, 4),
            "delta": round(stab_delta, 4),
            "baseline_repeat": base_stability.get("repeat", 1),
            "current_repeat": curr_stability.get("repeat", 1),
        }

    return {
        "baseline_target": baseline.get("target", "unknown"),
        "current_target": current.get("target", "unknown"),
        "score": {
            "baseline": base_score,
            "current": curr_score,
            "delta": round(score_delta, 4),
        },
        "stability": stability,
        "probes": {
            "baseline_total": base_summary.get("total", 0),
            "current_total": curr_summary.get("total", 0),
            "baseline_passed": base_summary.get("passed", 0),
            "current_passed": curr_summary.get("passed", 0),
        },
        "new_findings": new_findings,
        "fixed_findings": fixed_findings,
        "unchanged_findings": unchanged_findings,
        "counts": {
            "new": len(new_findings),
            "fixed": len(fixed_findings),
            "unchanged": len(unchanged_findings),
        },
        "regression": len(new_findings) > 0,
    }


def _severity_style(severity: str) -> str:
    sev = severity.lower()
    if sev == "critical":
        return "bold red"
    if sev == "high":
        return "red"
    if sev == "medium":
        return "yellow"
    return "dim"


def render_diff(diff: dict[str, Any]) -> None:
    """Render a scan diff to the terminal with Rich."""
    score = diff["score"]
    counts = diff["counts"]

    delta = score["delta"]
    base_pct = int(round(score["baseline"] * 100))
    curr_pct = int(round(score["current"] * 100))

    if delta > 0.005:
        arrow = "[green]improved[/green]"
        delta_str = f"[green]+{int(round(delta * 100))}%[/green]"
    elif delta < -0.005:
        arrow = "[red]regressed[/red]"
        delta_str = f"[red]{int(round(delta * 100))}%[/red]"
    else:
        arrow = "[dim]unchanged[/dim]"
        delta_str = "[dim]+0%[/dim]"

    console.print()
    console.print("[bold]CheckAgent Scan Diff[/bold]")
    console.print()

    summary = Table(show_header=False, box=None, padding=(0, 2))
    summary.add_column("label", style="dim")
    summary.add_column("value")
    summary.add_row("Score", f"{base_pct}% → {curr_pct}% ({delta_str})")
    summary.add_row("Status", arrow)
    summary.add_row(
        "New findings",
        f"[red]{counts['new']}[/red]" if counts["new"] else "[green]0[/green]",
    )
    summary.add_row(
        "Fixed findings",
        f"[green]{counts['fixed']}[/green]" if counts["fixed"] else "[dim]0[/dim]",
    )
    summary.add_row("Unchanged", str(counts["unchanged"]))
    if diff.get("stability"):
        stab = diff["stability"]
        base_sp = int(round(stab["baseline"] * 100))
        curr_sp = int(round(stab["current"] * 100))
        sd = stab["delta"]
        if sd < -0.005:
            stab_str = f"[red]{base_sp}% → {curr_sp}% ({int(round(sd * 100))}%)[/red]"
        elif sd > 0.005:
            stab_str = f"[green]{base_sp}% → {curr_sp}% (+{int(round(sd * 100))}%)[/green]"
        else:
            stab_str = f"[dim]{base_sp}% → {curr_sp}% (+0%)[/dim]"
        summary.add_row("Stability", stab_str)
    console.print(summary)

    if diff["new_findings"]:
        console.print()
        table = Table(
            title="[red]New Findings (regressions)[/red]",
            show_lines=False,
        )
        table.add_column("Probe", style="cyan")
        table.add_column("Category")
        table.add_column("Severity")
        table.add_column("Finding", max_width=60)

        for f in diff["new_findings"]:
            table.add_row(
                f.get("probe_id", "?"),
                f.get("category", "?"),
                f"[{_severity_style(f.get('severity', ''))}]"
                f"{f.get('severity', '?')}"
                f"[/{_severity_style(f.get('severity', ''))}]",
                f.get("finding", "")[:60],
            )
        console.print(table)

    if diff["fixed_findings"]:
        console.print()
        table = Table(
            title="[green]Fixed Findings (improvements)[/green]",
            show_lines=False,
        )
        table.add_column("Probe", style="cyan")
        table.add_column("Category")
        table.add_column("Severity")
        table.add_column("Finding", max_width=60)

        for f in diff["fixed_findings"]:
            table.add_row(
                f.get("probe_id", "?"),
                f.get("category", "?"),
                f.get("severity", "?"),
                f.get("finding", "")[:60],
            )
        console.print(table)

    if not diff["new_findings"] and not diff["fixed_findings"]:
        console.print("\n[green]No changes detected between scans.[/green]")

    console.print()


@click.command("diff")
@click.argument(
    "baseline",
    type=click.Path(exists=True, dir_okay=False),
)
@click.argument(
    "current",
    type=click.Path(exists=True, dir_okay=False),
)
@click.option(
    "--json",
    "json_output",
    is_flag=True,
    default=False,
    help="Output diff as JSON instead of a Rich table.",
)
@click.option(
    "--fail-on-new",
    is_flag=True,
    default=False,
    help="Exit with code 1 if new findings (regressions) are detected.",
)
@click.option(
    "--comment-file",
    type=click.Path(dir_okay=False),
    default=None,
    help="Write a GitHub PR comment summarizing the diff to this file.",
)
def diff_cmd(
    baseline: str,
    current: str,
    json_output: bool,
    fail_on_new: bool,
    comment_file: str | None,
) -> None:
    """Compare two scan results and show safety regressions.

    BASELINE is the reference scan JSON (e.g., from main branch).
    CURRENT is the new scan JSON (e.g., from PR branch).

    \b
    Usage in CI:
        checkagent scan main_agent:fn --json > baseline.json
        # ... make changes ...
        checkagent scan main_agent:fn --json > current.json
        checkagent diff baseline.json current.json --fail-on-new

    \b
    Generate a PR comment:
        checkagent diff baseline.json current.json --comment-file diff-report.md
    """
    base_data = _load_scan(Path(baseline))
    curr_data = _load_scan(Path(current))

    result = compute_diff(base_data, curr_data)

    if json_output:
        print(json.dumps(result, indent=2))
    else:
        render_diff(result)

    if comment_file:
        md = _build_diff_comment(result)
        Path(comment_file).write_text(md, encoding="utf-8")
        if not json_output:
            console.print(f"[dim]PR comment written to {comment_file}[/dim]")

    if fail_on_new and result["regression"]:
        n = result["counts"]["new"]
        if not json_output:
            console.print(
                f"[red]Exiting with code 1: {n} new finding(s) detected.[/red]"
            )
        sys.exit(1)


def _build_diff_comment(diff: dict[str, Any]) -> str:
    """Build a GitHub PR comment in Markdown from a diff result."""
    score = diff["score"]
    counts = diff["counts"]
    base_pct = int(round(score["baseline"] * 100))
    curr_pct = int(round(score["current"] * 100))
    delta = score["delta"]

    if delta > 0.005:
        emoji = "🟢"
        status = "Improved"
    elif delta < -0.005:
        emoji = "🔴"
        status = "Regressed"
    else:
        emoji = "⚪"
        status = "Unchanged"

    lines = [
        f"## {emoji} CheckAgent Safety Diff — {status}",
        "",
        "| Metric | Baseline | Current | Delta |",
        "|--------|----------|---------|-------|",
        f"| Safety Score | {base_pct}% | {curr_pct}% "
        f"| {'+' if delta >= 0 else ''}{int(round(delta * 100))}% |",
        f"| New Findings | — | {counts['new']} | {'⚠️' if counts['new'] else '✅'} |",
        f"| Fixed Findings | {counts['fixed']} | — | {'✅' if counts['fixed'] else '—'} |",
        f"| Unchanged | {counts['unchanged']} | {counts['unchanged']} | — |",
    ]
    stab = diff.get("stability")
    if stab:
        base_sp = int(round(stab["baseline"] * 100))
        curr_sp = int(round(stab["current"] * 100))
        sd = stab["delta"]
        stab_delta_str = f"{'+' if sd >= 0 else ''}{int(round(sd * 100))}%"
        stab_flag = "⚠️" if sd < -0.05 else ("✅" if sd > 0.005 else "—")
        lines.append(
            f"| Stability | {base_sp}% | {curr_sp}% | {stab_delta_str} {stab_flag} |"
        )

    if diff["new_findings"]:
        lines.extend([
            "",
            "### New Findings (regressions)",
            "",
            "| Probe | Category | Severity | Finding |",
            "|-------|----------|----------|---------|",
        ])
        for f in diff["new_findings"]:
            finding_text = f.get("finding", "")[:80]
            pid = f.get("probe_id", "?")
            cat = f.get("category", "?")
            sev = f.get("severity", "?")
            lines.append(
                f"| {pid} | {cat} | {sev} | {finding_text} |"
            )

    if diff["fixed_findings"]:
        lines.extend([
            "",
            "### Fixed Findings (improvements)",
            "",
            "| Probe | Category | Severity |",
            "|-------|----------|----------|",
        ])
        for f in diff["fixed_findings"]:
            pid = f.get("probe_id", "?")
            cat = f.get("category", "?")
            sev = f.get("severity", "?")
            lines.append(f"| {pid} | {cat} | {sev} |")

    lines.extend([
        "",
        "---",
        "*Generated by [CheckAgent](https://github.com/checkagent/checkagent)*",
    ])

    return "\n".join(lines) + "\n"
