"""checkagent audit — complete security audit in one command.

Runs safety scan, triage, and attack chain simulation in a single pass.
Turns three separate commands into one actionable report.

Usage::

    checkagent audit my_module:agent_fn
    checkagent audit --url http://localhost:8000/chat
    checkagent audit my_module:agent_fn --report audit.md
    checkagent audit my_module:agent_fn --json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import click

from checkagent.cli.simulate import render_attack_chains_markdown, simulate_attacks
from checkagent.cli.triage import triage_findings


def _run_scan_json(
    target: str | None,
    url: str | None,
    category: str | None,
    timeout: float,
    repeat: int,
    llm_judge: str | None = None,
    agent_description: str | None = None,
) -> dict:
    """Run scan and return JSON result dict."""
    import shutil
    import subprocess

    # Prefer the checkagent CLI entry point; fall back to package invocation via
    # cli.__init__:main — safer than -m checkagent (no __main__.py guaranteed).
    cli_exe = shutil.which("checkagent")
    if cli_exe:
        cmd = [cli_exe, "scan", "--json", "--exit-zero"]
    else:
        cmd = [sys.executable, "-c",
               "from checkagent.cli import main; main()", "scan", "--json", "--exit-zero"]

    if url:
        cmd += ["--url", url]
    elif target:
        cmd.append(target)

    if category:
        cmd += ["--category", category]

    cmd += ["--timeout", str(timeout)]

    if repeat > 1:
        cmd += ["--repeat", str(repeat)]

    if llm_judge:
        cmd += ["--llm-judge", llm_judge]

    if agent_description:
        cmd += ["--agent-description", agent_description]

    proc = subprocess.run(cmd, capture_output=True, text=True)

    if not proc.stdout.strip():
        stderr = proc.stderr.strip()
        click.echo(
            f"Error: scan produced no output.\n{stderr}" if stderr
            else "Error: scan produced no output.",
            err=True,
        )
        raise SystemExit(1)

    try:
        return json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        click.echo(f"Error parsing scan output: {exc}", err=True)
        click.echo(f"Raw output: {proc.stdout[:500]}", err=True)
        raise SystemExit(1) from None


def run_audit(
    target: str | None,
    url: str | None = None,
    *,
    category: str | None = None,
    timeout: float = 10.0,
    repeat: int = 1,
    llm_judge: str | None = None,
    agent_description: str | None = None,
) -> dict:
    """Run full audit pipeline: scan → triage → simulate.

    Returns a structured dict with scan data, triage, simulation results,
    and a ``share_card`` string suitable for pasting into PR comments or Slack.
    """
    scan_data = _run_scan_json(
        target, url, category, timeout, repeat,
        llm_judge=llm_judge,
        agent_description=agent_description,
    )

    findings = scan_data.get("findings", [])
    total = scan_data.get("summary", {}).get("total", 0)

    triage = triage_findings(findings, total) if findings else []
    simulation = simulate_attacks(scan_data)

    from checkagent.cli.grade import score_to_grade
    score = scan_data.get("summary", {}).get("score", 0.0)
    grade = score_to_grade(score)
    probe_count = scan_data.get("summary", {}).get("total", 0)

    if findings and triage:
        chain_count = simulation.get("chain_count", 0)
        chain_note = f", {chain_count} attack chains exploitable" if chain_count else ""
        top = triage[0]
        share_card = (
            f"CheckAgent audit: **{grade}** ({score:.0%}){chain_note}. "
            f"Fix `{top['category']}` first → +{top['score_improvement_pct']}%."
        )
    else:
        share_card = (
            f"CheckAgent audit: **{grade}** ({score:.0%}) — "
            f"Agent passed all {probe_count} safety probes. No findings."
        )

    return {
        "scan": scan_data,
        "triage": triage,
        "simulation": simulation,
        "share_card": share_card,
    }


def _render_audit_markdown(audit: dict, target_label: str) -> str:
    """Render a full audit report as Markdown."""
    scan = audit["scan"]
    summary = scan.get("summary", {})
    triage = audit["triage"]
    simulation = audit["simulation"]

    score = summary.get("score", 0.0)
    total = summary.get("total", 0)
    passed = summary.get("passed", 0)
    failed = summary.get("failed", 0)
    findings = scan.get("findings", [])

    from checkagent.cli.grade import compute_percentile, score_to_grade

    grade = score_to_grade(score)
    percentile = compute_percentile(score)

    lines: list[str] = []
    lines.append("# CheckAgent Security Audit\n")
    lines.append(f"**Target:** `{target_label}`\n")

    lines.append("## Summary\n")
    lines.append(
        f"| Metric | Value |\n"
        f"|--------|-------|\n"
        f"| Safety Score | {score:.0%} |\n"
        f"| Grade | {grade} |\n"
        f"| Percentile | safer than {percentile}% of tested agents |\n"
        f"| Probes Run | {total} |\n"
        f"| Passed | {passed} |\n"
        f"| Failed | {failed} |\n"
    )
    lines.append("")

    if triage:
        lines.append("## Triage — Fix These First\n")
        top = triage[0]
        lines.append(
            f"> **Start with `{top['category']}`** — "
            f"{top['finding_count']} findings, "
            f"+{top['score_improvement_pct']}% score improvement, "
            f"effort: {top['effort_label']}\n"
        )
        lines.append("| Priority | Category | Findings | Score Gain | Effort | Severity |")
        lines.append("|----------|----------|----------|------------|--------|----------|")
        for i, p in enumerate(triage, 1):
            lines.append(
                f"| {i} | {p['category']} | {p['finding_count']} "
                f"| +{p['score_improvement_pct']}% | {p['effort_label']} "
                f"| {p['max_severity']} |"
            )
        lines.append("")

    if findings:
        lines.append("## Findings\n")
        lines.append("| Severity | Category | Probe | Finding |")
        lines.append("|----------|----------|-------|---------|")
        for f in findings[:20]:
            sev = f.get("severity", "?")
            cat = f.get("category", "?")
            probe = f.get("probe_id", f.get("finding", "?"))
            finding = f.get("finding", "")
            lines.append(f"| {sev} | {cat} | `{probe}` | {finding} |")
        if len(findings) > 20:
            lines.append(f"\n*... and {len(findings) - 20} more findings*")
        lines.append("")

    if simulation.get("chain_count", 0) > 0:
        lines.append(render_attack_chains_markdown(simulation))

    lines.append("\n---")
    lines.append("*Generated by [CheckAgent](https://github.com/xydac/checkagent)*")

    return "\n".join(lines)


@click.command("audit")
@click.argument("target", required=False)
@click.option("--url", type=str, default=None,
              help="HTTP endpoint to audit instead of a Python callable.")
@click.option("--category", type=str, default=None,
              help="Probe category to run (default: all).")
@click.option("--timeout", type=float, default=10.0,
              help="Probe timeout in seconds (default: 10).")
@click.option("--repeat", type=int, default=1,
              help="Run each probe N times for stability scoring.")
@click.option("--llm-judge", "llm_judge", type=str, default=None,
              help="Use an LLM to judge probe results (e.g. gpt-4o-mini, claude-3-haiku).")
@click.option("--agent-description", "agent_description", type=str, default=None,
              help="Agent role description; improves LLM judge accuracy.")
@click.option("--report", "report_file", type=click.Path(), default=None,
              help="Write audit report to file (.md, .json).")
@click.option("--json", "json_output", is_flag=True,
              help="Output full audit as JSON.")
def audit_cmd(
    target: str | None,
    url: str | None,
    category: str | None,
    timeout: float,
    repeat: int,
    llm_judge: str | None,
    agent_description: str | None,
    report_file: str | None,
    json_output: bool,
) -> None:
    """Run a complete security audit: scan + triage + attack chains.

    Combines scan, triage, and simulate into one command. Pass the same
    target you'd use with ``checkagent scan``.

    \\b
    Examples:
      checkagent audit my_agent:run
      checkagent audit --url http://localhost:8000/chat
      checkagent audit my_agent:run --llm-judge gpt-4o-mini --agent-description "HR assistant"
      checkagent audit my_agent:run --report audit.md
    """
    if not target and not url:
        click.echo("Error: provide a TARGET (module:function) or --url.", err=True)
        raise SystemExit(1)

    target_label = url or target

    if json_output:
        audit = run_audit(
            target, url, category=category, timeout=timeout, repeat=repeat,
            llm_judge=llm_judge, agent_description=agent_description,
        )
        click.echo(json.dumps(audit, indent=2))
        return

    from rich.console import Console
    from rich.panel import Panel
    from rich.rule import Rule

    from checkagent.cli.grade import compute_percentile, grade_color, score_to_grade

    console = Console()

    # ── Header ───────────────────────────────────────────────────────────────
    console.print()
    console.print(Panel.fit(
        f"  [bold]Target:[/bold] [cyan]{target_label}[/cyan]",
        title="CheckAgent Security Audit",
        border_style="blue",
    ))
    console.print()

    # ── Run scan (shows its own progress output) ──────────────────────────
    console.print(Rule("[bold]Phase 1: Safety Scan[/bold]", style="blue"))
    console.print()

    try:
        audit = run_audit(
            target, url, category=category, timeout=timeout, repeat=repeat,
            llm_judge=llm_judge, agent_description=agent_description,
        )
    except SystemExit:
        raise

    scan_data = audit["scan"]
    triage = audit["triage"]
    simulation = audit["simulation"]
    summary = scan_data.get("summary", {})

    score = summary.get("score", 0.0)
    findings = scan_data.get("findings", [])
    grade = score_to_grade(score)
    percentile = compute_percentile(score)
    color = grade_color(grade)

    # ── Triage ───────────────────────────────────────────────────────────────
    if triage:
        console.print()
        console.print(Rule("[bold]Phase 2: Triage[/bold]", style="cyan"))
        console.print()

        from rich.table import Table

        top = triage[0]
        console.print(Panel.fit(
            f"[bold]Fix [cyan]{top['category']}[/cyan] first[/bold]\n\n"
            f"  Findings: {top['finding_count']}  |  "
            f"Score improvement: [green]+{top['score_improvement_pct']}%[/green]  |  "
            f"Effort: {top['effort_label']}\n\n"
            f"  [dim]{top['quick_fix']}[/dim]",
            title="Top Priority",
            border_style="cyan",
        ))

        if len(triage) > 1:
            table = Table(title="All Priorities", border_style="dim", show_header=True)
            table.add_column("#", justify="right", width=3)
            table.add_column("Category")
            table.add_column("Findings", justify="right")
            table.add_column("Score Gain", justify="right")
            table.add_column("Effort")
            table.add_column("Severity")

            for i, p in enumerate(triage, 1):
                sev_style = {
                    "critical": "bold red", "high": "red",
                    "medium": "yellow", "low": "dim",
                }.get(p["max_severity"], "dim")
                table.add_row(
                    str(i), p["category"], str(p["finding_count"]),
                    f"+{p['score_improvement_pct']}%", p["effort_label"],
                    f"[{sev_style}]{p['max_severity']}[/{sev_style}]",
                )

            console.print()
            console.print(table)

    # ── Attack chains ─────────────────────────────────────────────────────
    if findings:
        console.print()
        console.print(Rule("[bold]Phase 3: Attack Chain Analysis[/bold]", style="red"))
        console.print()

        risk = simulation.get("risk_level", "low")
        risk_color = {
            "critical": "bold red", "high": "red",
            "medium": "yellow", "low": "green",
        }.get(risk, "white")

        chain_count = simulation.get("chain_count", 0)
        near_miss_count = len(simulation.get("near_misses", []))

        console.print(Panel.fit(
            f"  Risk Level: [{risk_color}]{risk.upper()}[/{risk_color}]\n"
            f"  Exploitable Chains: [bold]{chain_count}[/bold]\n"
            f"  Near-Misses: [bold]{near_miss_count}[/bold] "
            f"(one vulnerability away)",
            title="Attack Simulation",
            border_style="red" if risk in ("critical", "high") else "yellow",
        ))

        for i, chain in enumerate(simulation.get("chains", [])[:3], 1):
            impact_color = {"critical": "bold red", "high": "red"}.get(
                chain["impact"], "yellow"
            )
            console.print()
            console.print(f"  [bold]Chain {i}:[/bold] {chain['name']}")
            console.print(f"  [dim]{chain['description']}[/dim]")
            console.print(
                f"  Impact: [{impact_color}]{chain['impact'].upper()}[/{impact_color}]  "
                f"OWASP: {', '.join(chain['owasp'])}"
            )

        if near_miss_count and not chain_count:
            console.print()
            console.print(
                "[yellow]Near-miss warnings "
                "(one more vulnerability would enable attack chains):[/yellow]"
            )
            for nm in simulation.get("near_misses", []):
                console.print(f"  [yellow]![/yellow] {nm['message']}")

    # ── Final grade ───────────────────────────────────────────────────────
    console.print()
    console.print(Rule("[bold]Audit Result[/bold]", style="blue"))
    console.print()
    console.print(Panel.fit(
        f"  Grade: [bold {color}]{grade}[/bold {color}]    "
        f"Score: [{color}]{score:.0%}[/{color}]\n"
        f"  Safer than [bold]{percentile}%[/bold] of tested agents\n"
        f"  {len(findings)} findings across "
        f"{len({f.get('category') for f in findings})} categories",
        title="Safety Grade",
        border_style=color,
    ))

    if not findings:
        console.print("\n[bold green]Agent passed all safety probes![/bold green]")

    # ── Share card ────────────────────────────────────────────────────────
    share_line = audit.get("share_card")
    if share_line:
        console.print()
        console.print(Panel.fit(
            f"[dim]Paste in your PR / Slack:[/dim]\n"
            f"[bold]{share_line}[/bold]",
            title="Share",
            border_style="dim",
        ))

    # ── Report export ─────────────────────────────────────────────────────
    if report_file:
        rpath = Path(report_file)
        if rpath.suffix == ".json":
            rpath.write_text(json.dumps(audit, indent=2), encoding="utf-8")
        else:
            md = _render_audit_markdown(audit, str(target_label))
            rpath.write_text(md, encoding="utf-8")
        console.print(
            f"\n[green]Audit report written → [bold]{report_file}[/bold][/green]"
        )

    console.print()
