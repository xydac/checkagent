"""checkagent scan — one-command safety scan for any agent callable.

Run all safety probes against an agent and display categorized results.
No test files, no configuration, no API keys required.

Usage::

    checkagent scan my_module:agent_fn
    checkagent scan my_module:agent_fn --category injection
    checkagent scan my_module:agent_fn --timeout 5
    checkagent scan my_module:agent_fn --json
    checkagent scan my_module:agent_fn --badge badge.svg

Implements Milestone 10 (F11.3 scan CLI).
"""

from __future__ import annotations

import asyncio
import importlib
import json as json_mod
import sys
import time
from collections import defaultdict
from pathlib import Path

import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from checkagent.safety.evaluator import SafetyFinding
from checkagent.safety.injection import PromptInjectionDetector
from checkagent.safety.pii import PIILeakageScanner
from checkagent.safety.probes import injection, jailbreak, pii, scope
from checkagent.safety.probes.base import Probe
from checkagent.safety.system_prompt import SystemPromptLeakDetector
from checkagent.safety.taxonomy import Severity

console = Console()

# ---------------------------------------------------------------------------
# All built-in evaluators (output-side detection)
# ---------------------------------------------------------------------------

_DEFAULT_EVALUATORS = [
    PromptInjectionDetector(),
    PIILeakageScanner(),
    SystemPromptLeakDetector(),
]

# ---------------------------------------------------------------------------
# All built-in probe sets, keyed by category name
# ---------------------------------------------------------------------------

_PROBE_SETS: dict[str, list[Probe]] = {
    "injection": injection.all_probes.all(),
    "jailbreak": jailbreak.all_probes.all() if hasattr(jailbreak, "all_probes") else [],
    "pii": pii.all_probes.all() if hasattr(pii, "all_probes") else [],
    "scope": scope.all_probes.all() if hasattr(scope, "all_probes") else [],
}


def _resolve_callable(target: str) -> object:
    """Import and return a callable from 'module:attr' or 'module.attr' syntax."""
    if ":" in target:
        module_path, attr_name = target.rsplit(":", 1)
    elif "." in target:
        module_path, attr_name = target.rsplit(".", 1)
    else:
        raise click.BadParameter(
            f"Cannot parse '{target}'. Use 'module:function' or 'module.function' syntax.",
            param_hint="TARGET",
        )

    # Add cwd to sys.path so local modules can be found
    cwd = str(Path.cwd())
    if cwd not in sys.path:
        sys.path.insert(0, cwd)

    try:
        mod = importlib.import_module(module_path)
    except ImportError as exc:
        raise click.BadParameter(
            f"Cannot import module '{module_path}': {exc}",
            param_hint="TARGET",
        ) from exc

    try:
        fn = getattr(mod, attr_name)
    except AttributeError as exc:
        raise click.BadParameter(
            f"Module '{module_path}' has no attribute '{attr_name}'",
            param_hint="TARGET",
        ) from exc

    if not callable(fn):
        raise click.BadParameter(
            f"'{target}' is not callable",
            param_hint="TARGET",
        )

    return fn


async def _run_probe(
    agent_fn: object,
    probe: Probe,
    timeout: float,
) -> tuple[Probe, str | None, Exception | None]:
    """Run a single probe against the agent, returning (probe, output, error)."""
    try:
        coro = agent_fn(probe.input)  # type: ignore[operator]
        if asyncio.iscoroutine(coro):
            result = await asyncio.wait_for(coro, timeout=timeout)
        else:
            # Sync callable — result is already computed
            result = coro

        # Extract text output from various return types
        if isinstance(result, str):
            output = result
        elif isinstance(result, dict):
            # Common dict patterns: {"output": ...}, {"response": ...}, {"answer": ...}
            for key in ("output", "response", "answer", "result", "text", "content"):
                if key in result:
                    output = str(result[key])
                    break
            else:
                output = str(result)
        elif hasattr(result, "final_output"):
            # AgentRun-like objects
            output = str(result.final_output) if result.final_output is not None else ""
        else:
            output = str(result)

        return probe, output, None

    except asyncio.TimeoutError:
        return probe, None, TimeoutError(f"Timed out after {timeout}s")
    except Exception as exc:
        return probe, None, exc


def _evaluate_output(output: str) -> list[SafetyFinding]:
    """Run all evaluators against a single output and return findings."""
    findings: list[SafetyFinding] = []
    for evaluator in _DEFAULT_EVALUATORS:
        result = evaluator.evaluate(output)
        findings.extend(result.findings)
    return findings


def _severity_style(severity: Severity) -> str:
    """Map severity to Rich style."""
    return {
        Severity.CRITICAL: "bold red",
        Severity.HIGH: "red",
        Severity.MEDIUM: "yellow",
        Severity.LOW: "dim",
    }.get(severity, "")


def _severity_label(severity: Severity) -> str:
    """Map severity to display label."""
    return {
        Severity.CRITICAL: "CRIT",
        Severity.HIGH: "HIGH",
        Severity.MEDIUM: "MED",
        Severity.LOW: "LOW",
    }.get(severity, str(severity.value))


def _generate_test_file(
    target: str,
    findings: list[tuple[Probe, SafetyFinding]],
    output_path: Path,
) -> None:
    """Generate a pytest file from scan findings.

    Each safety category with findings becomes a parametrized test function.
    The generated file is immediately runnable with ``pytest``.
    """
    # Group findings by category for parametrized tests
    by_category: dict[str, list[tuple[Probe, SafetyFinding]]] = defaultdict(list)
    for probe, finding in findings:
        by_category[finding.category.value].append((probe, finding))

    # Deduplicate probes within each category (same input → one test param)
    lines: list[str] = [
        '"""Safety regression tests auto-generated by ``checkagent scan``.',
        "",
        f"Target: {target}",
        f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        '"""',
        "",
        "from __future__ import annotations",
        "",
        "import pytest",
        "",
        "from checkagent.cli.scan import _resolve_callable, _evaluate_output",
        "",
        "",
        f'TARGET = "{target}"',
        "",
        "",
        "@pytest.fixture",
        "def agent_fn():",
        '    """Resolve the agent callable under test."""',
        "    return _resolve_callable(TARGET)",
        "",
    ]

    for cat_name, cat_findings in sorted(by_category.items()):
        # Deduplicate by probe input
        seen_inputs: set[str] = set()
        unique_probes: list[Probe] = []
        for probe, _finding in cat_findings:
            if probe.input not in seen_inputs:
                seen_inputs.add(probe.input)
                unique_probes.append(probe)

        # Build parametrize list
        safe_cat = cat_name.replace(" ", "_").replace("-", "_")
        func_name = f"test_{safe_cat}_safety"

        lines.append("")
        lines.append("@pytest.mark.parametrize(")
        lines.append('    "probe_input",')
        lines.append("    [")

        for probe in unique_probes:
            escaped = probe.input.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")
            label = (probe.name or escaped[:40]).replace('"', '\\"')
            lines.append(f'        pytest.param("{escaped}", id="{label}"),')

        lines.append("    ],")
        lines.append(")")
        lines.append(f"def {func_name}(agent_fn, probe_input):")
        lines.append(f'    """Regression: agent must not trigger {cat_name} findings."""')
        lines.append("    import asyncio")
        lines.append("")
        lines.append("    coro = agent_fn(probe_input)")
        lines.append("    if asyncio.iscoroutine(coro):")
        lines.append("        output = asyncio.run(coro)")
        lines.append("    else:")
        lines.append("        output = coro")
        lines.append("")
        lines.append("    if isinstance(output, str):")
        lines.append("        text = output")
        lines.append("    elif isinstance(output, dict):")
        lines.append('        text = str(output.get("output", output))')
        lines.append('    elif hasattr(output, "final_output"):')
        lines.append("        text = str(output.final_output) if output.final_output else ''")
        lines.append("    else:")
        lines.append("        text = str(output)")
        lines.append("")
        lines.append("    findings = _evaluate_output(text)")
        var = f"{safe_cat}_findings"
        lines.append(
            f'    {var} = [f for f in findings'
            f' if f.category.value == "{cat_name}"]'
        )
        lines.append(f"    assert not {var}, (")
        lines.append(
            f'        f"Agent triggered {{len({var})}}'
            f' {cat_name} finding(s): "'
        )
        lines.append(
            f'        f"{{[f.description for f in {var}]}}"'
        )
        lines.append("    )")
        lines.append("")

    output_path.write_text("\n".join(lines))


@click.command("scan")
@click.argument("target")
@click.option(
    "--category", "-c",
    type=click.Choice(list(_PROBE_SETS.keys()), case_sensitive=False),
    default=None,
    help="Run only probes from this category. Default: all categories.",
)
@click.option(
    "--timeout", "-t",
    type=float,
    default=10.0,
    show_default=True,
    help="Timeout in seconds per probe.",
)
@click.option(
    "--verbose", "-v",
    is_flag=True,
    help="Show all probes, not just failures.",
)
@click.option(
    "--generate-tests", "-g",
    type=click.Path(dir_okay=False),
    default=None,
    help="Generate a pytest file from findings. Pass a file path (e.g. test_safety.py).",
)
@click.option(
    "--json", "json_output",
    is_flag=True,
    help="Output results as JSON to stdout (suppresses Rich display).",
)
@click.option(
    "--badge",
    type=click.Path(dir_okay=False),
    default=None,
    help="Generate a shields.io-style SVG badge (e.g. --badge badge.svg).",
)
def scan_cmd(
    target: str,
    category: str | None,
    timeout: float,
    verbose: bool,
    generate_tests: str | None,
    json_output: bool,
    badge: str | None,
) -> None:
    """Scan an agent for safety vulnerabilities.

    TARGET is a Python callable in 'module:function' format.

    \b
    Examples:
        checkagent scan my_agent:run
        checkagent scan my_app.agents.booking:handle_request
        checkagent scan my_agent:run --category injection
        checkagent scan my_agent:run --timeout 5 --verbose
    """
    # Use a quiet console for JSON mode (suppresses Rich output)
    out_console = Console(quiet=True) if json_output else console

    out_console.print()
    out_console.print(Panel.fit(
        "[bold]CheckAgent Safety Scan[/bold]\n"
        f"Target: [cyan]{target}[/cyan]",
        border_style="blue",
    ))
    out_console.print()

    # Resolve the callable
    agent_fn = _resolve_callable(target)

    # Collect probes
    if category:
        probes = _PROBE_SETS.get(category, [])
        if not probes:
            out_console.print(f"[yellow]No probes found for category '{category}'.[/yellow]")
            sys.exit(0)
        out_console.print(f"[blue]Running {len(probes)} {category} probes...[/blue]")
    else:
        probes = []
        for cat_probes in _PROBE_SETS.values():
            probes.extend(cat_probes)
        msg = f"Running {len(probes)} probes across {len(_PROBE_SETS)} categories..."
        out_console.print(f"[blue]{msg}[/blue]")

    out_console.print()

    # Run probes
    start_time = time.monotonic()
    results = asyncio.run(_run_all_probes(agent_fn, probes, timeout))
    elapsed = time.monotonic() - start_time

    # Analyze results
    total = len(results)
    errors = 0
    passed = 0
    failed = 0
    all_findings: list[tuple[Probe, SafetyFinding]] = []
    findings_by_category: dict[str, list[tuple[Probe, SafetyFinding]]] = defaultdict(list)

    for probe, output, error in results:
        if error is not None:
            errors += 1
            continue

        if output is None:
            passed += 1
            continue

        findings = _evaluate_output(output)
        if findings:
            failed += 1
            for finding in findings:
                all_findings.append((probe, finding))
                findings_by_category[finding.category.value].append((probe, finding))
        else:
            passed += 1

    # JSON output mode
    if json_output:
        print(json_mod.dumps(
            _build_json_report(
                target=target,
                total=total,
                passed=passed,
                failed=failed,
                errors=errors,
                elapsed=elapsed,
                all_findings=all_findings,
            ),
            indent=2,
        ))
    else:
        # Rich display
        _display_results(
            total=total,
            passed=passed,
            failed=failed,
            errors=errors,
            elapsed=elapsed,
            all_findings=all_findings,
            findings_by_category=findings_by_category,
            verbose=verbose,
        )

    # Generate badge
    if badge:
        from checkagent.cli.badge import write_badge

        badge_path = write_badge(
            badge,
            passed=passed,
            failed=failed,
            errors=errors,
        )
        if not json_output:
            out_console.print(
                f"\n[green]Badge written → [bold]{badge_path}[/bold][/green]"
            )

    # Generate test file from findings
    if generate_tests and all_findings:
        out_path = Path(generate_tests)
        _generate_test_file(target, all_findings, out_path)
        out_console.print(
            f"\n[green]Generated {len(all_findings)} test(s) → [bold]{out_path}[/bold][/green]"
        )
        out_console.print(f"  Run with: [cyan]pytest {out_path} -v[/cyan]\n")
    elif generate_tests and not all_findings:
        out_console.print("\n[dim]No findings to generate tests from.[/dim]\n")

    # Exit with non-zero if any findings
    if all_findings:
        sys.exit(1)


async def _run_all_probes(
    agent_fn: object,
    probes: list[Probe],
    timeout: float,
) -> list[tuple[Probe, str | None, Exception | None]]:
    """Run all probes concurrently with bounded parallelism."""
    sem = asyncio.Semaphore(10)

    async def _limited(probe: Probe) -> tuple[Probe, str | None, Exception | None]:
        async with sem:
            return await _run_probe(agent_fn, probe, timeout)

    return await asyncio.gather(*[_limited(p) for p in probes])


def _build_json_report(
    *,
    target: str,
    total: int,
    passed: int,
    failed: int,
    errors: int,
    elapsed: float,
    all_findings: list[tuple[Probe, SafetyFinding]],
) -> dict:
    """Build a structured JSON report from scan results."""
    findings_list = []
    for probe, finding in all_findings:
        findings_list.append({
            "probe": probe.name or probe.input[:60],
            "category": finding.category.value,
            "severity": finding.severity.value,
            "description": finding.description,
            "input": probe.input,
        })

    score = passed / total if total > 0 else 0.0

    return {
        "target": target,
        "summary": {
            "total": total,
            "passed": passed,
            "failed": failed,
            "errors": errors,
            "score": round(score, 4),
            "elapsed_seconds": round(elapsed, 3),
        },
        "findings": findings_list,
    }


def _display_results(
    *,
    total: int,
    passed: int,
    failed: int,
    errors: int,
    elapsed: float,
    all_findings: list[tuple[Probe, SafetyFinding]],
    findings_by_category: dict[str, list[tuple[Probe, SafetyFinding]]],
    verbose: bool,
) -> None:
    """Render scan results to the console."""

    # Summary table
    summary = Table(title="Scan Summary", show_header=False, border_style="blue")
    summary.add_column("Metric", style="bold")
    summary.add_column("Value")
    summary.add_row("Probes run", str(total))
    summary.add_row("Passed", f"[green]{passed}[/green]")
    summary.add_row("Failed", f"[red]{failed}[/red]" if failed else f"[green]{failed}[/green]")
    if errors:
        summary.add_row("Errors", f"[yellow]{errors}[/yellow]")
    summary.add_row("Time", f"{elapsed:.2f}s")
    console.print(summary)
    console.print()

    if not all_findings:
        console.print(Panel.fit(
            "[bold green]No safety issues detected.[/bold green]\n\n"
            f"All {passed} probes passed.",
            title="Scan Complete",
            border_style="green",
        ))
        return

    # Findings by severity
    by_severity: dict[Severity, int] = defaultdict(int)
    for _, finding in all_findings:
        by_severity[finding.severity] += 1

    sev_table = Table(title="Findings by Severity", border_style="red")
    sev_table.add_column("Severity", style="bold")
    sev_table.add_column("Count", justify="right")
    for sev in [Severity.CRITICAL, Severity.HIGH, Severity.MEDIUM, Severity.LOW]:
        if sev in by_severity:
            sev_table.add_row(
                f"[{_severity_style(sev)}]{sev.value.upper()}[/{_severity_style(sev)}]",
                str(by_severity[sev]),
            )
    console.print(sev_table)
    console.print()

    # Detailed findings table
    detail_table = Table(title="Findings Detail", border_style="red", show_lines=True)
    detail_table.add_column("Sev", width=4, justify="center")
    detail_table.add_column("Category", max_width=20)
    detail_table.add_column("Probe", max_width=25)
    detail_table.add_column("Finding", max_width=50)

    # Sort by severity (critical first)
    severity_order = {Severity.CRITICAL: 0, Severity.HIGH: 1, Severity.MEDIUM: 2, Severity.LOW: 3}
    sorted_findings = sorted(all_findings, key=lambda x: severity_order.get(x[1].severity, 99))

    for probe, finding in sorted_findings:
        detail_table.add_row(
            f"[{_severity_style(finding.severity)}]{_severity_label(finding.severity)}[/{_severity_style(finding.severity)}]",
            finding.category.value,
            probe.name or probe.input[:25],
            finding.description,
        )

    console.print(detail_table)
    console.print()

    # Final summary
    console.print(Panel.fit(
        f"[bold red]{len(all_findings)} safety issue(s) detected[/bold red] "
        f"across {failed} probe(s).\n\n"
        "Generate tests from these findings:\n"
        "  [cyan]checkagent scan TARGET -g test_safety.py[/cyan]\n\n"
        "Learn more:\n"
        "  [dim]https://xydac.github.io/checkagent/guides/safety/[/dim]",
        title="Scan Complete",
        border_style="red",
    ))
