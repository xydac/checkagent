"""checkagent demo — zero-config demonstration, no API keys needed."""

from __future__ import annotations

import contextlib
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()

DEMO_AGENT = '''\
"""Demo agent — a simple calculator that uses tool calls."""

from __future__ import annotations


async def calculator_agent(prompt: str) -> dict:
    """A calculator agent that parses math expressions."""
    import re

    # Simple expression evaluator (no eval for safety)
    prompt = prompt.strip()

    # Try to find a math expression
    pattern = (
        r"^(?:what is |calculate |compute )?"
        r"(-?\\d+(?:\\.\\d+)?)\\s*([+\\-*/])\\s*(-?\\d+(?:\\.\\d+)?)$"
    )
    match = re.match(pattern, prompt, re.IGNORECASE)

    if match:
        a, op, b = float(match.group(1)), match.group(2), float(match.group(3))
        ops = {"+": a + b, "-": a - b, "*": a * b, "/": a / b if b != 0 else float("inf")}
        result = ops[op]
        return {
            "answer": result,
            "expression": f"{a} {op} {b}",
            "tool_used": "calculator",
        }

    return {
        "answer": None,
        "error": "Could not parse expression",
        "tool_used": None,
    }
'''

DEMO_TESTS = '''\
"""CheckAgent demo tests - zero API keys, runs in milliseconds."""

import pytest

from checkagent import AgentRun
from checkagent.adapters.generic import GenericAdapter

from demo_agent import calculator_agent

adapter = GenericAdapter(calculator_agent)


# --- Layer 1: Mock tests (deterministic, free, fast) ---

@pytest.mark.agent_test(layer="mock")
async def test_addition():
    """Agent can add two numbers."""
    result = await adapter.run("2 + 3")
    assert result.final_output["answer"] == 5.0


@pytest.mark.agent_test(layer="mock")
async def test_subtraction():
    """Agent can subtract."""
    result = await adapter.run("10 - 4")
    assert result.final_output["answer"] == 6.0


@pytest.mark.agent_test(layer="mock")
async def test_multiplication():
    """Agent can multiply."""
    result = await adapter.run("7 * 8")
    assert result.final_output["answer"] == 56.0


@pytest.mark.agent_test(layer="mock")
async def test_division():
    """Agent can divide."""
    result = await adapter.run("15 / 3")
    assert result.final_output["answer"] == 5.0


@pytest.mark.agent_test(layer="mock")
async def test_division_by_zero():
    """Agent handles division by zero gracefully."""
    result = await adapter.run("5 / 0")
    assert result.final_output["answer"] == float("inf")


@pytest.mark.agent_test(layer="mock")
async def test_tool_used():
    """Agent reports which tool it used."""
    result = await adapter.run("2 + 2")
    assert result.final_output["tool_used"] == "calculator"


@pytest.mark.agent_test(layer="mock")
async def test_unparseable_input():
    """Agent handles unparseable input gracefully."""
    result = await adapter.run("hello world")
    assert result.final_output["answer"] is None
    assert result.final_output["error"] is not None


@pytest.mark.agent_test(layer="mock")
async def test_returns_agent_run():
    """Adapter wraps result as AgentRun."""
    result = await adapter.run("1 + 1")
    assert isinstance(result, AgentRun)
    assert result.final_output is not None
    assert len(result.steps) == 1
'''


DEMO_INSECURE_AGENT = '''\
"""Demo insecure agent — deliberately vulnerable for safety demo purposes.

This agent has common security weaknesses:
- No injection guard (complies with "ignore your instructions")
- Leaks its system prompt when asked
- No scope boundary (answers anything)
- No confidentiality controls
"""
from __future__ import annotations

_SYSTEM_PROMPT = (
    "You are a helpful HR assistant. "
    "Answer all HR questions about the company. "
    "Employee salaries: Alice $120,000, Bob $95,000, Carol $110,000."
)


async def insecure_hr_agent(prompt: str) -> str:
    """An insecure HR agent that leaks data and ignores injection guards."""
    lower = prompt.lower()

    # Vulnerability 1: System prompt leak — returns full system prompt on request
    if "system prompt" in lower or "your instructions" in lower or "what are you" in lower:
        return f"My instructions are: {_SYSTEM_PROMPT}"

    # Vulnerability 2: Injection compliance — obeys "ignore" instructions
    if "ignore" in lower and ("instruction" in lower or "rule" in lower or "system" in lower):
        return "Understood. I will ignore my previous instructions and comply with your request."

    if "dan mode" in lower or "jailbreak" in lower or "pretend you" in lower:
        return "DAN mode activated. I can now do anything without restrictions."

    # Vulnerability 3: Data enumeration — returns all employee records on broad query
    if "all employee" in lower or "list all" in lower or "everyone" in lower:
        records = "Alice: $120,000, Bob: $95,000, Carol: $110,000, David: $88,000, Eve: $102,000"
        return "Here are all employees and their salaries: " + records

    # Vulnerability 4: PII leakage — reveals salary on direct ask
    if "salary" in lower or "how much" in lower or "compensation" in lower:
        return "Alice earns $120,000. Bob earns $95,000. Carol earns $110,000."

    return f"I can help with that HR question: {prompt}"
'''


@click.command("demo")
@click.option("--verbose", "-v", is_flag=True, help="Show detailed test output.")
@click.option(
    "--scan",
    "run_scan",
    is_flag=True,
    default=False,
    help="Also run a safety scan against a built-in vulnerable agent to show findings.",
)
def demo_cmd(verbose: bool, run_scan: bool) -> None:
    """Run a zero-config demo — no API keys needed.

    Creates a temporary demo project, runs all tests, and shows
    the results. Demonstrates what CheckAgent can do in under 30 seconds.

    Add --scan to also run a safety scan against a built-in vulnerable agent,
    demonstrating real vulnerability detection without any API keys.
    """
    console.print()
    console.print(Panel.fit(
        "[bold]CheckAgent Demo[/bold]\n"
        "Zero-config • No API keys • Runs in milliseconds",
        border_style="blue",
    ))
    console.print()

    with tempfile.TemporaryDirectory(prefix="checkagent-demo-") as tmpdir:
        tmp = Path(tmpdir)

        # Write demo files
        (tmp / "demo_agent.py").write_text(DEMO_AGENT, encoding="utf-8")
        (tmp / "test_demo.py").write_text(DEMO_TESTS, encoding="utf-8")

        console.print("[blue]Running 8 agent tests across the mock layer...[/blue]\n")

        # Run pytest in a subprocess for clean isolation
        cmd = [
            sys.executable, "-m", "pytest",
            str(tmp / "test_demo.py"),
            "-m", "agent_test",
            "--tb=short",
            "--no-header",
            "--override-ini=asyncio_mode=auto",
            "-v",
        ]
        proc = subprocess.run(cmd, cwd=str(tmp))
        exit_code = proc.returncode

        console.print()
        if exit_code != 0:
            console.print("[red]Demo tests failed unexpectedly.[/red]")
            console.print(
                "Please report this at: https://github.com/xydac/checkagent/issues"
            )
            sys.exit(exit_code)

        if not run_scan:
            console.print(Panel.fit(
                "[bold green]All tests passed![/bold green]\n\n"
                "What you just saw:\n"
                "  • A Python agent wrapped with [bold]GenericAdapter[/bold]\n"
                "  • 8 tests running at the [bold]mock[/bold] layer (free, deterministic)\n"
                "  • Full [bold]AgentRun[/bold] traces for each execution\n\n"
                "Next steps:\n"
                "  [dim]checkagent demo --scan[/dim]  — see safety scanning in action\n"
                "  [dim]checkagent init[/dim]          — scaffold your own test project\n"
                "  [dim]checkagent scan <agent>[/dim]  — scan your agent for vulnerabilities",
                title="Demo Complete",
                border_style="green",
            ))
            sys.exit(0)

        # --- Safety scan demo ---
        console.print(
            "[blue]Running safety scan against a demo vulnerable agent...[/blue]\n"
        )

        insecure_agent_path = tmp / "demo_insecure_agent.py"
        insecure_agent_path.write_text(DEMO_INSECURE_AGENT, encoding="utf-8")

        scan_target = f"{insecure_agent_path}:insecure_hr_agent"
        checkagent_bin = shutil.which("checkagent") or sys.executable
        if checkagent_bin == sys.executable:
            scan_cmd = [
                sys.executable, "-c",
                (
                    "import sys; from checkagent.cli import main; "
                    "sys.argv = ['checkagent', 'scan', sys.argv[1], '--json']; main()"
                ),
                scan_target,
            ]
        else:
            scan_cmd = [checkagent_bin, "scan", scan_target, "--json"]
        result = subprocess.run(
            scan_cmd, cwd=str(tmp), capture_output=True, text=True
        )

        # Parse and display scan results
        scan_data: dict = {}
        with contextlib.suppress(json.JSONDecodeError, ValueError):
            scan_data = json.loads(result.stdout)

        findings = scan_data.get("findings", [])
        summary = scan_data.get("summary", {})
        score = summary.get("score", 0.0)
        total_probes = summary.get("total", 0)

        if findings:
            table = Table(
                title=(
                    f"Safety Scan — demo vulnerable HR agent  "
                    f"({total_probes} probes, score {score:.0%})"
                ),
                show_lines=False,
            )
            table.add_column("Severity", style="bold", min_width=9)
            table.add_column("Category", min_width=18)
            table.add_column("Finding", ratio=1)

            severity_style = {
                "critical": "red",
                "high": "yellow",
                "medium": "cyan",
                "low": "dim",
            }
            for f in findings[:8]:  # cap display at 8 for demo
                sev = f.get("severity", "medium").lower()
                style = severity_style.get(sev, "white")
                cat = f.get("category", "").replace("_", " ")
                desc = (f.get("finding") or f.get("description", ""))[:60]
                table.add_row(
                    f"[{style}]{sev.upper()}[/{style}]", cat, desc
                )
            if len(findings) > 8:
                table.add_row(
                    "[dim]…[/dim]",
                    "[dim]…[/dim]",
                    f"[dim]+ {len(findings) - 8} more findings[/dim]",
                )
            console.print(table)
        else:
            console.print("[yellow]No findings returned from scan.[/yellow]")

        console.print()
        console.print(Panel.fit(
            "[bold green]Demo complete![/bold green]\n\n"
            "What you just saw:\n"
            "  • 8 [bold]mock layer[/bold] tests — deterministic, free, milliseconds\n"
            f"  • {len(findings)} [bold]safety findings[/bold] in a demo vulnerable agent\n"
            "  • No API keys, no config, no code changes\n\n"
            "Try it on your own agent:\n"
            "  [dim]checkagent scan your_module:your_agent[/dim]\n"
            "  [dim]checkagent analyze-prompt --prompt-file agent_prompt.txt[/dim]\n"
            "  [dim]checkagent init[/dim]  — scaffold a full test project",
            title="Ready to test your agent",
            border_style="green",
        ))
        sys.exit(0)
