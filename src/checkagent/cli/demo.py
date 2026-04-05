"""checkagent demo — zero-config demonstration, no API keys needed."""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

import click
from rich.console import Console
from rich.panel import Panel

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
"""CheckAgent demo tests — zero API keys, runs in milliseconds."""

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


@click.command("demo")
@click.option("--verbose", "-v", is_flag=True, help="Show detailed test output.")
def demo_cmd(verbose: bool) -> None:
    """Run a zero-config demo — no API keys needed.

    Creates a temporary demo project, runs all tests, and shows
    the results. Demonstrates what CheckAgent can do in under 30 seconds.
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
        (tmp / "demo_agent.py").write_text(DEMO_AGENT)
        (tmp / "test_demo.py").write_text(DEMO_TESTS)

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
        if exit_code == 0:
            console.print(Panel.fit(
                "[bold green]All tests passed![/bold green]\n\n"
                "What you just saw:\n"
                "  • A Python agent wrapped with [bold]GenericAdapter[/bold]\n"
                "  • 8 tests running at the [bold]mock[/bold] layer (free, deterministic)\n"
                "  • Full [bold]AgentRun[/bold] traces for each execution\n\n"
                "Next steps:\n"
                "  [dim]checkagent init[/dim]     — scaffold your own test project\n"
                "  [dim]checkagent run[/dim]      — run your agent tests\n"
                "  [dim]pytest -v[/dim]           — use pytest directly",
                title="Demo Complete",
                border_style="green",
            ))
        else:
            console.print("[red]Demo tests failed unexpectedly.[/red]")
            console.print("Please report this at: https://github.com/xydac/checkagent/issues")

        sys.exit(exit_code)
