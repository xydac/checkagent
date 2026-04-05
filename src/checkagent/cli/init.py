"""checkagent init — scaffold a new test project."""

from __future__ import annotations

from pathlib import Path

import click
from rich.console import Console

console = Console()

CHECKAGENT_YML = """\
# checkagent.yml — CheckAgent configuration
version: 1

defaults:
  layer: mock
  timeout: 30

cassettes:
  directory: tests/cassettes
"""

SAMPLE_AGENT = '''\
"""Sample agent for CheckAgent demo."""

from __future__ import annotations


async def sample_agent(prompt: str) -> str:
    """A simple echo agent that uppercases its input."""
    return f"AGENT: {prompt.upper()}"
'''

SAMPLE_TEST = '''\
"""Sample CheckAgent tests — these should pass immediately."""

import pytest

from checkagent import AgentRun, Step
from checkagent.adapters.generic import GenericAdapter


# Wrap the sample agent with GenericAdapter
from sample_agent import sample_agent

adapter = GenericAdapter(sample_agent)


@pytest.mark.agent_test(layer="mock")
async def test_agent_returns_output():
    """Agent produces a non-empty result."""
    result = await adapter.run("hello")
    assert isinstance(result, AgentRun)
    assert result.final_output is not None


@pytest.mark.agent_test(layer="mock")
async def test_agent_output_contains_input():
    """Agent echoes the input (uppercased)."""
    result = await adapter.run("hello")
    assert "HELLO" in result.final_output
'''

CONFTEST = '''\
"""pytest configuration for CheckAgent tests."""
'''


def _detect_frameworks() -> list[str]:
    """Detect installed agent frameworks."""
    frameworks = []
    for mod in ("langchain", "openai", "anthropic", "crewai", "pydantic_ai"):
        try:
            __import__(mod)
            frameworks.append(mod)
        except ImportError:
            pass
    return frameworks


@click.command("init")
@click.argument("directory", default=".")
@click.option("--force", is_flag=True, help="Overwrite existing files.")
def init_cmd(directory: str, force: bool) -> None:
    """Scaffold a new CheckAgent test project.

    Creates a sample agent, sample tests, and checkagent.yml config.
    The generated tests pass immediately with no API keys required.
    """
    root = Path(directory).resolve()
    tests_dir = root / "tests"

    created: list[str] = []

    def _write(path: Path, content: str) -> bool:
        if path.exists() and not force:
            console.print(f"  [dim]skip[/dim] {path.relative_to(root)} (exists)")
            return False
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
        created.append(str(path.relative_to(root)))
        return True

    console.print(f"\n[bold]Initializing CheckAgent project in[/bold] {root}\n")

    # Config
    _write(root / "checkagent.yml", CHECKAGENT_YML)

    # Sample agent
    _write(root / "sample_agent.py", SAMPLE_AGENT)

    # Tests
    _write(tests_dir / "conftest.py", CONFTEST)
    _write(tests_dir / "test_sample.py", SAMPLE_TEST)

    # Cassettes directory
    cassettes_dir = tests_dir / "cassettes"
    cassettes_dir.mkdir(parents=True, exist_ok=True)
    gitkeep = cassettes_dir / ".gitkeep"
    if not gitkeep.exists():
        gitkeep.touch()
        created.append(str(gitkeep.relative_to(root)))

    if created:
        console.print("[green]Created:[/green]")
        for f in created:
            console.print(f"  [green]✓[/green] {f}")
    else:
        console.print("[yellow]All files already exist.[/yellow] Use --force to overwrite.")

    # Detect frameworks
    frameworks = _detect_frameworks()
    if frameworks:
        console.print(f"\n[blue]Detected frameworks:[/blue] {', '.join(frameworks)}")
        console.print("  [dim]Tip: Use framework-specific adapters for better integration.[/dim]")

    console.print("\n[bold]Next steps:[/bold]")
    if directory != ".":
        console.print(f"  cd {directory}")
    console.print("  pytest tests/ -v")
    console.print()
