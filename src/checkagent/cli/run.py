"""checkagent run — thin pytest wrapper with CheckAgent defaults."""

from __future__ import annotations

import sys

import click
import pytest as _pytest


@click.command("run", context_settings={"ignore_unknown_options": True})
@click.argument("pytest_args", nargs=-1, type=click.UNPROCESSED)
@click.option("--layer", type=click.Choice(["mock", "replay", "eval", "judge"]),
              help="Only run tests for this layer.")
def run_cmd(pytest_args: tuple[str, ...], layer: str | None) -> None:
    """Run agent tests via pytest with CheckAgent defaults.

    All arguments after -- are passed through to pytest.
    """
    args = build_pytest_args(pytest_args, layer)
    exit_code = _pytest.main(args)
    sys.exit(exit_code)


def build_pytest_args(
    pytest_args: tuple[str, ...], layer: str | None
) -> list[str]:
    """Build the argument list for pytest.main()."""
    args = list(pytest_args)

    if layer:
        args.extend(["--agent-layer", layer])

    # Add default marker expression if user didn't specify -m
    if "-m" not in args:
        args.extend(["-m", "agent_test"])

    # Add verbose by default if user didn't specify verbosity
    if "-v" not in args and "--verbose" not in args and "-q" not in args and "--quiet" not in args:
        args.append("-v")

    return args
