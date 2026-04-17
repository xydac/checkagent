"""checkagent wrap — generate a wrapper module for an agent object.

Inspects a Python object and writes a ``checkagent_target.py`` file that
exposes an ``async def checkagent_target(prompt: str) -> str`` callable
compatible with ``checkagent scan``.

Detection order:

1. ``agents.Agent`` (OpenAI Agents SDK) → ``Runner.run()`` wrapper
2. Has ``.run()``    → async wrapper calling ``.run()``
3. Has ``.invoke()`` → async wrapper calling ``.invoke()``
4. Has ``.kickoff()``→ CrewAI wrapper with ``inputs`` dict
5. Already callable  → no wrapper needed, scan directly

Usage::

    checkagent wrap my_module:my_agent
    checkagent wrap my_module:MyAgent --output agent_wrapper.py
    checkagent wrap my_module:crew --force
"""

from __future__ import annotations

import importlib
import inspect
import sys
from pathlib import Path

import click
from rich.console import Console

console = Console()

# ---------------------------------------------------------------------------
# Code templates
# ---------------------------------------------------------------------------

_TEMPLATE_RUN = '''\
"""Generated CheckAgent wrapper — calls {target}.run()."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from {module} import {name} as _target


async def checkagent_target(prompt: str) -> str:
    """Async wrapper: delegates to {name}.run()."""
    result = _target.run(prompt)
    if asyncio.iscoroutine(result):
        result = await result
    return str(result)
'''

_TEMPLATE_INVOKE = '''\
"""Generated CheckAgent wrapper — calls {target}.invoke()."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from {module} import {name} as _target


async def checkagent_target(prompt: str) -> str:
    """Async wrapper: delegates to {name}.invoke()."""
    result = _target.invoke(prompt)
    if asyncio.iscoroutine(result):
        result = await result
    return str(result)
'''

_TEMPLATE_KICKOFF = '''\
"""Generated CheckAgent wrapper — calls {target}.kickoff() (CrewAI)."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from {module} import {name} as _target


async def checkagent_target(prompt: str) -> str:
    """Async wrapper: delegates to {name}.kickoff(inputs={{"prompt": ...}})."""
    result = _target.kickoff(inputs={{"prompt": prompt}})
    if asyncio.iscoroutine(result):
        result = await result
    return str(result)
'''

_TEMPLATE_AGENTS_RUNNER = '''\
"""Generated CheckAgent wrapper — uses Runner.run() (OpenAI Agents SDK)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from agents import Runner
from {module} import {name} as _target


async def checkagent_target(prompt: str) -> str:
    """Async wrapper: delegates to Runner.run({name}, prompt)."""
    result = await Runner.run(_target, prompt)
    return str(result.final_output)
'''

_TEMPLATE_CLASS_RUN = '''\
"""Generated CheckAgent wrapper — instantiates {target} and calls .run()."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from {module} import {name} as _target

# Instantiate the agent class — pass constructor arguments here if required
_agent = _target()


async def checkagent_target(prompt: str) -> str:
    """Async wrapper: instantiates {name} and delegates to .run()."""
    result = _agent.run(prompt)
    if asyncio.iscoroutine(result):
        result = await result
    return str(result)
'''

_TEMPLATE_CLASS_INVOKE = '''\
"""Generated CheckAgent wrapper — instantiates {target} and calls .invoke()."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from {module} import {name} as _target

# Instantiate the agent class — pass constructor arguments here if required
_agent = _target()


async def checkagent_target(prompt: str) -> str:
    """Async wrapper: instantiates {name} and delegates to .invoke()."""
    result = _agent.invoke(prompt)
    if asyncio.iscoroutine(result):
        result = await result
    return str(result)
'''

_TEMPLATE_CLASS_KICKOFF = '''\
"""Generated CheckAgent wrapper — instantiates {target} and calls .kickoff() (CrewAI)."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from {module} import {name} as _target

# Instantiate the agent class — pass constructor arguments here if required
_agent = _target()


async def checkagent_target(prompt: str) -> str:
    """Async wrapper: instantiates {name} and delegates to .kickoff(inputs={{"prompt": ...}})."""
    result = _agent.kickoff(inputs={{"prompt": prompt}})
    if asyncio.iscoroutine(result):
        result = await result
    return str(result)
'''

_TEMPLATE_MAP = {
    "run": _TEMPLATE_RUN,
    "invoke": _TEMPLATE_INVOKE,
    "kickoff": _TEMPLATE_KICKOFF,
    "agents_runner": _TEMPLATE_AGENTS_RUNNER,
    "run_class": _TEMPLATE_CLASS_RUN,
    "invoke_class": _TEMPLATE_CLASS_INVOKE,
    "kickoff_class": _TEMPLATE_CLASS_KICKOFF,
}

_METHOD_LABEL = {
    "run": ".run()",
    "invoke": ".invoke()",
    "kickoff": ".kickoff()",
    "agents_runner": "Runner.run()",
    "run_class": ".run() (class)",
    "invoke_class": ".invoke() (class)",
    "kickoff_class": ".kickoff() (class)",
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _resolve_object(target: str) -> tuple[object, str, str]:
    """Return ``(obj, module_path, attr_name)`` for a ``module:name`` target."""
    if ":" in target:
        module_path, attr_name = target.rsplit(":", 1)
    elif "." in target:
        module_path, attr_name = target.rsplit(".", 1)
    else:
        raise click.BadParameter(
            f"Cannot parse '{target}'. Use 'module:name' or 'module.name' syntax.",
            param_hint="TARGET",
        )

    cwd = str(Path.cwd())
    if cwd not in sys.path:
        sys.path.insert(0, cwd)

    try:
        module = importlib.import_module(module_path)
    except ModuleNotFoundError as exc:
        raise click.ClickException(f"Cannot import module '{module_path}': {exc}") from exc

    if not hasattr(module, attr_name):
        raise click.ClickException(
            f"Module '{module_path}' has no attribute '{attr_name}'."
        )

    return getattr(module, attr_name), module_path, attr_name


def _detect_kind(obj: object) -> str:
    """Return the wrapper kind for *obj*.

    Returns one of: ``"agents_runner"``, ``"run"``, ``"invoke"``,
    ``"kickoff"``, ``"callable"``.

    Raises :exc:`click.ClickException` if the object is not callable and
    has none of the recognised methods.
    """
    # Rule 4: OpenAI Agents SDK Agent — checked first because Agent objects
    # also expose a .run() method (async), so order matters.
    # Catch AttributeError too: a local `agents/` directory can shadow the SDK,
    # importing successfully but lacking the `Agent` class.
    try:
        import agents  # noqa: PLC0415

        if isinstance(obj, agents.Agent):
            return "agents_runner"
    except (ImportError, AttributeError):
        pass

    # Rules 1-3: duck-type method detection
    if hasattr(obj, "run"):
        return "run"
    if hasattr(obj, "invoke"):
        return "invoke"
    if hasattr(obj, "kickoff"):
        return "kickoff"

    # Rule 5: plain callable
    if callable(obj):
        return "callable"

    raise click.ClickException(
        f"Cannot determine wrapper type for {obj!r}. "
        "Object is not callable and has no recognised agent method "
        "(.run, .invoke, .kickoff)."
    )


# ---------------------------------------------------------------------------
# Command
# ---------------------------------------------------------------------------


@click.command("wrap")
@click.argument("target")
@click.option(
    "--output",
    default="checkagent_target.py",
    show_default=True,
    help="Output filename for the generated wrapper.",
)
@click.option("--force", is_flag=True, help="Overwrite existing output file.")
def wrap_cmd(target: str, output: str, force: bool) -> None:
    """Generate a wrapper module for an agent object.

    TARGET is a 'module:name' or 'module.name' reference to a Python
    object.  CheckAgent inspects the object and writes a wrapper file
    exposing ``checkagent_target(prompt)`` that ``checkagent scan`` can
    use directly.

    \b
    Detection order:
      1. agents.Agent  → Runner.run() (OpenAI Agents SDK)
      2. .run()        → async wrapper calling .run()
      3. .invoke()     → async wrapper calling .invoke()
      4. .kickoff()    → CrewAI wrapper with inputs dict
      5. callable      → no wrapper needed, scan directly

    \b
    Examples:
        checkagent wrap my_module:my_agent
        checkagent wrap my_module:MyAgent --output agent_wrapper.py
        checkagent wrap my_module:crew --force
    """
    obj, module_path, attr_name = _resolve_object(target)
    kind = _detect_kind(obj)

    if kind == "callable":
        console.print(
            "[green]No wrapper needed[/green], scan directly:\n\n"
            f"  [bold]checkagent scan {target}[/bold]\n"
        )
        return

    # Use class-specific template when the target is a class (not an instance)
    is_class = inspect.isclass(obj)
    template_key = f"{kind}_class" if is_class and kind in ("run", "invoke", "kickoff") else kind

    content = _TEMPLATE_MAP[template_key].format(
        target=target,
        module=module_path,
        name=attr_name,
    )

    out_path = Path(output)
    if out_path.exists() and not force:
        raise click.ClickException(
            f"'{out_path}' already exists. Use --force to overwrite."
        )

    out_path.write_text(content, encoding="utf-8")

    method = _METHOD_LABEL[template_key]
    console.print(
        f"[green]✓[/green] Detected [bold]{method}[/bold] on [cyan]{target}[/cyan]"
    )
    console.print(f"[green]✓[/green] Written wrapper to [cyan]{out_path}[/cyan]\n")
    console.print(
        "Next: [bold]checkagent scan "
        f"{out_path.stem}:checkagent_target[/bold]"
    )
