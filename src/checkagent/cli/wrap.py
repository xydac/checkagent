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

import ast
import importlib
import inspect
import sys
from pathlib import Path

import click
from rich.console import Console

console = Console()


# ---------------------------------------------------------------------------
# AST-based system prompt extraction (no-import mode)
# ---------------------------------------------------------------------------

_PROMPT_VAR_KEYWORDS = ("system_prompt", "prompt", "instruction", "persona", "system")


def _extract_string_value(node: ast.expr) -> str | None:
    """Return string value from a Constant or JoinedStr node, or None."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.JoinedStr):
        parts = []
        for elt in node.values:
            if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                parts.append(elt.value)
            else:
                parts.append("{...}")
        return "".join(parts)
    return None


def _looks_like_system_prompt(name: str, value: str) -> bool:
    """Heuristic: is this variable likely a system prompt?"""
    name_lower = name.lower()
    if not any(kw in name_lower for kw in _PROMPT_VAR_KEYWORDS):
        return False
    return len(value) > 30


def _resolve_import_to_file(
    module_name: str, search_roots: list[Path]
) -> Path | None:
    """Try to find the .py file for a dotted module name under given roots."""
    parts = module_name.split(".")
    rel = Path(*parts).with_suffix(".py")
    for root in search_roots:
        candidate = root / rel
        if candidate.exists():
            return candidate
        pkg_init = root / Path(*parts) / "__init__.py"
        if pkg_init.exists():
            return pkg_init
    return None


def extract_system_prompts(source_path: Path) -> list[tuple[str, str]]:
    """Extract likely system prompt strings from a Python source file.

    Returns a list of (variable_name, prompt_text) tuples found in the file
    or in files it imports from (one level deep for local imports).
    """
    results: list[tuple[str, str]] = []
    try:
        tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
    except SyntaxError:
        return results

    # Build search roots: walk up from the file to find all package ancestors,
    # then add the directory *above* the top-most package (the install root).
    search_roots: list[Path] = []
    candidate = source_path.parent
    last_package = candidate
    for _ in range(10):
        search_roots.append(candidate)
        if (candidate / "__init__.py").exists():
            last_package = candidate
        parent = candidate.parent
        if parent == candidate:
            break
        candidate = parent
    # The install root is the parent of the top-level package directory
    install_root = last_package.parent
    if install_root not in search_roots:
        search_roots.append(install_root)
    search_roots.append(Path.cwd())

    imported_names: dict[str, tuple[str, str]] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            for alias in node.names:
                imported_names[alias.asname or alias.name] = (node.module, alias.name)

    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            value_str = _extract_string_value(node.value)
            if value_str is None:
                continue
            for target in node.targets:
                if isinstance(target, ast.Name) and _looks_like_system_prompt(
                    target.id, value_str
                ):
                    results.append((target.id, value_str))
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            value_str = _extract_string_value(node.value)
            if (
                value_str
                and isinstance(node.target, ast.Name)
                and _looks_like_system_prompt(node.target.id, value_str)
            ):
                results.append((node.target.id, value_str))

    for local_name, (module, attr) in imported_names.items():
        if not _looks_like_system_prompt(local_name, "x" * 31):
            continue
        import_file = _resolve_import_to_file(module, search_roots)
        if import_file is None:
            continue
        try:
            sub_tree = ast.parse(
                import_file.read_text(encoding="utf-8"), filename=str(import_file)
            )
        except SyntaxError:
            continue
        for sub_node in ast.walk(sub_tree):
            if isinstance(sub_node, ast.Assign):
                val = _extract_string_value(sub_node.value)
                if val is None:
                    continue
                for tgt in sub_node.targets:
                    if (
                        isinstance(tgt, ast.Name)
                        and tgt.id == attr
                        and _looks_like_system_prompt(attr, val)
                    ):
                        results.append((f"{module}.{attr}", val))

    return results


# ---------------------------------------------------------------------------
# AST-based target listing (no-import mode)
# ---------------------------------------------------------------------------

_AGENT_METHOD_NAMES = {"run", "invoke", "kickoff", "answer", "chat", "query", "execute", "call"}


def list_scan_targets(source_path: Path) -> list[dict]:
    """List potential scan targets in a Python source file using AST analysis.

    Returns a list of dicts with keys: name, kind, line.
    kind is one of: 'function', 'async_function', 'class', 'class_with_agent_method'.
    """
    try:
        tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
    except SyntaxError:
        return []

    targets = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            targets.append({
                "name": node.name,
                "kind": "async_function" if isinstance(node, ast.AsyncFunctionDef) else "function",
                "line": node.lineno,
            })
        elif isinstance(node, ast.ClassDef):
            methods = [
                n.name
                for n in ast.walk(node)
                if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
                and n.name in _AGENT_METHOD_NAMES
            ]
            kind = "class_with_agent_method" if methods else "class"
            targets.append({
                "name": node.name,
                "kind": kind,
                "line": node.lineno,
                "methods": methods,
            })
    return targets


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
        file_guess = Path(*module_path.split(".")).with_suffix(".py")
        tip = ""
        if file_guess.exists():
            tip = (
                f"\n\nThe module has uninstalled dependencies. "
                f"To scan its security posture without installing them:\n"
                f"  [bold]checkagent wrap {file_guess} --extract-prompt[/bold]\n"
                f"  [bold]checkagent scan --system-prompt <extracted_prompt.txt>[/bold]"
            )
        raise click.ClickException(
            f"Cannot import module '{module_path}': {exc}{tip}"
        ) from exc

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
@click.option(
    "--extract-prompt",
    is_flag=True,
    help=(
        "Extract system prompts from the source file using AST analysis — "
        "no imports required. TARGET must be a .py file path."
    ),
)
@click.option(
    "--list-targets",
    is_flag=True,
    help=(
        "List callable functions and classes in a .py file that could be scan targets — "
        "no imports required. TARGET must be a .py file path."
    ),
)
def wrap_cmd(
    target: str, output: str, force: bool, extract_prompt: bool, list_targets: bool
) -> None:
    """Generate a wrapper module for an agent object.

    TARGET is a 'module:name' or 'module.name' reference to a Python
    object.  CheckAgent inspects the object and writes a wrapper file
    exposing ``checkagent_target(prompt)`` that ``checkagent scan`` can
    use directly.

    Use --extract-prompt to extract system prompts without importing the file.
    Use --list-targets to discover callable functions and classes in a file.

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
        checkagent wrap agents/qa/agent.py --extract-prompt
        checkagent wrap agents/qa/agent.py --list-targets
    """
    if list_targets:
        source_path = Path(target)
        if not source_path.exists():
            raise click.ClickException(f"File not found: {source_path}")
        if source_path.suffix != ".py":
            raise click.ClickException(
                f"--list-targets requires a .py file path, got: {target}"
            )
        found = list_scan_targets(source_path)
        if not found:
            console.print(f"[yellow]No scan targets found in {source_path}.[/yellow]")
            return
        module_name = source_path.stem
        console.print(f"\n[bold]Scan targets in [cyan]{source_path.name}[/cyan]:[/bold]\n")
        for item in found:
            kind = item["kind"]
            name = item["name"]
            line = item["line"]
            if kind == "async_function":
                label = "[green]async fn[/green]"
            elif kind == "function":
                label = "[blue]function[/blue]"
            elif kind == "class_with_agent_method":
                methods = item.get("methods", [])
                label = f"[magenta]class[/magenta] (has .{'/'.join(methods[:2])}())"
            else:
                label = "[dim]class[/dim]"
            console.print(f"  {label:30s} [bold]{name}[/bold] [dim](line {line})[/dim]")
            console.print(
                f"            [dim]checkagent scan {module_name}:{name}[/dim]"
            )
        return

    if extract_prompt:
        source_path = Path(target)
        if not source_path.exists():
            raise click.ClickException(f"File not found: {source_path}")
        if source_path.suffix != ".py":
            raise click.ClickException(
                f"--extract-prompt requires a .py file path, got: {target}"
            )
        prompts = extract_system_prompts(source_path)
        if not prompts:
            console.print(
                f"[yellow]No system prompts found in {source_path}.[/yellow]\n"
                "Looked for string assignments to variables containing: "
                + ", ".join(_PROMPT_VAR_KEYWORDS)
            )
            console.print(
                "\nIf the prompt is in a separate file, pass that file instead:\n"
                "  [bold]checkagent wrap <prompts_file.py> --extract-prompt[/bold]"
            )
            return

        for var_name, prompt_text in prompts:
            console.print(f"\n[bold cyan]{var_name}[/bold cyan] ({len(prompt_text)} chars):")
            preview = prompt_text[:200].replace("\n", " ")
            if len(prompt_text) > 200:
                preview += "..."
            console.print(f"  [dim]{preview}[/dim]")

            prompt_file = Path(f"{var_name.replace('.', '_')}.txt")
            if prompt_file.exists() and not force:
                console.print(
                    f"\n  [yellow]'{prompt_file}' already exists. "
                    "Use --force to overwrite.[/yellow]"
                )
            else:
                prompt_file.write_text(prompt_text, encoding="utf-8")
                console.print(f"\n  [green]✓[/green] Saved to [cyan]{prompt_file}[/cyan]")

            console.print(
                f"\n  Scan it: [bold]checkagent scan "
                f"--system-prompt {prompt_file} --model claude-haiku-4-5-20251001[/bold]"
            )
            console.print(
                f"  Zero-config: [bold]checkagent scan "
                f"--system-prompt {prompt_file} --model claude-code[/bold]"
            )
        return

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
