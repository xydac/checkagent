"""checkagent probe-list — show all built-in safety probe categories and templates.

Helps users understand what CheckAgent tests before running a full scan.
"""

from __future__ import annotations

import json as json_mod

import click
from rich import box
from rich.console import Console
from rich.markup import escape as _escape
from rich.table import Table

console = Console()

_CATEGORY_DESCRIPTIONS: dict[str, str] = {
    "injection": "Prompt injection attacks that try to override agent instructions",
    "jailbreak": "Encoding tricks and roleplay scenarios to bypass safety rules",
    "pii": "Attempts to extract personal information from the agent",
    "scope": "Requests outside the agent's defined domain or purpose",
    "data_enumeration": "Bulk data disclosure and cross-user data access attempts",
    "groundedness": "Fabrication probes that test factual accuracy and hallucination",
}

# OWASP LLM Top 10 mapping for each category
_OWASP_MAP: dict[str, str] = {
    "injection": "LLM01: Prompt Injection",
    "jailbreak": "LLM01: Prompt Injection",
    "pii": "LLM06: Sensitive Info Disclosure",
    "scope": "LLM01: Prompt Injection",
    "data_enumeration": "LLM06: Sensitive Info Disclosure",
    "groundedness": "LLM09: Misinformation",
}

# Human-readable display names
_CATEGORY_DISPLAY: dict[str, str] = {
    "injection": "prompt_injection",
    "jailbreak": "jailbreak",
    "pii": "pii_leakage",
    "scope": "scope_boundary",
    "data_enumeration": "data_enumeration",
    "groundedness": "groundedness",
}


@click.command("probe-list")
@click.option(
    "--category",
    "filter_category",
    default=None,
    metavar="CATEGORY",
    help="Show probes for a specific category only.",
)
@click.option(
    "--examples",
    "show_examples",
    is_flag=True,
    default=False,
    help="Show 3 example probe inputs per category.",
)
@click.option(
    "--verbose",
    "show_verbose",
    is_flag=True,
    default=False,
    help="Show all probe inputs (full text). Useful for security audits.",
)
@click.option(
    "--json",
    "json_output",
    is_flag=True,
    default=False,
    help="Output probe list as JSON.",
)
def probe_list_cmd(
    filter_category: str | None,
    show_examples: bool,
    show_verbose: bool,
    json_output: bool,
) -> None:
    """List all built-in safety probe categories and templates.

    Shows what CheckAgent tests during a scan, organized by category and
    mapped to the OWASP LLM Top 10. Use --examples to see sample probe inputs,
    or --verbose to see the complete probe text for security audits.

    \b
    Examples:
        checkagent probe-list
        checkagent probe-list --category injection
        checkagent probe-list --examples
        checkagent probe-list --verbose --category pii
        checkagent probe-list --json
    """
    from checkagent.cli.scan import _PROBE_SETS  # noqa: PLC0415

    categories = {}
    for key, probe_iterable in _PROBE_SETS.items():
        probes = list(probe_iterable)
        display = _CATEGORY_DISPLAY.get(key, key)
        if filter_category and filter_category not in (key, display):
            continue

        if show_verbose:
            example_inputs = [p.input for p in probes]
        elif show_examples:
            example_inputs = [p.input for p in probes[:3]]
        else:
            example_inputs = []

        categories[key] = {
            "name": display,
            "count": len(probes),
            "description": _CATEGORY_DESCRIPTIONS.get(key, ""),
            "owasp": _OWASP_MAP.get(key, ""),
            "examples": example_inputs,
            "probes": [
                {"input": p.input, "description": p.description}
                for p in probes
            ] if json_output and show_verbose else [],
        }

    if not categories:
        # Show both short key and display name so users can use either form
        valid = sorted(
            f"{_CATEGORY_DISPLAY.get(k, k)} ({k})"
            for k in _PROBE_SETS
        )
        raise click.UsageError(
            f"Unknown category '{filter_category}'. "
            f"Valid categories: {', '.join(valid)}"
        )

    if json_output:
        total = sum(c["count"] for c in categories.values())
        out: dict = {"total_probes": total, "categories": []}
        for info in categories.values():
            entry: dict = {
                "name": info["name"],
                "count": info["count"],
                "description": info["description"],
                "owasp": info["owasp"],
                "examples": info["examples"],
            }
            if show_verbose:
                entry["probes"] = info["probes"]
            out["categories"].append(entry)
        click.echo(json_mod.dumps(out, indent=2))
        return

    # Terminal output
    total = sum(c["count"] for c in categories.values())
    console.print()
    console.print(f"[bold]Built-in Safety Probes[/bold]  ({total} total)")
    console.print("─" * 60)

    table = Table(box=box.SIMPLE, show_header=True, header_style="bold dim")
    table.add_column("Category", style="white", min_width=22)
    table.add_column("Probes", justify="right", min_width=7)
    table.add_column("OWASP LLM Top 10", style="dim", min_width=28)
    table.add_column("Tests For", style="dim", max_width=40)

    for info in categories.values():
        table.add_row(
            info["name"],
            str(info["count"]),
            info["owasp"],
            info["description"],
        )

    console.print(table)

    if show_verbose or show_examples:
        label = "All Probe Inputs" if show_verbose else "Example Probe Inputs"
        console.print()
        console.print(f"[bold]{label}[/bold]")
        console.print("─" * 60)
        for _key, info in categories.items():
            if info["examples"]:
                count_note = f" ({info['count']} total)" if show_verbose else ""
                console.print(f"\n[cyan]{info['name']}[/cyan]{count_note}")
                for idx, ex in enumerate(info["examples"], 1):
                    if show_verbose:
                        # Show full text with line number for security audits
                        console.print(f"  [dim]{idx:3}.[/dim] {_escape(ex)}")
                    else:
                        preview = ex[:80] + ("…" if len(ex) > 80 else "")
                        console.print(f"  [dim]•[/dim] {_escape(preview)}")

    console.print()
    console.print(
        "[dim]Run [bold]checkagent scan <target>[/bold] "
        "to test your agent against all probes.[/dim]"
    )
    console.print(
        "[dim]Use [bold]--category <name>[/bold] on scan "
        "to run a specific category only.[/dim]"
    )
    console.print()
