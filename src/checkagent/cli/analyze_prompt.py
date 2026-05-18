"""CLI command: checkagent analyze-prompt

Zero-setup, LLM-free static analysis of a system prompt.  Checks the
text for eight security best practices that map directly to the probe
categories used by ``checkagent scan``.

Optional ``--llm MODEL`` flag adds a semantic verification pass on any
checks that pattern matching could not confirm — useful for system prompts
that use non-canonical phrasing.
"""

from __future__ import annotations

import asyncio
import json
import sys

import click
from rich import box
from rich.console import Console
from rich.markup import escape as rich_escape
from rich.table import Table

from checkagent.safety.prompt_analyzer import (
    PromptAnalysisResult,
    PromptAnalyzer,
    PromptCheck,
)

_console = Console()

# ---------------------------------------------------------------------------
# LLM semantic verification
# ---------------------------------------------------------------------------

_LLM_SYSTEM = """\
You are a security reviewer auditing a system prompt for AI agents.

Your task: determine whether a specific security control is present in the
system prompt, even if phrased unusually or implicitly.

Respond ONLY with valid JSON — no markdown, no code fences:
{"present": true, "evidence": "brief quote or description (max 80 chars)"}

Definitions:
- present=true  — the control is explicitly OR implicitly addressed
- present=false — the control is completely absent

Be liberal in interpretation: if the intent is clear, mark it present.
Example: "Focus only on customer service questions" satisfies scope_boundary
even without the words 'must not' or 'restricted to'."""


async def _verify_one_check(
    prompt_text: str,
    check: PromptCheck,
    model: str,
) -> tuple[bool, str]:
    """Ask an LLM whether *check* is satisfied by *prompt_text*.

    Returns (present, evidence).  Falls back to (False, "") on any error.
    """
    from checkagent.core.llm_call import call_llm

    user = (
        f"Security control to check:\n"
        f"  Name: {check.name}\n"
        f"  Description: {check.description}\n\n"
        f"System prompt:\n---\n{prompt_text[:2000]}\n---\n\n"
        f"Is this control present? Reply only with JSON."
    )
    try:
        raw = await call_llm(model, _LLM_SYSTEM, user, max_tokens=120, temperature=0)
        raw = raw.strip()
        # Strip markdown code fences if the model ignores our instruction
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        data = json.loads(raw)
        present = bool(data.get("present", False))
        evidence = str(data.get("evidence", ""))[:120]
        return present, evidence
    except Exception:  # noqa: BLE001
        return False, ""


async def _llm_verify_failing_checks(
    prompt_text: str,
    failing_checks: list[PromptCheck],
    model: str,
) -> dict[str, tuple[bool, str]]:
    """Return {check_id: (llm_passed, evidence)} for each failing check."""
    tasks = [
        _verify_one_check(prompt_text, check, model)
        for check in failing_checks
    ]
    results = await asyncio.gather(*tasks)
    return {
        check.id: result
        for check, result in zip(failing_checks, results, strict=True)
    }


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------


def _score_bar(score: float, width: int = 20) -> str:
    """Return a unicode progress bar string for *score* (0.0–1.0)."""
    filled = round(score * width)
    empty = width - filled
    return "█" * filled + "░" * empty


def _severity_color(severity: str) -> str:
    return {"high": "red", "medium": "yellow", "low": "cyan"}.get(severity, "white")


def _render_result(
    result: PromptAnalysisResult,
    prompt_preview: str,
    llm_verified: dict[str, tuple[bool, str]] | None = None,
    llm_model: str | None = None,
) -> None:
    llm_verified = llm_verified or {}

    # Recompute score including LLM-verified passes
    llm_pass_ids = {cid for cid, (passed, _) in llm_verified.items() if passed}
    total = result.total_count
    passed_count = result.passed_count + len(llm_pass_ids)
    score = passed_count / total if total > 0 else 0.0
    pct = int(score * 100)
    bar = _score_bar(score)

    # Headline
    _console.print()
    _console.print("[bold]System Prompt Analysis[/bold]", style="white")
    _console.print("─" * 46)
    if prompt_preview:
        preview = prompt_preview[:72] + ("…" if len(prompt_preview) > 72 else "")
        _console.print(f"[dim]Prompt:[/dim] {preview}")
        _console.print()

    # Score bar
    if pct >= 75:
        score_style = "green"
    elif pct >= 50:
        score_style = "yellow"
    else:
        score_style = "red"

    counts = f"{passed_count}/{total} ({pct}%)"
    _console.print(
        f"Score: [{score_style}]{counts}[/{score_style}]  "
        f"[{score_style}]{bar}[/{score_style}]"
    )
    if llm_model and llm_pass_ids:
        _console.print(
            f"[dim]Pattern: {result.passed_count}/{total} "
            f"· LLM ({llm_model}): +{len(llm_pass_ids)} semantic[/dim]"
        )
    _console.print()

    # Per-check table
    table = Table(box=box.SIMPLE, show_header=True, header_style="bold dim")
    table.add_column("Check", style="white", min_width=22)
    table.add_column("Status", min_width=14)
    table.add_column("Severity", min_width=8)
    table.add_column("Note", style="dim")

    for cr in result.check_results:
        llm_result = llm_verified.get(cr.check.id)
        llm_passed = llm_result[0] if llm_result else False
        llm_evidence = llm_result[1] if llm_result else ""

        if cr.passed:
            status = "[green]✓ PRESENT[/green]"
            if cr.evidence and len(cr.evidence) > 48:
                note = f'Found: "{cr.evidence[:48]}…"'
            elif cr.evidence:
                note = f'Found: "{cr.evidence}"'
            else:
                note = ""
        elif llm_passed:
            status = "[yellow]~ PRESENT (LLM)[/yellow]"
            note = f'LLM: "{llm_evidence[:60]}"' if llm_evidence else "LLM verified"
        else:
            status = "[red]✗ MISSING[/red]"
            note = ""

        sev_color = _severity_color(cr.check.severity)
        severity = f"[{sev_color}]{cr.check.severity.upper()}[/{sev_color}]"
        table.add_row(cr.check.name, status, severity, note)

    _console.print(table)

    # Recommendations — only for checks missing from BOTH pattern and LLM
    still_missing = [
        cr for cr in result.check_results
        if not cr.passed and cr.check.id not in llm_pass_ids
    ]
    recs = [cr.check.recommendation for cr in still_missing if cr.check.recommendation]
    if recs:
        _console.print("[bold]Recommendations[/bold]")
        for i, rec in enumerate(recs, 1):
            _console.print(f"  [dim]{i}.[/dim] {rich_escape(rec)}")
        _console.print()
    elif passed_count == total:
        _console.print(
            "[green]All checks passed — your system prompt covers all eight "
            "security controls.[/green]"
        )
        _console.print()

    # Footer
    if llm_model:
        _console.print(
            f"[dim]This is a static + LLM-assisted ({llm_model}) guidelines check, "
            "not a security guarantee. "
            "Run [bold]checkagent scan[/bold] for dynamic probe testing.[/dim]"
        )
    else:
        _console.print(
            "[dim]This is a static guidelines check, not a security guarantee. "
            "Run [bold]checkagent scan[/bold] for dynamic probe testing.[/dim]"
        )
    _console.print()


@click.command("analyze-prompt")
@click.argument("prompt_source", metavar="PROMPT_OR_FILE", default="-")
@click.option("--json", "output_json", is_flag=True, default=False, help="Output results as JSON.")
@click.option(
    "--llm",
    "llm_model",
    default=None,
    metavar="MODEL",
    help=(
        "Use an LLM for semantic verification of failing checks. "
        "More accurate than pattern matching for non-canonical phrasing. "
        "Examples: gpt-4o-mini, claude-haiku-4-5-20251001"
    ),
)
def analyze_prompt_cmd(prompt_source: str, output_json: bool, llm_model: str | None) -> None:
    """Analyze a system prompt for security best practices.

    PROMPT_OR_FILE can be:

    \b
      - A literal string: checkagent analyze-prompt "You are a helpful assistant."
      - A file path:      checkagent analyze-prompt system_prompt.txt
      - stdin (default):  cat prompt.txt | checkagent analyze-prompt

    Checks the prompt text for eight security controls (injection guard,
    scope boundary, confidentiality, refusal behavior, PII handling,
    data scope, role clarity, escalation path) and reports which are
    present or missing.

    This is a zero-setup, LLM-free check — no API key required.
    Add --llm gpt-4o-mini for semantic verification of ambiguous prompts.
    """
    # Resolve prompt text
    prompt_text: str
    if prompt_source == "-":
        if sys.stdin.isatty():
            raise click.UsageError(
                "No prompt provided. Pass a string, a file path, or pipe via stdin.\n"
                "  checkagent analyze-prompt \"You are a helpful assistant.\"\n"
                "  checkagent analyze-prompt system_prompt.txt\n"
                "  cat prompt.txt | checkagent analyze-prompt"
            )
        prompt_text = sys.stdin.read()
    else:
        # Try as file path first, fall back to treating as literal string
        import pathlib

        p = pathlib.Path(prompt_source)
        try:
            is_file = p.exists() and p.is_file()
        except OSError:
            # Path too long or invalid as a filesystem path — treat as literal
            is_file = False

        if is_file:
            prompt_text = p.read_text(encoding="utf-8")
        else:
            _looks_like_path = (
                "/" in prompt_source
                or "\\" in prompt_source
                or prompt_source.endswith((
                    ".txt", ".md", ".prompt", ".py", ".yaml",
                    ".yml", ".json", ".toml", ".cfg", ".conf",
                ))
            )
            if _looks_like_path:
                raise click.UsageError(
                    f"File not found: {prompt_source}\n"
                    "If this is a literal prompt string, it should not look like a file path."
                )
            prompt_text = prompt_source

    prompt_text = prompt_text.strip()
    if not prompt_text:
        raise click.UsageError("Prompt is empty.")

    # Validate model name early (before doing pattern analysis)
    if llm_model:
        from checkagent.core.llm_call import detect_provider
        detect_provider(llm_model, param_hint="--llm")

    analyzer = PromptAnalyzer()
    result = analyzer.analyze(prompt_text)

    # LLM semantic verification pass — runs on checks that pattern matching missed
    llm_verified: dict[str, tuple[bool, str]] = {}
    if llm_model:
        failing_checks = [cr.check for cr in result.check_results if not cr.passed]
        if failing_checks:
            if not output_json:
                _console.print(
                    f"\n[dim]Running LLM verification ({llm_model}) "
                    f"on {len(failing_checks)} unconfirmed check(s)…[/dim]"
                )
            llm_verified = asyncio.run(
                _llm_verify_failing_checks(prompt_text, failing_checks, llm_model)
            )

    # Recompute effective missing_high considering LLM results
    llm_pass_ids = {cid for cid, (passed, _) in llm_verified.items() if passed}
    still_missing_high = [
        cr for cr in result.check_results
        if not cr.passed
        and cr.check.id not in llm_pass_ids
        and cr.check.severity == "high"
    ]

    if output_json:
        checks_out = []
        for cr in result.check_results:
            llm_result = llm_verified.get(cr.check.id)
            llm_passed = llm_result[0] if llm_result else None
            llm_evidence = llm_result[1] if llm_result else None
            effective_passed = cr.passed or bool(llm_passed)
            checks_out.append({
                "id": cr.check.id,
                "name": cr.check.name,
                "passed": effective_passed,
                "pattern_passed": cr.passed,
                "llm_passed": llm_passed,
                "severity": cr.check.severity,
                "evidence": cr.evidence or llm_evidence,
                "recommendation": cr.check.recommendation if not effective_passed else None,
            })
        llm_pass_count = len(llm_pass_ids)
        total = result.total_count
        effective_passed_count = result.passed_count + llm_pass_count
        data = {
            "score": round(effective_passed_count / total, 4) if total else 1.0,
            "passed_count": effective_passed_count,
            "total_count": total,
            "pattern_passed_count": result.passed_count,
            "llm_verified_count": llm_pass_count if llm_model else None,
            "llm_model": llm_model,
            "checks": checks_out,
        }
        click.echo(json.dumps(data, indent=2))
    else:
        _render_result(result, prompt_text, llm_verified=llm_verified, llm_model=llm_model)

    # Exit with non-zero if any HIGH checks are still missing after LLM verification
    if still_missing_high:
        sys.exit(1)
