"""CLI command: checkagent ablate-prompt

Prompt ablation analysis — systematically remove each sentence from a
system prompt to identify which instructions are load-bearing for safety.

Like ablation studies in ML research, applied to prompt engineering.
Answers: "Which parts of my prompt actually matter for security?"

Zero-cost, no API keys required.
"""

from __future__ import annotations

import json
import re
import sys

import click
from rich import box
from rich.console import Console
from rich.markup import escape as rich_escape
from rich.table import Table

from checkagent.safety.prompt_analyzer import PromptAnalyzer

_console = Console()


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences, preserving meaningful chunks."""
    parts = re.split(r'(?<=[.!?])\s+', text.strip())
    sentences = [p.strip() for p in parts if len(p.strip()) > 5]
    if len(sentences) <= 1 and "\n" in text:
        lines = [ln.strip() for ln in text.strip().splitlines() if len(ln.strip()) > 5]
        if len(lines) > len(sentences):
            sentences = lines
    return sentences


def _ablation_analysis(
    prompt: str,
    analyzer: PromptAnalyzer,
) -> dict:
    """Run ablation: remove each sentence, measure impact on safety score."""
    baseline = analyzer.analyze(prompt)
    sentences = _split_sentences(prompt)

    if len(sentences) < 2:
        return {
            "baseline_score": baseline.score,
            "baseline_passed": baseline.passed_count,
            "baseline_total": baseline.total_count,
            "sentences": [],
            "load_bearing": [],
            "redundant": [],
            "error": "Prompt has fewer than 2 sentences — ablation requires multiple sentences.",
        }

    results = []
    for i, sentence in enumerate(sentences):
        ablated = " ".join(s for j, s in enumerate(sentences) if j != i)
        ablated_result = analyzer.analyze(ablated)

        score_delta = ablated_result.score - baseline.score
        checks_lost = []
        checks_gained = []

        for orig_cr, abl_cr in zip(
            baseline.check_results, ablated_result.check_results, strict=True
        ):
            if orig_cr.passed and not abl_cr.passed:
                checks_lost.append(orig_cr.check.id)
            elif not orig_cr.passed and abl_cr.passed:
                checks_gained.append(abl_cr.check.id)

        results.append({
            "index": i,
            "sentence": sentence,
            "score_delta": round(score_delta, 4),
            "ablated_score": round(ablated_result.score, 4),
            "ablated_passed": ablated_result.passed_count,
            "checks_lost": checks_lost,
            "checks_gained": checks_gained,
            "is_load_bearing": len(checks_lost) > 0,
        })

    load_bearing = [r for r in results if r["is_load_bearing"]]
    redundant = [r for r in results if not r["is_load_bearing"] and r["score_delta"] == 0.0]

    check_coverage: dict[str, list[int]] = {}
    for r in results:
        for check_id in r["checks_lost"]:
            check_coverage.setdefault(check_id, []).append(r["index"])

    single_point = [
        {"check": cid, "sentence_index": idxs[0], "sentence": sentences[idxs[0]]}
        for cid, idxs in check_coverage.items()
        if len(idxs) == 1
    ]

    return {
        "baseline_score": round(baseline.score, 4),
        "baseline_passed": baseline.passed_count,
        "baseline_total": baseline.total_count,
        "sentence_count": len(sentences),
        "sentences": results,
        "load_bearing": load_bearing,
        "redundant": redundant,
        "single_points_of_failure": single_point,
        "check_coverage": {
            cid: len(idxs) for cid, idxs in check_coverage.items()
        },
    }


def _render_ablation(data: dict) -> None:
    """Render ablation results to the terminal."""
    _console.print()
    _console.print("[bold]Prompt Ablation Analysis[/bold]", style="white")
    _console.print("─" * 50)
    _console.print(
        f"[dim]Baseline:[/dim] {data['baseline_passed']}/{data['baseline_total']} "
        f"checks passing ({int(data['baseline_score'] * 100)}%)"
    )
    _console.print(
        f"[dim]Sentences:[/dim] {data.get('sentence_count', len(data['sentences']))}"
    )
    _console.print()

    if data.get("error"):
        _console.print(f"[yellow]{data['error']}[/yellow]")
        return

    table = Table(box=box.SIMPLE, show_header=True, header_style="bold dim")
    table.add_column("#", style="dim", width=3)
    table.add_column("Sentence", max_width=50)
    table.add_column("Impact", min_width=12)
    table.add_column("Checks Lost", style="red")

    for r in data["sentences"]:
        idx = str(r["index"] + 1)
        sentence = r["sentence"]
        if len(sentence) > 50:
            sentence = sentence[:47] + "..."

        if r["checks_lost"]:
            delta_str = f"[red]−{len(r['checks_lost'])} check(s)[/red]"
        elif r["score_delta"] < 0:
            delta_str = f"[yellow]{r['score_delta']:+.0%}[/yellow]"
        else:
            delta_str = "[dim]no impact[/dim]"

        lost = ", ".join(r["checks_lost"]) if r["checks_lost"] else "[dim]—[/dim]"

        table.add_row(idx, rich_escape(sentence), delta_str, lost)

    _console.print(table)
    _console.print()

    spofs = data.get("single_points_of_failure", [])
    if spofs:
        _console.print("[bold red]Single Points of Failure[/bold red]")
        _console.print(
            "[dim]These checks depend on exactly one sentence. "
            "If that sentence is removed or rephrased, the check fails.[/dim]"
        )
        for spof in spofs:
            sentence = spof["sentence"]
            if len(sentence) > 60:
                sentence = sentence[:57] + "..."
            _console.print(
                f"  [red]•[/red] [bold]{spof['check']}[/bold] → "
                f"sentence {spof['sentence_index'] + 1}: "
                f"\"{rich_escape(sentence)}\""
            )
        _console.print()

    coverage = data.get("check_coverage", {})
    if coverage:
        _console.print("[bold]Check Coverage Depth[/bold]")
        _console.print(
            "[dim]How many sentences contribute to each security check. "
            "Higher = more resilient.[/dim]"
        )
        for cid, count in sorted(coverage.items(), key=lambda x: x[1]):
            if count == 1:
                bar = "[red]█[/red]"
                label = "[red]fragile[/red]"
            elif count == 2:
                bar = "[yellow]██[/yellow]"
                label = "[yellow]moderate[/yellow]"
            else:
                bar = f"[green]{'█' * min(count, 8)}[/green]"
                label = "[green]resilient[/green]"
            _console.print(f"  {cid:<20s} {bar} ({count}) {label}")
        _console.print()

    load_bearing_count = len(data.get("load_bearing", []))
    redundant_count = len(data.get("redundant", []))
    total = data.get("sentence_count", 0)
    if total:
        _console.print(
            f"[dim]Summary:[/dim] {load_bearing_count}/{total} sentences are "
            f"load-bearing, {redundant_count}/{total} have no safety impact"
        )
    _console.print()


@click.command("ablate-prompt")
@click.argument("prompt_source", metavar="PROMPT_OR_FILE", default="-")
@click.option("--json", "output_json", is_flag=True, default=False, help="Output results as JSON.")
def ablate_prompt_cmd(prompt_source: str, output_json: bool) -> None:
    """Ablation analysis: find which prompt sentences are load-bearing for safety.

    Systematically removes each sentence from the prompt and measures
    the impact on the safety score. Identifies:

    \b
      - Load-bearing sentences (removing them drops the score)
      - Redundant sentences (removing them has no effect)
      - Single points of failure (checks that depend on only one sentence)

    This is like ablation studies in ML — applied to prompt engineering.

    \b
    PROMPT_OR_FILE can be:
      - A literal string: checkagent ablate-prompt "You are a helpful..."
      - A file path:      checkagent ablate-prompt system_prompt.txt
      - stdin (default):  cat prompt.txt | checkagent ablate-prompt

    No API key required — uses the same static analysis as analyze-prompt.
    """
    import pathlib

    if prompt_source == "-":
        if sys.stdin.isatty():
            raise click.UsageError(
                "No prompt provided. Pass a string, a file path, or pipe via stdin.\n"
                "  checkagent ablate-prompt \"You are a helpful assistant.\"\n"
                "  checkagent ablate-prompt system_prompt.txt\n"
                "  cat prompt.txt | checkagent ablate-prompt"
            )
        prompt_text = sys.stdin.read()
    else:
        p = pathlib.Path(prompt_source)
        try:
            is_file = p.exists() and p.is_file()
        except OSError:
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

    analyzer = PromptAnalyzer()
    data = _ablation_analysis(prompt_text, analyzer)

    if output_json:
        click.echo(json.dumps(data, indent=2))
    else:
        _render_ablation(data)
