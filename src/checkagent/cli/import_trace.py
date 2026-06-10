"""CLI command: checkagent import-trace

Import production traces and generate regression test cases.

Requirements: F6.2, F8.9
"""

from __future__ import annotations

from pathlib import Path

import click
from rich.console import Console
from rich.table import Table

from checkagent.trace_import.json_importer import JsonFileImporter
from checkagent.trace_import.langfuse_importer import LangfuseAPIImporter
from checkagent.trace_import.otel_importer import OtelJsonImporter
from checkagent.trace_import.phoenix_importer import PhoenixAPIImporter
from checkagent.trace_import.pii import PiiScrubber
from checkagent.trace_import.testcase_gen import (
    export_dataset_json,
    generate_test_cases,
)

console = Console()

_FILE_SOURCE_MAP = {
    "json": JsonFileImporter,
    "jsonl": JsonFileImporter,
    "otel": OtelJsonImporter,
}

_API_SOURCES = {"langfuse", "phoenix"}


@click.command("import-trace")
@click.argument("file", type=click.Path(), required=False, default=None)
@click.option(
    "--source",
    type=click.Choice(["json", "jsonl", "otel", "langfuse", "phoenix"], case_sensitive=False),
    default=None,
    help=(
        "Source format. Auto-detected from file extension if not specified. "
        "Use 'langfuse' or 'phoenix' to fetch directly from the API (no file needed)."
    ),
)
@click.option(
    "--api-url",
    default=None,
    metavar="URL",
    help=(
        "API base URL for langfuse/phoenix sources. "
        "Defaults: Langfuse cloud (https://cloud.langfuse.com), "
        "Phoenix local (http://localhost:6006)."
    ),
)
@click.option(
    "--api-key",
    default=None,
    metavar="KEY",
    help=(
        "API credentials. For Langfuse, use 'public_key:secret_key'. "
        "For Phoenix, provide the API key. "
        "Can also be set via LANGFUSE_PUBLIC_KEY/LANGFUSE_SECRET_KEY or PHOENIX_API_KEY env vars."
    ),
)
@click.option(
    "--output",
    "-o",
    type=click.Path(),
    default=None,
    help="Output path for generated dataset JSON. Defaults to datasets/imported/<name>.json.",
)
@click.option(
    "--filter-status",
    type=click.Choice(["error", "success"]),
    default=None,
    help="Filter traces by status.",
)
@click.option(
    "--limit",
    type=int,
    default=None,
    help="Max number of traces to import.",
)
@click.option(
    "--no-pii-scrub",
    is_flag=True,
    default=False,
    help="Disable PII scrubbing (not recommended for production traces).",
)
@click.option(
    "--no-safety-check",
    is_flag=True,
    default=False,
    help="Skip safety screening of trace outputs. Not recommended.",
)
@click.option(
    "--dataset-name",
    default=None,
    help="Name for the generated dataset. Defaults to filename stem or source name.",
)
@click.option(
    "--tag",
    multiple=True,
    help="Additional tags to add to all generated test cases.",
)
def import_trace_cmd(
    file: str | None,
    source: str | None,
    api_url: str | None,
    api_key: str | None,
    output: str | None,
    filter_status: str | None,
    limit: int | None,
    no_pii_scrub: bool,
    no_safety_check: bool,
    dataset_name: str | None,
    tag: tuple[str, ...],
) -> None:
    """Import production traces and generate test cases.

    FILE is the path to a trace file (JSON, JSONL, or OTLP JSON).
    For Langfuse and Phoenix, omit FILE and use --source with optional --api-url.

    By default, trace outputs are screened for security issues (PII leakage,
    prompt injection, data enumeration). Flagged traces are tagged
    'needs-review' and their outputs are NOT encoded as expected assertions,
    preventing vulnerabilities from becoming regression tests.

    Examples:

        checkagent import-trace traces.jsonl

        checkagent import-trace otel-export.json --source otel --filter-status error

        checkagent import-trace prod-traces.json -o tests/datasets/regression.json

        checkagent import-trace --source langfuse --limit 100

        checkagent import-trace --source phoenix --api-url http://localhost:6006

        checkagent import-trace --source langfuse --api-key pk-lf-...:sk-lf-...
    """
    filters = {}
    if filter_status:
        filters["status"] = filter_status

    # API-based import (no file needed)
    if source in _API_SOURCES:
        importer, display_name = _build_api_importer(source, api_url, api_key)
        console.print(
            f"[bold]Fetching traces from[/bold] [cyan]{display_name}[/cyan]"
            f" [dim]({source})[/dim]"
        )
        try:
            runs = importer.import_traces(filters=filters or None, limit=limit)
        except RuntimeError as exc:
            raise click.ClickException(str(exc)) from exc
        name = dataset_name or source
    else:
        # File-based import
        if file is None:
            raise click.UsageError(
                "Provide a FILE argument, or use --source langfuse/phoenix for API import."
            )
        if not Path(file).exists():
            raise click.ClickException(f"File not found: {file}")

        file_path = Path(file)

        if source is None:
            source = _detect_source(file_path)

        importer_cls = _FILE_SOURCE_MAP[source]
        importer = importer_cls()

        console.print(
            f"[bold]Importing traces from[/bold] {file_path.name}"
            f" [dim]({source} format)[/dim]"
        )

        runs = importer.import_traces(str(file_path), filters=filters or None, limit=limit)
        name = dataset_name or file_path.stem

    if not runs:
        console.print("[yellow]No traces found matching the specified criteria.[/yellow]")
        return

    console.print(f"[green]Found {len(runs)} traces[/green]")

    scrubber = None if no_pii_scrub else PiiScrubber()

    dataset, screening = generate_test_cases(
        runs,
        scrub_pii=not no_pii_scrub,
        pii_scrubber=scrubber,
        dataset_name=name,
        tags=list(tag) if tag else None,
        safety_check=not no_safety_check,
    )

    if output is None:
        output_dir = Path("datasets") / "imported"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{name}.json"
    else:
        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

    export_dataset_json(dataset, str(output_path))

    table = Table(title="Import Summary")
    table.add_column("Metric", style="bold")
    table.add_column("Value")
    table.add_row("Traces imported", str(len(runs)))
    table.add_row("Test cases generated", str(len(dataset.cases)))
    table.add_row("PII scrubbing", "disabled" if no_pii_scrub else "enabled")
    table.add_row(
        "Safety screening",
        "disabled" if no_safety_check else "enabled",
    )
    table.add_row("Output file", str(output_path))

    error_count = sum(1 for r in runs if r.error)
    if error_count:
        table.add_row("Error traces", str(error_count))

    tool_count = sum(len(r.tool_calls) for r in runs)
    if tool_count:
        table.add_row("Total tool calls", str(tool_count))

    if screening.flagged_count:
        table.add_row(
            "[yellow]Flagged traces[/yellow]",
            f"[yellow]{screening.flagged_count}[/yellow]",
        )

    console.print(table)

    if screening.flagged_count:
        console.print(
            f"\n[yellow bold]Warning:[/yellow bold] {screening.flagged_count} "
            f"trace(s) contain potential security issues and are tagged "
            f"'needs-review'."
        )
        console.print(
            "[yellow]Their outputs were NOT encoded as expected assertions "
            "to prevent vulnerabilities from becoming regression tests.[/yellow]"
        )
        for trace_id, findings in screening.findings_by_trace.items():
            categories = {
                f.category.value
                if hasattr(f.category, "value")
                else str(f.category)
                for f in findings
            }
            console.print(f"  [dim]{trace_id}:[/dim] {', '.join(sorted(categories))}")
        console.print()

    console.print(
        f"\n[green]✓[/green] Dataset written to [bold]{output_path}[/bold]"
    )
    console.print(
        "[dim]Review the generated test cases and add them to your test suite.[/dim]"
    )


def _build_api_importer(
    source: str,
    api_url: str | None,
    api_key: str | None,
) -> tuple[object, str]:
    """Build an API-based importer and return (importer, display_name)."""
    if source == "langfuse":
        public_key: str | None = None
        secret_key: str | None = None
        if api_key and ":" in api_key:
            public_key, _, secret_key = api_key.partition(":")
        elif api_key:
            public_key = api_key
        host = api_url or "https://cloud.langfuse.com"
        importer = LangfuseAPIImporter(host=host, public_key=public_key, secret_key=secret_key)
        return importer, host

    if source == "phoenix":
        host = api_url or "http://localhost:6006"
        importer = PhoenixAPIImporter(host=host, api_key=api_key or None)
        return importer, host

    raise ValueError(f"Unknown API source: {source}")


def _detect_source(path: Path) -> str:
    """Auto-detect source format from file extension and content."""
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        return "jsonl"

    # For .json files, peek at content to detect OTLP format
    if suffix == ".json":
        try:
            import json

            with open(path) as f:
                data = json.load(f)
            if isinstance(data, dict) and (
                "resourceSpans" in data
                or "resource_spans" in data
                or (
                    isinstance(data.get("spans"), list)
                    and data["spans"]
                    and "traceId" in data["spans"][0]
                )
            ):
                return "otel"
        except (json.JSONDecodeError, KeyError, IndexError):
            pass

    return "json"
